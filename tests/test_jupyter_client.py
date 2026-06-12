"""Unit tests for JupyterLabClient — output parsers and the WebSocket
execution paths (kernel death, cancellation, reconnect), driven by a
scripted fake WebSocket. No JupyterLab needed."""

import base64
import io
import json
from unittest.mock import patch

import websocket
from PIL import Image

from jlab_mcp.jupyter_client import JupyterLabClient


def _make_jpeg_b64(width: int, height: int) -> str:
    img = Image.new("RGB", (width, height), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Fake WebSocket plumbing
# ---------------------------------------------------------------------------

def _msg(msg_type, content, parent_msg_id="PARENT"):
    """Build a serialized Jupyter protocol message."""
    return json.dumps({
        "header": {"msg_type": msg_type},
        "msg_type": msg_type,
        "parent_header": {"msg_id": parent_msg_id} if parent_msg_id else {},
        "content": content,
    })


class FakeWS:
    """Scripted WebSocket: replays a list of messages/exceptions.

    Replaces the parent msg_id placeholder 'PARENT' with the msg_id of
    the execute_request it receives, so scripted replies match the filter.
    Answers the kernel_info readiness handshake out-of-band (the reply is
    injected at the front of the script without consuming it).
    """

    def __init__(self, script):
        self.script = list(script)
        self.connected = True
        self.sent = []
        self.closed = False
        self._msg_id = None

    def send(self, raw):
        msg = json.loads(raw)
        if msg["header"]["msg_type"] == "kernel_info_request":
            self.script.insert(0, json.dumps({
                "header": {"msg_type": "kernel_info_reply"},
                "msg_type": "kernel_info_reply",
                "parent_header": {"msg_id": msg["header"]["msg_id"]},
                "content": {},
            }))
            return
        self.sent.append(raw)
        self._msg_id = msg["header"]["msg_id"]

    def settimeout(self, t):
        pass

    def recv(self):
        if not self.script:
            raise AssertionError("FakeWS script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item.replace("PARENT", self._msg_id or "")

    def close(self):
        self.closed = True
        self.connected = False


class NeverReadyWS(FakeWS):
    """A socket whose kernel never answers — simulates the window where
    the WS upgrade succeeded but the kernel's ZMQ channels aren't bridged."""

    def __init__(self):
        super().__init__([])

    def send(self, raw):
        pass  # everything sent here vanishes

    def recv(self):
        raise websocket.WebSocketTimeoutException()


def _client_with_ws(kernel_id, ws):
    client = JupyterLabClient("localhost", 8888, "tok")
    client._ws_cache[kernel_id] = ws
    return client


IDLE = _msg("status", {"execution_state": "idle"})
STREAM = _msg("stream", {"name": "stdout", "text": "hello\n"})
STDERR = _msg("stream", {"name": "stderr", "text": "warn\n"})
RESULT = _msg("execute_result", {"data": {"text/plain": "42"}})
DEAD_BROADCAST = _msg(
    "status", {"execution_state": "restarting"}, parent_msg_id=None
)


class TestExecuteOnFakeWS:
    def test_collects_outputs_until_idle(self):
        ws = FakeWS([STREAM, STDERR, RESULT, IDLE])
        client = _client_with_ws("k1", ws)
        outputs = client.execute_code("k1", "print('hello')")
        assert outputs == [
            {"type": "text", "content": "hello\n", "name": "stdout"},
            {"type": "text", "content": "warn\n", "name": "stderr"},
            {"type": "text", "content": "42", "result": True},
        ]
        # WS stays cached for reuse
        assert client._ws_cache["k1"] is ws

    def test_death_broadcast_returns_kernel_died_and_evicts_ws(self):
        """A restarting/dead broadcast (no parent header) must end the
        execution AND evict the cached WS so buffered death messages
        can't poison the next execution."""
        ws = FakeWS([DEAD_BROADCAST])
        client = _client_with_ws("k1", ws)
        outputs = client.execute_code("k1", "boom()")
        assert outputs[-1]["ename"] == "KernelDied"
        assert "k1" not in client._ws_cache
        assert ws.closed

    def test_connection_closed_mid_recv_reports_kernel_died(self):
        ws = FakeWS([websocket.WebSocketConnectionClosedException()])
        client = _client_with_ws("k1", ws)
        outputs = client.execute_code("k1", "x = 1")
        assert outputs[-1]["ename"] == "KernelDied"
        assert "k1" not in client._ws_cache

    def test_cancelled_execution_reports_cancelled_not_dead(self):
        """When cancel_execution closed the socket on purpose, the result
        must say 'cancelled', not 'kernel died', and the client must not
        reconnect and re-send the cancelled code."""
        ws = FakeWS([websocket.WebSocketConnectionClosedException()])
        client = _client_with_ws("k1", ws)

        real_recv = ws.recv

        def recv_and_cancel():
            client.cancel_execution("k1")
            return real_recv()

        ws.recv = recv_and_cancel
        with patch("jlab_mcp.jupyter_client.websocket.create_connection") as cc:
            outputs = client.execute_code("k1", "long_running()")
            cc.assert_not_called()  # no reconnect after cancellation
        assert outputs[-1]["ename"] == "ExecutionCancelled"

    def test_cancel_flag_cleared_on_next_execution(self):
        client = _client_with_ws("k1", FakeWS([]))
        client.cancel_execution("k1")
        # Next execution must run normally despite the earlier cancel
        client._ws_cache["k1"] = FakeWS([STREAM, IDLE])
        outputs = client.execute_code("k1", "print('hello')")
        assert outputs[0]["content"] == "hello\n"

    def test_reconnects_once_on_stale_cached_ws(self):
        """A dead cached socket triggers exactly one reconnect attempt."""
        stale = FakeWS([])
        stale.connected = False
        fresh = FakeWS([STREAM, IDLE])
        client = _client_with_ws("k1", stale)
        with patch(
            "jlab_mcp.jupyter_client.websocket.create_connection",
            return_value=fresh,
        ) as cc:
            outputs = client.execute_code("k1", "print('hello')")
        assert cc.call_count == 1
        assert outputs[0]["content"] == "hello\n"
        assert client._ws_cache["k1"] is fresh

    def test_channels_never_ready_fails_instead_of_hanging(self):
        """If the kernel never answers on a fresh socket, execute_code must
        return an error instead of waiting forever on a request that was
        silently dropped."""
        client = JupyterLabClient("localhost", 8888, "tok")
        with patch(
            "jlab_mcp.jupyter_client.websocket.create_connection",
            side_effect=lambda *a, **k: NeverReadyWS(),
        ):
            outputs = client.execute_code("k1", "x = 1")
        assert outputs[-1]["ename"] == "KernelDied"

    def test_fresh_connection_waits_for_channel_ready(self):
        """A fresh socket is handed out only after the kernel answers the
        kernel_info handshake."""
        fresh = FakeWS([STREAM, IDLE])
        client = JupyterLabClient("localhost", 8888, "tok")
        with patch(
            "jlab_mcp.jupyter_client.websocket.create_connection",
            return_value=fresh,
        ):
            outputs = client.execute_code("k1", "print('hello')")
        assert outputs[0]["content"] == "hello\n"

    def test_handshake_404_means_kernel_gone(self):
        """A 404 on the channels handshake = kernel no longer exists.
        Must return the distinct KernelGone error, not retry forever or
        leak a raw exception."""
        client = JupyterLabClient("localhost", 8888, "tok")
        err = websocket.WebSocketBadStatusException("Handshake status %d %s", 404)
        with patch(
            "jlab_mcp.jupyter_client.websocket.create_connection",
            side_effect=err,
        ):
            outputs = client.execute_code("missing-kernel", "x = 1")
        assert outputs[-1]["ename"] == "KernelGone"

    def test_executions_on_same_kernel_serialized(self):
        """Two concurrent executions on one kernel must not interleave on
        the shared WebSocket (stolen idle/output messages)."""
        import threading

        order = []

        class SlowWS(FakeWS):
            def recv(self):
                import time as _t

                _t.sleep(0.05)
                return super().recv()

        ws = SlowWS([STREAM, IDLE, STREAM, IDLE])
        client = _client_with_ws("k1", ws)

        def run_exec(tag):
            client.execute_code("k1", f"print('{tag}')")
            order.append(tag)

        t1 = threading.Thread(target=run_exec, args=("a",))
        t2 = threading.Thread(target=run_exec, args=("b",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        # Both completed (no stolen-idle hang), strictly one after the other
        assert sorted(order) == ["a", "b"]
        assert len(ws.sent) == 2


class TestParseJpeg:
    def test_returns_image_dict(self):
        out = JupyterLabClient._parse_jpeg(_make_jpeg_b64(100, 50))
        assert out["type"] == "image"
        assert isinstance(out["content"], str)

    def test_content_is_png(self):
        out = JupyterLabClient._parse_jpeg(_make_jpeg_b64(100, 50))
        png_bytes = base64.b64decode(out["content"])
        # PNG magic number: 89 50 4E 47 0D 0A 1A 0A
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_preserves_dimensions(self):
        out = JupyterLabClient._parse_jpeg(_make_jpeg_b64(123, 45))
        png_bytes = base64.b64decode(out["content"])
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (123, 45)
