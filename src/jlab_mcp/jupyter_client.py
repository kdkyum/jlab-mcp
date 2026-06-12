import base64
import json
import logging
import threading
import time
import uuid

import requests
import websocket


logger = logging.getLogger("jlab-mcp")

# WebSocket recv wakes up every RECV_POLL_INTERVAL seconds so the
# thread stays responsive and the MCP stdio connection is never
# considered dead by Claude Code during long-running executions.
RECV_POLL_INTERVAL = 30

# Connection errors that indicate a broken WebSocket.
# WebSocketException covers handshake failures (e.g. WebSocketBadStatusException
# when the kernel no longer exists) and timeouts, which are not OSErrors.
_WS_CONNECTION_ERRORS = (
    websocket.WebSocketException,
    ConnectionError,
    OSError,
)


def _kernel_died_error(detail: str = "") -> dict:
    """Build a standardised 'KernelDied' error output dict."""
    reason = detail if detail else "likely OOM or crash"
    return {
        "type": "error",
        "ename": "KernelDied",
        "evalue": (
            f"Kernel died during execution ({reason}). "
            "All in-memory state is lost. "
            "Start a new session to continue."
        ),
        "traceback": [],
    }


def _kernel_gone_error() -> dict:
    """Error output for a kernel that no longer exists on the server."""
    return {
        "type": "error",
        "ename": "KernelGone",
        "evalue": (
            "Kernel no longer exists on the JupyterLab server "
            "(it was shut down or removed after repeated crashes). "
            "Use shutdown_session and start a new session to recover."
        ),
        "traceback": [],
    }


def _execution_cancelled_error() -> dict:
    """Error output for an execution aborted by user cancellation."""
    return {
        "type": "error",
        "ename": "ExecutionCancelled",
        "evalue": "Execution was cancelled by the user.",
        "traceback": [],
    }


class JupyterLabClient:
    """HTTP + WebSocket client to JupyterLab on a compute node."""

    def __init__(self, host: str, port: int | str, token: str):
        self.host = host
        self.port = int(port)
        self.token = token
        self.base_url = f"http://{self.host}:{self.port}"
        self._headers = {"Authorization": f"token {self.token}"}
        self._ws_cache: dict[str, websocket.WebSocket] = {}
        # Guards _ws_cache, _exec_locks and _cancelled — never held across
        # network calls (connect/close happen outside the lock).
        self._ws_lock = threading.Lock()
        # One execution conversation per kernel at a time: a kernel executes
        # serially anyway, and two recv loops on one WebSocket steal each
        # other's messages (lost outputs, missed idle -> infinite wait).
        self._exec_locks: dict[str, threading.Lock] = {}
        # Kernels whose in-flight execution was cancelled via cancel_execution;
        # the executing thread must not reconnect-and-resend after the close.
        self._cancelled: set[str] = set()

    def health_check(self) -> bool:
        """Check if JupyterLab is responding."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/status",
                headers=self._headers,
                timeout=10,
            )
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def start_kernel(self, kernel_name: str = "python3") -> str:
        """Start a new kernel. Returns kernel_id."""
        resp = requests.post(
            f"{self.base_url}/api/kernels",
            headers=self._headers,
            json={"name": kernel_name},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def shutdown_kernel(self, kernel_id: str) -> None:
        """Shutdown a kernel and close its cached WebSocket."""
        self._close_ws(kernel_id)
        requests.delete(
            f"{self.base_url}/api/kernels/{kernel_id}",
            headers=self._headers,
            timeout=10,
        )

    def interrupt_kernel(self, kernel_id: str) -> None:
        """Interrupt a running kernel."""
        resp = requests.post(
            f"{self.base_url}/api/kernels/{kernel_id}/interrupt",
            headers=self._headers,
            timeout=10,
        )
        resp.raise_for_status()

    def list_kernels(self) -> list[dict]:
        """List all running kernels."""
        resp = requests.get(
            f"{self.base_url}/api/kernels",
            headers=self._headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_ws(self, kernel_id: str) -> websocket.WebSocket:
        """Get or create a cached WebSocket connection for a kernel.

        The connect happens outside _ws_lock so a slow handshake (wedged
        node) cannot stall threads working on other kernels or the
        cancellation path.
        """
        with self._ws_lock:
            ws = self._ws_cache.get(kernel_id)
            stale = None
            if ws is not None and ws.connected:
                return ws
            if ws is not None:
                stale = self._ws_cache.pop(kernel_id, None)
        if stale is not None:
            try:
                stale.close()
            except Exception:
                pass
        ws_url = (
            f"ws://{self.host}:{self.port}"
            f"/api/kernels/{kernel_id}/channels"
            f"?token={self.token}"
        )
        ws = websocket.create_connection(ws_url, timeout=RECV_POLL_INTERVAL)
        # The server accepts the WS upgrade before the kernel's ZMQ channels
        # are necessarily wired up; anything sent in that window is silently
        # dropped. Handshake before handing the socket out.
        if not self._wait_channel_ready(ws):
            try:
                ws.close()
            except Exception:
                pass
            raise ConnectionError(
                f"Kernel {kernel_id} channels not ready (no kernel_info reply)"
            )
        with self._ws_lock:
            existing = self._ws_cache.get(kernel_id)
            if existing is not None and existing.connected:
                # Another thread connected first — use theirs.
                ws_extra = ws
                ws = existing
            else:
                self._ws_cache[kernel_id] = ws
                ws_extra = None
        if ws_extra is not None:
            try:
                ws_extra.close()
            except Exception:
                pass
        return ws

    def _close_ws(self, kernel_id: str) -> None:
        """Close and remove a cached WebSocket connection."""
        with self._ws_lock:
            ws = self._ws_cache.pop(kernel_id, None)
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    @staticmethod
    def _wait_channel_ready(ws: websocket.WebSocket, attempts: int = 3) -> bool:
        """Confirm the kernel actually answers on a fresh channels WebSocket.

        Sends kernel_info_request and waits for any reply parented to it.
        A freshly started kernel can take a moment before its ZMQ channels
        are bridged — until then, requests vanish without a trace, which
        would leave the execute recv-loop waiting forever.
        """
        for _ in range(attempts):
            msg_id = str(uuid.uuid4())
            ws.send(json.dumps({
                "header": {
                    "msg_id": msg_id,
                    "msg_type": "kernel_info_request",
                    "username": "",
                    "session": str(uuid.uuid4()),
                    "version": "5.3",
                },
                "parent_header": {},
                "metadata": {},
                "content": {},
                "buffers": [],
                "channel": "shell",
            }))
            ws.settimeout(2)
            deadline = time.time() + 4
            try:
                while time.time() < deadline:
                    try:
                        msg = json.loads(ws.recv())
                    except websocket.WebSocketTimeoutException:
                        break  # re-send the request
                    if msg.get("parent_header", {}).get("msg_id") == msg_id:
                        return True
            finally:
                ws.settimeout(RECV_POLL_INTERVAL)
        return False

    def _get_exec_lock(self, kernel_id: str) -> threading.Lock:
        """Get (or create) the per-kernel execution lock."""
        with self._ws_lock:
            lock = self._exec_locks.get(kernel_id)
            if lock is None:
                lock = threading.Lock()
                self._exec_locks[kernel_id] = lock
            return lock

    def cancel_execution(self, kernel_id: str) -> None:
        """Abort an in-flight execution on this kernel.

        Closes the cached WebSocket (unblocking a thread stuck in recv)
        and marks the kernel cancelled so the execution thread does not
        reconnect and re-send the code.
        """
        with self._ws_lock:
            self._cancelled.add(kernel_id)
        self._close_ws(kernel_id)

    def execute_code(
        self, kernel_id: str, code: str, timeout: int | None = None
    ) -> list[dict]:
        """Execute code via kernel WebSocket. Returns list of output dicts.

        Reuses a cached WebSocket per kernel. If the connection is broken,
        it reconnects once automatically. Executions on the same kernel are
        serialized (the kernel runs code serially anyway, and concurrent
        recv loops on one WebSocket would steal each other's messages).

        Each output dict has:
          - type: "text" | "image" | "error"
          - content: str (text or base64 image data)
          - For text: optional name ("stdout"/"stderr") and result flag
          - For errors: ename, evalue, traceback
        """
        with self._get_exec_lock(kernel_id):
            with self._ws_lock:
                self._cancelled.discard(kernel_id)
            max_attempts = 2
            for attempt in range(max_attempts):
                try:
                    ws = self._get_ws(kernel_id)
                    return self._execute_on_ws(ws, kernel_id, code, timeout)
                except websocket.WebSocketBadStatusException as e:
                    # 404 on the channels handshake: the kernel id is gone
                    # (culled, removed after repeated crashes, or shut down).
                    # Retrying cannot help.
                    self._close_ws(kernel_id)
                    if getattr(e, "status_code", None) == 404:
                        return [_kernel_gone_error()]
                    if attempt < max_attempts - 1 and not self._was_cancelled(kernel_id):
                        continue
                    return [_kernel_died_error(str(e))]
                except _WS_CONNECTION_ERRORS:
                    self._close_ws(kernel_id)
                    if self._was_cancelled(kernel_id):
                        # cancel_execution closed the socket on purpose —
                        # do not reconnect and re-send the cancelled code.
                        return [_execution_cancelled_error()]
                    if attempt < max_attempts - 1:
                        continue
                    return [_kernel_died_error()]

    def _was_cancelled(self, kernel_id: str) -> bool:
        with self._ws_lock:
            return kernel_id in self._cancelled

    @staticmethod
    def _parse_png(png_base64: str) -> dict:
        """Wrap a base64 PNG image into an output dict (no resize)."""
        return {"type": "image", "content": png_base64}

    @staticmethod
    def _parse_jpeg(jpeg_base64: str) -> dict:
        """Decode a base64 JPEG, re-encode as PNG, return output dict.

        Downstream (_format_outputs) assumes PNG, so normalise here.
        """
        import io as _io

        from PIL import Image as _PILImage

        img = _PILImage.open(_io.BytesIO(base64.b64decode(jpeg_base64)))
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return {"type": "image", "content": base64.b64encode(buf.getvalue()).decode("ascii")}

    def _execute_on_ws(
        self,
        ws: websocket.WebSocket,
        kernel_id: str,
        code: str,
        timeout: int | None,
    ) -> list[dict]:
        if self._was_cancelled(kernel_id):
            return [_execution_cancelled_error()]
        msg_id = str(uuid.uuid4())
        execute_msg = {
            "header": {
                "msg_id": msg_id,
                "msg_type": "execute_request",
                "username": "",
                "session": str(uuid.uuid4()),
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "buffers": [],
            "channel": "shell",
        }
        ws.send(json.dumps(execute_msg))

        outputs: list[dict] = []
        ws.settimeout(RECV_POLL_INTERVAL)
        start_time = time.time()
        saw_reply = False  # any message parented to our request

        while True:
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                elapsed = time.time() - start_time
                if not saw_reply and elapsed >= 60:
                    # The kernel echoes status/execute_input immediately for
                    # every request, even for long-running code. Total
                    # silence means the request was lost (e.g. channels not
                    # bridged yet) — retry on a fresh socket rather than
                    # waiting forever.
                    raise ConnectionError(
                        "no kernel response to execute_request after 60s"
                    )
                if timeout is not None and elapsed >= timeout:
                    outputs.append({
                        "type": "error",
                        "ename": "ExecutionTimeout",
                        "evalue": (
                            f"Execution timed out after {timeout} seconds. "
                            "The kernel is still running — use interrupt_kernel "
                            "to cancel, or increase the timeout."
                        ),
                        "traceback": [],
                    })
                    break
                logger.debug("Waiting for kernel output (%.0fs elapsed)", elapsed)
                continue
            except _WS_CONNECTION_ERRORS:
                # Evict the broken socket so the next execution starts fresh.
                self._close_ws(kernel_id)
                if self._was_cancelled(kernel_id):
                    outputs.append(_execution_cancelled_error())
                else:
                    outputs.append(_kernel_died_error())
                break

            msg = json.loads(raw)

            # Detect kernel death/restart BEFORE filtering by parent_msg_id.
            # JupyterLab broadcasts these on IOPub with no parent_header,
            # so the parent_msg_id filter would skip them.
            msg_type = msg.get("msg_type") or msg.get("header", {}).get(
                "msg_type", ""
            )
            if msg_type == "status":
                exec_state = msg["content"].get("execution_state")
                if exec_state in ("restarting", "dead"):
                    # Evict the WS: death broadcasts buffered in this socket
                    # would otherwise poison the next execution, which could
                    # send its request and then immediately read a stale
                    # 'restarting' message.
                    self._close_ws(kernel_id)
                    outputs.append(
                        _kernel_died_error(f"kernel {exec_state}")
                    )
                    break

            # Only process messages that are replies to our request
            parent_msg_id = msg.get("parent_header", {}).get("msg_id", "")
            if parent_msg_id != msg_id:
                continue
            saw_reply = True

            if msg_type == "stream":
                text = msg["content"].get("text", "")
                if text:
                    outputs.append({
                        "type": "text",
                        "content": text,
                        "name": msg["content"].get("name", "stdout"),
                    })

            elif msg_type == "execute_result":
                data = msg["content"].get("data", {})
                if "text/plain" in data:
                    outputs.append(
                        {"type": "text", "content": data["text/plain"],
                         "result": True}
                    )
                if "image/png" in data:
                    outputs.append(self._parse_png(data["image/png"]))
                elif "image/jpeg" in data:
                    outputs.append(self._parse_jpeg(data["image/jpeg"]))

            elif msg_type == "display_data":
                data = msg["content"].get("data", {})
                if "image/png" in data:
                    outputs.append(self._parse_png(data["image/png"]))
                elif "image/jpeg" in data:
                    outputs.append(self._parse_jpeg(data["image/jpeg"]))
                elif "text/plain" in data:
                    outputs.append(
                        {"type": "text", "content": data["text/plain"]}
                    )

            elif msg_type == "error":
                outputs.append(
                    {
                        "type": "error",
                        "ename": msg["content"].get("ename", ""),
                        "evalue": msg["content"].get("evalue", ""),
                        "traceback": msg["content"].get("traceback", []),
                    }
                )

            elif msg_type == "status":
                if msg["content"].get("execution_state") == "idle":
                    break

        return outputs
