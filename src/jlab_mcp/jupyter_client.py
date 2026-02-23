import base64
import json
import logging
import threading
import time
import uuid

import requests
import websocket

from jlab_mcp.image_utils import resize_image_if_needed

logger = logging.getLogger("jlab-mcp")

# WebSocket recv wakes up every RECV_POLL_INTERVAL seconds so the
# thread stays responsive and the MCP stdio connection is never
# considered dead by Claude Code during long-running executions.
RECV_POLL_INTERVAL = 30


class JupyterLabClient:
    """HTTP + WebSocket client to JupyterLab on a compute node."""

    def __init__(self, host: str, port: int | str, token: str):
        self.host = host
        self.port = int(port)
        self.token = token
        self.base_url = f"http://{self.host}:{self.port}"
        self._headers = {"Authorization": f"token {self.token}"}
        self._ws_cache: dict[str, websocket.WebSocket] = {}
        self._ws_lock = threading.Lock()

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
        """Get or create a cached WebSocket connection for a kernel."""
        with self._ws_lock:
            ws = self._ws_cache.get(kernel_id)
            if ws is not None and ws.connected:
                return ws
            # Close stale connection if any
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
            ws_url = (
                f"ws://{self.host}:{self.port}"
                f"/api/kernels/{kernel_id}/channels"
                f"?token={self.token}"
            )
            ws = websocket.create_connection(ws_url, timeout=RECV_POLL_INTERVAL)
            self._ws_cache[kernel_id] = ws
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

    def execute_code(
        self, kernel_id: str, code: str, timeout: int | None = None
    ) -> list[dict]:
        """Execute code via kernel WebSocket. Returns list of output dicts.

        Reuses a cached WebSocket per kernel. If the connection is broken,
        it reconnects automatically.

        Each output dict has:
          - type: "text" | "image" | "error"
          - content: str (text or base64 image data)
          - For errors: ename, evalue, traceback
        """
        try:
            ws = self._get_ws(kernel_id)
            return self._execute_on_ws(ws, code, timeout)
        except (
            websocket.WebSocketConnectionClosedException,
            ConnectionError,
            OSError,
        ):
            # Connection was broken, close cached and retry once
            self._close_ws(kernel_id)
            try:
                ws = self._get_ws(kernel_id)
                return self._execute_on_ws(ws, code, timeout)
            except (
                websocket.WebSocketConnectionClosedException,
                ConnectionError,
                OSError,
            ):
                self._close_ws(kernel_id)
                return [{
                    "type": "error",
                    "ename": "KernelDied",
                    "evalue": (
                        "Kernel died during execution (likely OOM or crash). "
                        "All in-memory state is lost. "
                        "Start a new session to continue."
                    ),
                    "traceback": [],
                }]

    @staticmethod
    def _resize_png(png_base64: str) -> dict:
        """Decode, resize, and re-encode a base64 PNG image."""
        image_data = base64.b64decode(png_base64)
        resized = resize_image_if_needed(image_data)
        return {"type": "image", "content": base64.b64encode(resized).decode()}

    def _execute_on_ws(
        self, ws: websocket.WebSocket, code: str, timeout: int | None
    ) -> list[dict]:
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

        while True:
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                # Check if caller-specified timeout exceeded
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
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
                # No timeout or not yet exceeded — keep waiting
                elapsed = time.time() - start_time
                logger.debug("Waiting for kernel output (%.0fs elapsed)", elapsed)
                continue
            except (
                websocket.WebSocketConnectionClosedException,
                ConnectionError,
                OSError,
            ):
                outputs.append({
                    "type": "error",
                    "ename": "KernelDied",
                    "evalue": (
                        "Kernel died during execution (likely OOM or crash). "
                        "All in-memory state is lost. "
                        "Start a new session to continue."
                    ),
                    "traceback": [],
                })
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
                    outputs.append({
                        "type": "error",
                        "ename": "KernelDied",
                        "evalue": (
                            f"Kernel {exec_state} during execution "
                            "(likely OOM or crash). "
                            "All in-memory state is lost. "
                            "Start a new session to continue."
                        ),
                        "traceback": [],
                    })
                    break

            # Only process messages that are replies to our request
            parent_msg_id = msg.get("parent_header", {}).get("msg_id", "")
            if parent_msg_id != msg_id:
                continue

            if msg_type == "stream":
                text = msg["content"].get("text", "")
                if text:
                    outputs.append({"type": "text", "content": text})

            elif msg_type == "execute_result":
                data = msg["content"].get("data", {})
                if "text/plain" in data:
                    outputs.append(
                        {"type": "text", "content": data["text/plain"]}
                    )
                if "image/png" in data:
                    outputs.append(self._resize_png(data["image/png"]))

            elif msg_type == "display_data":
                data = msg["content"].get("data", {})
                if "image/png" in data:
                    outputs.append(self._resize_png(data["image/png"]))
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
