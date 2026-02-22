import base64
import json
import uuid

import requests
import websocket

from jlab_mcp.image_utils import resize_image_if_needed


class JupyterLabClient:
    """HTTP + WebSocket client to JupyterLab on a compute node."""

    def __init__(self, host: str, port: int | str, token: str):
        self.host = host
        self.port = int(port)
        self.token = token
        self.base_url = f"http://{self.host}:{self.port}"
        self._headers = {"Authorization": f"token {self.token}"}

    def health_check(self) -> bool:
        """Check if JupyterLab is responding."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/status",
                headers=self._headers,
                timeout=10,
            )
            return resp.status_code == 200
        except requests.ConnectionError:
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
        """Shutdown a kernel."""
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

    def execute_code(
        self, kernel_id: str, code: str, timeout: int = 300
    ) -> list[dict]:
        """Execute code via kernel WebSocket. Returns list of output dicts.

        Each output dict has:
          - type: "text" | "image" | "error"
          - content: str (text or base64 image data)
          - For errors: ename, evalue, traceback
        """
        ws_url = (
            f"ws://{self.host}:{self.port}"
            f"/api/kernels/{kernel_id}/channels"
            f"?token={self.token}"
        )

        ws = websocket.create_connection(ws_url, timeout=timeout)
        try:
            return self._execute_on_ws(ws, code, timeout)
        finally:
            ws.close()

    @staticmethod
    def _resize_png(png_base64: str) -> dict:
        """Decode, resize, and re-encode a base64 PNG image."""
        image_data = base64.b64decode(png_base64)
        resized = resize_image_if_needed(image_data)
        return {"type": "image", "content": base64.b64encode(resized).decode()}

    def _execute_on_ws(
        self, ws: websocket.WebSocket, code: str, timeout: int
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
        ws.settimeout(timeout)

        while True:
            try:
                raw = ws.recv()
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
