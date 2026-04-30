"""Unit tests for JupyterLabClient output parsers."""

import base64
import io

from PIL import Image

from jlab_mcp.jupyter_client import JupyterLabClient


def _make_jpeg_b64(width: int, height: int) -> str:
    img = Image.new("RGB", (width, height), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
