"""Unit tests for image_utils.py — local only."""

import io

import pytest
from PIL import Image

from jlab_mcp.image_utils import _calculate_resize_dimensions, resize_image_if_needed


def _make_png(width: int, height: int) -> bytes:
    """Create a PNG image of given dimensions."""
    img = Image.new("RGB", (width, height), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestCalculateResizeDimensions:
    def test_small_image_unchanged(self):
        assert _calculate_resize_dimensions(100, 100) == (100, 100)

    def test_square_at_limit(self):
        assert _calculate_resize_dimensions(512, 512) == (512, 512)

    def test_wide_image(self):
        w, h = _calculate_resize_dimensions(1024, 512)
        assert w == 512
        assert h == 256

    def test_tall_image(self):
        w, h = _calculate_resize_dimensions(512, 1024)
        assert w == 256
        assert h == 512

    def test_very_large(self):
        w, h = _calculate_resize_dimensions(4000, 3000)
        assert w == 512
        assert h == 384

    def test_maintains_aspect_ratio(self):
        w, h = _calculate_resize_dimensions(1000, 500, max_dim=200)
        assert w == 200
        assert h == 100

    def test_custom_max_dim(self):
        w, h = _calculate_resize_dimensions(1000, 1000, max_dim=256)
        assert w == 256
        assert h == 256


class TestResizeImageIfNeeded:
    def test_small_image_passthrough(self):
        original = _make_png(100, 100)
        result = resize_image_if_needed(original)
        assert result == original

    def test_at_limit_passthrough(self):
        original = _make_png(512, 512)
        result = resize_image_if_needed(original)
        assert result == original

    def test_large_image_resized(self):
        original = _make_png(1024, 768)
        result = resize_image_if_needed(original)
        assert result != original
        img = Image.open(io.BytesIO(result))
        assert img.width <= 512
        assert img.height <= 512

    def test_maintains_aspect_ratio(self):
        original = _make_png(2000, 1000)
        result = resize_image_if_needed(original)
        img = Image.open(io.BytesIO(result))
        assert img.width == 512
        assert img.height == 256

    def test_invalid_data_returns_original(self):
        bad_data = b"not an image"
        result = resize_image_if_needed(bad_data)
        assert result == bad_data

    def test_custom_max_dim(self):
        original = _make_png(1000, 1000)
        result = resize_image_if_needed(original, max_dim=256)
        img = Image.open(io.BytesIO(result))
        assert img.width == 256
        assert img.height == 256
