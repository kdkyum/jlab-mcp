import io

from PIL import Image


def _calculate_resize_dimensions(
    width: int, height: int, max_dim: int = 512
) -> tuple[int, int]:
    """Calculate new dimensions maintaining aspect ratio."""
    if width <= max_dim and height <= max_dim:
        return width, height

    if width >= height:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    else:
        new_height = max_dim
        new_width = int(width * (max_dim / height))

    return max(1, new_width), max(1, new_height)


def resize_image_if_needed(
    image_data: bytes, max_dim: int = 512
) -> bytes:
    """Resize image if any dimension exceeds max_dim. Returns PNG bytes."""
    try:
        img = Image.open(io.BytesIO(image_data))
        width, height = img.size

        if width <= max_dim and height <= max_dim:
            return image_data

        new_w, new_h = _calculate_resize_dimensions(width, height, max_dim)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return image_data
