from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def make_default_noise_image(output_path: Path, size: int = 256, seed: int = 0) -> Path:
    """Create a deterministic random-noise dummy image when the caller omits --image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(output_path)
    return output_path


def prepare_image(image_path: str | None, output_dir: Path, image_size: int) -> Path:
    """Return a square RGB image with the requested size.

    If image_path is omitted, a deterministic random-noise image is generated.
    If image_path is provided, the file is opened, converted to RGB, resized to
    image_size x image_size, and copied into the run directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if image_path is None:
        return make_default_noise_image(output_dir / f"default_noise_{image_size}.png", image_size)
    src = Path(image_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Image not found: {src}")
    img = Image.open(src).convert("RGB").resize((image_size, image_size))
    out = output_dir / f"resized_{src.stem}_{image_size}.png"
    img.save(out)
    return out
