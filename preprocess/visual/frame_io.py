from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def load_frame_source(path: str | Path, frames_key: str = "frames") -> np.ndarray:
    """Load a frame source into [V, T, H, W, C] uint8 frames."""
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() != ".npz":
            raise ValueError(f"frame file source must be a .npz file, got '{source}'")
        return _load_npz_frames(source, frames_key=frames_key)
    if source.is_dir():
        return _load_directory_frames(source)
    raise FileNotFoundError(f"frame source does not exist: {source}")


def _load_npz_frames(path: Path, frames_key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as src:
        if frames_key not in src:
            raise ValueError(f"frame npz '{path}' is missing key '{frames_key}'")
        frames = np.asarray(src[frames_key])
    if frames.ndim == 4:
        frames = frames[None]
    _validate_frame_tensor(frames, source=path)
    return _to_uint8(frames)


def _load_directory_frames(path: Path) -> np.ndarray:
    direct_images = _image_files(path)
    if direct_images:
        return _load_single_view(direct_images)[None]

    view_dirs = [child for child in sorted(path.iterdir()) if child.is_dir()]
    view_image_lists = [(view_dir, _image_files(view_dir)) for view_dir in view_dirs]
    view_image_lists = [(view_dir, images) for view_dir, images in view_image_lists if images]
    if not view_image_lists:
        raise ValueError(f"frame directory '{path}' does not contain image frames")

    frame_counts = {len(images) for _, images in view_image_lists}
    if len(frame_counts) != 1:
        raise ValueError("all multiview directories must contain the same number of frames")

    views = [_load_single_view(images) for _, images in view_image_lists]
    shapes = {tuple(view.shape[1:]) for view in views}
    if len(shapes) != 1:
        raise ValueError("all multiview frames must share the same image shape")
    return np.stack(views, axis=0)


def _load_single_view(image_paths: Sequence[Path]) -> np.ndarray:
    frames = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return np.stack(frames, axis=0)


def _image_files(path: Path) -> list[Path]:
    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    )


def _validate_frame_tensor(frames: np.ndarray, source: Path) -> None:
    if frames.ndim != 5:
        raise ValueError(f"frames from '{source}' must have shape [V,T,H,W,C], got {frames.shape}")
    if frames.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"frames from '{source}' must have 1, 3, or 4 channels, got {frames.shape}")
    if min(frames.shape[:4]) <= 0:
        raise ValueError(f"frames from '{source}' must have non-empty view/time/height/width dimensions")


def _to_uint8(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames
    return np.clip(frames, 0, 255).astype(np.uint8)
