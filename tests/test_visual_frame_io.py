import numpy as np
import pytest
from PIL import Image

from preprocess.visual.frame_io import load_frame_source


def write_rgb_image(path, value, size=(4, 6)):
    height, width = size
    image = np.full((height, width, 3), value, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def test_load_frame_source_reads_npz_frames(tmp_path):
    frames = np.arange(2 * 3 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 3, 4, 5, 3)
    source = tmp_path / "frames.npz"
    np.savez(source, frames=frames)

    loaded = load_frame_source(source)

    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, frames)


def test_load_frame_source_reads_single_view_directory_in_filename_order(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    write_rgb_image(frames_dir / "0002.png", 22)
    write_rgb_image(frames_dir / "0001.png", 11)

    loaded = load_frame_source(frames_dir)

    assert loaded.shape == (1, 2, 4, 6, 3)
    assert loaded[0, 0, 0, 0, 0] == 11
    assert loaded[0, 1, 0, 0, 0] == 22


def test_load_frame_source_reads_multiview_directory_in_view_order(tmp_path):
    root = tmp_path / "multiview"
    view_b = root / "view_b"
    view_a = root / "view_a"
    view_a.mkdir(parents=True)
    view_b.mkdir(parents=True)
    write_rgb_image(view_b / "0001.png", 21)
    write_rgb_image(view_b / "0002.png", 22)
    write_rgb_image(view_a / "0001.png", 11)
    write_rgb_image(view_a / "0002.png", 12)

    loaded = load_frame_source(root)

    assert loaded.shape == (2, 2, 4, 6, 3)
    assert loaded[0, 0, 0, 0, 0] == 11
    assert loaded[0, 1, 0, 0, 0] == 12
    assert loaded[1, 0, 0, 0, 0] == 21
    assert loaded[1, 1, 0, 0, 0] == 22


def test_load_frame_source_rejects_mismatched_multiview_frame_counts(tmp_path):
    root = tmp_path / "multiview"
    view_a = root / "view_a"
    view_b = root / "view_b"
    view_a.mkdir(parents=True)
    view_b.mkdir(parents=True)
    write_rgb_image(view_a / "0001.png", 11)
    write_rgb_image(view_a / "0002.png", 12)
    write_rgb_image(view_b / "0001.png", 21)

    with pytest.raises(ValueError, match="same number of frames"):
        load_frame_source(root)
