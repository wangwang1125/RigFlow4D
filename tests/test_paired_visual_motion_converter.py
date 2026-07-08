import json

import numpy as np
import pytest
from PIL import Image

from data.adapters.normalized_npz import NormalizedNpzAdapter
from preprocess.converters.paired_visual_motion import convert_paired_visual_motion_manifest
from preprocess.visual.token_cache import DeterministicVisualBackbone, VisualTokenCacheConfig


def write_rgb_image(path, value):
    image = np.full((4, 4, 3), value, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def write_normalized_motion_npz(path, frames=3, joints=4):
    np.savez(
        path,
        dataset_name=np.array("bedlam"),
        input_type=np.array("video"),
        source_label_type=np.array("smplx"),
        camera_mode=np.array("unknown"),
        parents=np.array([-1, 0, 1, 2], dtype=np.int64)[:joints],
        rest_offsets=np.zeros((joints, 3), dtype=np.float32),
        joint_names=np.array([f"joint_{i}" for i in range(joints)]),
        chain_ids=np.arange(joints, dtype=np.int64),
        chain_coordinates=np.linspace(0.0, 1.0, joints, dtype=np.float32),
        positions=np.zeros((frames, joints, 3), dtype=np.float32),
        local_rotations_6d=np.zeros((frames, joints, 6), dtype=np.float32),
        root_translation=np.zeros((frames, 3), dtype=np.float32),
    )


def write_prepared_pair(root, frame_count=3, motion_frames=3):
    frames_dir = root / "raw" / "seq_001" / "frames"
    frames_dir.mkdir(parents=True)
    for index in range(frame_count):
        write_rgb_image(frames_dir / f"{index:04d}.png", 10 + index)
    motion_path = root / "raw" / "seq_001" / "motion.npz"
    write_normalized_motion_npz(motion_path, frames=motion_frames)
    manifest_path = root / "paired_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "seq_001",
                        "frames": "raw/seq_001/frames",
                        "motion": "raw/seq_001/motion.npz",
                        "out": "samples/seq_001.npz",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def make_backbone():
    return DeterministicVisualBackbone(
        config=VisualTokenCacheConfig(
            backbone_name="dinov3_dummy",
            feature_dim=6,
            patch_grid=(2, 2),
        )
    )


def test_convert_paired_visual_motion_manifest_writes_adapter_loadable_dataset(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    manifest_path = write_prepared_pair(input_root)
    output_root = tmp_path / "BEDLAM_RigFlow4D"

    normalized_manifest = convert_paired_visual_motion_manifest(
        input_manifest_path=manifest_path,
        output_root=output_root,
        backbone=make_backbone(),
    )

    adapter = NormalizedNpzAdapter(root=output_root, manifest_path=normalized_manifest)
    sample = adapter[0]

    assert normalized_manifest == output_root / "manifest.json"
    assert len(adapter) == 1
    assert sample.dataset_name == "bedlam"
    assert sample.visual.backbone_name == "dinov3_dummy"
    assert sample.visual.tokens.shape == (1, 3, 4, 6)


def test_convert_paired_visual_motion_manifest_can_skip_invalid_pairs(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    manifest_path = write_prepared_pair(input_root, frame_count=2, motion_frames=3)
    output_root = tmp_path / "BEDLAM_RigFlow4D"

    normalized_manifest = convert_paired_visual_motion_manifest(
        input_manifest_path=manifest_path,
        output_root=output_root,
        backbone=make_backbone(),
        skip_invalid=True,
    )

    manifest = json.loads(normalized_manifest.read_text(encoding="utf-8"))
    assert manifest["samples"] == []


def test_convert_paired_visual_motion_manifest_raises_on_invalid_pairs_by_default(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    manifest_path = write_prepared_pair(input_root, frame_count=2, motion_frames=3)

    with pytest.raises(ValueError, match="frame"):
        convert_paired_visual_motion_manifest(
            input_manifest_path=manifest_path,
            output_root=tmp_path / "BEDLAM_RigFlow4D",
            backbone=make_backbone(),
        )
