import json

import numpy as np
import pytest

from data import create_default_registry
from data.adapters.normalized_npz import NormalizedNpzAdapter
from data.dataset_registry import DatasetAdapterRegistry
from data.schema import (
    CameraMode,
    InputType,
    RigDefinition,
    RigFlowSample,
    SourceLabelType,
    VisualTokenCache,
)
from data.window_dataset import MotionWindowDataset, collate_motion_windows
from preprocess.converters.motion_npz import convert_motion_npz_directory


def make_rig(joint_count=3):
    parents = np.arange(joint_count, dtype=np.int64) - 1
    parents[0] = -1
    return RigDefinition(
        parents=parents,
        rest_offsets=np.zeros((joint_count, 3), dtype=np.float32),
        joint_names=tuple(f"joint_{i}" for i in range(joint_count)),
        chain_ids=np.zeros((joint_count,), dtype=np.int64),
        chain_coordinates=np.linspace(0.0, 1.0, joint_count, dtype=np.float32),
    )


def make_sample(**overrides):
    frames = overrides.pop("frames", 2)
    joint_count = overrides.pop("joint_count", 3)
    fields = {
        "dataset_name": "unit",
        "input_type": InputType.VIDEO,
        "source_label_type": SourceLabelType.RIG_NATIVE,
        "camera_mode": CameraMode.UNKNOWN,
        "rig": make_rig(joint_count=joint_count),
        "positions": np.zeros((frames, joint_count, 3), dtype=np.float32),
        "local_rotations_6d": np.zeros((frames, joint_count, 6), dtype=np.float32),
        "root_translation": np.zeros((frames, 3), dtype=np.float32),
    }
    fields.update(overrides)
    return RigFlowSample(**fields)


def test_video_sample_accepts_missing_camera_parameters():
    sample = make_sample(
        visual=VisualTokenCache(
            tokens=np.zeros((1, 2, 4, 8), dtype=np.float32),
            backbone_name="dinov3_vitl16",
            feature_dim=8,
            patch_grid=(2, 2),
        ),
    )

    sample.validate()


def test_rejects_position_joint_count_mismatch():
    sample = make_sample(
        positions=np.zeros((2, 4, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="positions"):
        sample.validate()


def test_calibrated_camera_requires_intrinsics_and_extrinsics():
    sample = make_sample(
        input_type=InputType.MULTIVIEW_VIDEO,
        source_label_type=SourceLabelType.SMPLX,
        camera_mode=CameraMode.CALIBRATED,
    )

    with pytest.raises(ValueError, match="calibrated"):
        sample.validate()


def test_dataset_registry_registers_and_builds_adapter():
    registry = DatasetAdapterRegistry()

    class DummyAdapter:
        def __init__(self, root, split):
            self.root = root
            self.split = split

    registry.register("dummy", DummyAdapter)
    adapter = registry.build("dummy", root="data_root", split="train")

    assert isinstance(adapter, DummyAdapter)
    assert adapter.root == "data_root"
    assert adapter.split == "train"


def test_dataset_registry_rejects_duplicate_names():
    registry = DatasetAdapterRegistry()
    registry.register("dummy", object)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("dummy", object)


def test_dataset_registry_rejects_unknown_names():
    registry = DatasetAdapterRegistry()

    with pytest.raises(ValueError, match="Unknown dataset adapter"):
        registry.build("missing")


def write_normalized_npz(path):
    np.savez(
        path,
        dataset_name=np.array("unit"),
        input_type=np.array("video"),
        source_label_type=np.array("rig_native"),
        camera_mode=np.array("unknown"),
        parents=np.array([-1, 0, 1], dtype=np.int64),
        rest_offsets=np.zeros((3, 3), dtype=np.float32),
        joint_names=np.array(["root", "spine", "head"]),
        chain_ids=np.array([0, 0, 1], dtype=np.int64),
        chain_coordinates=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        positions=np.zeros((2, 3, 3), dtype=np.float32),
        local_rotations_6d=np.zeros((2, 3, 6), dtype=np.float32),
        root_translation=np.zeros((2, 3), dtype=np.float32),
        visual_tokens=np.zeros((1, 2, 4, 8), dtype=np.float32),
        visual_backbone_name=np.array("dinov3_vitl16"),
        visual_feature_dim=np.array(8),
        visual_patch_grid=np.array([2, 2], dtype=np.int64),
    )


def write_manifest(path, sample_path):
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample_0001",
                        "path": sample_path.name,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def write_motion_source_npz(path):
    np.savez(
        path,
        parents=np.array([-1, 0, 1], dtype=np.int64),
        rest_offsets=np.zeros((3, 3), dtype=np.float32),
        joint_names=np.array(["root", "spine", "head"]),
        chain_ids=np.array([0, 0, 1], dtype=np.int64),
        chain_coordinates=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        positions=np.zeros((2, 3, 3), dtype=np.float32),
        local_axis_angle=np.zeros((2, 3, 3), dtype=np.float32),
        root_translation=np.zeros((2, 3), dtype=np.float32),
    )


def test_normalized_npz_adapter_loads_sample(tmp_path):
    sample_path = tmp_path / "sample_0001.npz"
    manifest_path = tmp_path / "manifest.json"
    write_normalized_npz(sample_path)
    write_manifest(manifest_path, sample_path)

    adapter = NormalizedNpzAdapter(root=tmp_path, manifest_path=manifest_path)
    sample = adapter[0]

    assert len(adapter) == 1
    assert sample.dataset_name == "unit"
    assert sample.input_type == InputType.VIDEO
    assert sample.camera_mode == CameraMode.UNKNOWN
    assert sample.rig.joint_names == ("root", "spine", "head")
    assert sample.visual.backbone_name == "dinov3_vitl16"


def test_default_registry_builds_normalized_npz_adapter():
    registry = create_default_registry()

    assert "normalized_npz" in registry.names()


def test_motion_npz_converter_writes_normalized_manifest(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    write_motion_source_npz(src / "motion_0001.npz")

    manifest_path = convert_motion_npz_directory(src, out, dataset_name="amass")

    adapter = NormalizedNpzAdapter(root=out, manifest_path=manifest_path)
    sample = adapter[0]

    assert len(adapter) == 1
    assert sample.dataset_name == "amass"
    assert sample.source_label_type == SourceLabelType.MOTION_ONLY
    assert sample.local_rotations_6d.shape == (2, 3, 6)


class InMemoryAdapter:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_motion_window_dataset_creates_strided_windows():
    positions = np.arange(5 * 3 * 3, dtype=np.float32).reshape(5, 3, 3)
    sample = make_sample(
        frames=5,
        joint_count=3,
        positions=positions,
        local_rotations_6d=np.zeros((5, 3, 6), dtype=np.float32),
        root_translation=np.zeros((5, 3), dtype=np.float32),
    )
    dataset = MotionWindowDataset(InMemoryAdapter([sample]), window_size=3, stride=2)

    first = dataset[0]
    second = dataset[1]

    assert len(dataset) == 2
    np.testing.assert_array_equal(first.positions, positions[0:3])
    np.testing.assert_array_equal(second.positions, positions[2:5])
    np.testing.assert_array_equal(first.time_mask, np.array([True, True, True]))
    assert second.start == 2


def test_motion_window_dataset_slices_visual_tokens():
    visual_tokens = np.arange(2 * 5 * 3 * 4, dtype=np.float32).reshape(2, 5, 3, 4)
    sample = make_sample(
        frames=5,
        joint_count=3,
        visual=VisualTokenCache(
            tokens=visual_tokens,
            backbone_name="dinov3_vitl16",
            feature_dim=4,
            patch_grid=(1, 3),
        ),
    )
    dataset = MotionWindowDataset(InMemoryAdapter([sample]), window_size=3, stride=2)

    second = dataset[1]

    assert second.visual_tokens.shape == (2, 3, 3, 4)
    np.testing.assert_array_equal(second.visual_tokens, visual_tokens[:, 2:5])
    np.testing.assert_array_equal(second.visual_mask, np.ones((2, 3, 3), dtype=bool))


def test_motion_window_dataset_pads_short_visual_tokens():
    visual_tokens = np.ones((1, 2, 2, 4), dtype=np.float32)
    sample = make_sample(
        frames=2,
        joint_count=3,
        visual=VisualTokenCache(
            tokens=visual_tokens,
            backbone_name="dinov3_vitl16",
            feature_dim=4,
            patch_grid=(1, 2),
        ),
    )
    dataset = MotionWindowDataset(InMemoryAdapter([sample]), window_size=4, stride=4)

    window = dataset[0]

    assert window.visual_tokens.shape == (1, 4, 2, 4)
    np.testing.assert_array_equal(window.visual_tokens[:, :2], visual_tokens)
    np.testing.assert_array_equal(window.visual_tokens[:, 2:], np.zeros((1, 2, 2, 4), dtype=np.float32))
    np.testing.assert_array_equal(
        window.visual_mask,
        np.array([[[True, True], [True, True], [False, False], [False, False]]]),
    )


def test_collate_motion_windows_pads_joint_dimension():
    sample_a = make_sample(frames=3, joint_count=3)
    sample_b = make_sample(frames=3, joint_count=5)
    dataset = MotionWindowDataset(InMemoryAdapter([sample_a, sample_b]), window_size=3, stride=3)

    batch = collate_motion_windows([dataset[0], dataset[1]])

    assert batch["positions"].shape == (2, 3, 5, 3)
    assert batch["local_rotations_6d"].shape == (2, 3, 5, 6)
    assert batch["root_translation"].shape == (2, 3, 3)
    np.testing.assert_array_equal(batch["joint_mask"][0], np.array([True, True, True, False, False]))
    np.testing.assert_array_equal(batch["joint_mask"][1], np.array([True, True, True, True, True]))


def test_collate_motion_windows_includes_visual_tokens():
    sample_a = make_sample(
        frames=3,
        joint_count=3,
        visual=VisualTokenCache(
            tokens=np.ones((1, 3, 2, 4), dtype=np.float32),
            backbone_name="dinov3_vitl16",
            feature_dim=4,
            patch_grid=(1, 2),
        ),
    )
    sample_b = make_sample(frames=3, joint_count=3)
    dataset = MotionWindowDataset(InMemoryAdapter([sample_a, sample_b]), window_size=3, stride=3)

    batch = collate_motion_windows([dataset[0], dataset[1]])

    assert batch["visual_tokens"].shape == (2, 1, 3, 2, 4)
    assert batch["visual_mask"].shape == (2, 1, 3, 2)
    np.testing.assert_array_equal(batch["visual_mask"][0], np.ones((1, 3, 2), dtype=bool))
    np.testing.assert_array_equal(batch["visual_mask"][1], np.zeros((1, 3, 2), dtype=bool))
