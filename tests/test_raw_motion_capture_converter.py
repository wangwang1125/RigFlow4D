import numpy as np
import pytest

from data.adapters.normalized_npz import NormalizedNpzAdapter
from preprocess.converters.raw_motion_capture import (
    convert_raw_motion_capture_directory,
    parse_args,
    parse_raw_motion_capture_npz,
)


def write_amass_like_npz(path, frames=3):
    poses = np.zeros((frames, 72), dtype=np.float32)
    poses[:, 3:6] = np.array([0.0, 0.0, 0.1], dtype=np.float32)
    trans = np.stack(
        [
            np.linspace(0.0, 0.2, frames),
            np.zeros((frames,), dtype=np.float32),
            np.ones((frames,), dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    np.savez(path, poses=poses, trans=trans, mocap_framerate=np.array(60.0))


def write_aist_like_npz(path, frames=4):
    poses = np.zeros((frames, 24, 3), dtype=np.float32)
    poses[:, 1] = np.array([0.0, 0.2, 0.0], dtype=np.float32)
    trans = np.zeros((frames, 3), dtype=np.float32)
    trans[:, 1] = np.linspace(0.0, 0.3, frames)
    np.savez(path, smpl_poses=poses, smpl_trans=trans)


def test_parse_amass_like_npz_outputs_motion_contract(tmp_path):
    source = tmp_path / "amass_sample.npz"
    write_amass_like_npz(source)

    parsed = parse_raw_motion_capture_npz(source)

    assert parsed["local_axis_angle"].shape == (3, 24, 3)
    assert parsed["positions"].shape == (3, 24, 3)
    assert parsed["root_translation"].shape == (3, 3)
    assert parsed["joint_names"].shape == (24,)
    np.testing.assert_allclose(parsed["positions"][:, 0], parsed["root_translation"])


def test_parse_aist_like_npz_accepts_smpl_key_names(tmp_path):
    source = tmp_path / "aist_sample.npz"
    write_aist_like_npz(source)

    parsed = parse_raw_motion_capture_npz(source)

    assert parsed["local_axis_angle"].shape == (4, 24, 3)
    np.testing.assert_allclose(parsed["root_translation"][:, 1], np.linspace(0.0, 0.3, 4))


def test_parse_raw_motion_capture_rejects_invalid_pose_shape(tmp_path):
    source = tmp_path / "bad_sample.npz"
    np.savez(
        source,
        poses=np.zeros((2, 71), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="pose"):
        parse_raw_motion_capture_npz(source)


def test_convert_raw_motion_capture_directory_writes_normalized_manifest(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "normalized"
    raw_dir.mkdir()
    write_amass_like_npz(raw_dir / "amass_sample.npz")

    manifest_path = convert_raw_motion_capture_directory(
        input_dir=raw_dir,
        output_dir=out_dir,
        dataset_name="amass",
    )
    adapter = NormalizedNpzAdapter(root=out_dir, manifest_path=manifest_path)
    sample = adapter[0]

    assert len(adapter) == 1
    assert sample.dataset_name == "amass"
    assert sample.positions.shape == (3, 24, 3)
    assert sample.local_rotations_6d.shape == (3, 24, 6)
    assert sample.root_translation.shape == (3, 3)


def test_convert_raw_motion_capture_directory_recurses_nested_amass_layout(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "normalized"
    nested_a = raw_dir / "ACCAD" / "s007"
    nested_b = raw_dir / "ACCAD" / "s008"
    nested_a.mkdir(parents=True)
    nested_b.mkdir(parents=True)
    write_amass_like_npz(nested_a / "neutral_stagei.npz")
    write_amass_like_npz(nested_b / "neutral_stagei.npz")

    manifest_path = convert_raw_motion_capture_directory(
        input_dir=raw_dir,
        output_dir=out_dir,
        dataset_name="amass",
    )
    adapter = NormalizedNpzAdapter(root=out_dir, manifest_path=manifest_path)

    assert len(adapter) == 2
    output_names = sorted(record["path"] for record in adapter.samples)
    assert output_names == [
        "ACCAD__s007__neutral_stagei.npz",
        "ACCAD__s008__neutral_stagei.npz",
    ]


def test_parse_args_builds_converter_config(tmp_path):
    config = parse_args(
        [
            "--input-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "normalized"),
            "--dataset-name",
            "aistpp",
            "--source-format",
            "aistpp",
        ]
    )

    assert config.input_dir == tmp_path / "raw"
    assert config.output_dir == tmp_path / "normalized"
    assert config.dataset_name == "aistpp"
    assert config.source_format == "aistpp"
