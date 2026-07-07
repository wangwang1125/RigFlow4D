import subprocess
import sys
import warnings

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


def write_amass_smplh_like_npz(path, frames=3):
    poses = np.zeros((frames, 156), dtype=np.float32)
    poses[:, 3:6] = np.array([0.0, 0.0, 0.1], dtype=np.float32)
    poses[:, 66:69] = np.array([0.0, 0.2, 0.0], dtype=np.float32)
    poses[:, 111:114] = np.array([0.0, -0.2, 0.0], dtype=np.float32)
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


def write_non_motion_npz(path):
    np.savez(path, metadata=np.array("not a motion file"), fps=np.array(30))


def write_broken_npz(path):
    path.write_bytes(b"this is not a zip npz file")


def write_custom_skeleton_template(path):
    path.write_text(
        """
{
  "parents": [-1, 0, 1, 1, 3],
  "rest_offsets": [
    [0.0, 0.0, 0.0],
    [0.0, 0.2, 0.0],
    [-0.1, 0.1, 0.0],
    [0.1, 0.1, 0.0],
    [0.2, 0.0, 0.0]
  ],
  "joint_names": ["root", "spine", "left_tip", "right_shoulder", "right_tip"],
  "chain_ids": [0, 1, 1, 3, 3],
  "chain_coordinates": [0.0, 0.25, 0.5, 0.5, 1.0]
}
""".strip(),
        encoding="utf-8",
    )


def write_custom_topology_npz(path, frames=2):
    poses = np.zeros((frames, 5, 3), dtype=np.float32)
    poses[:, 1] = np.array([0.0, 0.0, 0.1], dtype=np.float32)
    trans = np.stack(
        [
            np.linspace(0.0, 0.1, frames),
            np.linspace(0.2, 0.3, frames),
            np.zeros((frames,), dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    np.savez(path, poses=poses, trans=trans)


def test_parse_amass_like_npz_outputs_motion_contract(tmp_path):
    source = tmp_path / "amass_sample.npz"
    write_amass_like_npz(source)

    parsed = parse_raw_motion_capture_npz(source)

    assert parsed["local_axis_angle"].shape == (3, 24, 3)
    assert parsed["positions"].shape == (3, 24, 3)
    assert parsed["root_translation"].shape == (3, 3)
    assert parsed["joint_names"].shape == (24,)
    np.testing.assert_allclose(parsed["positions"][:, 0], parsed["root_translation"])


def test_parse_amass_smplh_like_npz_keeps_52_joint_hand_topology(tmp_path):
    source = tmp_path / "amass_smplh_sample.npz"
    write_amass_smplh_like_npz(source)

    parsed = parse_raw_motion_capture_npz(source, source_format="amass")

    assert parsed["local_axis_angle"].shape == (3, 52, 3)
    assert parsed["positions"].shape == (3, 52, 3)
    assert parsed["joint_names"].shape == (52,)
    assert parsed["joint_names"][22] == "left_index1"
    assert parsed["joint_names"][51] == "right_thumb3"
    assert parsed["parents"][22] == 20
    assert parsed["parents"][37] == 21
    np.testing.assert_allclose(parsed["positions"][:, 0], parsed["root_translation"])


def test_parse_smplh_source_format_uses_builtin_52_joint_template(tmp_path):
    source = tmp_path / "smplh_sample.npz"
    write_amass_smplh_like_npz(source)

    parsed = parse_raw_motion_capture_npz(source, source_format="smplh")

    assert parsed["local_axis_angle"].shape == (3, 52, 3)
    assert parsed["rest_offsets"].shape == (52, 3)


def test_parse_aist_like_npz_accepts_smpl_key_names(tmp_path):
    source = tmp_path / "aist_sample.npz"
    write_aist_like_npz(source)

    parsed = parse_raw_motion_capture_npz(source)

    assert parsed["local_axis_angle"].shape == (4, 24, 3)
    np.testing.assert_allclose(parsed["root_translation"][:, 1], np.linspace(0.0, 0.3, 4))


def test_parse_raw_motion_capture_uses_skeleton_template_for_custom_topology(tmp_path):
    source = tmp_path / "custom_sample.npz"
    template = tmp_path / "custom_skeleton.json"
    write_custom_topology_npz(source)
    write_custom_skeleton_template(template)

    parsed = parse_raw_motion_capture_npz(
        source,
        source_format="generic",
        skeleton_template=template,
    )

    assert parsed["local_axis_angle"].shape == (2, 5, 3)
    assert parsed["positions"].shape == (2, 5, 3)
    np.testing.assert_array_equal(parsed["parents"], np.array([-1, 0, 1, 1, 3]))
    np.testing.assert_allclose(parsed["rest_offsets"][4], np.array([0.2, 0.0, 0.0]))
    assert parsed["joint_names"].tolist() == [
        "root",
        "spine",
        "left_tip",
        "right_shoulder",
        "right_tip",
    ]
    np.testing.assert_allclose(parsed["positions"][:, 0], parsed["root_translation"])


def test_parse_raw_motion_capture_requires_template_for_non_smpl_topology(tmp_path):
    source = tmp_path / "smplx_like_sample.npz"
    poses = np.zeros((2, 30, 3), dtype=np.float32)
    np.savez(source, poses=poses)

    with pytest.raises(ValueError, match="skeleton template"):
        parse_raw_motion_capture_npz(source, source_format="smplx")


def test_parse_raw_motion_capture_rejects_invalid_pose_shape(tmp_path):
    source = tmp_path / "bad_sample.npz"
    np.savez(
        source,
        poses=np.zeros((2, 71), dtype=np.float32),
        trans=np.zeros((2, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="pose"):
        parse_raw_motion_capture_npz(source)


def test_parse_raw_motion_capture_reports_file_and_keys_for_non_motion_npz(tmp_path):
    source = tmp_path / "not_motion.npz"
    write_non_motion_npz(source)

    with pytest.raises(ValueError) as exc_info:
        parse_raw_motion_capture_npz(source, source_format="amass")

    message = str(exc_info.value)
    assert "not_motion.npz" in message
    assert "metadata" in message
    assert "fps" in message


def test_parse_raw_motion_capture_reports_file_for_broken_npz(tmp_path):
    source = tmp_path / "broken.npz"
    write_broken_npz(source)

    with pytest.raises(ValueError) as exc_info:
        parse_raw_motion_capture_npz(source, source_format="amass")

    message = str(exc_info.value)
    assert "broken.npz" in message
    assert "not a readable npz" in message


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


def test_convert_raw_motion_capture_directory_preserves_amass_smplh_hand_joints(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "normalized"
    raw_dir.mkdir()
    write_amass_smplh_like_npz(raw_dir / "amass_smplh_sample.npz")

    manifest_path = convert_raw_motion_capture_directory(
        input_dir=raw_dir,
        output_dir=out_dir,
        dataset_name="amass",
        source_format="amass",
    )
    adapter = NormalizedNpzAdapter(root=out_dir, manifest_path=manifest_path)
    sample = adapter[0]

    assert sample.positions.shape == (3, 52, 3)
    assert sample.local_rotations_6d.shape == (3, 52, 6)
    assert sample.rig.joint_names[22] == "left_index1"
    assert sample.rig.joint_names[51] == "right_thumb3"


def test_convert_raw_motion_capture_directory_preserves_custom_topology(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "normalized"
    template = tmp_path / "custom_skeleton.json"
    raw_dir.mkdir()
    write_custom_topology_npz(raw_dir / "custom_sample.npz")
    write_custom_skeleton_template(template)

    manifest_path = convert_raw_motion_capture_directory(
        input_dir=raw_dir,
        output_dir=out_dir,
        dataset_name="custom",
        source_format="generic",
        skeleton_template=template,
    )
    adapter = NormalizedNpzAdapter(root=out_dir, manifest_path=manifest_path)
    sample = adapter[0]

    assert len(adapter) == 1
    assert sample.dataset_name == "custom"
    assert sample.positions.shape == (2, 5, 3)
    assert sample.local_rotations_6d.shape == (2, 5, 6)
    assert sample.rig.joint_names == (
        "root",
        "spine",
        "left_tip",
        "right_shoulder",
        "right_tip",
    )
    np.testing.assert_array_equal(sample.rig.parents, np.array([-1, 0, 1, 1, 3]))


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


def test_convert_raw_motion_capture_directory_can_skip_invalid_npz(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "normalized"
    raw_dir.mkdir()
    write_amass_like_npz(raw_dir / "valid_motion.npz")
    write_non_motion_npz(raw_dir / "not_motion.npz")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manifest_path = convert_raw_motion_capture_directory(
            input_dir=raw_dir,
            output_dir=out_dir,
            dataset_name="amass",
            source_format="amass",
            skip_invalid=True,
        )
    adapter = NormalizedNpzAdapter(root=out_dir, manifest_path=manifest_path)

    assert caught == []
    assert len(adapter) == 1
    assert adapter.samples[0]["path"] == "valid_motion.npz"
    assert (out_dir / "skipped_raw_motion_npz.json").exists()


def test_convert_raw_motion_capture_directory_can_skip_broken_npz(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "normalized"
    raw_dir.mkdir()
    write_amass_like_npz(raw_dir / "valid_motion.npz")
    write_broken_npz(raw_dir / "broken.npz")

    manifest_path = convert_raw_motion_capture_directory(
        input_dir=raw_dir,
        output_dir=out_dir,
        dataset_name="amass",
        source_format="amass",
        skip_invalid=True,
    )
    adapter = NormalizedNpzAdapter(root=out_dir, manifest_path=manifest_path)

    assert len(adapter) == 1
    assert adapter.samples[0]["path"] == "valid_motion.npz"


def test_convert_raw_motion_capture_directory_can_report_verbose_skips(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "normalized"
    raw_dir.mkdir()
    write_amass_like_npz(raw_dir / "valid_motion.npz")
    write_non_motion_npz(raw_dir / "not_motion.npz")

    with pytest.warns(RuntimeWarning, match="Skipping invalid raw motion npz"):
        convert_raw_motion_capture_directory(
            input_dir=raw_dir,
            output_dir=out_dir,
            dataset_name="amass",
            source_format="amass",
            skip_invalid=True,
            verbose_skips=True,
        )


def test_convert_raw_motion_capture_directory_clears_stale_parsed_cache(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "normalized"
    raw_dir.mkdir()
    write_amass_like_npz(raw_dir / "first.npz")
    write_amass_like_npz(raw_dir / "second.npz")

    convert_raw_motion_capture_directory(
        input_dir=raw_dir,
        output_dir=out_dir,
        dataset_name="amass",
        source_format="amass",
    )
    (raw_dir / "second.npz").unlink()
    manifest_path = convert_raw_motion_capture_directory(
        input_dir=raw_dir,
        output_dir=out_dir,
        dataset_name="amass",
        source_format="amass",
    )

    adapter = NormalizedNpzAdapter(root=out_dir, manifest_path=manifest_path)
    assert [record["path"] for record in adapter.samples] == ["first.npz"]


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
    assert config.skip_invalid is False


def test_parse_args_accepts_skeleton_template_for_generic_topology(tmp_path):
    config = parse_args(
        [
            "--input-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "normalized"),
            "--dataset-name",
            "smplx",
            "--source-format",
            "smplx",
            "--skeleton-template",
            str(tmp_path / "smplx_template.json"),
        ]
    )

    assert config.source_format == "smplx"
    assert config.skeleton_template == tmp_path / "smplx_template.json"


def test_parse_args_accepts_skip_invalid(tmp_path):
    config = parse_args(
        [
            "--input-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "normalized"),
            "--dataset-name",
            "amass",
            "--skip-invalid",
        ]
    )

    assert config.skip_invalid is True


def test_parse_args_accepts_verbose_skips(tmp_path):
    config = parse_args(
        [
            "--input-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "normalized"),
            "--dataset-name",
            "amass",
            "--skip-invalid",
            "--verbose-skips",
        ]
    )

    assert config.skip_invalid is True
    assert config.verbose_skips is True


def test_raw_motion_capture_module_runs_without_preimport_warning():
    result = subprocess.run(
        [sys.executable, "-m", "preprocess.converters.raw_motion_capture", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "RuntimeWarning" not in result.stderr
