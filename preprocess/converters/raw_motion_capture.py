from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np

from .motion_npz import convert_motion_npz_directory


SMPL24_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

SMPL24_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

SMPL24_REST_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [-0.09, -0.09, 0.0],
        [0.09, -0.09, 0.0],
        [0.0, 0.10, 0.0],
        [0.0, -0.42, 0.0],
        [0.0, -0.42, 0.0],
        [0.0, 0.12, 0.0],
        [0.0, -0.42, 0.0],
        [0.0, -0.42, 0.0],
        [0.0, 0.12, 0.0],
        [0.0, -0.08, 0.12],
        [0.0, -0.08, 0.12],
        [0.0, 0.12, 0.0],
        [-0.06, 0.09, 0.0],
        [0.06, 0.09, 0.0],
        [0.0, 0.12, 0.0],
        [-0.16, 0.02, 0.0],
        [0.16, 0.02, 0.0],
        [-0.28, 0.0, 0.0],
        [0.28, 0.0, 0.0],
        [-0.25, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        [-0.08, 0.0, 0.0],
        [0.08, 0.0, 0.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class RawMotionCaptureConfig:
    input_dir: Path
    output_dir: Path
    dataset_name: str
    source_format: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_dir", Path(self.input_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.source_format not in {"auto", "amass", "aistpp"}:
            raise ValueError("source_format must be one of: auto, amass, aistpp")
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty")


def parse_raw_motion_capture_npz(
    source_file: str | Path,
    source_format: str = "auto",
) -> dict[str, np.ndarray]:
    path = Path(source_file)
    if source_format not in {"auto", "amass", "aistpp"}:
        raise ValueError("source_format must be one of: auto, amass, aistpp")
    with np.load(path, allow_pickle=False) as src:
        local_axis_angle = _read_local_axis_angle(src, source_format)
        root_translation = _read_root_translation(src, local_axis_angle.shape[0], source_format)

    joint_count = local_axis_angle.shape[1]
    parents = SMPL24_PARENTS[:joint_count].copy()
    rest_offsets = SMPL24_REST_OFFSETS[:joint_count].copy()
    positions = forward_kinematics_positions(
        local_axis_angle=local_axis_angle,
        root_translation=root_translation,
        parents=parents,
        rest_offsets=rest_offsets,
    )
    return {
        "parents": parents,
        "rest_offsets": rest_offsets,
        "joint_names": np.asarray(SMPL24_JOINT_NAMES[:joint_count]),
        "chain_ids": _default_chain_ids(parents),
        "chain_coordinates": _default_chain_coordinates(parents),
        "positions": positions,
        "root_translation": root_translation,
        "local_axis_angle": local_axis_angle,
    }


def convert_raw_motion_capture_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    source_format: str = "auto",
) -> Path:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    parsed_path = output_path / "_parsed_motion"
    parsed_path.mkdir(parents=True, exist_ok=True)

    for source_file in sorted(input_path.glob("*.npz")):
        parsed = parse_raw_motion_capture_npz(source_file, source_format=source_format)
        np.savez(parsed_path / source_file.name, **parsed)

    return convert_motion_npz_directory(
        input_dir=parsed_path,
        output_dir=output_path,
        dataset_name=dataset_name,
        source_label_type="motion_only",
    )


def forward_kinematics_positions(
    local_axis_angle: np.ndarray,
    root_translation: np.ndarray,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
) -> np.ndarray:
    rotations = axis_angle_to_matrix(local_axis_angle)
    frames, joints = local_axis_angle.shape[:2]
    positions = np.zeros((frames, joints, 3), dtype=np.float32)
    global_rotations = np.zeros((frames, joints, 3, 3), dtype=np.float32)

    for joint_index in range(joints):
        parent = int(parents[joint_index])
        if parent < 0:
            positions[:, joint_index] = root_translation
            global_rotations[:, joint_index] = rotations[:, joint_index]
            continue
        global_rotations[:, joint_index] = np.einsum(
            "tij,tjk->tik",
            global_rotations[:, parent],
            rotations[:, joint_index],
        )
        rotated_offset = np.einsum(
            "tij,j->ti",
            global_rotations[:, parent],
            rest_offsets[joint_index],
        )
        positions[:, joint_index] = positions[:, parent] + rotated_offset
    return positions


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    vectors = np.asarray(axis_angle, dtype=np.float32)
    if vectors.ndim != 3 or vectors.shape[-1] != 3:
        raise ValueError(f"axis_angle must have shape [T, J, 3], got {vectors.shape}")

    flat = vectors.reshape(-1, 3)
    angles = np.linalg.norm(flat, axis=1, keepdims=True)
    axes = np.divide(flat, angles, out=np.zeros_like(flat), where=angles > 1e-8)
    x = axes[:, 0]
    y = axes[:, 1]
    z = axes[:, 2]
    cos = np.cos(angles[:, 0])
    sin = np.sin(angles[:, 0])
    one_minus_cos = 1.0 - cos

    rot = np.empty((flat.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = cos + x * x * one_minus_cos
    rot[:, 0, 1] = x * y * one_minus_cos - z * sin
    rot[:, 0, 2] = x * z * one_minus_cos + y * sin
    rot[:, 1, 0] = y * x * one_minus_cos + z * sin
    rot[:, 1, 1] = cos + y * y * one_minus_cos
    rot[:, 1, 2] = y * z * one_minus_cos - x * sin
    rot[:, 2, 0] = z * x * one_minus_cos - y * sin
    rot[:, 2, 1] = z * y * one_minus_cos + x * sin
    rot[:, 2, 2] = cos + z * z * one_minus_cos

    identity_mask = angles[:, 0] <= 1e-8
    if np.any(identity_mask):
        rot[identity_mask] = np.eye(3, dtype=np.float32)
    return rot.reshape(*vectors.shape[:2], 3, 3)


def parse_args(argv: Optional[Iterable[str]] = None) -> RawMotionCaptureConfig:
    parser = argparse.ArgumentParser(description="Convert AMASS/AIST++ raw motion npz files")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-format", choices=["auto", "amass", "aistpp"], default="auto")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return RawMotionCaptureConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        source_format=args.source_format,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    config = parse_args(argv)
    manifest_path = convert_raw_motion_capture_directory(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        dataset_name=config.dataset_name,
        source_format=config.source_format,
    )
    print(manifest_path)
    return 0


def _read_local_axis_angle(src: Mapping[str, np.ndarray], source_format: str) -> np.ndarray:
    pose_key = _select_pose_key(src, source_format)
    poses = np.asarray(src[pose_key], dtype=np.float32)
    if poses.ndim == 2:
        if poses.shape[1] % 3 != 0:
            raise ValueError(f"pose vector dimension must be divisible by 3, got {poses.shape}")
        poses = poses.reshape(poses.shape[0], poses.shape[1] // 3, 3)
    elif poses.ndim != 3 or poses.shape[-1] != 3:
        raise ValueError(f"pose array must have shape [T, J*3] or [T, J, 3], got {poses.shape}")
    if poses.shape[1] < 24:
        raise ValueError(f"pose array must contain at least 24 joints, got {poses.shape[1]}")
    return poses[:, :24].astype(np.float32)


def _read_root_translation(
    src: Mapping[str, np.ndarray],
    frames: int,
    source_format: str,
) -> np.ndarray:
    for key in _translation_keys(source_format):
        if key in src:
            translation = np.asarray(src[key], dtype=np.float32)
            if translation.shape != (frames, 3):
                raise ValueError(
                    f"{key} must have shape ({frames}, 3), got {translation.shape}"
                )
            return translation
    return np.zeros((frames, 3), dtype=np.float32)


def _select_pose_key(src: Mapping[str, np.ndarray], source_format: str) -> str:
    for key in _pose_keys(source_format):
        if key in src:
            return key
    raise ValueError(f"raw motion npz is missing pose keys for source_format={source_format}")


def _pose_keys(source_format: str) -> tuple[str, ...]:
    if source_format == "amass":
        return ("poses",)
    if source_format == "aistpp":
        return ("smpl_poses", "poses")
    return ("poses", "smpl_poses")


def _translation_keys(source_format: str) -> tuple[str, ...]:
    if source_format == "amass":
        return ("trans", "transl", "root_translation")
    if source_format == "aistpp":
        return ("smpl_trans", "trans", "transl", "root_translation")
    return ("trans", "smpl_trans", "transl", "root_translation")


def _default_chain_ids(parents: np.ndarray) -> np.ndarray:
    chain_ids = np.zeros((parents.shape[0],), dtype=np.int64)
    for joint_index in range(1, parents.shape[0]):
        parent = int(parents[joint_index])
        chain_ids[joint_index] = joint_index if parent == 0 else chain_ids[parent]
    return chain_ids


def _default_chain_coordinates(parents: np.ndarray) -> np.ndarray:
    depth = np.zeros((parents.shape[0],), dtype=np.float32)
    for joint_index in range(1, parents.shape[0]):
        parent = int(parents[joint_index])
        depth[joint_index] = depth[parent] + 1.0
    max_depth = max(float(depth.max()), 1.0)
    return depth / max_depth


if __name__ == "__main__":
    raise SystemExit(main())
