from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional
import warnings
import zipfile

import numpy as np

from .motion_npz import convert_motion_npz_directory


SOURCE_FORMATS = {"auto", "amass", "aistpp", "smpl24", "smplh", "smplx", "generic"}
TEMPLATE_REQUIRED_FORMATS = {"smplx", "generic"}

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

SMPLH52_JOINT_NAMES = SMPL24_JOINT_NAMES[:22] + (
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
)

SMPLH52_PARENTS = np.asarray(
    list(SMPL24_PARENTS[:22])
    + [
        20,
        22,
        23,
        20,
        25,
        26,
        20,
        28,
        29,
        20,
        31,
        32,
        20,
        34,
        35,
        21,
        37,
        38,
        21,
        40,
        41,
        21,
        43,
        44,
        21,
        46,
        47,
        21,
        49,
        50,
    ],
    dtype=np.int64,
)

_LEFT_HAND_REST_OFFSETS = np.asarray(
    [
        [-0.045, 0.010, 0.030],
        [-0.030, 0.000, 0.010],
        [-0.024, 0.000, 0.006],
        [-0.050, 0.005, 0.000],
        [-0.034, 0.000, 0.000],
        [-0.026, 0.000, 0.000],
        [-0.036, -0.002, -0.035],
        [-0.026, 0.000, -0.008],
        [-0.020, 0.000, -0.006],
        [-0.045, 0.000, -0.018],
        [-0.031, 0.000, -0.005],
        [-0.024, 0.000, -0.004],
        [-0.020, -0.010, 0.040],
        [-0.022, -0.006, 0.018],
        [-0.018, -0.004, 0.014],
    ],
    dtype=np.float32,
)

_RIGHT_HAND_REST_OFFSETS = _LEFT_HAND_REST_OFFSETS.copy()
_RIGHT_HAND_REST_OFFSETS[:, 0] *= -1.0

SMPLH52_REST_OFFSETS = np.vstack(
    [
        SMPL24_REST_OFFSETS[:22],
        _LEFT_HAND_REST_OFFSETS,
        _RIGHT_HAND_REST_OFFSETS,
    ]
).astype(np.float32)


@dataclass(frozen=True)
class RawMotionCaptureConfig:
    input_dir: Path
    output_dir: Path
    dataset_name: str
    source_format: str = "auto"
    skeleton_template: Optional[Path] = None
    skip_invalid: bool = False
    verbose_skips: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_dir", Path(self.input_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.skeleton_template is not None:
            object.__setattr__(self, "skeleton_template", Path(self.skeleton_template))
        _validate_source_format(self.source_format)
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty")


def parse_raw_motion_capture_npz(
    source_file: str | Path,
    source_format: str = "auto",
    skeleton_template: str | Path | Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    path = Path(source_file)
    _validate_source_format(source_format)
    template = load_skeleton_template(skeleton_template) if skeleton_template is not None else None
    with _open_npz(path) as src:
        try:
            raw_axis_angle = _read_local_axis_angle(src, source_format)
            local_axis_angle, skeleton = _select_axis_angle_and_skeleton(
                raw_axis_angle=raw_axis_angle,
                source_format=source_format,
                skeleton_template=template,
            )
            root_translation = _read_root_translation(src, local_axis_angle.shape[0], source_format)
        except ValueError as exc:
            raise ValueError(
                f"failed to parse raw motion npz '{path}': {exc}; "
                f"available keys: {_format_npz_keys(src)}"
            ) from exc

    parents = skeleton["parents"]
    rest_offsets = skeleton["rest_offsets"]
    positions = forward_kinematics_positions(
        local_axis_angle=local_axis_angle,
        root_translation=root_translation,
        parents=parents,
        rest_offsets=rest_offsets,
    )
    return {
        "parents": parents,
        "rest_offsets": rest_offsets,
        "joint_names": skeleton["joint_names"],
        "chain_ids": skeleton["chain_ids"],
        "chain_coordinates": skeleton["chain_coordinates"],
        "positions": positions,
        "root_translation": root_translation,
        "local_axis_angle": local_axis_angle,
    }


def convert_raw_motion_capture_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    source_format: str = "auto",
    skeleton_template: str | Path | Mapping[str, np.ndarray] | None = None,
    skip_invalid: bool = False,
    verbose_skips: bool = False,
) -> Path:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    template = load_skeleton_template(skeleton_template) if skeleton_template is not None else None
    parsed_path = output_path / "_parsed_motion"
    parsed_path.mkdir(parents=True, exist_ok=True)
    _clear_generated_npz_cache(parsed_path)

    parsed_count = 0
    skipped_records: list[dict[str, str]] = []
    for source_file in sorted(input_path.rglob("*.npz")):
        try:
            parsed = parse_raw_motion_capture_npz(
                source_file,
                source_format=source_format,
                skeleton_template=template,
            )
        except ValueError as exc:
            if not skip_invalid:
                raise
            skipped_records.append(
                {
                    "path": source_file.relative_to(input_path).as_posix(),
                    "reason": str(exc),
                }
            )
            if verbose_skips:
                warnings.warn(
                    f"Skipping invalid raw motion npz '{source_file}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue
        relative_name = _safe_relative_npz_name(source_file.relative_to(input_path))
        np.savez(parsed_path / relative_name, **parsed)
        parsed_count += 1

    if parsed_count == 0:
        raise ValueError(f"no valid raw motion npz files found under '{input_path}'")
    _write_skip_report(output_path, skipped_records)

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
    parser = argparse.ArgumentParser(description="Convert raw motion npz files")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-format", choices=sorted(SOURCE_FORMATS), default="auto")
    parser.add_argument(
        "--skeleton-template",
        type=Path,
        help="JSON/NPZ template with parents, rest_offsets, joint_names, and optional chain metadata.",
    )
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--verbose-skips", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return RawMotionCaptureConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        source_format=args.source_format,
        skeleton_template=args.skeleton_template,
        skip_invalid=args.skip_invalid,
        verbose_skips=args.verbose_skips,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    config = parse_args(argv)
    manifest_path = convert_raw_motion_capture_directory(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        dataset_name=config.dataset_name,
        source_format=config.source_format,
        skeleton_template=config.skeleton_template,
        skip_invalid=config.skip_invalid,
        verbose_skips=config.verbose_skips,
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
    return poses.astype(np.float32)


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


def _format_npz_keys(src: Mapping[str, np.ndarray]) -> str:
    keys = list(src.keys())
    return ", ".join(keys) if keys else "<none>"


def _open_npz(path: Path) -> np.lib.npyio.NpzFile:
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
        raise ValueError(f"failed to parse raw motion npz '{path}': not a readable npz ({exc})") from exc


def _write_skip_report(output_path: Path, skipped_records: list[dict[str, str]]) -> None:
    report_path = output_path / "skipped_raw_motion_npz.json"
    if not skipped_records:
        if report_path.exists():
            report_path.unlink()
        return
    report_path.write_text(
        json.dumps({"skipped": skipped_records}, indent=2),
        encoding="utf-8",
    )


def _clear_generated_npz_cache(parsed_path: Path) -> None:
    for stale_file in parsed_path.glob("*.npz"):
        stale_file.unlink()


def _pose_keys(source_format: str) -> tuple[str, ...]:
    if source_format in {"amass", "smpl24", "smplh", "smplx", "generic"}:
        return ("poses",)
    if source_format == "aistpp":
        return ("smpl_poses", "poses")
    return ("poses", "smpl_poses")


def _translation_keys(source_format: str) -> tuple[str, ...]:
    if source_format in {"amass", "smpl24", "smplh", "smplx", "generic"}:
        return ("trans", "transl", "root_translation")
    if source_format == "aistpp":
        return ("smpl_trans", "trans", "transl", "root_translation")
    return ("trans", "smpl_trans", "transl", "root_translation")


def _safe_relative_npz_name(relative_path: Path) -> str:
    stem_parts = list(relative_path.with_suffix("").parts)
    safe_stem = "__".join(_sanitize_name_part(part) for part in stem_parts)
    return f"{safe_stem}.npz"


def _sanitize_name_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def load_skeleton_template(template_path: str | Path | Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(template_path, Mapping):
        raw_template = template_path
    else:
        path = Path(template_path)
        if path.suffix.lower() == ".json":
            raw_template = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as src:
                raw_template = {key: src[key] for key in src.keys()}
        else:
            raise ValueError(f"skeleton template must be a JSON or NPZ file, got '{path}'")
    return _normalize_skeleton_template(raw_template)


def _normalize_skeleton_template(template: Mapping[str, object]) -> dict[str, np.ndarray]:
    try:
        parents = np.asarray(template["parents"], dtype=np.int64)
        rest_offsets = np.asarray(template["rest_offsets"], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"skeleton template is missing required key: {exc.args[0]}") from exc

    if parents.ndim != 1:
        raise ValueError(f"skeleton template parents must have shape [J], got {parents.shape}")
    joint_count = int(parents.shape[0])
    if joint_count == 0:
        raise ValueError("skeleton template must contain at least one joint")
    if rest_offsets.shape != (joint_count, 3):
        raise ValueError(
            f"skeleton template rest_offsets must have shape ({joint_count}, 3), "
            f"got {rest_offsets.shape}"
        )
    if int(parents[0]) >= 0:
        raise ValueError("skeleton template root joint at index 0 must have parent -1")
    for joint_index in range(1, joint_count):
        parent = int(parents[joint_index])
        if parent < 0 or parent >= joint_index:
            raise ValueError(
                "skeleton template parents must reference an earlier parent joint "
                f"for joint {joint_index}, got {parent}"
            )

    if "joint_names" in template:
        joint_names = np.asarray(template["joint_names"])
    else:
        joint_names = np.asarray([f"joint_{index}" for index in range(joint_count)])
    if joint_names.shape != (joint_count,):
        raise ValueError(
            f"skeleton template joint_names must have shape ({joint_count},), "
            f"got {joint_names.shape}"
        )

    chain_ids = (
        np.asarray(template["chain_ids"], dtype=np.int64)
        if "chain_ids" in template
        else _default_chain_ids(parents)
    )
    if chain_ids.shape != (joint_count,):
        raise ValueError(
            f"skeleton template chain_ids must have shape ({joint_count},), got {chain_ids.shape}"
        )
    chain_coordinates = (
        np.asarray(template["chain_coordinates"], dtype=np.float32)
        if "chain_coordinates" in template
        else _default_chain_coordinates(parents)
    )
    if chain_coordinates.shape != (joint_count,):
        raise ValueError(
            "skeleton template chain_coordinates must have shape "
            f"({joint_count},), got {chain_coordinates.shape}"
        )

    return {
        "parents": parents,
        "rest_offsets": rest_offsets,
        "joint_names": joint_names,
        "chain_ids": chain_ids,
        "chain_coordinates": chain_coordinates,
    }


def _select_axis_angle_and_skeleton(
    raw_axis_angle: np.ndarray,
    source_format: str,
    skeleton_template: Mapping[str, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw_joint_count = int(raw_axis_angle.shape[1])
    if skeleton_template is not None:
        skeleton = _normalize_skeleton_template(skeleton_template)
        template_joint_count = int(skeleton["parents"].shape[0])
        if raw_joint_count < template_joint_count:
            raise ValueError(
                f"pose array contains {raw_joint_count} joints but skeleton template "
                f"defines {template_joint_count} joints"
            )
        return raw_axis_angle[:, :template_joint_count].astype(np.float32), skeleton

    if source_format in TEMPLATE_REQUIRED_FORMATS:
        raise ValueError(
            f"skeleton template is required for source_format={source_format}; "
            "non-SMPL24 topologies must provide parents/rest_offsets/joint_names"
        )
    if source_format == "smplh":
        if raw_joint_count < 52:
            raise ValueError(f"SMPL-H pose array must contain at least 52 joints, got {raw_joint_count}")
        skeleton = _smplh52_skeleton()
        return raw_axis_angle[:, :52].astype(np.float32), skeleton
    if source_format in {"auto", "amass"} and raw_joint_count >= 52:
        skeleton = _smplh52_skeleton()
        return raw_axis_angle[:, :52].astype(np.float32), skeleton
    if raw_joint_count < 24:
        raise ValueError(f"pose array must contain at least 24 joints, got {raw_joint_count}")

    skeleton = _smpl24_skeleton()
    return raw_axis_angle[:, :24].astype(np.float32), skeleton


def _validate_source_format(source_format: str) -> None:
    if source_format not in SOURCE_FORMATS:
        valid_formats = ", ".join(sorted(SOURCE_FORMATS))
        raise ValueError(f"source_format must be one of: {valid_formats}")


def _smpl24_skeleton() -> dict[str, np.ndarray]:
    parents = SMPL24_PARENTS.copy()
    return {
        "parents": parents,
        "rest_offsets": SMPL24_REST_OFFSETS.copy(),
        "joint_names": np.asarray(SMPL24_JOINT_NAMES),
        "chain_ids": _default_chain_ids(parents),
        "chain_coordinates": _default_chain_coordinates(parents),
    }


def _smplh52_skeleton() -> dict[str, np.ndarray]:
    parents = SMPLH52_PARENTS.copy()
    return {
        "parents": parents,
        "rest_offsets": SMPLH52_REST_OFFSETS.copy(),
        "joint_names": np.asarray(SMPLH52_JOINT_NAMES),
        "chain_ids": _default_chain_ids(parents),
        "chain_coordinates": _default_chain_coordinates(parents),
    }


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
