from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_RIG_KEYS = (
    "parents",
    "rest_offsets",
    "joint_names",
    "chain_ids",
    "chain_coordinates",
)


def axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    vectors = np.asarray(axis_angle, dtype=np.float32)
    if vectors.ndim != 3 or vectors.shape[-1] != 3:
        raise ValueError(f"local_axis_angle must have shape [T, J, 3], got {vectors.shape}")

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

    rot6d = rot[:, :, :2].reshape(*vectors.shape[:2], 6)
    return rot6d.astype(np.float32)


def convert_motion_npz_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    source_label_type: str = "motion_only",
) -> Path:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for source_file in sorted(input_path.glob("*.npz")):
        out_file = output_path / source_file.name
        _convert_one(source_file, out_file, dataset_name, source_label_type)
        records.append({"sample_id": source_file.stem, "path": out_file.name})

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": records}, indent=2), encoding="utf-8")
    return manifest_path


def _convert_one(
    source_file: Path,
    out_file: Path,
    dataset_name: str,
    source_label_type: str,
) -> None:
    with np.load(source_file, allow_pickle=False) as src:
        _require_keys(src, source_file)
        positions = np.asarray(src["positions"], dtype=np.float32)
        local_rotations_6d = _read_rotations(src)
        root_translation = _read_root_translation(src, positions.shape[0])

        payload = {
            "dataset_name": np.array(dataset_name),
            "input_type": np.array("video"),
            "source_label_type": np.array(source_label_type),
            "camera_mode": np.array("unknown"),
            "parents": np.asarray(src["parents"]),
            "rest_offsets": np.asarray(src["rest_offsets"], dtype=np.float32),
            "joint_names": np.asarray(src["joint_names"]),
            "chain_ids": np.asarray(src["chain_ids"]),
            "chain_coordinates": np.asarray(src["chain_coordinates"], dtype=np.float32),
            "positions": positions,
            "local_rotations_6d": local_rotations_6d,
            "root_translation": root_translation,
        }
        if "contact_labels" in src:
            payload["contact_labels"] = np.asarray(src["contact_labels"])

    np.savez(out_file, **payload)


def _require_keys(src: np.lib.npyio.NpzFile, source_file: Path) -> None:
    missing = [key for key in (*REQUIRED_RIG_KEYS, "positions") if key not in src]
    if missing:
        raise ValueError(f"{source_file} is missing required keys: {', '.join(missing)}")


def _read_rotations(src: np.lib.npyio.NpzFile) -> np.ndarray:
    if "local_rotations_6d" in src:
        rotations = np.asarray(src["local_rotations_6d"], dtype=np.float32)
        if rotations.ndim != 3 or rotations.shape[-1] != 6:
            raise ValueError(f"local_rotations_6d must have shape [T, J, 6], got {rotations.shape}")
        return rotations
    if "local_axis_angle" in src:
        return axis_angle_to_rot6d(np.asarray(src["local_axis_angle"], dtype=np.float32))
    raise ValueError("motion npz must contain local_rotations_6d or local_axis_angle")


def _read_root_translation(src: np.lib.npyio.NpzFile, frames: int) -> np.ndarray:
    if "root_translation" in src:
        root_translation = np.asarray(src["root_translation"], dtype=np.float32)
        if root_translation.shape != (frames, 3):
            raise ValueError(f"root_translation must have shape ({frames}, 3), got {root_translation.shape}")
        return root_translation
    return np.zeros((frames, 3), dtype=np.float32)
