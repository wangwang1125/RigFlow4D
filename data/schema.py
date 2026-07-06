from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

import numpy as np


class InputType(str, Enum):
    IMAGE = "image"
    MULTIVIEW_IMAGE = "multiview_image"
    VIDEO = "video"
    MULTIVIEW_VIDEO = "multiview_video"


class CameraMode(str, Enum):
    CALIBRATED = "calibrated"
    WEAK_CALIBRATED = "weak_calibrated"
    UNKNOWN = "unknown"


class SourceLabelType(str, Enum):
    RIG_NATIVE = "rig_native"
    SMPL = "smpl"
    SMPLX = "smplx"
    MOTION_ONLY = "motion_only"


def _as_array(value: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    return value


def _expect_shape(array: np.ndarray, name: str, rank: int, last_dim: Optional[int] = None) -> None:
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {array.shape}")
    if last_dim is not None and array.shape[-1] != last_dim:
        raise ValueError(f"{name} last dimension must be {last_dim}, got shape {array.shape}")


@dataclass(frozen=True)
class RigDefinition:
    parents: np.ndarray
    rest_offsets: np.ndarray
    joint_names: Sequence[str]
    chain_ids: np.ndarray
    chain_coordinates: np.ndarray

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def validate(self) -> None:
        parents = _as_array(self.parents, "parents")
        rest_offsets = _as_array(self.rest_offsets, "rest_offsets")
        chain_ids = _as_array(self.chain_ids, "chain_ids")
        chain_coordinates = _as_array(self.chain_coordinates, "chain_coordinates")
        joint_count = self.num_joints

        if parents.shape != (joint_count,):
            raise ValueError(f"parents must have shape ({joint_count},), got {parents.shape}")
        if rest_offsets.shape != (joint_count, 3):
            raise ValueError(f"rest_offsets must have shape ({joint_count}, 3), got {rest_offsets.shape}")
        if chain_ids.shape != (joint_count,):
            raise ValueError(f"chain_ids must have shape ({joint_count},), got {chain_ids.shape}")
        if chain_coordinates.shape != (joint_count,):
            raise ValueError(
                f"chain_coordinates must have shape ({joint_count},), got {chain_coordinates.shape}"
            )


@dataclass(frozen=True)
class VisualTokenCache:
    tokens: np.ndarray
    backbone_name: str
    feature_dim: int
    patch_grid: Optional[Tuple[int, int]] = None
    has_cls: bool = True
    num_registers: int = 0

    def validate(self, expected_frames: Optional[int] = None) -> None:
        tokens = _as_array(self.tokens, "visual.tokens")
        _expect_shape(tokens, "visual.tokens", rank=4)
        if tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                f"visual.tokens feature dimension must match feature_dim={self.feature_dim}, "
                f"got shape {tokens.shape}"
            )
        if expected_frames is not None and tokens.shape[1] != expected_frames:
            raise ValueError(
                f"visual.tokens frame dimension must match T={expected_frames}, got shape {tokens.shape}"
            )
        if not self.backbone_name:
            raise ValueError("visual.backbone_name must be non-empty")
        if self.patch_grid is not None and len(self.patch_grid) != 2:
            raise ValueError("visual.patch_grid must be a pair of integers")


@dataclass(frozen=True)
class RigFlowSample:
    dataset_name: str
    input_type: InputType
    source_label_type: SourceLabelType
    camera_mode: CameraMode
    rig: RigDefinition
    positions: np.ndarray
    local_rotations_6d: np.ndarray
    root_translation: np.ndarray
    visual: Optional[VisualTokenCache] = None
    camera_intrinsics: Optional[np.ndarray] = None
    camera_extrinsics: Optional[np.ndarray] = None
    camera_valid_mask: Optional[np.ndarray] = None
    view_ids: Optional[np.ndarray] = None
    frame_times: Optional[np.ndarray] = None
    observed_time_mask: Optional[np.ndarray] = None
    contact_labels: Optional[np.ndarray] = None

    def validate(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty")

        self.rig.validate()
        joint_count = self.rig.num_joints

        positions = _as_array(self.positions, "positions")
        rotations = _as_array(self.local_rotations_6d, "local_rotations_6d")
        root_translation = _as_array(self.root_translation, "root_translation")

        _expect_shape(positions, "positions", rank=3, last_dim=3)
        if positions.shape[1] != joint_count:
            raise ValueError(
                f"positions joint count must match rig joints={joint_count}, got shape {positions.shape}"
            )

        _expect_shape(rotations, "local_rotations_6d", rank=3, last_dim=6)
        if rotations.shape[:2] != positions.shape[:2]:
            raise ValueError(
                "local_rotations_6d must share [T, J] with positions, "
                f"got {rotations.shape} vs {positions.shape}"
            )

        _expect_shape(root_translation, "root_translation", rank=2, last_dim=3)
        if root_translation.shape[0] != positions.shape[0]:
            raise ValueError(
                f"root_translation frame count must match positions T={positions.shape[0]}, "
                f"got shape {root_translation.shape}"
            )

        if self.visual is not None:
            self.visual.validate(expected_frames=positions.shape[0])

        self._validate_camera_fields()

    def _validate_camera_fields(self) -> None:
        if self.camera_mode == CameraMode.CALIBRATED:
            if self.camera_intrinsics is None or self.camera_extrinsics is None:
                raise ValueError(
                    "calibrated camera mode requires camera_intrinsics and camera_extrinsics"
                )
        if self.camera_intrinsics is not None:
            _expect_shape(_as_array(self.camera_intrinsics, "camera_intrinsics"), "camera_intrinsics", 3)
        if self.camera_extrinsics is not None:
            _expect_shape(_as_array(self.camera_extrinsics, "camera_extrinsics"), "camera_extrinsics", 3)
