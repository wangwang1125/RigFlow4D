from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class RigFlowConditionOutput:
    condition: Tensor
    components: Dict[str, Tensor]


def masked_mean(
    values: Tensor,
    mask: Optional[Tensor],
    reduce_dims: Tuple[int, ...],
    eps: float = 1e-6,
) -> Tensor:
    if mask is None:
        return values.mean(dim=reduce_dims)
    expected_mask_shape = tuple(values.shape[:-1])
    if tuple(mask.shape) != expected_mask_shape:
        raise ValueError(
            f"mask must have shape {expected_mask_shape} for values {tuple(values.shape)}, "
            f"got {tuple(mask.shape)}"
        )
    mask = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    numerator = (values * mask).sum(dim=reduce_dims)
    denominator = mask.sum(dim=reduce_dims).clamp_min(eps)
    return numerator / denominator


class RigFlowConditionEncoder(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        camera_dim: int,
        rig_dim: int,
        pose_seed_dim: int,
        condition_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        _require_positive("visual_dim", visual_dim)
        _require_positive("camera_dim", camera_dim)
        _require_positive("rig_dim", rig_dim)
        _require_positive("pose_seed_dim", pose_seed_dim)
        _require_positive("condition_dim", condition_dim)
        _require_positive("hidden_dim", hidden_dim)

        self.visual_dim = visual_dim
        self.camera_dim = camera_dim
        self.rig_dim = rig_dim
        self.pose_seed_dim = pose_seed_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim

        self.visual_proj = _projection(visual_dim, hidden_dim)
        self.camera_proj = _projection(camera_dim, hidden_dim)
        self.rig_proj = _projection(rig_dim, hidden_dim)
        self.pose_seed_proj = _projection(pose_seed_dim, hidden_dim)

        self.visual_missing = nn.Parameter(torch.zeros(hidden_dim))
        self.camera_missing = nn.Parameter(torch.zeros(hidden_dim))
        self.rig_missing = nn.Parameter(torch.zeros(hidden_dim))
        self.pose_seed_missing = nn.Parameter(torch.zeros(hidden_dim))

        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, condition_dim),
        )

    def forward(
        self,
        visual_tokens: Optional[Tensor] = None,
        visual_mask: Optional[Tensor] = None,
        camera_features: Optional[Tensor] = None,
        camera_mask: Optional[Tensor] = None,
        rig_features: Optional[Tensor] = None,
        joint_mask: Optional[Tensor] = None,
        pose_seed: Optional[Tensor] = None,
        time_mask: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> RigFlowConditionOutput:
        batch = self._infer_batch_size(
            batch_size,
            visual_tokens,
            camera_features,
            rig_features,
            pose_seed,
        )
        device, dtype = self._infer_device_dtype(
            visual_tokens,
            camera_features,
            rig_features,
            pose_seed,
        )

        visual = self._encode_visual(visual_tokens, visual_mask, batch, device, dtype)
        camera = self._encode_camera(camera_features, camera_mask, batch, device, dtype)
        rig = self._encode_rig(rig_features, joint_mask, batch, device, dtype)
        pose = self._encode_pose_seed(pose_seed, time_mask, joint_mask, batch, device, dtype)
        fused = torch.cat([visual, camera, rig, pose], dim=-1)

        return RigFlowConditionOutput(
            condition=self.fuse(fused),
            components={
                "visual": visual,
                "camera": camera,
                "rig": rig,
                "pose_seed": pose,
            },
        )

    def _encode_visual(
        self,
        visual_tokens: Optional[Tensor],
        visual_mask: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if visual_tokens is None:
            return self._missing(self.visual_missing, batch_size, device, dtype)
        _expect_rank("visual_tokens", visual_tokens, 5)
        _expect_batch("visual_tokens", visual_tokens, batch_size)
        _expect_last_dim("visual_tokens", visual_tokens, self.visual_dim)
        pooled = masked_mean(visual_tokens, visual_mask, reduce_dims=(1, 2, 3))
        return self.visual_proj(pooled)

    def _encode_camera(
        self,
        camera_features: Optional[Tensor],
        camera_mask: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if camera_features is None:
            return self._missing(self.camera_missing, batch_size, device, dtype)
        if camera_features.ndim == 3:
            reduce_dims = (1,)
        elif camera_features.ndim == 4:
            reduce_dims = (1, 2)
        else:
            raise ValueError(
                "camera_features must have shape [B, V, Dc] or [B, V, T, Dc], "
                f"got {tuple(camera_features.shape)}"
            )
        _expect_batch("camera_features", camera_features, batch_size)
        _expect_last_dim("camera_features", camera_features, self.camera_dim)
        pooled = masked_mean(camera_features, camera_mask, reduce_dims=reduce_dims)
        return self.camera_proj(pooled)

    def _encode_rig(
        self,
        rig_features: Optional[Tensor],
        joint_mask: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if rig_features is None:
            return self._missing(self.rig_missing, batch_size, device, dtype)
        _expect_rank("rig_features", rig_features, 3)
        _expect_batch("rig_features", rig_features, batch_size)
        _expect_last_dim("rig_features", rig_features, self.rig_dim)
        pooled = masked_mean(rig_features, joint_mask, reduce_dims=(1,))
        return self.rig_proj(pooled)

    def _encode_pose_seed(
        self,
        pose_seed: Optional[Tensor],
        time_mask: Optional[Tensor],
        joint_mask: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if pose_seed is None:
            return self._missing(self.pose_seed_missing, batch_size, device, dtype)
        _expect_rank("pose_seed", pose_seed, 4)
        _expect_batch("pose_seed", pose_seed, batch_size)
        _expect_last_dim("pose_seed", pose_seed, self.pose_seed_dim)
        pose_mask = _build_pose_mask(pose_seed, time_mask, joint_mask)
        pooled = masked_mean(pose_seed, pose_mask, reduce_dims=(1, 2))
        return self.pose_seed_proj(pooled)

    def _missing(
        self,
        token: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return token.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)

    def _infer_batch_size(self, batch_size: Optional[int], *tensors: Optional[Tensor]) -> int:
        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            return int(batch_size)
        for tensor in tensors:
            if tensor is not None:
                return int(tensor.shape[0])
        raise ValueError("batch_size is required when all condition inputs are missing")

    def _infer_device_dtype(self, *tensors: Optional[Tensor]) -> Tuple[torch.device, torch.dtype]:
        for tensor in tensors:
            if tensor is not None:
                return tensor.device, tensor.dtype
        return self.visual_missing.device, self.visual_missing.dtype


def _projection(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
    )


def _build_pose_mask(
    pose_seed: Tensor,
    time_mask: Optional[Tensor],
    joint_mask: Optional[Tensor],
) -> Optional[Tensor]:
    batch, frames, joints = pose_seed.shape[:3]
    mask = None
    if time_mask is not None:
        if tuple(time_mask.shape) != (batch, frames):
            raise ValueError(
                f"time_mask must have shape {(batch, frames)}, got {tuple(time_mask.shape)}"
            )
        mask = time_mask.to(device=pose_seed.device, dtype=torch.bool)[:, :, None].expand(
            batch, frames, joints
        )
    if joint_mask is not None:
        if tuple(joint_mask.shape) != (batch, joints):
            raise ValueError(
                f"joint_mask must have shape {(batch, joints)}, got {tuple(joint_mask.shape)}"
            )
        joint_mask = joint_mask.to(device=pose_seed.device, dtype=torch.bool)[:, None, :].expand(
            batch, frames, joints
        )
        mask = joint_mask if mask is None else mask & joint_mask
    return mask


def _expect_rank(name: str, tensor: Tensor, rank: int) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")


def _expect_batch(name: str, tensor: Tensor, batch_size: int) -> None:
    if tensor.shape[0] != batch_size:
        raise ValueError(f"{name} batch size must be {batch_size}, got {tensor.shape[0]}")


def _expect_last_dim(name: str, tensor: Tensor, expected_dim: int) -> None:
    if tensor.shape[-1] != expected_dim:
        raise ValueError(
            f"{name} feature dimension must be {expected_dim}, got shape {tuple(tensor.shape)}"
        )


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
