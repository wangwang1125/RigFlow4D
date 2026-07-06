from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class LatentFlowPair:
    z_t: Tensor
    t: Tensor
    target_velocity: Tensor
    z0: Tensor
    z1: Tensor


class LatentFlowMatcher(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        input_dim = latent_dim + condition_dim + 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: Tensor, t: Tensor, condition: Tensor) -> Tensor:
        if z_t.ndim != 2 or z_t.shape[-1] != self.latent_dim:
            raise ValueError(f"z_t must have shape [B, {self.latent_dim}], got {tuple(z_t.shape)}")
        if condition.ndim != 2 or condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"condition must have shape [B, {self.condition_dim}], got {tuple(condition.shape)}"
            )
        if condition.shape[0] != z_t.shape[0]:
            raise ValueError("condition batch dimension must match z_t")
        time_features = make_time_features(t.to(device=z_t.device, dtype=z_t.dtype))
        if time_features.shape[0] != z_t.shape[0]:
            raise ValueError("t batch dimension must match z_t")
        return self.net(torch.cat([z_t, time_features, condition], dim=-1))


def sample_latent_flow_pair(
    z0: Tensor,
    z1: Tensor,
    t: Optional[Tensor] = None,
) -> LatentFlowPair:
    if z0.shape != z1.shape:
        raise ValueError(f"z0 and z1 must share shape, got {tuple(z0.shape)} and {tuple(z1.shape)}")
    if z0.ndim != 2:
        raise ValueError(f"z0 and z1 must have shape [B, Dz], got {tuple(z0.shape)}")
    if t is None:
        t = torch.rand(z0.shape[0], device=z0.device, dtype=z0.dtype)
    else:
        t = t.to(device=z0.device, dtype=z0.dtype)
        if t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, 0]
    if t.shape != (z0.shape[0],):
        raise ValueError(f"t must have shape [B], got {tuple(t.shape)}")
    t_expanded = t[:, None]
    z_t = (1.0 - t_expanded) * z0 + t_expanded * z1
    target_velocity = z1 - z0
    return LatentFlowPair(
        z_t=z_t,
        t=t,
        target_velocity=target_velocity,
        z0=z0,
        z1=z1,
    )


def latent_flow_matching_loss(
    model: LatentFlowMatcher,
    z0: Tensor,
    z1: Tensor,
    condition: Tensor,
    t: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    pair = sample_latent_flow_pair(z0=z0, z1=z1, t=t)
    velocity = model(z_t=pair.z_t, t=pair.t, condition=condition)
    velocity_mse = F.mse_loss(velocity, pair.target_velocity)
    return {
        "loss": velocity_mse,
        "velocity_mse": velocity_mse,
    }


def make_time_features(t: Tensor) -> Tensor:
    if t.ndim == 2 and t.shape[-1] == 1:
        t = t[:, 0]
    if t.ndim != 1:
        raise ValueError(f"t must have shape [B] or [B, 1], got {tuple(t.shape)}")
    return torch.stack(
        [
            t,
            t.square(),
            torch.sin(math.pi * t),
            torch.cos(math.pi * t),
        ],
        dim=-1,
    )
