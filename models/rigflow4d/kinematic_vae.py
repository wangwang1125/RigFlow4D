from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class KinematicVAEOutput:
    positions: Tensor
    local_rotations_6d: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor


class KinematicVAE(nn.Module):
    def __init__(
        self,
        feature_dim: int = 9,
        hidden_dim: int = 256,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, batch: Dict[str, Tensor]) -> KinematicVAEOutput:
        positions = batch["positions"]
        rotations = batch["local_rotations_6d"]
        time_mask = batch.get("time_mask")
        joint_mask = batch.get("joint_mask")
        features = torch.cat([positions, rotations], dim=-1)

        pooled = self._masked_pool(features, time_mask=time_mask, joint_mask=joint_mask)
        hidden = self.encoder(pooled)
        mu = self.to_mu(hidden)
        logvar = self.to_logvar(hidden).clamp(min=-20.0, max=20.0)
        z = self.reparameterize(mu, logvar)

        decoded_feature = self.decoder(z)
        decoded = decoded_feature[:, None, None, :].expand(
            positions.shape[0],
            positions.shape[1],
            positions.shape[2],
            self.feature_dim,
        )
        return KinematicVAEOutput(
            positions=decoded[..., :3],
            local_rotations_6d=decoded[..., 3:9],
            mu=mu,
            logvar=logvar,
            z=z,
        )

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _masked_pool(features: Tensor, time_mask: Tensor | None, joint_mask: Tensor | None) -> Tensor:
        mask = torch.ones_like(features[..., :1])
        if time_mask is not None:
            mask = mask * time_mask[:, :, None, None].to(dtype=features.dtype, device=features.device)
        if joint_mask is not None:
            mask = mask * joint_mask[:, None, :, None].to(dtype=features.dtype, device=features.device)
        denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
        return (features * mask).sum(dim=(1, 2)) / denom


def kinematic_vae_loss(
    output: KinematicVAEOutput,
    batch: Dict[str, Tensor],
    beta: float = 1e-3,
) -> Dict[str, Tensor]:
    time_mask = batch.get("time_mask")
    joint_mask = batch.get("joint_mask")
    position_mask = _motion_mask(batch["positions"], time_mask, joint_mask)
    rotation_mask = _motion_mask(batch["local_rotations_6d"], time_mask, joint_mask)

    recon_position = _masked_mse(output.positions, batch["positions"], position_mask)
    recon_rotation = _masked_mse(
        output.local_rotations_6d,
        batch["local_rotations_6d"],
        rotation_mask,
    )
    kl = -0.5 * torch.mean(1.0 + output.logvar - output.mu.pow(2) - output.logvar.exp())
    loss = recon_position + recon_rotation + beta * kl
    return {
        "loss": loss,
        "recon_position": recon_position,
        "recon_rotation": recon_rotation,
        "kl": kl,
    }


def _motion_mask(target: Tensor, time_mask: Tensor | None, joint_mask: Tensor | None) -> Tensor:
    mask = torch.ones_like(target)
    if time_mask is not None:
        mask = mask * time_mask[:, :, None, None].to(dtype=target.dtype, device=target.device)
    if joint_mask is not None:
        mask = mask * joint_mask[:, None, :, None].to(dtype=target.dtype, device=target.device)
    return mask


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    squared = F.mse_loss(prediction, target, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (squared * mask).sum() / denom
