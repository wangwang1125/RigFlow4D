from __future__ import annotations

from dataclasses import dataclass
import math
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


class FactorizedTemporalSpatialBlock(nn.Module):
    """Alternates temporal attention per joint with spatial attention per frame."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x: Tensor, time_mask: Tensor, joint_mask: Tensor) -> Tensor:
        batch_size, frames, joints, hidden_dim = x.shape
        temporal_x = x.permute(0, 2, 1, 3).reshape(batch_size * joints, frames, hidden_dim)
        temporal_pad = (~time_mask[:, None, :]).expand(batch_size, joints, frames)
        temporal_x = self.temporal(
            temporal_x,
            src_key_padding_mask=temporal_pad.reshape(batch_size * joints, frames),
        )
        x = temporal_x.reshape(batch_size, joints, frames, hidden_dim).permute(0, 2, 1, 3)
        x = _apply_token_mask(x, time_mask=time_mask, joint_mask=joint_mask)

        spatial_x = x.reshape(batch_size * frames, joints, hidden_dim)
        spatial_pad = (~joint_mask[:, None, :]).expand(batch_size, frames, joints)
        spatial_x = self.spatial(
            spatial_x,
            src_key_padding_mask=spatial_pad.reshape(batch_size * frames, joints),
        )
        x = spatial_x.reshape(batch_size, frames, joints, hidden_dim)
        return _apply_token_mask(x, time_mask=time_mask, joint_mask=joint_mask)


class KinematicVAE(nn.Module):
    def __init__(
        self,
        feature_dim: int = 9,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if feature_dim != 9:
            raise ValueError("KinematicVAE expects feature_dim=9: xyz position + 6D rotation")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            [
                FactorizedTemporalSpatialBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        self.decoder_blocks = nn.ModuleList(
            [
                FactorizedTemporalSpatialBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, feature_dim)

    def forward(self, batch: Dict[str, Tensor]) -> KinematicVAEOutput:
        positions = batch["positions"]
        rotations = batch["local_rotations_6d"]
        time_mask = _mask_or_ones(batch.get("time_mask"), positions.shape[:2], positions.device)
        joint_mask = _mask_or_ones(batch.get("joint_mask"), (positions.shape[0], positions.shape[2]), positions.device)
        features = torch.cat([positions, rotations], dim=-1)

        x = self.input_projection(features)
        x = x + _time_embedding(positions.shape[1], self.hidden_dim, positions.device, x.dtype)
        x = x + _joint_embedding(positions.shape[2], self.hidden_dim, positions.device, x.dtype)
        x = _apply_token_mask(x, time_mask=time_mask, joint_mask=joint_mask)
        for block in self.encoder_blocks:
            x = block(x, time_mask=time_mask, joint_mask=joint_mask)

        pooled = self._masked_pool(self.encoder_norm(x), time_mask=time_mask, joint_mask=joint_mask)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled).clamp(min=-20.0, max=20.0)
        z = self.reparameterize(mu, logvar)

        decoded = self._decode(z, frames=positions.shape[1], joints=positions.shape[2], time_mask=time_mask, joint_mask=joint_mask)
        return KinematicVAEOutput(
            positions=decoded[..., :3],
            local_rotations_6d=decoded[..., 3:9],
            mu=mu,
            logvar=logvar,
            z=z,
        )

    def _decode(self, z: Tensor, frames: int, joints: int, time_mask: Tensor, joint_mask: Tensor) -> Tensor:
        x = self.latent_projection(z)[:, None, None, :].expand(z.shape[0], frames, joints, self.hidden_dim)
        x = x + _time_embedding(frames, self.hidden_dim, z.device, x.dtype)
        x = x + _joint_embedding(joints, self.hidden_dim, z.device, x.dtype)
        x = _apply_token_mask(x, time_mask=time_mask, joint_mask=joint_mask)
        for block in self.decoder_blocks:
            x = block(x, time_mask=time_mask, joint_mask=joint_mask)
        x = self.decoder_norm(x)
        return self.output_projection(x)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _masked_pool(features: Tensor, time_mask: Tensor, joint_mask: Tensor) -> Tensor:
        mask = _motion_mask(features, time_mask, joint_mask)
        denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
        return (features * mask).sum(dim=(1, 2)) / denom


def kinematic_vae_loss(
    output: KinematicVAEOutput,
    batch: Dict[str, Tensor],
    beta: float = 1e-3,
    velocity_weight: float = 0.1,
    acceleration_weight: float = 0.01,
    bone_length_weight: float = 0.1,
    root_velocity_weight: float = 0.05,
) -> Dict[str, Tensor]:
    positions = batch["positions"]
    rotations = batch["local_rotations_6d"]
    time_mask = _mask_or_ones(batch.get("time_mask"), positions.shape[:2], positions.device)
    joint_mask = _mask_or_ones(batch.get("joint_mask"), (positions.shape[0], positions.shape[2]), positions.device)

    recon_position = _masked_mse(output.positions, positions, _motion_mask(positions, time_mask, joint_mask))
    recon_rotation = _masked_mse(
        output.local_rotations_6d,
        rotations,
        _motion_mask(rotations, time_mask, joint_mask),
    )
    kl = -0.5 * torch.mean(1.0 + output.logvar - output.mu.pow(2) - output.logvar.exp())
    velocity = _temporal_difference_loss(output.positions, positions, time_mask, joint_mask, order=1)
    acceleration = _temporal_difference_loss(output.positions, positions, time_mask, joint_mask, order=2)
    bone_length = _bone_length_loss(
        prediction=output.positions,
        target=positions,
        parents=batch.get("parents"),
        time_mask=time_mask,
        joint_mask=joint_mask,
    )
    root_velocity = _root_velocity_loss(
        prediction=output.positions,
        target_positions=positions,
        target_root_translation=batch.get("root_translation"),
        time_mask=time_mask,
    )

    loss = (
        recon_position
        + recon_rotation
        + beta * kl
        + velocity_weight * velocity
        + acceleration_weight * acceleration
        + bone_length_weight * bone_length
        + root_velocity_weight * root_velocity
    )
    return {
        "loss": loss,
        "recon_position": recon_position,
        "recon_rotation": recon_rotation,
        "kl": kl,
        "velocity": velocity,
        "acceleration": acceleration,
        "bone_length": bone_length,
        "root_velocity": root_velocity,
    }


def _time_embedding(length: int, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return _sinusoidal_embedding(length, hidden_dim, device, dtype).view(1, length, 1, hidden_dim)


def _joint_embedding(length: int, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return _sinusoidal_embedding(length, hidden_dim, device, dtype).view(1, 1, length, hidden_dim)


def _sinusoidal_embedding(length: int, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, hidden_dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / hidden_dim)
    )
    embedding = torch.zeros(length, hidden_dim, device=device, dtype=dtype)
    embedding[:, 0::2] = torch.sin(positions * div)
    embedding[:, 1::2] = torch.cos(positions * div[: embedding[:, 1::2].shape[1]])
    return embedding


def _mask_or_ones(mask: Tensor | None, shape: tuple[int, ...], device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones(shape, device=device, dtype=torch.bool)
    return mask.to(device=device, dtype=torch.bool)


def _apply_token_mask(x: Tensor, time_mask: Tensor, joint_mask: Tensor) -> Tensor:
    token_mask = time_mask[:, :, None, None] & joint_mask[:, None, :, None]
    return x * token_mask.to(dtype=x.dtype, device=x.device)


def _motion_mask(target: Tensor, time_mask: Tensor, joint_mask: Tensor) -> Tensor:
    return (time_mask[:, :, None, None] & joint_mask[:, None, :, None]).to(dtype=target.dtype, device=target.device)


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    squared = F.mse_loss(prediction, target, reduction="none")
    expanded_mask = mask.expand_as(target)
    denom = expanded_mask.sum().clamp_min(1.0)
    return (squared * expanded_mask).sum() / denom


def _temporal_difference_loss(
    prediction: Tensor,
    target: Tensor,
    time_mask: Tensor,
    joint_mask: Tensor,
    order: int,
) -> Tensor:
    if prediction.shape[1] <= order:
        return prediction.new_zeros(())
    pred_diff = torch.diff(prediction, n=order, dim=1)
    target_diff = torch.diff(target, n=order, dim=1)
    valid_time = time_mask[:, order:].clone()
    for offset in range(order):
        valid_time = valid_time & time_mask[:, offset : offset + valid_time.shape[1]]
    mask = _motion_mask(pred_diff, valid_time, joint_mask)
    return _masked_mse(pred_diff, target_diff, mask)


def _bone_length_loss(
    prediction: Tensor,
    target: Tensor,
    parents: Tensor | None,
    time_mask: Tensor,
    joint_mask: Tensor,
) -> Tensor:
    if parents is None:
        return prediction.new_zeros(())
    batch_size, _, joints, _ = prediction.shape
    parents = parents.to(device=prediction.device, dtype=torch.long)
    if parents.ndim == 1:
        parents = parents[None, :].expand(batch_size, joints)

    total = prediction.new_zeros(())
    denom = prediction.new_zeros(())
    joint_indices = torch.arange(joints, device=prediction.device)
    for batch_index in range(batch_size):
        parent_index = parents[batch_index].clamp(min=0, max=max(joints - 1, 0))
        valid = parents[batch_index] >= 0
        valid = valid & joint_mask[batch_index]
        valid = valid & joint_mask[batch_index, parent_index]
        if not torch.any(valid):
            continue
        child = joint_indices[valid]
        parent = parent_index[valid]
        pred_len = torch.linalg.norm(prediction[batch_index, :, child] - prediction[batch_index, :, parent], dim=-1)
        target_len = torch.linalg.norm(target[batch_index, :, child] - target[batch_index, :, parent], dim=-1)
        mask = time_mask[batch_index, :, None].to(dtype=prediction.dtype)
        squared = F.mse_loss(pred_len, target_len, reduction="none")
        total = total + (squared * mask).sum()
        denom = denom + mask.expand_as(squared).sum()
    return total / denom.clamp_min(1.0)


def _root_velocity_loss(
    prediction: Tensor,
    target_positions: Tensor,
    target_root_translation: Tensor | None,
    time_mask: Tensor,
) -> Tensor:
    if prediction.shape[1] <= 1:
        return prediction.new_zeros(())
    pred_root = prediction[:, :, 0]
    target_root = target_root_translation.to(prediction.device, prediction.dtype) if target_root_translation is not None else target_positions[:, :, 0]
    pred_velocity = pred_root[:, 1:] - pred_root[:, :-1]
    target_velocity = target_root[:, 1:] - target_root[:, :-1]
    valid_time = time_mask[:, 1:] & time_mask[:, :-1]
    mask = valid_time[:, :, None].to(dtype=prediction.dtype, device=prediction.device)
    return _masked_mse(pred_velocity, target_velocity, mask)
