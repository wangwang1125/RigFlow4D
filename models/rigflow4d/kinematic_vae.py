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
    root_translation: Tensor
    root_relative_positions: Tensor
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
        use_graph_mixer: bool = True,
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
        self.graph_mixer = SkeletonGraphMixer(hidden_dim=hidden_dim, dropout=dropout) if use_graph_mixer else None

    def forward(self, x: Tensor, time_mask: Tensor, joint_mask: Tensor, parents: Tensor | None = None) -> Tensor:
        batch_size, frames, joints, hidden_dim = x.shape
        temporal_x = x.permute(0, 2, 1, 3).reshape(batch_size * joints, frames, hidden_dim)
        temporal_pad = (~time_mask[:, None, :]).expand(batch_size, joints, frames)
        temporal_x = self.temporal(
            temporal_x,
            src_key_padding_mask=temporal_pad.reshape(batch_size * joints, frames),
        )
        x = temporal_x.reshape(batch_size, joints, frames, hidden_dim).permute(0, 2, 1, 3)
        x = _apply_token_mask(x, time_mask=time_mask, joint_mask=joint_mask)
        if self.graph_mixer is not None and parents is not None:
            x = self.graph_mixer(x, parents=parents, joint_mask=joint_mask)
            x = _apply_token_mask(x, time_mask=time_mask, joint_mask=joint_mask)

        spatial_x = x.reshape(batch_size * frames, joints, hidden_dim)
        spatial_pad = (~joint_mask[:, None, :]).expand(batch_size, frames, joints)
        spatial_x = self.spatial(
            spatial_x,
            src_key_padding_mask=spatial_pad.reshape(batch_size * frames, joints),
        )
        x = spatial_x.reshape(batch_size, frames, joints, hidden_dim)
        return _apply_token_mask(x, time_mask=time_mask, joint_mask=joint_mask)


class SkeletonGraphMixer(nn.Module):
    """Lightweight parent-child message passing for arbitrary skeleton trees."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, parents: Tensor, joint_mask: Tensor) -> Tensor:
        batch_size, _, joints, _ = x.shape
        parents = _parents_or_default(parents, batch_size=batch_size, joints=joints, device=x.device)
        parents_cpu = parents.detach().cpu().tolist()
        joint_mask_cpu = joint_mask.detach().cpu().tolist()

        neighbor_sum = torch.zeros_like(x)
        degree = x.new_zeros((batch_size, 1, joints, 1))
        for batch_index in range(batch_size):
            for child_index, parent_index in enumerate(parents_cpu[batch_index]):
                if parent_index < 0 or parent_index >= joints:
                    continue
                if not joint_mask_cpu[batch_index][child_index] or not joint_mask_cpu[batch_index][parent_index]:
                    continue
                neighbor_sum[batch_index, :, child_index] += x[batch_index, :, parent_index]
                neighbor_sum[batch_index, :, parent_index] += x[batch_index, :, child_index]
                degree[batch_index, :, child_index] += 1.0
                degree[batch_index, :, parent_index] += 1.0

        neighbor_mean = neighbor_sum / degree.clamp_min(1.0)
        message = self.projection(self.norm(neighbor_mean))
        has_neighbors = (degree > 0).to(dtype=x.dtype, device=x.device)
        return x + self.dropout(message) * has_neighbors


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
        use_topology_conditioning: bool = True,
        use_graph_mixer: bool = True,
        use_joint_index_embedding: bool = False,
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
        self.use_topology_conditioning = use_topology_conditioning
        self.use_graph_mixer = use_graph_mixer
        self.use_joint_index_embedding = use_joint_index_embedding

        motion_feature_dim = feature_dim + 3
        self.input_projection = nn.Linear(motion_feature_dim, hidden_dim)
        self.topology_projection = nn.Linear(7, hidden_dim) if use_topology_conditioning else None
        self.encoder_blocks = nn.ModuleList(
            [
                FactorizedTemporalSpatialBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    use_graph_mixer=use_graph_mixer,
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
                    use_graph_mixer=use_graph_mixer,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, motion_feature_dim)

    def forward(self, batch: Dict[str, Tensor], sample_posterior: bool = True) -> KinematicVAEOutput:
        positions = batch["positions"]
        rotations = batch["local_rotations_6d"]
        time_mask = _mask_or_ones(batch.get("time_mask"), positions.shape[:2], positions.device)
        joint_mask = _mask_or_ones(batch.get("joint_mask"), (positions.shape[0], positions.shape[2]), positions.device)
        root_translation = _target_root_translation(batch, positions)
        root_origin = root_translation[:, :1]
        root_delta = root_translation - root_origin
        root_relative_positions = positions - root_translation[:, :, None, :]
        root_delta_tokens = root_delta[:, :, None, :].expand(-1, -1, positions.shape[2], -1)
        features = torch.cat([root_relative_positions, root_delta_tokens, rotations], dim=-1)
        topology_embedding = self._topology_embedding(batch=batch, positions=positions, joint_mask=joint_mask)
        parents = batch.get("parents")

        x = self.input_projection(features)
        x = x + _time_embedding(positions.shape[1], self.hidden_dim, positions.device, x.dtype)
        if self.use_joint_index_embedding:
            x = x + _joint_embedding(positions.shape[2], self.hidden_dim, positions.device, x.dtype)
        if topology_embedding is not None:
            x = x + topology_embedding[:, None]
        x = _apply_token_mask(x, time_mask=time_mask, joint_mask=joint_mask)
        for block in self.encoder_blocks:
            x = block(x, time_mask=time_mask, joint_mask=joint_mask, parents=parents)

        pooled = self._masked_pool(self.encoder_norm(x), time_mask=time_mask, joint_mask=joint_mask)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled).clamp(min=-20.0, max=20.0)
        z = self.reparameterize(mu, logvar) if sample_posterior else mu

        decoded = self._decode(
            z,
            frames=positions.shape[1],
            joints=positions.shape[2],
            time_mask=time_mask,
            joint_mask=joint_mask,
            topology_embedding=topology_embedding,
            parents=parents,
        )
        predicted_relative_positions = decoded[..., :3]
        predicted_root_delta_tokens = decoded[..., 3:6]
        predicted_root_delta = _masked_joint_mean(predicted_root_delta_tokens, joint_mask=joint_mask)
        predicted_root_translation = root_origin + predicted_root_delta
        return KinematicVAEOutput(
            positions=predicted_relative_positions + predicted_root_translation[:, :, None, :],
            root_translation=predicted_root_translation,
            root_relative_positions=predicted_relative_positions,
            local_rotations_6d=decoded[..., 6:12],
            mu=mu,
            logvar=logvar,
            z=z,
        )

    def _decode(
        self,
        z: Tensor,
        frames: int,
        joints: int,
        time_mask: Tensor,
        joint_mask: Tensor,
        topology_embedding: Tensor | None,
        parents: Tensor | None,
    ) -> Tensor:
        x = self.latent_projection(z)[:, None, None, :].expand(z.shape[0], frames, joints, self.hidden_dim)
        x = x + _time_embedding(frames, self.hidden_dim, z.device, x.dtype)
        if self.use_joint_index_embedding:
            x = x + _joint_embedding(joints, self.hidden_dim, z.device, x.dtype)
        if topology_embedding is not None:
            x = x + topology_embedding[:, None]
        x = _apply_token_mask(x, time_mask=time_mask, joint_mask=joint_mask)
        for block in self.decoder_blocks:
            x = block(x, time_mask=time_mask, joint_mask=joint_mask, parents=parents)
        x = self.decoder_norm(x)
        return self.output_projection(x)

    def _topology_embedding(self, batch: Dict[str, Tensor], positions: Tensor, joint_mask: Tensor) -> Tensor | None:
        if self.topology_projection is None:
            return None
        topology = _topology_features(batch=batch, positions=positions, joint_mask=joint_mask)
        return self.topology_projection(topology)

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
    root_position_weight: float = 1.0,
    velocity_weight: float = 0.1,
    acceleration_weight: float = 0.01,
    bone_length_weight: float = 0.1,
    root_velocity_weight: float = 0.05,
) -> Dict[str, Tensor]:
    positions = batch["positions"]
    rotations = batch["local_rotations_6d"]
    time_mask = _mask_or_ones(batch.get("time_mask"), positions.shape[:2], positions.device)
    joint_mask = _mask_or_ones(batch.get("joint_mask"), (positions.shape[0], positions.shape[2]), positions.device)
    target_root_translation = _target_root_translation(batch, positions)
    target_root_relative_positions = positions - target_root_translation[:, :, None, :]

    recon_position = _masked_mse(
        output.root_relative_positions,
        target_root_relative_positions,
        _motion_mask(positions, time_mask, joint_mask),
    )
    recon_rotation = _masked_mse(
        output.local_rotations_6d,
        rotations,
        _motion_mask(rotations, time_mask, joint_mask),
    )
    root_position = _root_position_loss(
        prediction=output.root_translation,
        target=target_root_translation,
        time_mask=time_mask,
    )
    absolute_position = _masked_mse(output.positions, positions, _motion_mask(positions, time_mask, joint_mask))
    kl = -0.5 * torch.mean(1.0 + output.logvar - output.mu.pow(2) - output.logvar.exp())
    velocity = _temporal_difference_loss(
        output.root_relative_positions,
        target_root_relative_positions,
        time_mask,
        joint_mask,
        order=1,
    )
    acceleration = _temporal_difference_loss(
        output.root_relative_positions,
        target_root_relative_positions,
        time_mask,
        joint_mask,
        order=2,
    )
    bone_length = _bone_length_loss(
        prediction=output.positions,
        target=positions,
        parents=batch.get("parents"),
        time_mask=time_mask,
        joint_mask=joint_mask,
    )
    root_velocity = _root_velocity_loss(
        prediction=output.positions,
        predicted_root_translation=output.root_translation,
        target_positions=positions,
        target_root_translation=target_root_translation,
        time_mask=time_mask,
    )

    loss = (
        recon_position
        + recon_rotation
        + beta * kl
        + root_position_weight * root_position
        + velocity_weight * velocity
        + acceleration_weight * acceleration
        + bone_length_weight * bone_length
        + root_velocity_weight * root_velocity
    )
    return {
        "loss": loss,
        "recon_position": recon_position,
        "recon_rotation": recon_rotation,
        "root_position": root_position,
        "absolute_position": absolute_position,
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


def _target_root_translation(batch: Dict[str, Tensor], positions: Tensor) -> Tensor:
    root_translation = batch.get("root_translation")
    if root_translation is None:
        return positions[:, :, 0]
    return root_translation.to(device=positions.device, dtype=positions.dtype)


def _topology_features(batch: Dict[str, Tensor], positions: Tensor, joint_mask: Tensor) -> Tensor:
    batch_size, _, joints, _ = positions.shape
    device = positions.device
    dtype = positions.dtype
    parents = _parents_or_default(batch.get("parents"), batch_size=batch_size, joints=joints, device=device)
    rest_offsets = batch.get("rest_offsets")
    if rest_offsets is None:
        rest_offsets = torch.zeros((batch_size, joints, 3), device=device, dtype=dtype)
    else:
        rest_offsets = rest_offsets.to(device=device, dtype=dtype)
        if rest_offsets.ndim == 2:
            rest_offsets = rest_offsets[None].expand(batch_size, joints, 3)
    bone_lengths = torch.linalg.norm(rest_offsets, dim=-1, keepdim=True)
    depth = _normalized_parent_depth(parents=parents, joint_mask=joint_mask, dtype=dtype)
    chain_coordinates = batch.get("chain_coordinates")
    if chain_coordinates is None:
        chain_coordinates = depth.squeeze(-1)
    else:
        chain_coordinates = chain_coordinates.to(device=device, dtype=dtype)
        if chain_coordinates.ndim == 1:
            chain_coordinates = chain_coordinates[None].expand(batch_size, joints)
    root_flag = (parents < 0).to(device=device, dtype=dtype).unsqueeze(-1)
    return torch.cat(
        [
            rest_offsets,
            bone_lengths,
            depth,
            chain_coordinates.unsqueeze(-1),
            root_flag,
        ],
        dim=-1,
    )


def _parents_or_default(
    parents: Tensor | None,
    batch_size: int,
    joints: int,
    device: torch.device,
) -> Tensor:
    if parents is None:
        default = torch.arange(joints, device=device, dtype=torch.long) - 1
        default[0] = -1
        return default[None].expand(batch_size, joints)
    parents = parents.to(device=device, dtype=torch.long)
    if parents.ndim == 1:
        parents = parents[None].expand(batch_size, joints)
    return parents[:, :joints]


def _normalized_parent_depth(parents: Tensor, joint_mask: Tensor, dtype: torch.dtype) -> Tensor:
    parents_cpu = parents.detach().cpu().tolist()
    joint_mask_cpu = joint_mask.detach().cpu().tolist()
    depths: list[list[float]] = []
    for batch_index, batch_parents in enumerate(parents_cpu):
        batch_depths: list[float] = []
        for joint_index, _ in enumerate(batch_parents):
            if not joint_mask_cpu[batch_index][joint_index]:
                batch_depths.append(0.0)
                continue
            depth = 0
            parent = batch_parents[joint_index]
            visited = {joint_index}
            while parent >= 0 and parent < len(batch_parents) and parent not in visited:
                depth += 1
                visited.add(parent)
                parent = batch_parents[parent]
            batch_depths.append(float(depth))
        max_depth = max(max(batch_depths), 1.0)
        depths.append([value / max_depth for value in batch_depths])
    return torch.tensor(depths, device=parents.device, dtype=dtype).unsqueeze(-1)


def _masked_joint_mean(tokens: Tensor, joint_mask: Tensor) -> Tensor:
    mask = joint_mask[:, None, :, None].to(device=tokens.device, dtype=tokens.dtype)
    denom = mask.sum(dim=2).clamp_min(1.0)
    return (tokens * mask).sum(dim=2) / denom


def _root_position_loss(prediction: Tensor, target: Tensor, time_mask: Tensor) -> Tensor:
    mask = time_mask[:, :, None].to(dtype=prediction.dtype, device=prediction.device)
    return _masked_mse(prediction, target, mask)


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
    predicted_root_translation: Tensor | None,
    target_positions: Tensor,
    target_root_translation: Tensor | None,
    time_mask: Tensor,
) -> Tensor:
    if prediction.shape[1] <= 1:
        return prediction.new_zeros(())
    pred_root = predicted_root_translation if predicted_root_translation is not None else prediction[:, :, 0]
    target_root = target_root_translation.to(prediction.device, prediction.dtype) if target_root_translation is not None else target_positions[:, :, 0]
    pred_velocity = pred_root[:, 1:] - pred_root[:, :-1]
    target_velocity = target_root[:, 1:] - target_root[:, :-1]
    valid_time = time_mask[:, 1:] & time_mask[:, :-1]
    mask = valid_time[:, :, None].to(dtype=prediction.dtype, device=prediction.device)
    return _masked_mse(pred_velocity, target_velocity, mask)
