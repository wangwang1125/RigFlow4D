from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .condition_encoder import RigFlowConditionEncoder, RigFlowConditionOutput
from .kinematic_vae import KinematicVAE, KinematicVAEOutput
from .latent_flow import LatentFlowMatcher, LatentFlowPair, sample_latent_flow_pair


@dataclass
class RigFlowLatentRefinerOutput:
    loss: Tensor
    losses: Dict[str, Tensor]
    condition: Tensor
    condition_output: RigFlowConditionOutput
    vae_output: KinematicVAEOutput
    flow_pair: LatentFlowPair
    predicted_velocity: Tensor


class RigFlowLatentRefiner(nn.Module):
    def __init__(
        self,
        vae: KinematicVAE,
        condition_encoder: RigFlowConditionEncoder,
        flow_matcher: LatentFlowMatcher,
        detach_vae_target: bool = True,
    ) -> None:
        super().__init__()
        if vae.latent_dim != flow_matcher.latent_dim:
            raise ValueError(
                f"vae latent_dim={vae.latent_dim} must match flow latent_dim={flow_matcher.latent_dim}"
            )
        if condition_encoder.condition_dim != flow_matcher.condition_dim:
            raise ValueError(
                "condition encoder output dimension must match flow condition dimension: "
                f"{condition_encoder.condition_dim} vs {flow_matcher.condition_dim}"
            )
        self.vae = vae
        self.condition_encoder = condition_encoder
        self.flow_matcher = flow_matcher
        self.detach_vae_target = detach_vae_target

    def forward(
        self,
        motion_batch: Dict[str, Tensor],
        visual_tokens: Optional[Tensor] = None,
        visual_mask: Optional[Tensor] = None,
        camera_features: Optional[Tensor] = None,
        camera_mask: Optional[Tensor] = None,
        rig_features: Optional[Tensor] = None,
        pose_seed: Optional[Tensor] = None,
        z0: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        detach_vae_target: Optional[bool] = None,
    ) -> RigFlowLatentRefinerOutput:
        vae_output = self.vae(motion_batch)
        should_detach = self.detach_vae_target if detach_vae_target is None else detach_vae_target
        z1 = vae_output.mu.detach() if should_detach else vae_output.mu
        if z0 is None:
            z0 = torch.randn_like(z1)
        else:
            z0 = z0.to(device=z1.device, dtype=z1.dtype)

        condition_output = self.condition_encoder(
            visual_tokens=visual_tokens,
            visual_mask=visual_mask,
            camera_features=camera_features,
            camera_mask=camera_mask,
            rig_features=rig_features,
            joint_mask=motion_batch.get("joint_mask"),
            pose_seed=pose_seed,
            time_mask=motion_batch.get("time_mask"),
            batch_size=z1.shape[0],
        )
        flow_pair = sample_latent_flow_pair(z0=z0, z1=z1, t=t)
        predicted_velocity = self.flow_matcher(
            z_t=flow_pair.z_t,
            t=flow_pair.t,
            condition=condition_output.condition,
        )
        flow_loss = torch.nn.functional.mse_loss(predicted_velocity, flow_pair.target_velocity)

        return RigFlowLatentRefinerOutput(
            loss=flow_loss,
            losses={
                "loss": flow_loss,
                "flow_loss": flow_loss,
            },
            condition=condition_output.condition,
            condition_output=condition_output,
            vae_output=vae_output,
            flow_pair=flow_pair,
            predicted_velocity=predicted_velocity,
        )
