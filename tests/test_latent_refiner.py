import torch

from models.rigflow4d import (
    KinematicVAE,
    LatentFlowMatcher,
    RigFlowConditionEncoder,
)
from models.rigflow4d.latent_refiner import RigFlowLatentRefiner


def make_motion_batch(batch_size=2, frames=4, joints=5):
    return {
        "positions": torch.randn(batch_size, frames, joints, 3),
        "local_rotations_6d": torch.randn(batch_size, frames, joints, 6),
        "time_mask": torch.ones(batch_size, frames, dtype=torch.bool),
        "joint_mask": torch.ones(batch_size, joints, dtype=torch.bool),
    }


def make_refiner(latent_dim=8, condition_dim=12):
    return RigFlowLatentRefiner(
        vae=KinematicVAE(feature_dim=9, hidden_dim=32, latent_dim=latent_dim),
        condition_encoder=RigFlowConditionEncoder(
            visual_dim=6,
            camera_dim=4,
            rig_dim=5,
            pose_seed_dim=9,
            hidden_dim=16,
            condition_dim=condition_dim,
        ),
        flow_matcher=LatentFlowMatcher(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            hidden_dim=32,
        ),
    )


def test_latent_refiner_forward_returns_finite_flow_loss():
    batch_size = 2
    frames = 4
    joints = 5
    refiner = make_refiner()

    output = refiner(
        motion_batch=make_motion_batch(batch_size=batch_size, frames=frames, joints=joints),
        visual_tokens=torch.randn(batch_size, 2, frames, 3, 6),
        camera_features=torch.randn(batch_size, 2, 4),
        rig_features=torch.randn(batch_size, joints, 5),
        pose_seed=torch.randn(batch_size, frames, joints, 9),
    )

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.losses["flow_loss"])
    assert output.condition.shape == (batch_size, 12)
    assert output.flow_pair.z0.shape == (batch_size, 8)
    assert output.flow_pair.z1.shape == (batch_size, 8)


def test_latent_refiner_accepts_motion_only_batch():
    batch_size = 2
    refiner = make_refiner()

    output = refiner(motion_batch=make_motion_batch(batch_size=batch_size))

    assert output.condition.shape == (batch_size, 12)
    assert torch.isfinite(output.loss)


def test_latent_refiner_detaches_vae_target_by_default():
    refiner = make_refiner()

    output = refiner(motion_batch=make_motion_batch())
    output.loss.backward()

    vae_grads = [param.grad for param in refiner.vae.parameters()]
    assert all(grad is None or torch.allclose(grad, torch.zeros_like(grad)) for grad in vae_grads)


def test_latent_refiner_training_step_updates_flow_parameters():
    refiner = make_refiner()
    optimizer = torch.optim.Adam(refiner.parameters(), lr=1e-3)
    before = refiner.flow_matcher.net[0].weight.detach().clone()

    output = refiner(motion_batch=make_motion_batch())
    output.loss.backward()
    optimizer.step()

    after = refiner.flow_matcher.net[0].weight.detach()
    assert not torch.allclose(before, after)


def test_latent_refiner_is_exported_from_package():
    from models.rigflow4d import RigFlowLatentRefiner as ExportedRigFlowLatentRefiner

    assert ExportedRigFlowLatentRefiner is RigFlowLatentRefiner
