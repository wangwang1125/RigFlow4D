import torch

from models.rigflow4d.kinematic_vae import KinematicVAE, kinematic_vae_loss


def make_batch(batch_size=2, frames=4, joints=5):
    return {
        "positions": torch.randn(batch_size, frames, joints, 3),
        "local_rotations_6d": torch.randn(batch_size, frames, joints, 6),
        "time_mask": torch.ones(batch_size, frames, dtype=torch.bool),
        "joint_mask": torch.ones(batch_size, joints, dtype=torch.bool),
    }


def test_kinematic_vae_forward_shapes():
    batch = make_batch()
    model = KinematicVAE(feature_dim=9, hidden_dim=32, latent_dim=12)

    output = model(batch)

    assert output.positions.shape == batch["positions"].shape
    assert output.local_rotations_6d.shape == batch["local_rotations_6d"].shape
    assert output.mu.shape == (2, 12)
    assert output.logvar.shape == (2, 12)
    assert output.z.shape == (2, 12)


def test_kinematic_vae_loss_is_finite_with_masks():
    batch = make_batch()
    batch["joint_mask"][0, -2:] = False
    model = KinematicVAE(feature_dim=9, hidden_dim=32, latent_dim=12)

    output = model(batch)
    losses = kinematic_vae_loss(output, batch, beta=0.01)

    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["recon_position"])
    assert torch.isfinite(losses["recon_rotation"])
    assert torch.isfinite(losses["kl"])


def test_kinematic_vae_training_step_updates_parameters():
    batch = make_batch()
    model = KinematicVAE(feature_dim=9, hidden_dim=32, latent_dim=12)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = next(model.parameters()).detach().clone()

    output = model(batch)
    loss = kinematic_vae_loss(output, batch)["loss"]
    loss.backward()
    optimizer.step()

    after = next(model.parameters()).detach()
    assert not torch.allclose(before, after)
