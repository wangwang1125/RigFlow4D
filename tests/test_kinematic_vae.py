import torch

from models.rigflow4d.kinematic_vae import KinematicVAE, kinematic_vae_loss


def make_batch(batch_size=2, frames=4, joints=5):
    return {
        "positions": torch.randn(batch_size, frames, joints, 3),
        "local_rotations_6d": torch.randn(batch_size, frames, joints, 6),
        "root_translation": torch.randn(batch_size, frames, 3),
        "time_mask": torch.ones(batch_size, frames, dtype=torch.bool),
        "joint_mask": torch.ones(batch_size, joints, dtype=torch.bool),
        "parents": torch.tensor([[-1, 0, 1, 2, 3]] * batch_size, dtype=torch.long),
    }


def test_kinematic_vae_forward_shapes():
    batch = make_batch()
    model = KinematicVAE(
        feature_dim=9,
        hidden_dim=32,
        latent_dim=12,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )

    output = model(batch)

    assert output.positions.shape == batch["positions"].shape
    assert output.local_rotations_6d.shape == batch["local_rotations_6d"].shape
    assert output.mu.shape == (2, 12)
    assert output.logvar.shape == (2, 12)
    assert output.z.shape == (2, 12)


def test_kinematic_vae_decoder_keeps_time_and_joint_structure():
    torch.manual_seed(3)
    batch = make_batch(batch_size=1, frames=4, joints=5)
    model = KinematicVAE(
        feature_dim=9,
        hidden_dim=32,
        latent_dim=12,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    model.eval()

    output = model(batch)

    assert not torch.allclose(output.positions[:, 0, 0], output.positions[:, 1, 0])
    assert not torch.allclose(output.positions[:, 0, 0], output.positions[:, 0, 1])


def test_kinematic_vae_loss_is_finite_with_masks():
    batch = make_batch()
    batch["joint_mask"][0, -2:] = False
    model = KinematicVAE(
        feature_dim=9,
        hidden_dim=32,
        latent_dim=12,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )

    output = model(batch)
    losses = kinematic_vae_loss(
        output,
        batch,
        beta=0.01,
        velocity_weight=0.1,
        acceleration_weight=0.01,
        bone_length_weight=0.2,
        root_velocity_weight=0.05,
    )

    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["recon_position"])
    assert torch.isfinite(losses["recon_rotation"])
    assert torch.isfinite(losses["kl"])
    assert torch.isfinite(losses["velocity"])
    assert torch.isfinite(losses["acceleration"])
    assert torch.isfinite(losses["bone_length"])
    assert torch.isfinite(losses["root_velocity"])


def test_kinematic_vae_loss_weights_motion_terms_into_total():
    batch = make_batch()
    model = KinematicVAE(
        feature_dim=9,
        hidden_dim=32,
        latent_dim=12,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )

    output = model(batch)
    losses = kinematic_vae_loss(
        output,
        batch,
        beta=0.01,
        velocity_weight=0.5,
        acceleration_weight=0.25,
        bone_length_weight=0.125,
        root_velocity_weight=0.0625,
    )
    expected = (
        losses["recon_position"]
        + losses["recon_rotation"]
        + 0.01 * losses["kl"]
        + 0.5 * losses["velocity"]
        + 0.25 * losses["acceleration"]
        + 0.125 * losses["bone_length"]
        + 0.0625 * losses["root_velocity"]
    )

    torch.testing.assert_close(losses["loss"], expected)


def test_kinematic_vae_training_step_updates_parameters():
    batch = make_batch()
    model = KinematicVAE(
        feature_dim=9,
        hidden_dim=32,
        latent_dim=12,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = next(model.parameters()).detach().clone()

    output = model(batch)
    loss = kinematic_vae_loss(output, batch)["loss"]
    loss.backward()
    optimizer.step()

    after = next(model.parameters()).detach()
    assert not torch.allclose(before, after)
