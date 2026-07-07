import torch

from models.rigflow4d.kinematic_vae import KinematicVAE, KinematicVAEOutput, kinematic_vae_loss


def make_batch(batch_size=2, frames=4, joints=5):
    return {
        "positions": torch.randn(batch_size, frames, joints, 3),
        "local_rotations_6d": torch.randn(batch_size, frames, joints, 6),
        "root_translation": torch.randn(batch_size, frames, 3),
        "time_mask": torch.ones(batch_size, frames, dtype=torch.bool),
        "joint_mask": torch.ones(batch_size, joints, dtype=torch.bool),
        "parents": torch.tensor([[-1, 0, 1, 2, 3]] * batch_size, dtype=torch.long),
        "rest_offsets": torch.randn(batch_size, joints, 3),
        "chain_ids": torch.arange(joints).repeat(batch_size, 1),
        "chain_coordinates": torch.linspace(0.0, 1.0, joints).repeat(batch_size, 1),
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
    assert output.root_translation.shape == batch["root_translation"].shape
    assert output.root_relative_positions.shape == batch["positions"].shape
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


def test_kinematic_vae_can_reconstruct_from_posterior_mean_deterministically():
    torch.manual_seed(5)
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

    first = model(batch, sample_posterior=False)
    second = model(batch, sample_posterior=False)

    torch.testing.assert_close(first.z, first.mu)
    torch.testing.assert_close(first.positions, second.positions)


def test_kinematic_vae_conditions_on_skeleton_topology():
    torch.manual_seed(7)
    batch = make_batch(batch_size=1, frames=4, joints=5)
    model = KinematicVAE(
        feature_dim=9,
        hidden_dim=32,
        latent_dim=12,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        use_topology_conditioning=True,
        use_graph_mixer=True,
    )
    model.eval()

    alternate = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    alternate["parents"] = torch.tensor([[-1, 0, 0, 2, 2]], dtype=torch.long)
    alternate["rest_offsets"] = batch["rest_offsets"] * torch.tensor([[[1.0, -1.0, 0.5]]])
    alternate["chain_coordinates"] = torch.flip(batch["chain_coordinates"], dims=(1,))

    first = model(batch, sample_posterior=False)
    second = model(alternate, sample_posterior=False)

    assert not torch.allclose(first.positions, second.positions)


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
    assert torch.isfinite(losses["root_position"])
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
        + losses["root_position"]
        + 0.5 * losses["velocity"]
        + 0.25 * losses["acceleration"]
        + 0.125 * losses["bone_length"]
        + 0.0625 * losses["root_velocity"]
    )

    torch.testing.assert_close(losses["loss"], expected)


def test_kinematic_vae_loss_separates_body_pose_from_global_root_shift():
    rotations = torch.zeros(1, 3, 3, 6)
    root = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    local_pose = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.1, 1.1, 0.0], [0.0, 2.1, 0.0]],
                [[0.0, 0.0, 0.0], [0.2, 1.2, 0.0], [0.0, 2.2, 0.0]],
            ]
        ]
    )
    shifted_root = root + torch.tensor([[[10.0, 0.0, 0.0]]])
    batch = {
        "positions": root[:, :, None, :] + local_pose,
        "local_rotations_6d": rotations,
        "root_translation": root,
        "time_mask": torch.ones(1, 3, dtype=torch.bool),
        "joint_mask": torch.ones(1, 3, dtype=torch.bool),
        "parents": torch.tensor([[-1, 0, 1]], dtype=torch.long),
    }
    output = KinematicVAEOutput(
        positions=shifted_root[:, :, None, :] + local_pose,
        root_translation=shifted_root,
        root_relative_positions=local_pose,
        local_rotations_6d=rotations,
        mu=torch.zeros(1, 4),
        logvar=torch.zeros(1, 4),
        z=torch.zeros(1, 4),
    )

    losses = kinematic_vae_loss(
        output,
        batch,
        beta=0.0,
        velocity_weight=0.0,
        acceleration_weight=0.0,
        bone_length_weight=0.0,
        root_velocity_weight=0.0,
    )

    torch.testing.assert_close(losses["recon_position"], torch.tensor(0.0))
    assert losses["root_position"] > 0.0


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
