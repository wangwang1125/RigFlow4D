import torch

from models.rigflow4d.latent_flow import (
    LatentFlowMatcher,
    latent_flow_matching_loss,
    sample_latent_flow_pair,
)


def test_latent_flow_matcher_is_exported_from_package():
    from models.rigflow4d import LatentFlowMatcher as ExportedLatentFlowMatcher

    assert ExportedLatentFlowMatcher is LatentFlowMatcher


def test_sample_latent_flow_pair_uses_linear_path():
    z0 = torch.zeros(2, 4)
    z1 = torch.ones(2, 4)
    t = torch.tensor([0.25, 0.75])

    pair = sample_latent_flow_pair(z0, z1, t=t)

    expected = t[:, None] * torch.ones_like(z1)
    torch.testing.assert_close(pair.z_t, expected)
    torch.testing.assert_close(pair.target_velocity, torch.ones_like(z1))
    torch.testing.assert_close(pair.t, t)


def test_latent_flow_matcher_forward_shape():
    model = LatentFlowMatcher(latent_dim=8, condition_dim=5, hidden_dim=32)
    z_t = torch.randn(3, 8)
    t = torch.rand(3)
    condition = torch.randn(3, 5)

    velocity = model(z_t=z_t, t=t, condition=condition)

    assert velocity.shape == (3, 8)


def test_latent_flow_matching_loss_is_finite():
    model = LatentFlowMatcher(latent_dim=8, condition_dim=5, hidden_dim=32)
    z0 = torch.randn(3, 8)
    z1 = torch.randn(3, 8)
    condition = torch.randn(3, 5)

    losses = latent_flow_matching_loss(model, z0=z0, z1=z1, condition=condition)

    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["velocity_mse"])


def test_latent_flow_training_step_updates_parameters():
    model = LatentFlowMatcher(latent_dim=8, condition_dim=5, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = next(model.parameters()).detach().clone()
    z0 = torch.randn(3, 8)
    z1 = torch.randn(3, 8)
    condition = torch.randn(3, 5)

    loss = latent_flow_matching_loss(model, z0=z0, z1=z1, condition=condition)["loss"]
    loss.backward()
    optimizer.step()

    after = next(model.parameters()).detach()
    assert not torch.allclose(before, after)
