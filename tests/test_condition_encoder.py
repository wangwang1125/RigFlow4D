import pytest
import torch

from models.rigflow4d.condition_encoder import RigFlowConditionEncoder, masked_mean
from models.rigflow4d.latent_flow import LatentFlowMatcher


def test_masked_mean_respects_binary_mask():
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ]
    )
    mask = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)

    pooled = masked_mean(values, mask=mask, reduce_dims=(1,))

    expected = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    torch.testing.assert_close(pooled, expected)


def test_condition_encoder_forward_shapes_with_all_modalities():
    batch_size = 2
    hidden_dim = 16
    condition_dim = 24
    encoder = RigFlowConditionEncoder(
        visual_dim=6,
        camera_dim=4,
        rig_dim=5,
        pose_seed_dim=9,
        hidden_dim=hidden_dim,
        condition_dim=condition_dim,
    )

    output = encoder(
        visual_tokens=torch.randn(batch_size, 2, 3, 4, 6),
        visual_mask=torch.ones(batch_size, 2, 3, 4, dtype=torch.bool),
        camera_features=torch.randn(batch_size, 2, 4),
        camera_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        rig_features=torch.randn(batch_size, 7, 5),
        joint_mask=torch.ones(batch_size, 7, dtype=torch.bool),
        pose_seed=torch.randn(batch_size, 3, 7, 9),
        time_mask=torch.ones(batch_size, 3, dtype=torch.bool),
    )

    assert output.condition.shape == (batch_size, condition_dim)
    assert output.components["visual"].shape == (batch_size, hidden_dim)
    assert output.components["camera"].shape == (batch_size, hidden_dim)
    assert output.components["rig"].shape == (batch_size, hidden_dim)
    assert output.components["pose_seed"].shape == (batch_size, hidden_dim)


def test_condition_encoder_accepts_missing_camera_features():
    batch_size = 2
    encoder = RigFlowConditionEncoder(
        visual_dim=6,
        camera_dim=4,
        rig_dim=5,
        pose_seed_dim=9,
        hidden_dim=16,
        condition_dim=24,
    )

    output = encoder(
        visual_tokens=torch.randn(batch_size, 1, 2, 3, 6),
        camera_features=None,
        rig_features=torch.randn(batch_size, 5, 5),
        pose_seed=torch.randn(batch_size, 2, 5, 9),
    )

    assert output.condition.shape == (batch_size, 24)
    assert torch.isfinite(output.condition).all()


def test_condition_encoder_rejects_wrong_visual_feature_dim():
    encoder = RigFlowConditionEncoder(
        visual_dim=6,
        camera_dim=4,
        rig_dim=5,
        pose_seed_dim=9,
        hidden_dim=16,
        condition_dim=24,
    )

    with pytest.raises(ValueError, match="visual_tokens"):
        encoder(visual_tokens=torch.randn(2, 1, 2, 3, 7))


def test_condition_encoder_export_feeds_latent_flow_matcher():
    from models.rigflow4d import RigFlowConditionEncoder as ExportedRigFlowConditionEncoder

    batch_size = 3
    condition_dim = 12
    encoder = ExportedRigFlowConditionEncoder(
        visual_dim=6,
        camera_dim=4,
        rig_dim=5,
        pose_seed_dim=9,
        hidden_dim=16,
        condition_dim=condition_dim,
    )
    condition = encoder(
        visual_tokens=torch.randn(batch_size, 1, 2, 3, 6),
        camera_features=None,
        rig_features=torch.randn(batch_size, 4, 5),
        pose_seed=torch.randn(batch_size, 2, 4, 9),
    ).condition

    flow = LatentFlowMatcher(latent_dim=8, condition_dim=condition_dim, hidden_dim=32)
    velocity = flow(
        z_t=torch.randn(batch_size, 8),
        t=torch.rand(batch_size),
        condition=condition,
    )

    assert velocity.shape == (batch_size, 8)
