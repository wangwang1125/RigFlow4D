import json
import math

import numpy as np
import torch

from train.rigflow4d_latent_refiner import (
    LatentRefinerSmokeConfig,
    build_latent_refiner,
    build_motion_dataloader,
    motion_batch_to_torch,
    parse_args,
    run_latent_refiner_smoke_training,
    train_latent_refiner_step,
)


def write_normalized_dataset(tmp_path, frames=6, joints=4, include_visual=False, visual_dim=6):
    sample_path = tmp_path / "sample_0000.npz"
    manifest_path = tmp_path / "manifest.json"
    payload = {
        "dataset_name": np.array("unit"),
        "input_type": np.array("video"),
        "source_label_type": np.array("motion_only"),
        "camera_mode": np.array("unknown"),
        "parents": np.array([-1, 0, 1, 2], dtype=np.int64)[:joints],
        "rest_offsets": np.zeros((joints, 3), dtype=np.float32),
        "joint_names": np.array([f"joint_{i}" for i in range(joints)]),
        "chain_ids": np.arange(joints, dtype=np.int64),
        "chain_coordinates": np.linspace(0.0, 1.0, joints, dtype=np.float32),
        "positions": np.random.randn(frames, joints, 3).astype(np.float32),
        "local_rotations_6d": np.random.randn(frames, joints, 6).astype(np.float32),
        "root_translation": np.zeros((frames, 3), dtype=np.float32),
    }
    if include_visual:
        payload.update(
            {
                "visual_tokens": np.random.randn(2, frames, 3, visual_dim).astype(np.float32),
                "visual_backbone_name": np.array("dinov3_vitl16"),
                "visual_feature_dim": np.array(visual_dim),
                "visual_patch_grid": np.array([1, 3], dtype=np.int64),
            }
        )
    np.savez(sample_path, **payload)
    manifest_path.write_text(
        json.dumps({"samples": [{"sample_id": "sample_0000", "path": sample_path.name}]}),
        encoding="utf-8",
    )
    return tmp_path, manifest_path


def make_config(tmp_path, max_steps=2, include_visual=False, visual_dim=6):
    root, manifest_path = write_normalized_dataset(
        tmp_path,
        include_visual=include_visual,
        visual_dim=visual_dim,
    )
    return LatentRefinerSmokeConfig(
        data_root=root,
        manifest_path=manifest_path,
        window_size=3,
        stride=2,
        batch_size=2,
        max_steps=max_steps,
        latent_dim=8,
        condition_dim=12,
        model_hidden_dim=32,
        condition_hidden_dim=16,
        visual_dim=visual_dim,
        lr=1e-3,
        seed=7,
    )


def test_build_motion_dataloader_reads_normalized_npz(tmp_path):
    config = make_config(tmp_path, max_steps=1)

    dataloader = build_motion_dataloader(config)
    batch = next(iter(dataloader))

    assert len(dataloader.dataset) > 0
    assert batch["positions"].shape == (2, 3, 4, 3)
    assert batch["local_rotations_6d"].shape == (2, 3, 4, 6)
    assert batch["time_mask"].dtype == np.bool_
    assert batch["joint_mask"].dtype == np.bool_


def test_motion_batch_to_torch_converts_expected_dtypes(tmp_path):
    config = make_config(tmp_path, max_steps=1)
    batch = next(iter(build_motion_dataloader(config)))

    torch_batch = motion_batch_to_torch(batch, device=torch.device("cpu"))

    assert torch_batch["positions"].dtype == torch.float32
    assert torch_batch["local_rotations_6d"].dtype == torch.float32
    assert torch_batch["time_mask"].dtype == torch.bool
    assert torch_batch["joint_mask"].dtype == torch.bool


def test_train_latent_refiner_step_updates_flow_parameters(tmp_path):
    config = make_config(tmp_path, max_steps=1)
    model = build_latent_refiner(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    batch = next(iter(build_motion_dataloader(config)))
    torch_batch = motion_batch_to_torch(batch, device=torch.device("cpu"))
    before = model.flow_matcher.net[0].weight.detach().clone()

    loss = train_latent_refiner_step(model, torch_batch, optimizer)

    after = model.flow_matcher.net[0].weight.detach()
    assert math.isfinite(loss)
    assert not torch.allclose(before, after)


def test_train_latent_refiner_step_uses_visual_tokens(tmp_path):
    config = make_config(tmp_path, max_steps=1, include_visual=True, visual_dim=6)
    model = build_latent_refiner(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    batch = next(iter(build_motion_dataloader(config)))

    torch_batch = motion_batch_to_torch(batch, device=torch.device("cpu"))
    loss = train_latent_refiner_step(model, torch_batch, optimizer)

    assert batch["visual_tokens"].shape == (2, 2, 3, 3, 6)
    assert torch_batch["visual_tokens"].dtype == torch.float32
    assert torch_batch["visual_mask"].dtype == torch.bool
    assert math.isfinite(loss)


def test_run_latent_refiner_smoke_training_returns_finite_losses(tmp_path):
    config = make_config(tmp_path, max_steps=2)

    result = run_latent_refiner_smoke_training(config)

    assert result.steps == 2
    assert len(result.loss_history) == 2
    assert all(math.isfinite(loss) for loss in result.loss_history)


def test_parse_args_builds_smoke_config(tmp_path):
    _, manifest_path = write_normalized_dataset(tmp_path)

    config = parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--manifest",
            str(manifest_path),
            "--window-size",
            "3",
            "--stride",
            "2",
            "--batch-size",
            "2",
            "--max-steps",
            "5",
            "--latent-dim",
            "8",
            "--condition-dim",
            "12",
            "--visual-dim",
            "6",
        ]
    )

    assert config.data_root == tmp_path
    assert config.manifest_path == manifest_path
    assert config.window_size == 3
    assert config.stride == 2
    assert config.batch_size == 2
    assert config.max_steps == 5
    assert config.latent_dim == 8
    assert config.condition_dim == 12
    assert config.visual_dim == 6
