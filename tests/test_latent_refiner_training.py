import json
import math
import subprocess
import sys

import numpy as np
import torch

from train.rigflow4d_latent_refiner import (
    LatentRefinerSmokeConfig,
    build_latent_refiner,
    build_motion_dataloader,
    motion_batch_to_torch,
    parse_args,
    run_stage2_latent_flow_training,
    run_latent_refiner_smoke_training,
    train_latent_refiner_step,
)
from train.rigflow4d_stage1_vae import Stage1VAEConfig, build_stage1_vae, save_stage1_vae_checkpoint


def write_normalized_dataset(tmp_path, frames=6, joints=4, include_visual=False, visual_dim=6):
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def write_stage1_checkpoint(tmp_path, data_root, manifest_path, latent_dim=8):
    config = Stage1VAEConfig(
        data_root=data_root,
        manifest_path=manifest_path,
        output_dir=tmp_path / "stage1",
        window_size=3,
        stride=2,
        batch_size=2,
        max_steps=1,
        latent_dim=latent_dim,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        device="cpu",
    )
    model = build_stage1_vae(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint_path = config.output_dir / "vae_best.pt"
    save_stage1_vae_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=1,
        config=config,
        metrics={"val_loss": 1.0},
        best_val_loss=1.0,
    )
    return checkpoint_path


def test_build_motion_dataloader_reads_normalized_npz(tmp_path):
    config = make_config(tmp_path, max_steps=1)

    dataloader = build_motion_dataloader(config)
    batch = next(iter(dataloader))

    assert len(dataloader.dataset) > 0
    assert batch["positions"].shape == (2, 3, 4, 3)
    assert batch["local_rotations_6d"].shape == (2, 3, 4, 6)
    assert batch["parents"].shape == (2, 4)
    np.testing.assert_array_equal(batch["parents"][0], np.array([-1, 0, 1, 2], dtype=np.int64))
    assert batch["time_mask"].dtype == np.bool_
    assert batch["joint_mask"].dtype == np.bool_


def test_motion_batch_to_torch_converts_expected_dtypes(tmp_path):
    config = make_config(tmp_path, max_steps=1)
    batch = next(iter(build_motion_dataloader(config)))

    torch_batch = motion_batch_to_torch(batch, device=torch.device("cpu"))

    assert torch_batch["positions"].dtype == torch.float32
    assert torch_batch["local_rotations_6d"].dtype == torch.float32
    assert torch_batch["parents"].dtype == torch.long
    assert torch_batch["rest_offsets"].dtype == torch.float32
    assert torch_batch["chain_ids"].dtype == torch.long
    assert torch_batch["chain_coordinates"].dtype == torch.float32
    assert torch_batch["pose_seed"].shape == (2, 3, 4, 9)
    assert torch_batch["time_mask"].dtype == torch.bool
    assert torch_batch["joint_mask"].dtype == torch.bool


def test_build_latent_refiner_loads_and_freezes_stage1_checkpoint(tmp_path):
    config = make_config(tmp_path / "data", max_steps=1)
    checkpoint_path = write_stage1_checkpoint(tmp_path, config.data_root, config.manifest_path)
    config = LatentRefinerSmokeConfig(
        **{
            **config.to_dict(),
            "stage1_checkpoint_path": checkpoint_path,
            "freeze_vae": True,
        }
    )

    model = build_latent_refiner(config, device=torch.device("cpu"))

    assert model.vae.latent_dim == 8
    assert all(not parameter.requires_grad for parameter in model.vae.parameters())
    assert any(parameter.requires_grad for parameter in model.flow_matcher.parameters())


def test_train_latent_refiner_step_updates_flow_parameters(tmp_path):
    config = make_config(tmp_path, max_steps=1)
    model = build_latent_refiner(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    batch = next(iter(build_motion_dataloader(config)))
    torch_batch = motion_batch_to_torch(batch, device=torch.device("cpu"))
    before = model.flow_matcher.net[0].weight.detach().clone()
    vae_before = next(model.vae.parameters()).detach().clone()

    loss = train_latent_refiner_step(model, torch_batch, optimizer)

    after = model.flow_matcher.net[0].weight.detach()
    vae_after = next(model.vae.parameters()).detach()
    assert math.isfinite(loss)
    assert not torch.allclose(before, after)
    assert torch.allclose(vae_before, vae_after)


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


def test_stage2_config_infers_visual_dim_from_dataset(tmp_path):
    base_config = make_config(tmp_path, max_steps=1, include_visual=True, visual_dim=7)
    config = LatentRefinerSmokeConfig(
        **{
            **base_config.to_dict(),
            "visual_dim": None,
        }
    )

    model = build_latent_refiner(config)

    assert model.condition_encoder.visual_dim == 7


def test_run_latent_refiner_smoke_training_returns_finite_losses(tmp_path):
    config = make_config(tmp_path, max_steps=2)

    result = run_latent_refiner_smoke_training(config)

    assert result.steps == 2
    assert len(result.loss_history) == 2
    assert all(math.isfinite(loss) for loss in result.loss_history)


def test_run_stage2_latent_flow_training_saves_checkpoints_and_metrics(tmp_path):
    base_config = make_config(tmp_path / "data", max_steps=1)
    checkpoint_path = write_stage1_checkpoint(tmp_path, base_config.data_root, base_config.manifest_path)
    config = LatentRefinerSmokeConfig(
        **{
            **base_config.to_dict(),
            "stage1_checkpoint_path": checkpoint_path,
            "output_dir": tmp_path / "stage2",
            "max_steps": 2,
            "eval_every": 1,
            "save_every": 1,
            "device": "cpu",
        }
    )

    result = run_stage2_latent_flow_training(config)

    latest_path = config.output_dir / "flow_latest.pt"
    best_path = config.output_dir / "flow_best.pt"
    metrics_path = config.output_dir / "metrics.jsonl"
    latest = torch.load(latest_path, map_location="cpu")

    assert result.steps == 2
    assert latest_path.exists()
    assert best_path.exists()
    assert metrics_path.exists()
    assert latest["step"] == 2
    assert "model_state" in latest
    assert "optimizer_state" in latest
    assert "stage1_checkpoint_path" in latest["config"]
    assert math.isfinite(result.best_val_loss)
    assert "val_loss" in result.last_metrics
    assert metrics_path.read_text(encoding="utf-8").strip()


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
            "--output-dir",
            str(tmp_path / "stage2"),
            "--stage1-checkpoint",
            str(tmp_path / "stage1" / "vae_best.pt"),
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
    assert config.output_dir == tmp_path / "stage2"
    assert config.stage1_checkpoint_path == tmp_path / "stage1" / "vae_best.pt"


def test_latent_refiner_script_help_runs_from_file_path():
    result = subprocess.run(
        [sys.executable, "train/rigflow4d_latent_refiner.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "RigFlow4D Stage 2 latent flow training" in result.stdout
