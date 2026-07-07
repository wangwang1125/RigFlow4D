import json
import math
import subprocess
import sys

import numpy as np
import torch

from train.rigflow4d_stage1_vae import (
    Stage1VAEConfig,
    build_stage1_vae,
    parse_args,
    run_stage1_vae_training,
)


def write_normalized_dataset(tmp_path, frames=10, joints=4):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sample_path = tmp_path / "sample_0000.npz"
    manifest_path = tmp_path / "manifest.json"
    np.savez(
        sample_path,
        dataset_name=np.array("unit"),
        input_type=np.array("video"),
        source_label_type=np.array("motion_only"),
        camera_mode=np.array("unknown"),
        parents=np.array([-1, 0, 1, 2], dtype=np.int64)[:joints],
        rest_offsets=np.zeros((joints, 3), dtype=np.float32),
        joint_names=np.array([f"joint_{i}" for i in range(joints)]),
        chain_ids=np.arange(joints, dtype=np.int64),
        chain_coordinates=np.linspace(0.0, 1.0, joints, dtype=np.float32),
        positions=np.random.randn(frames, joints, 3).astype(np.float32),
        local_rotations_6d=np.random.randn(frames, joints, 6).astype(np.float32),
        root_translation=np.zeros((frames, 3), dtype=np.float32),
    )
    manifest_path.write_text(
        json.dumps({"samples": [{"sample_id": "sample_0000", "path": sample_path.name}]}),
        encoding="utf-8",
    )
    return tmp_path, manifest_path


def make_config(tmp_path, max_steps=3, resume_from=None):
    data_root, manifest_path = write_normalized_dataset(tmp_path / "data")
    return Stage1VAEConfig(
        data_root=data_root,
        manifest_path=manifest_path,
        output_dir=tmp_path / "checkpoints",
        window_size=4,
        stride=2,
        batch_size=2,
        max_steps=max_steps,
        lr=1e-3,
        beta=1e-3,
        latent_dim=6,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
        val_fraction=0.25,
        log_every=1,
        eval_every=2,
        save_every=2,
        seed=11,
        device="cpu",
        resume_from=resume_from,
    )


def test_stage1_vae_training_saves_latest_best_and_metrics(tmp_path):
    config = make_config(tmp_path, max_steps=3)

    result = run_stage1_vae_training(config)

    latest_path = config.output_dir / "vae_latest.pt"
    best_path = config.output_dir / "vae_best.pt"
    metrics_path = config.output_dir / "metrics.jsonl"
    latest = torch.load(latest_path, map_location="cpu")

    assert result.steps == 3
    assert latest_path.exists()
    assert best_path.exists()
    assert metrics_path.exists()
    assert latest["step"] == 3
    assert "model_state" in latest
    assert "optimizer_state" in latest
    assert math.isfinite(result.best_val_loss)
    assert "train_loss" in result.last_metrics
    assert "val_loss" in result.last_metrics
    assert metrics_path.read_text(encoding="utf-8").strip()


def test_stage1_vae_training_can_resume_from_latest(tmp_path):
    first_config = make_config(tmp_path, max_steps=2)
    run_stage1_vae_training(first_config)
    latest_path = first_config.output_dir / "vae_latest.pt"

    resumed_config = Stage1VAEConfig(
        **{
            **first_config.to_dict(),
            "max_steps": 4,
            "resume_from": latest_path,
        }
    )
    result = run_stage1_vae_training(resumed_config)
    latest = torch.load(latest_path, map_location="cpu")

    assert result.steps == 4
    assert latest["step"] == 4


def test_stage1_vae_parse_args_builds_formal_config(tmp_path):
    _, manifest_path = write_normalized_dataset(tmp_path / "data")

    config = parse_args(
        [
            "--data-root",
            str(manifest_path.parent),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "ckpt"),
            "--window-size",
            "8",
            "--stride",
            "4",
            "--batch-size",
            "3",
            "--max-steps",
            "7",
            "--num-layers",
            "2",
            "--num-heads",
            "4",
            "--ffn-dim",
            "64",
            "--dropout",
            "0.2",
            "--velocity-weight",
            "0.3",
            "--acceleration-weight",
            "0.04",
            "--bone-length-weight",
            "0.5",
            "--root-velocity-weight",
            "0.06",
            "--device",
            "cpu",
        ]
    )

    assert config.data_root == manifest_path.parent
    assert config.manifest_path == manifest_path
    assert config.output_dir == tmp_path / "ckpt"
    assert config.window_size == 8
    assert config.stride == 4
    assert config.batch_size == 3
    assert config.max_steps == 7
    assert config.num_layers == 2
    assert config.num_heads == 4
    assert config.ffn_dim == 64
    assert config.dropout == 0.2
    assert config.velocity_weight == 0.3
    assert config.acceleration_weight == 0.04
    assert config.bone_length_weight == 0.5
    assert config.root_velocity_weight == 0.06
    assert config.device == "cpu"


def test_stage1_vae_script_help_runs_from_file_path():
    result = subprocess.run(
        [sys.executable, "train/rigflow4d_stage1_vae.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "RigFlow4D Stage 1 kinematic VAE training" in result.stdout


def test_build_stage1_vae_uses_config_dimensions(tmp_path):
    config = make_config(tmp_path, max_steps=1)

    model = build_stage1_vae(config)

    assert model.hidden_dim == config.hidden_dim
    assert model.latent_dim == config.latent_dim
    assert model.num_layers == config.num_layers
    assert model.num_heads == config.num_heads
    assert model.ffn_dim == config.ffn_dim
