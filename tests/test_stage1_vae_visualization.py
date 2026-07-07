import json
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from inference.visualize_stage1_vae import Stage1VisualizationConfig, parse_args, run_stage1_vae_visualization
from train.rigflow4d_stage1_vae import Stage1VAEConfig, build_stage1_vae, save_stage1_vae_checkpoint


def write_normalized_dataset(tmp_path, frames=10, joints=4, positions=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sample_path = tmp_path / "sample_0000.npz"
    manifest_path = tmp_path / "manifest.json"
    if positions is None:
        positions = np.random.randn(frames, joints, 3).astype(np.float32)
    else:
        positions = np.asarray(positions, dtype=np.float32)
        frames, joints = positions.shape[:2]
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
        positions=positions,
        local_rotations_6d=np.random.randn(frames, joints, 6).astype(np.float32),
        root_translation=np.zeros((frames, 3), dtype=np.float32),
    )
    manifest_path.write_text(
        json.dumps({"samples": [{"sample_id": "sample_0000", "path": sample_path.name}]}),
        encoding="utf-8",
    )
    return tmp_path, manifest_path


def write_checkpoint(tmp_path, data_root, manifest_path):
    config = Stage1VAEConfig(
        data_root=data_root,
        manifest_path=manifest_path,
        output_dir=tmp_path / "ckpt",
        window_size=4,
        stride=2,
        batch_size=1,
        max_steps=1,
        latent_dim=8,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
        device="cpu",
    )
    model = build_stage1_vae(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "ckpt" / "vae_best.pt"
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


def test_stage1_vae_visualization_writes_reconstruction_artifacts(tmp_path):
    data_root, manifest_path = write_normalized_dataset(tmp_path / "data", frames=6, joints=4)
    checkpoint_path = write_checkpoint(tmp_path, data_root, manifest_path)
    output_dir = tmp_path / "vis"

    result = run_stage1_vae_visualization(
        Stage1VisualizationConfig(
            data_root=data_root,
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            window_size=4,
            stride=2,
            sample_indices=[0],
            fps=6,
            width=320,
            height=220,
            view="multi",
            device="cpu",
        )
    )

    assert len(result.gif_paths) == 1
    assert result.gif_paths[0].exists()
    assert result.reconstruction_paths[0].exists()
    assert result.metrics_path.exists()
    with Image.open(result.gif_paths[0]) as image:
        assert image.n_frames == 4
        assert image.size == (640, 660)
    with np.load(result.reconstruction_paths[0]) as recon:
        assert recon["input_positions"].shape == (4, 4, 3)
        assert recon["reconstructed_positions"].shape == (4, 4, 3)
        assert recon["parents"].shape == (4,)
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["samples"][0]["mpjpe"] >= 0
    assert metrics["view"] == "multi"
    assert metrics["views"] == ["front", "side", "top"]


def test_stage1_vae_visualization_defaults_to_multi_view():
    config = parse_args([])

    assert config.view == "multi"
    assert config.selection == "motion"
    assert config.trail_frames == 12


def test_stage1_vae_visualization_defaults_to_motionful_windows(tmp_path):
    positions = np.zeros((12, 4, 3), dtype=np.float32)
    positions[:, :, 1] = np.linspace(0.0, 0.3, 4, dtype=np.float32)
    positions[4:, :, 0] = np.linspace(0.0, 4.0, 8, dtype=np.float32)[:, None]
    data_root, manifest_path = write_normalized_dataset(tmp_path / "data", positions=positions)
    checkpoint_path = write_checkpoint(tmp_path, data_root, manifest_path)
    output_dir = tmp_path / "vis"

    result = run_stage1_vae_visualization(
        Stage1VisualizationConfig(
            data_root=data_root,
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            window_size=4,
            stride=2,
            num_samples=1,
            width=320,
            height=220,
            view="front",
            device="cpu",
        )
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["selection"] == "motion"
    assert metrics["trail_frames"] == 12
    assert metrics["samples"][0]["start"] > 0


def test_stage1_vae_visualization_help_runs_from_file_path():
    result = subprocess.run(
        [sys.executable, "inference/visualize_stage1_vae.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Visualize RigFlow4D Stage 1 VAE reconstruction" in result.stdout
