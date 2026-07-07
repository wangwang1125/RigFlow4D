from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.adapters.normalized_npz import NormalizedNpzAdapter
from data.window_dataset import MotionWindowDataset, collate_motion_windows
from train.rigflow4d_stage1_vae import Stage1VAEConfig, build_stage1_vae, motion_batch_to_torch


@dataclass(frozen=True)
class Stage1VisualizationConfig:
    data_root: str | Path = Path("datasets/AMASS_RigFlow4D")
    manifest_path: str | Path = Path("manifest.json")
    checkpoint_path: str | Path = Path("checkpoints/rigflow4d_stage1_tgvae/vae_best.pt")
    output_dir: str | Path = Path("outputs/visualize_stage1_vae")
    window_size: int | None = None
    stride: int | None = None
    num_samples: int = 3
    sample_indices: list[int] | None = None
    fps: int = 12
    width: int = 480
    height: int = 360
    device: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.window_size is not None and self.window_size <= 0:
            raise ValueError("window_size must be positive when provided")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be positive when provided")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")


@dataclass(frozen=True)
class Stage1VisualizationResult:
    gif_paths: list[Path]
    reconstruction_paths: list[Path]
    metrics_path: Path


def run_stage1_vae_visualization(config: Stage1VisualizationConfig) -> Stage1VisualizationResult:
    device = _resolve_device(config.device)
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model_config = _checkpoint_stage1_config(checkpoint)
    window_size = config.window_size or model_config.window_size
    stride = config.stride or model_config.stride

    adapter = NormalizedNpzAdapter(root=config.data_root, manifest_path=config.manifest_path)
    dataset = MotionWindowDataset(adapter=adapter, window_size=window_size, stride=stride)
    if len(dataset) == 0:
        raise ValueError("motion dataset produced no windows")

    model = build_stage1_vae(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = _select_sample_indices(config, dataset_len=len(dataset))
    gif_paths: list[Path] = []
    reconstruction_paths: list[Path] = []
    sample_metrics: list[dict[str, object]] = []

    for sample_index in sample_indices:
        window = dataset[sample_index]
        batch = collate_motion_windows([window])
        motion_batch = motion_batch_to_torch(batch, device=device)
        with torch.no_grad():
            output = model(motion_batch, sample_posterior=False)

        time_mask = batch["time_mask"][0].astype(bool)
        valid_frames = int(time_mask.sum())
        input_positions = batch["positions"][0, :valid_frames]
        input_rotations = batch["local_rotations_6d"][0, :valid_frames]
        recon_positions = output.positions[0, :valid_frames].detach().cpu().numpy()
        recon_rotations = output.local_rotations_6d[0, :valid_frames].detach().cpu().numpy()
        parents = batch["parents"][0, : input_positions.shape[1]].astype(np.int64)

        stem = f"sample_{sample_index:05d}_start_{int(batch['start'][0]):05d}"
        gif_path = config.output_dir / f"{stem}_gt_vs_recon.gif"
        reconstruction_path = config.output_dir / f"{stem}_reconstruction.npz"
        frames = render_skeleton_comparison_frames(
            input_positions=input_positions,
            reconstructed_positions=recon_positions,
            parents=parents,
            width=config.width,
            height=config.height,
        )
        save_gif(frames=frames, path=gif_path, fps=config.fps)
        np.savez(
            reconstruction_path,
            input_positions=input_positions.astype(np.float32),
            reconstructed_positions=recon_positions.astype(np.float32),
            input_local_rotations_6d=input_rotations.astype(np.float32),
            reconstructed_local_rotations_6d=recon_rotations.astype(np.float32),
            parents=parents,
            source_sample_index=np.array(int(batch["source_sample_index"][0]), dtype=np.int64),
            start=np.array(int(batch["start"][0]), dtype=np.int64),
        )

        metrics = compute_reconstruction_metrics(
            input_positions=input_positions,
            reconstructed_positions=recon_positions,
            parents=parents,
        )
        metrics.update(
            {
                "window_index": int(sample_index),
                "source_sample_index": int(batch["source_sample_index"][0]),
                "start": int(batch["start"][0]),
                "gif": gif_path.name,
                "reconstruction": reconstruction_path.name,
            }
        )
        sample_metrics.append(metrics)
        gif_paths.append(gif_path)
        reconstruction_paths.append(reconstruction_path)

    metrics_path = config.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": str(config.checkpoint_path),
                "data_root": str(config.data_root),
                "manifest": str(config.manifest_path),
                "window_size": window_size,
                "stride": stride,
                "samples": sample_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return Stage1VisualizationResult(
        gif_paths=gif_paths,
        reconstruction_paths=reconstruction_paths,
        metrics_path=metrics_path,
    )


def render_skeleton_comparison_frames(
    input_positions: np.ndarray,
    reconstructed_positions: np.ndarray,
    parents: np.ndarray,
    width: int = 480,
    height: int = 360,
) -> list[Image.Image]:
    if input_positions.shape != reconstructed_positions.shape:
        raise ValueError("input_positions and reconstructed_positions must have the same shape")
    if input_positions.ndim != 3 or input_positions.shape[-1] != 3:
        raise ValueError("positions must have shape [T, J, 3]")

    bounds = _projection_bounds(input_positions, reconstructed_positions)
    frames: list[Image.Image] = []
    for frame_index in range(input_positions.shape[0]):
        image = Image.new("RGB", (width * 2, height), color=(250, 250, 248))
        draw = ImageDraw.Draw(image)
        _draw_panel(
            draw=draw,
            positions=input_positions[frame_index],
            parents=parents,
            bounds=bounds,
            origin_x=0,
            width=width,
            height=height,
            title="Input / GT before VAE",
            color=(41, 102, 204),
        )
        _draw_panel(
            draw=draw,
            positions=reconstructed_positions[frame_index],
            parents=parents,
            bounds=bounds,
            origin_x=width,
            width=width,
            height=height,
            title="Encoder + Decoder reconstruction",
            color=(224, 113, 38),
        )
        draw.line((width, 0, width, height), fill=(210, 210, 210), width=1)
        draw.text((width - 56, height - 24), f"frame {frame_index + 1}/{input_positions.shape[0]}", fill=(70, 70, 70))
        frames.append(image)
    return frames


def save_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("frames must be non-empty")
    duration_ms = max(1, int(round(1000 / fps)))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def compute_reconstruction_metrics(
    input_positions: np.ndarray,
    reconstructed_positions: np.ndarray,
    parents: np.ndarray,
) -> dict[str, float]:
    diff = reconstructed_positions - input_positions
    metrics = {
        "mpjpe": float(np.linalg.norm(diff, axis=-1).mean()),
        "position_mse": float(np.square(diff).mean()),
    }
    if input_positions.shape[0] > 1:
        metrics["velocity_mse"] = float(np.square(np.diff(reconstructed_positions, axis=0) - np.diff(input_positions, axis=0)).mean())
    else:
        metrics["velocity_mse"] = 0.0
    if input_positions.shape[0] > 2:
        metrics["acceleration_mse"] = float(
            np.square(np.diff(reconstructed_positions, n=2, axis=0) - np.diff(input_positions, n=2, axis=0)).mean()
        )
    else:
        metrics["acceleration_mse"] = 0.0
    metrics["bone_length_mse"] = _bone_length_mse(input_positions, reconstructed_positions, parents)
    return metrics


def parse_args(argv: Optional[Iterable[str]] = None) -> Stage1VisualizationConfig:
    parser = argparse.ArgumentParser(description="Visualize RigFlow4D Stage 1 VAE reconstruction")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/AMASS_RigFlow4D"))
    parser.add_argument("--manifest", type=Path, default=Path("manifest.json"), dest="manifest_path")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/rigflow4d_stage1_tgvae/vae_best.pt"), dest="checkpoint_path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualize_stage1_vae"))
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--sample-index", action="append", type=int, dest="sample_indices")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return Stage1VisualizationConfig(
        data_root=args.data_root,
        manifest_path=args.manifest_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        num_samples=args.num_samples,
        sample_indices=args.sample_indices,
        fps=args.fps,
        width=args.width,
        height=args.height,
        device=args.device,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    config = parse_args(argv)
    result = run_stage1_vae_visualization(config)
    print(f"wrote {len(result.gif_paths)} visualization gif(s) to {config.output_dir}")
    print(f"metrics: {result.metrics_path}")
    return 0


def _checkpoint_stage1_config(checkpoint: dict[str, object]) -> Stage1VAEConfig:
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        return Stage1VAEConfig()
    return Stage1VAEConfig(**raw_config)


def _select_sample_indices(config: Stage1VisualizationConfig, dataset_len: int) -> list[int]:
    if config.sample_indices is not None:
        indices = config.sample_indices
    else:
        indices = list(range(min(config.num_samples, dataset_len)))
    for index in indices:
        if index < 0 or index >= dataset_len:
            raise IndexError(f"sample index {index} is out of range for {dataset_len} windows")
    return indices


def _projection_bounds(input_positions: np.ndarray, reconstructed_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate([input_positions[..., [0, 2]].reshape(-1, 2), reconstructed_positions[..., [0, 2]].reshape(-1, 2)], axis=0)
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    extent = max(float((points.max(axis=0) - points.min(axis=0)).max()), 1e-3)
    half = 0.5 * extent * 1.2
    return center - half, center + half


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    positions: np.ndarray,
    parents: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    origin_x: int,
    width: int,
    height: int,
    title: str,
    color: tuple[int, int, int],
) -> None:
    draw.rectangle((origin_x, 0, origin_x + width, height), outline=(220, 220, 220), width=1)
    draw.text((origin_x + 14, 12), title, fill=(35, 35, 35), font=ImageFont.load_default())
    projected = _project_points(positions, bounds=bounds, origin_x=origin_x, width=width, height=height)
    for joint_index, parent_index in enumerate(parents.tolist()):
        if parent_index < 0 or parent_index >= len(projected):
            continue
        draw.line((*projected[parent_index], *projected[joint_index]), fill=color, width=3)
    for point_index, point in enumerate(projected):
        radius = 5 if point_index == 0 else 3
        fill = (30, 30, 30) if point_index == 0 else color
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=fill)


def _project_points(
    positions: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    origin_x: int,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    low, high = bounds
    xy = positions[:, [0, 2]]
    normalized = (xy - low) / np.maximum(high - low, 1e-6)
    padding = 32
    x = origin_x + padding + normalized[:, 0] * (width - 2 * padding)
    y = padding + (1.0 - normalized[:, 1]) * (height - 2 * padding)
    return [(int(round(px)), int(round(py))) for px, py in zip(x, y)]


def _bone_length_mse(input_positions: np.ndarray, reconstructed_positions: np.ndarray, parents: np.ndarray) -> float:
    values: list[float] = []
    for joint_index, parent_index in enumerate(parents.tolist()):
        if parent_index < 0 or parent_index >= input_positions.shape[1]:
            continue
        input_length = np.linalg.norm(input_positions[:, joint_index] - input_positions[:, parent_index], axis=-1)
        recon_length = np.linalg.norm(reconstructed_positions[:, joint_index] - reconstructed_positions[:, parent_index], axis=-1)
        values.append(float(np.square(recon_length - input_length).mean()))
    if not values:
        return 0.0
    return float(np.mean(values))


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


if __name__ == "__main__":
    raise SystemExit(main())
