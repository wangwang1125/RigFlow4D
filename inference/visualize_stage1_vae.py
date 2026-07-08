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
from train.rigflow4d_stage1_vae import (
    Stage1VAEConfig,
    build_stage1_vae,
    motion_batch_to_torch,
    split_dataset_indices,
)


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
    view: str = "multi"
    split: str = "val"
    selection: str = "motion"
    candidate_windows: int = 512
    trail_frames: int = 12
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
        if self.view not in {"multi", "front", "side", "top"}:
            raise ValueError("view must be one of: multi, front, side, top")
        if self.split not in {"val", "train", "all"}:
            raise ValueError("split must be one of: val, train, all")
        if self.selection not in {"motion", "first"}:
            raise ValueError("selection must be one of: motion, first")
        if self.candidate_windows <= 0:
            raise ValueError("candidate_windows must be positive")
        if self.trail_frames < 0:
            raise ValueError("trail_frames must be non-negative")


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
    split_indices, split_summary = _split_indices_for_visualization(
        config=config,
        dataset_len=len(dataset),
        model_config=model_config,
    )

    model = build_stage1_vae(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    selected_samples = _select_sample_indices(
        config,
        dataset=dataset,
        split_indices=split_indices,
    )
    gif_paths: list[Path] = []
    reconstruction_paths: list[Path] = []
    sample_metrics: list[dict[str, object]] = []

    for sample_index, selection_score, selected_split in selected_samples:
        window = dataset[sample_index]
        batch = collate_motion_windows([window])
        motion_batch = motion_batch_to_torch(batch, device=device)
        with torch.no_grad():
            output = model(motion_batch, sample_posterior=False)

        time_mask = batch["time_mask"][0].astype(bool)
        valid_frames = int(time_mask.sum())
        input_positions = batch["positions"][0, :valid_frames]
        input_rotations = batch["local_rotations_6d"][0, :valid_frames]
        input_root_translation = batch["root_translation"][0, :valid_frames]
        recon_positions = output.positions[0, :valid_frames].detach().cpu().numpy()
        recon_rotations = output.local_rotations_6d[0, :valid_frames].detach().cpu().numpy()
        recon_root_translation = output.root_translation[0, :valid_frames].detach().cpu().numpy()
        parents = batch["parents"][0, : input_positions.shape[1]].astype(np.int64)
        source_sample = adapter[int(batch["source_sample_index"][0])]
        joint_names = np.asarray(source_sample.rig.joint_names[: input_positions.shape[1]])
        joint_count = int(input_positions.shape[1])

        stem = f"sample_{sample_index:05d}_start_{int(batch['start'][0]):05d}"
        gif_path = config.output_dir / f"{stem}_gt_vs_recon.gif"
        reconstruction_path = config.output_dir / f"{stem}_reconstruction.npz"
        frames = render_skeleton_comparison_frames(
            input_positions=input_positions,
            reconstructed_positions=recon_positions,
            parents=parents,
            width=config.width,
            height=config.height,
            view=config.view,
            trail_frames=config.trail_frames,
        )
        save_gif(frames=frames, path=gif_path, fps=config.fps)
        np.savez(
            reconstruction_path,
            input_positions=input_positions.astype(np.float32),
            reconstructed_positions=recon_positions.astype(np.float32),
            input_root_translation=input_root_translation.astype(np.float32),
            reconstructed_root_translation=recon_root_translation.astype(np.float32),
            input_local_rotations_6d=input_rotations.astype(np.float32),
            reconstructed_local_rotations_6d=recon_rotations.astype(np.float32),
            parents=parents,
            joint_count=np.array(joint_count, dtype=np.int64),
            joint_names=joint_names,
            source_sample_index=np.array(int(batch["source_sample_index"][0]), dtype=np.int64),
            start=np.array(int(batch["start"][0]), dtype=np.int64),
        )

        metrics = compute_reconstruction_metrics(
            input_positions=input_positions,
            reconstructed_positions=recon_positions,
            input_root_translation=input_root_translation,
            reconstructed_root_translation=recon_root_translation,
            parents=parents,
        )
        metrics.update(
            {
                "window_index": int(sample_index),
                "source_sample_index": int(batch["source_sample_index"][0]),
                "start": int(batch["start"][0]),
                "gif": gif_path.name,
                "reconstruction": reconstruction_path.name,
                "selection_score": float(selection_score),
                "split": selected_split,
                "joint_count": joint_count,
                "joint_names_head": [str(name) for name in joint_names[: min(joint_count, 12)]],
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
                "split": "explicit" if config.sample_indices is not None else config.split,
                "total_window_count": len(dataset),
                "split_window_count": len(split_indices),
                **split_summary,
                "selection": "explicit" if config.sample_indices is not None else config.selection,
                "candidate_windows": config.candidate_windows,
                "trail_frames": config.trail_frames,
                "view": config.view,
                "views": list(_view_sequence(config.view)),
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
    view: str = "multi",
    trail_frames: int = 12,
) -> list[Image.Image]:
    if input_positions.shape != reconstructed_positions.shape:
        raise ValueError("input_positions and reconstructed_positions must have the same shape")
    if input_positions.ndim != 3 or input_positions.shape[-1] != 3:
        raise ValueError("positions must have shape [T, J, 3]")

    views = _view_sequence(view)
    frames: list[Image.Image] = []
    for frame_index in range(input_positions.shape[0]):
        image = Image.new("RGB", (width * 2, height * len(views)), color=(250, 250, 248))
        draw = ImageDraw.Draw(image)
        for view_row, view_name in enumerate(views):
            axes = _view_axes(view_name)
            bounds = _projection_bounds(input_positions, reconstructed_positions, axes=axes)
            origin_y = view_row * height
            _draw_panel(
                draw=draw,
                positions=input_positions[frame_index],
                parents=parents,
                bounds=bounds,
                axes=axes,
                origin_x=0,
                origin_y=origin_y,
                width=width,
                height=height,
                title=f"{view_name} - Input / GT before VAE",
                color=(41, 102, 204),
                history_positions=input_positions[: frame_index + 1],
                trail_frames=trail_frames,
            )
            _draw_panel(
                draw=draw,
                positions=reconstructed_positions[frame_index],
                parents=parents,
                bounds=bounds,
                axes=axes,
                origin_x=width,
                origin_y=origin_y,
                width=width,
                height=height,
                title=f"{view_name} - Encoder + Decoder reconstruction",
                color=(224, 113, 38),
                history_positions=reconstructed_positions[: frame_index + 1],
                trail_frames=trail_frames,
            )
            draw.line((width, origin_y, width, origin_y + height), fill=(210, 210, 210), width=1)
            if view_row > 0:
                draw.line((0, origin_y, width * 2, origin_y), fill=(210, 210, 210), width=1)
        draw.text(
            (width - 56, height * len(views) - 24),
            f"frame {frame_index + 1}/{input_positions.shape[0]}",
            fill=(70, 70, 70),
        )
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
    input_root_translation: np.ndarray | None = None,
    reconstructed_root_translation: np.ndarray | None = None,
) -> dict[str, float]:
    input_root = input_positions[:, 0] if input_root_translation is None else input_root_translation
    reconstructed_root = reconstructed_positions[:, 0] if reconstructed_root_translation is None else reconstructed_root_translation
    diff = reconstructed_positions - input_positions
    relative_diff = (reconstructed_positions - reconstructed_root[:, None, :]) - (input_positions - input_root[:, None, :])
    root_diff = reconstructed_root - input_root
    metrics = {
        "mpjpe": float(np.linalg.norm(diff, axis=-1).mean()),
        "root_mpjpe": float(np.linalg.norm(root_diff, axis=-1).mean()),
        "root_relative_mpjpe": float(np.linalg.norm(relative_diff, axis=-1).mean()),
        "position_mse": float(np.square(diff).mean()),
        "root_relative_position_mse": float(np.square(relative_diff).mean()),
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
    parser.add_argument("--view", choices=("multi", "front", "side", "top"), default="multi")
    parser.add_argument("--split", choices=("val", "train", "all"), default="val")
    parser.add_argument("--selection", choices=("motion", "first"), default="motion")
    parser.add_argument("--candidate-windows", type=int, default=512)
    parser.add_argument("--trail-frames", type=int, default=12)
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
        view=args.view,
        split=args.split,
        selection=args.selection,
        candidate_windows=args.candidate_windows,
        trail_frames=args.trail_frames,
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


def _split_indices_for_visualization(
    config: Stage1VisualizationConfig,
    dataset_len: int,
    model_config: Stage1VAEConfig,
) -> tuple[list[int], dict[str, int]]:
    train_indices, val_indices = split_dataset_indices(
        dataset_len=dataset_len,
        val_fraction=model_config.val_fraction,
        seed=model_config.seed,
    )
    if config.split == "train":
        split_indices = train_indices
    elif config.split == "val":
        split_indices = val_indices
    else:
        split_indices = list(range(dataset_len))
    if config.sample_indices is None and not split_indices:
        raise ValueError(
            f"{config.split} split produced no windows; use --split train or --split all"
        )
    return split_indices, {
        "train_window_count": len(train_indices),
        "val_window_count": len(val_indices),
    }


def _select_sample_indices(
    config: Stage1VisualizationConfig,
    dataset: MotionWindowDataset,
    split_indices: list[int],
) -> list[tuple[int, float, str]]:
    dataset_len = len(dataset)
    if config.sample_indices is not None:
        indices = [
            (index, _motion_score(dataset[index]), "explicit")
            for index in config.sample_indices
        ]
    elif config.selection == "first":
        indices = [
            (index, _motion_score(dataset[index]), config.split)
            for index in split_indices[: min(config.num_samples, len(split_indices))]
        ]
    else:
        indices = _select_motionful_sample_indices(
            dataset=dataset,
            num_samples=config.num_samples,
            candidate_windows=config.candidate_windows,
            candidate_indices=split_indices,
            split_name=config.split,
        )
    for index, _, _ in indices:
        if index < 0 or index >= dataset_len:
            raise IndexError(f"sample index {index} is out of range for {dataset_len} windows")
    return indices


def _select_motionful_sample_indices(
    dataset: MotionWindowDataset,
    num_samples: int,
    candidate_windows: int,
    candidate_indices: list[int],
    split_name: str,
) -> list[tuple[int, float, str]]:
    candidate_count = min(len(candidate_indices), max(num_samples, candidate_windows))
    if candidate_count <= 0:
        return []
    candidate_positions = np.linspace(0, len(candidate_indices) - 1, candidate_count, dtype=np.int64)
    scored = [
        (int(candidate_indices[int(position)]), _motion_score(dataset[int(candidate_indices[int(position)])]), split_name)
        for position in np.unique(candidate_positions)
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:num_samples]


def _motion_score(window: object) -> float:
    positions = np.asarray(window.positions)
    valid_positions = positions[np.asarray(window.time_mask, dtype=bool)]
    if valid_positions.shape[0] <= 1:
        return 0.0
    velocities = np.diff(valid_positions, axis=0)
    return float(np.linalg.norm(velocities, axis=-1).mean())


def _projection_bounds(
    input_positions: np.ndarray,
    reconstructed_positions: np.ndarray,
    axes: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(
        [
            input_positions[..., list(axes)].reshape(-1, 2),
            reconstructed_positions[..., list(axes)].reshape(-1, 2),
        ],
        axis=0,
    )
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    extent = max(float((points.max(axis=0) - points.min(axis=0)).max()), 1e-3)
    half = 0.5 * extent * 1.2
    return center - half, center + half


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    positions: np.ndarray,
    parents: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    axes: tuple[int, int],
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
    title: str,
    color: tuple[int, int, int],
    history_positions: np.ndarray | None = None,
    trail_frames: int = 0,
) -> None:
    draw.rectangle((origin_x, origin_y, origin_x + width, origin_y + height), outline=(220, 220, 220), width=1)
    draw.text((origin_x + 14, origin_y + 12), title, fill=(35, 35, 35), font=ImageFont.load_default())
    projected = _project_points(
        positions,
        bounds=bounds,
        axes=axes,
        origin_x=origin_x,
        origin_y=origin_y,
        width=width,
        height=height,
    )
    if history_positions is not None and trail_frames > 0:
        _draw_motion_trail(
            draw=draw,
            history_positions=history_positions,
            parents=parents,
            bounds=bounds,
            axes=axes,
            origin_x=origin_x,
            origin_y=origin_y,
            width=width,
            height=height,
            color=color,
            trail_frames=trail_frames,
        )
    for joint_index, parent_index in enumerate(parents.tolist()):
        if parent_index < 0 or parent_index >= len(projected):
            continue
        draw.line((*projected[parent_index], *projected[joint_index]), fill=color, width=3)
    for point_index, point in enumerate(projected):
        radius = 5 if point_index == 0 else 3
        fill = (30, 30, 30) if point_index == 0 else color
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=fill)


def _draw_motion_trail(
    draw: ImageDraw.ImageDraw,
    history_positions: np.ndarray,
    parents: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    axes: tuple[int, int],
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
    trail_frames: int,
) -> None:
    if history_positions.shape[0] <= 1:
        return
    start = max(0, history_positions.shape[0] - trail_frames - 1)
    trail = history_positions[start:-1]
    root_points: list[tuple[int, int]] = []
    for ghost_index, ghost_positions in enumerate(trail):
        fade = 0.12 + 0.18 * float(ghost_index + 1) / max(1, trail.shape[0])
        ghost_color = _blend_color(color, (250, 250, 248), fade)
        projected = _project_points(
            ghost_positions,
            bounds=bounds,
            axes=axes,
            origin_x=origin_x,
            origin_y=origin_y,
            width=width,
            height=height,
        )
        root_points.append(projected[0])
        for joint_index, parent_index in enumerate(parents.tolist()):
            if parent_index < 0 or parent_index >= len(projected):
                continue
            draw.line((*projected[parent_index], *projected[joint_index]), fill=ghost_color, width=1)
    current_root = _project_points(
        history_positions[-1],
        bounds=bounds,
        axes=axes,
        origin_x=origin_x,
        origin_y=origin_y,
        width=width,
        height=height,
    )[0]
    root_points.append(current_root)
    if len(root_points) > 1:
        draw.line(root_points, fill=_blend_color(color, (250, 250, 248), 0.45), width=2)


def _blend_color(
    foreground: tuple[int, int, int],
    background: tuple[int, int, int],
    foreground_weight: float,
) -> tuple[int, int, int]:
    foreground_weight = max(0.0, min(1.0, foreground_weight))
    return tuple(
        int(round(background_channel * (1.0 - foreground_weight) + foreground_channel * foreground_weight))
        for foreground_channel, background_channel in zip(foreground, background)
    )


def _project_points(
    positions: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    axes: tuple[int, int],
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    low, high = bounds
    xy = positions[:, list(axes)]
    normalized = (xy - low) / np.maximum(high - low, 1e-6)
    padding = 32
    x = origin_x + padding + normalized[:, 0] * (width - 2 * padding)
    y = origin_y + padding + (1.0 - normalized[:, 1]) * (height - 2 * padding)
    return [(int(round(px)), int(round(py))) for px, py in zip(x, y)]


def _view_sequence(view: str) -> tuple[str, ...]:
    if view == "multi":
        return ("front", "side", "top")
    if view in {"front", "side", "top"}:
        return (view,)
    raise ValueError("view must be one of: multi, front, side, top")


def _view_axes(view: str) -> tuple[int, int]:
    if view == "front":
        return (0, 1)
    if view == "side":
        return (2, 1)
    if view == "top":
        return (0, 2)
    raise ValueError("view must be one of: front, side, top")


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
