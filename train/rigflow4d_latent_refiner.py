from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from data.adapters.normalized_npz import NormalizedNpzAdapter
from data.window_dataset import MotionWindowDataset, collate_motion_windows
from models.rigflow4d import (
    KinematicVAE,
    LatentFlowMatcher,
    RigFlowConditionEncoder,
    RigFlowLatentRefiner,
)


@dataclass(frozen=True)
class LatentRefinerSmokeConfig:
    data_root: str | Path
    manifest_path: str | Path
    window_size: int = 8
    stride: int = 8
    batch_size: int = 2
    max_steps: int = 2
    lr: float = 1e-3
    latent_dim: int = 32
    condition_dim: int = 64
    model_hidden_dim: int = 128
    condition_hidden_dim: int = 64
    visual_dim: int = 1
    camera_dim: int = 1
    rig_dim: int = 1
    pose_seed_dim: int = 9
    seed: int = 0
    num_workers: int = 0
    device: str = "cpu"
    drop_short: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        _require_positive("window_size", self.window_size)
        _require_positive("stride", self.stride)
        _require_positive("batch_size", self.batch_size)
        _require_positive("max_steps", self.max_steps)
        _require_positive("latent_dim", self.latent_dim)
        _require_positive("condition_dim", self.condition_dim)
        _require_positive("model_hidden_dim", self.model_hidden_dim)
        _require_positive("condition_hidden_dim", self.condition_hidden_dim)
        _require_positive("visual_dim", self.visual_dim)
        _require_positive("camera_dim", self.camera_dim)
        _require_positive("rig_dim", self.rig_dim)
        _require_positive("pose_seed_dim", self.pose_seed_dim)
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


@dataclass(frozen=True)
class LatentRefinerSmokeResult:
    steps: int
    loss_history: list[float]


def build_motion_dataloader(config: LatentRefinerSmokeConfig) -> DataLoader:
    adapter = NormalizedNpzAdapter(root=config.data_root, manifest_path=config.manifest_path)
    dataset = MotionWindowDataset(
        adapter=adapter,
        window_size=config.window_size,
        stride=config.stride,
        drop_short=config.drop_short,
    )
    if len(dataset) == 0:
        raise ValueError("motion dataset produced no windows")
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_motion_windows,
    )


def build_latent_refiner(config: LatentRefinerSmokeConfig) -> RigFlowLatentRefiner:
    vae = KinematicVAE(
        feature_dim=9,
        hidden_dim=config.model_hidden_dim,
        latent_dim=config.latent_dim,
    )
    condition_encoder = RigFlowConditionEncoder(
        visual_dim=config.visual_dim,
        camera_dim=config.camera_dim,
        rig_dim=config.rig_dim,
        pose_seed_dim=config.pose_seed_dim,
        hidden_dim=config.condition_hidden_dim,
        condition_dim=config.condition_dim,
    )
    flow_matcher = LatentFlowMatcher(
        latent_dim=config.latent_dim,
        condition_dim=config.condition_dim,
        hidden_dim=config.model_hidden_dim,
    )
    return RigFlowLatentRefiner(
        vae=vae,
        condition_encoder=condition_encoder,
        flow_matcher=flow_matcher,
    )


def motion_batch_to_torch(
    batch: Mapping[str, object],
    device: torch.device,
) -> dict[str, Tensor]:
    torch_batch = {
        "positions": _as_tensor(batch["positions"], device=device, dtype=torch.float32),
        "local_rotations_6d": _as_tensor(
            batch["local_rotations_6d"],
            device=device,
            dtype=torch.float32,
        ),
        "root_translation": _as_tensor(batch["root_translation"], device=device, dtype=torch.float32),
        "time_mask": _as_tensor(batch["time_mask"], device=device, dtype=torch.bool),
        "joint_mask": _as_tensor(batch["joint_mask"], device=device, dtype=torch.bool),
    }
    if "visual_tokens" in batch:
        torch_batch["visual_tokens"] = _as_tensor(
            batch["visual_tokens"],
            device=device,
            dtype=torch.float32,
        )
        torch_batch["visual_mask"] = _as_tensor(
            batch["visual_mask"],
            device=device,
            dtype=torch.bool,
        )
    return torch_batch


def train_latent_refiner_step(
    model: RigFlowLatentRefiner,
    motion_batch: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    motion_only_batch = {
        "positions": motion_batch["positions"],
        "local_rotations_6d": motion_batch["local_rotations_6d"],
        "root_translation": motion_batch["root_translation"],
        "time_mask": motion_batch["time_mask"],
        "joint_mask": motion_batch["joint_mask"],
    }
    output = model(
        motion_batch=motion_only_batch,
        visual_tokens=motion_batch.get("visual_tokens"),
        visual_mask=motion_batch.get("visual_mask"),
    )
    output.loss.backward()
    optimizer.step()
    return float(output.loss.detach().cpu())


def run_latent_refiner_smoke_training(
    config: LatentRefinerSmokeConfig,
) -> LatentRefinerSmokeResult:
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    dataloader = build_motion_dataloader(config)
    model = build_latent_refiner(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    losses: list[float] = []
    while len(losses) < config.max_steps:
        for batch in dataloader:
            motion_batch = motion_batch_to_torch(batch, device=device)
            loss = train_latent_refiner_step(model, motion_batch, optimizer)
            losses.append(loss)
            if len(losses) >= config.max_steps:
                break
    return LatentRefinerSmokeResult(steps=len(losses), loss_history=losses)


def parse_args(argv: Optional[Iterable[str]] = None) -> LatentRefinerSmokeConfig:
    parser = argparse.ArgumentParser(description="RigFlow4D latent refiner smoke training")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path, dest="manifest_path")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--model-hidden-dim", type=int, default=128)
    parser.add_argument("--condition-hidden-dim", type=int, default=64)
    parser.add_argument("--visual-dim", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return LatentRefinerSmokeConfig(
        data_root=args.data_root,
        manifest_path=args.manifest_path,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
        model_hidden_dim=args.model_hidden_dim,
        condition_hidden_dim=args.condition_hidden_dim,
        visual_dim=args.visual_dim,
        seed=args.seed,
        device=args.device,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    config = parse_args(argv)
    result = run_latent_refiner_smoke_training(config)
    for step, loss in enumerate(result.loss_history, start=1):
        print(f"step={step} loss={loss:.6f}")
    return 0


def _as_tensor(value: object, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
