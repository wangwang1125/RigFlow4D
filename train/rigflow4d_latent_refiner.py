from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from data.adapters.normalized_npz import NormalizedNpzAdapter
from data.window_dataset import MotionWindowDataset, collate_motion_windows
from models.rigflow4d import (
    KinematicVAE,
    LatentFlowMatcher,
    RigFlowConditionEncoder,
    RigFlowLatentRefiner,
)
from train.rigflow4d_stage1_vae import Stage1VAEConfig, build_stage1_vae, split_dataset_indices


@dataclass(frozen=True)
class LatentRefinerSmokeConfig:
    data_root: str | Path = Path("datasets/AMASS_RigFlow4D")
    manifest_path: str | Path = Path("manifest.json")
    output_dir: str | Path = Path("checkpoints/rigflow4d_stage2_latent_flow")
    stage1_checkpoint_path: str | Path | None = Path("checkpoints/rigflow4d_stage1_tgvae/vae_best.pt")
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
    val_fraction: float = 0.05
    log_every: int = 50
    eval_every: int = 1000
    save_every: int = 1000
    seed: int = 0
    num_workers: int = 0
    device: str = "cpu"
    drop_short: bool = False
    freeze_vae: bool = True
    grad_clip_norm: float | None = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.stage1_checkpoint_path is not None:
            object.__setattr__(self, "stage1_checkpoint_path", Path(self.stage1_checkpoint_path))
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
        _require_positive("log_every", self.log_every)
        _require_positive("eval_every", self.eval_every)
        _require_positive("save_every", self.save_every)
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive when provided")

    def to_dict(self) -> dict[str, object]:
        return {
            "data_root": str(self.data_root),
            "manifest_path": str(self.manifest_path),
            "output_dir": str(self.output_dir),
            "stage1_checkpoint_path": (
                str(self.stage1_checkpoint_path) if self.stage1_checkpoint_path is not None else None
            ),
            "window_size": self.window_size,
            "stride": self.stride,
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "lr": self.lr,
            "latent_dim": self.latent_dim,
            "condition_dim": self.condition_dim,
            "model_hidden_dim": self.model_hidden_dim,
            "condition_hidden_dim": self.condition_hidden_dim,
            "visual_dim": self.visual_dim,
            "camera_dim": self.camera_dim,
            "rig_dim": self.rig_dim,
            "pose_seed_dim": self.pose_seed_dim,
            "val_fraction": self.val_fraction,
            "log_every": self.log_every,
            "eval_every": self.eval_every,
            "save_every": self.save_every,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "device": self.device,
            "drop_short": self.drop_short,
            "freeze_vae": self.freeze_vae,
            "grad_clip_norm": self.grad_clip_norm,
        }


@dataclass(frozen=True)
class LatentRefinerSmokeResult:
    steps: int
    loss_history: list[float]


@dataclass(frozen=True)
class LatentRefinerTrainingResult:
    steps: int
    best_val_loss: float
    last_metrics: dict[str, float]
    output_dir: Path


def build_motion_dataloader(config: LatentRefinerSmokeConfig) -> DataLoader:
    dataset = build_motion_dataset(config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_motion_windows,
    )


def build_motion_dataset(config: LatentRefinerSmokeConfig) -> MotionWindowDataset:
    adapter = NormalizedNpzAdapter(root=config.data_root, manifest_path=config.manifest_path)
    dataset = MotionWindowDataset(
        adapter=adapter,
        window_size=config.window_size,
        stride=config.stride,
        drop_short=config.drop_short,
    )
    if len(dataset) == 0:
        raise ValueError("motion dataset produced no windows")
    return dataset


def build_stage2_dataloaders(
    config: LatentRefinerSmokeConfig,
) -> tuple[DataLoader, DataLoader | None]:
    dataset = build_motion_dataset(config)
    train_indices, val_indices = split_dataset_indices(
        dataset_len=len(dataset),
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_motion_windows,
        generator=generator,
    )
    val_loader = None
    if val_indices:
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_motion_windows,
        )
    return train_loader, val_loader


def build_latent_refiner(
    config: LatentRefinerSmokeConfig,
    device: torch.device | None = None,
) -> RigFlowLatentRefiner:
    device = torch.device("cpu") if device is None else device
    vae = _build_or_load_vae(config=config, device=device)
    condition_encoder = RigFlowConditionEncoder(
        visual_dim=config.visual_dim,
        camera_dim=config.camera_dim,
        rig_dim=config.rig_dim,
        pose_seed_dim=config.pose_seed_dim,
        hidden_dim=config.condition_hidden_dim,
        condition_dim=config.condition_dim,
    )
    flow_matcher = LatentFlowMatcher(
        latent_dim=vae.latent_dim,
        condition_dim=config.condition_dim,
        hidden_dim=config.model_hidden_dim,
    )
    refiner = RigFlowLatentRefiner(
        vae=vae,
        condition_encoder=condition_encoder,
        flow_matcher=flow_matcher,
    )
    if config.freeze_vae:
        refiner.vae.eval()
        for parameter in refiner.vae.parameters():
            parameter.requires_grad_(False)
    return refiner


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
        "parents": _as_tensor(batch["parents"], device=device, dtype=torch.long),
        "rest_offsets": _as_tensor(batch["rest_offsets"], device=device, dtype=torch.float32),
        "chain_ids": _as_tensor(batch["chain_ids"], device=device, dtype=torch.long),
        "chain_coordinates": _as_tensor(batch["chain_coordinates"], device=device, dtype=torch.float32),
        "time_mask": _as_tensor(batch["time_mask"], device=device, dtype=torch.bool),
        "joint_mask": _as_tensor(batch["joint_mask"], device=device, dtype=torch.bool),
    }
    torch_batch["pose_seed"] = _pose_seed_from_motion_batch(torch_batch)
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
    grad_clip_norm: float | None = None,
) -> float:
    model.train()
    if not any(parameter.requires_grad for parameter in model.vae.parameters()):
        model.vae.eval()
    optimizer.zero_grad(set_to_none=True)
    motion_only_batch = {
        "positions": motion_batch["positions"],
        "local_rotations_6d": motion_batch["local_rotations_6d"],
        "root_translation": motion_batch["root_translation"],
        "parents": motion_batch["parents"],
        "rest_offsets": motion_batch["rest_offsets"],
        "chain_ids": motion_batch["chain_ids"],
        "chain_coordinates": motion_batch["chain_coordinates"],
        "time_mask": motion_batch["time_mask"],
        "joint_mask": motion_batch["joint_mask"],
    }
    output = model(
        motion_batch=motion_only_batch,
        visual_tokens=motion_batch.get("visual_tokens"),
        visual_mask=motion_batch.get("visual_mask"),
        pose_seed=motion_batch.get("pose_seed"),
    )
    output.loss.backward()
    if grad_clip_norm is not None:
        clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            grad_clip_norm,
        )
    optimizer.step()
    return float(output.loss.detach().cpu())


@torch.no_grad()
def evaluate_latent_refiner(
    model: RigFlowLatentRefiner,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch in dataloader:
        motion_batch = motion_batch_to_torch(batch, device=device)
        output = model(
            motion_batch={
                "positions": motion_batch["positions"],
                "local_rotations_6d": motion_batch["local_rotations_6d"],
                "root_translation": motion_batch["root_translation"],
                "parents": motion_batch["parents"],
                "rest_offsets": motion_batch["rest_offsets"],
                "chain_ids": motion_batch["chain_ids"],
                "chain_coordinates": motion_batch["chain_coordinates"],
                "time_mask": motion_batch["time_mask"],
                "joint_mask": motion_batch["joint_mask"],
            },
            visual_tokens=motion_batch.get("visual_tokens"),
            visual_mask=motion_batch.get("visual_mask"),
            pose_seed=motion_batch.get("pose_seed"),
        )
        metrics = {key: float(value.detach().cpu()) for key, value in output.losses.items()}
        if not totals:
            totals = {key: 0.0 for key in metrics}
        for key in metrics:
            totals[key] += metrics[key]
        batches += 1
    if batches == 0:
        raise ValueError("evaluation dataloader produced no batches")
    return {key: value / batches for key, value in totals.items()}


def run_latent_refiner_smoke_training(
    config: LatentRefinerSmokeConfig,
) -> LatentRefinerSmokeResult:
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    dataloader = build_motion_dataloader(config)
    model = build_latent_refiner(config, device=device).to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.lr,
    )

    losses: list[float] = []
    while len(losses) < config.max_steps:
        for batch in dataloader:
            motion_batch = motion_batch_to_torch(batch, device=device)
            loss = train_latent_refiner_step(
                model,
                motion_batch,
                optimizer,
                grad_clip_norm=config.grad_clip_norm,
            )
            losses.append(loss)
            if len(losses) >= config.max_steps:
                break
    return LatentRefinerSmokeResult(steps=len(losses), loss_history=losses)


def run_stage2_latent_flow_training(
    config: LatentRefinerSmokeConfig,
) -> LatentRefinerTrainingResult:
    if config.stage1_checkpoint_path is None or not config.stage1_checkpoint_path.exists():
        raise FileNotFoundError(
            "Stage 2 training requires a Stage 1 VAE checkpoint. "
            f"Missing: {config.stage1_checkpoint_path}"
        )
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_stage2_dataloaders(config)
    model = build_latent_refiner(config, device=device).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("latent refiner has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.lr)

    metrics_path = config.output_dir / "metrics.jsonl"
    best_val_loss = float("inf")
    last_metrics: dict[str, float] = {}
    train_iter = iter(train_loader)
    step = 0
    while step < config.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        step += 1
        motion_batch = motion_batch_to_torch(batch, device=device)
        train_loss = train_latent_refiner_step(
            model=model,
            motion_batch=motion_batch,
            optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm,
        )
        last_metrics = {"train_loss": train_loss, "train_flow_loss": train_loss}

        should_eval = step % config.eval_every == 0 or step == config.max_steps
        if should_eval:
            eval_loader = val_loader if val_loader is not None else train_loader
            eval_metrics = evaluate_latent_refiner(model=model, dataloader=eval_loader, device=device)
            last_metrics.update({f"val_{key}": value for key, value in eval_metrics.items()})
            current_val_loss = eval_metrics["loss"]
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                save_latent_refiner_checkpoint(
                    path=config.output_dir / "flow_best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=config,
                    metrics=last_metrics,
                    best_val_loss=best_val_loss,
                )

        if step % config.save_every == 0 or step == config.max_steps:
            save_latent_refiner_checkpoint(
                path=config.output_dir / "flow_latest.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                config=config,
                metrics=last_metrics,
                best_val_loss=best_val_loss,
            )
        if step % config.log_every == 0 or step == config.max_steps or should_eval:
            _append_metrics(metrics_path, step=step, metrics=last_metrics)

    return LatentRefinerTrainingResult(
        steps=step,
        best_val_loss=best_val_loss,
        last_metrics=last_metrics,
        output_dir=config.output_dir,
    )


def save_latent_refiner_checkpoint(
    path: Path,
    model: RigFlowLatentRefiner,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: LatentRefinerSmokeConfig,
    metrics: Mapping[str, float],
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.to_dict(),
            "metrics": dict(metrics),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> LatentRefinerSmokeConfig:
    parser = argparse.ArgumentParser(description="RigFlow4D Stage 2 latent flow training")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/AMASS_RigFlow4D"))
    parser.add_argument("--manifest", type=Path, default=Path("manifest.json"), dest="manifest_path")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/rigflow4d_stage2_latent_flow"))
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=Path("checkpoints/rigflow4d_stage1_tgvae/vae_best.pt"),
        dest="stage1_checkpoint_path",
    )
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
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--drop-short", action="store_true")
    parser.add_argument("--unfreeze-vae", action="store_false", dest="freeze_vae")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return LatentRefinerSmokeConfig(
        data_root=args.data_root,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        stage1_checkpoint_path=args.stage1_checkpoint_path,
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
        val_fraction=args.val_fraction,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        drop_short=args.drop_short,
        freeze_vae=args.freeze_vae,
        grad_clip_norm=args.grad_clip_norm,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    config = parse_args(argv)
    result = run_stage2_latent_flow_training(config)
    print(
        f"steps={result.steps} best_val_loss={result.best_val_loss:.6f} "
        f"output_dir={result.output_dir}"
    )
    return 0


def _build_or_load_vae(config: LatentRefinerSmokeConfig, device: torch.device) -> KinematicVAE:
    checkpoint_path = config.stage1_checkpoint_path
    if checkpoint_path is None or not checkpoint_path.exists():
        return KinematicVAE(
            feature_dim=9,
            hidden_dim=config.model_hidden_dim,
            latent_dim=config.latent_dim,
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError(f"stage1 checkpoint '{checkpoint_path}' is missing config")
    stage1_config = Stage1VAEConfig(**raw_config)
    vae = build_stage1_vae(stage1_config)
    vae.load_state_dict(checkpoint["model_state"])
    return vae


def _pose_seed_from_motion_batch(batch: Mapping[str, Tensor]) -> Tensor:
    positions = batch["positions"]
    root_translation = batch["root_translation"]
    rotations = batch["local_rotations_6d"]
    root_relative_positions = positions - root_translation[:, :, None, :]
    return torch.cat([root_relative_positions, rotations], dim=-1)


def _append_metrics(path: Path, step: int, metrics: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"step": step, **{key: float(value) for key, value in metrics.items()}}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _as_tensor(value: object, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
