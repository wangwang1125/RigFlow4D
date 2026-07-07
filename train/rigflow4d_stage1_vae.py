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
from models.rigflow4d import KinematicVAE, kinematic_vae_loss


@dataclass(frozen=True)
class Stage1VAEConfig:
    data_root: str | Path = Path("datasets/AMASS_RigFlow4D")
    manifest_path: str | Path = Path("manifest.json")
    output_dir: str | Path = Path("checkpoints/rigflow4d_stage1_tgvae")
    window_size: int = 64
    stride: int = 32
    batch_size: int = 16
    max_steps: int = 100000
    lr: float = 1e-4
    beta: float = 1e-3
    latent_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int | None = 1024
    dropout: float = 0.1
    velocity_weight: float = 0.1
    acceleration_weight: float = 0.01
    bone_length_weight: float = 0.1
    root_velocity_weight: float = 0.05
    val_fraction: float = 0.05
    log_every: int = 50
    eval_every: int = 1000
    save_every: int = 1000
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"
    resume_from: str | Path | None = None
    drop_short: bool = False
    grad_clip_norm: float | None = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.resume_from is not None:
            object.__setattr__(self, "resume_from", Path(self.resume_from))

        _require_positive("window_size", self.window_size)
        _require_positive("stride", self.stride)
        _require_positive("batch_size", self.batch_size)
        _require_positive("max_steps", self.max_steps)
        _require_positive("latent_dim", self.latent_dim)
        _require_positive("hidden_dim", self.hidden_dim)
        _require_positive("num_layers", self.num_layers)
        _require_positive("num_heads", self.num_heads)
        _require_positive("log_every", self.log_every)
        _require_positive("eval_every", self.eval_every)
        _require_positive("save_every", self.save_every)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.ffn_dim is not None and self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive when provided")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        for name, value in {
            "velocity_weight": self.velocity_weight,
            "acceleration_weight": self.acceleration_weight,
            "bone_length_weight": self.bone_length_weight,
            "root_velocity_weight": self.root_velocity_weight,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
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
            "window_size": self.window_size,
            "stride": self.stride,
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "lr": self.lr,
            "beta": self.beta,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "velocity_weight": self.velocity_weight,
            "acceleration_weight": self.acceleration_weight,
            "bone_length_weight": self.bone_length_weight,
            "root_velocity_weight": self.root_velocity_weight,
            "val_fraction": self.val_fraction,
            "log_every": self.log_every,
            "eval_every": self.eval_every,
            "save_every": self.save_every,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "device": self.device,
            "resume_from": str(self.resume_from) if self.resume_from is not None else None,
            "drop_short": self.drop_short,
            "grad_clip_norm": self.grad_clip_norm,
        }


@dataclass(frozen=True)
class Stage1VAEResult:
    steps: int
    best_val_loss: float
    last_metrics: dict[str, float]
    output_dir: Path


def build_stage1_vae(config: Stage1VAEConfig) -> KinematicVAE:
    return KinematicVAE(
        feature_dim=9,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
    )


def build_stage1_vae_dataloaders(config: Stage1VAEConfig) -> tuple[DataLoader, DataLoader | None]:
    adapter = NormalizedNpzAdapter(root=config.data_root, manifest_path=config.manifest_path)
    dataset = MotionWindowDataset(
        adapter=adapter,
        window_size=config.window_size,
        stride=config.stride,
        drop_short=config.drop_short,
    )
    if len(dataset) == 0:
        raise ValueError("motion dataset produced no windows")

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


def split_dataset_indices(dataset_len: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if dataset_len <= 0:
        raise ValueError("dataset_len must be positive")
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(dataset_len, generator=generator).tolist()
    if dataset_len == 1 or val_fraction == 0.0:
        return shuffled, []

    val_count = max(1, int(round(dataset_len * val_fraction)))
    val_count = min(val_count, dataset_len - 1)
    return shuffled[val_count:], shuffled[:val_count]


def motion_batch_to_torch(batch: Mapping[str, object], device: torch.device) -> dict[str, Tensor]:
    return {
        "positions": _as_tensor(batch["positions"], device=device, dtype=torch.float32),
        "local_rotations_6d": _as_tensor(
            batch["local_rotations_6d"],
            device=device,
            dtype=torch.float32,
        ),
        "root_translation": _as_tensor(batch["root_translation"], device=device, dtype=torch.float32),
        "parents": _as_tensor(batch["parents"], device=device, dtype=torch.long),
        "time_mask": _as_tensor(batch["time_mask"], device=device, dtype=torch.bool),
        "joint_mask": _as_tensor(batch["joint_mask"], device=device, dtype=torch.bool),
    }


def train_stage1_vae_step(
    model: KinematicVAE,
    motion_batch: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    beta: float,
    velocity_weight: float,
    acceleration_weight: float,
    bone_length_weight: float,
    root_velocity_weight: float,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(dict(motion_batch))
    losses = kinematic_vae_loss(
        output,
        dict(motion_batch),
        beta=beta,
        velocity_weight=velocity_weight,
        acceleration_weight=acceleration_weight,
        bone_length_weight=bone_length_weight,
        root_velocity_weight=root_velocity_weight,
    )
    losses["loss"].backward()
    if grad_clip_norm is not None:
        clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return _float_metrics(losses)


@torch.no_grad()
def evaluate_stage1_vae(
    model: KinematicVAE,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    velocity_weight: float,
    acceleration_weight: float,
    bone_length_weight: float,
    root_velocity_weight: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch in dataloader:
        motion_batch = motion_batch_to_torch(batch, device=device)
        output = model(motion_batch)
        losses = kinematic_vae_loss(
            output,
            motion_batch,
            beta=beta,
            velocity_weight=velocity_weight,
            acceleration_weight=acceleration_weight,
            bone_length_weight=bone_length_weight,
            root_velocity_weight=root_velocity_weight,
        )
        metrics = _float_metrics(losses)
        if not totals:
            totals = {key: 0.0 for key in metrics}
        for key in metrics:
            totals[key] += metrics[key]
        batches += 1
    if batches == 0:
        raise ValueError("evaluation dataloader produced no batches")
    return {key: value / batches for key, value in totals.items()}


def run_stage1_vae_training(config: Stage1VAEConfig) -> Stage1VAEResult:
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_stage1_vae_dataloaders(config)
    model = build_stage1_vae(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    start_step = 0
    best_val_loss = float("inf")
    if config.resume_from is not None:
        start_step, best_val_loss = load_stage1_vae_checkpoint(
            path=config.resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    metrics_path = config.output_dir / "metrics.jsonl"
    last_metrics: dict[str, float] = {}
    train_iter = iter(train_loader)
    step = start_step
    while step < config.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step += 1
        motion_batch = motion_batch_to_torch(batch, device=device)
        train_metrics = train_stage1_vae_step(
            model=model,
            motion_batch=motion_batch,
            optimizer=optimizer,
            beta=config.beta,
            velocity_weight=config.velocity_weight,
            acceleration_weight=config.acceleration_weight,
            bone_length_weight=config.bone_length_weight,
            root_velocity_weight=config.root_velocity_weight,
            grad_clip_norm=config.grad_clip_norm,
        )
        last_metrics = _prefix_metrics("train", train_metrics)

        should_eval = step % config.eval_every == 0 or step == config.max_steps
        if should_eval:
            eval_loader = val_loader if val_loader is not None else train_loader
            eval_metrics = evaluate_stage1_vae(
                model=model,
                dataloader=eval_loader,
                device=device,
                beta=config.beta,
                velocity_weight=config.velocity_weight,
                acceleration_weight=config.acceleration_weight,
                bone_length_weight=config.bone_length_weight,
                root_velocity_weight=config.root_velocity_weight,
            )
            last_metrics.update(_prefix_metrics("val", eval_metrics))
            current_val_loss = eval_metrics["loss"]
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                save_stage1_vae_checkpoint(
                    path=config.output_dir / "vae_best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=config,
                    metrics=last_metrics,
                    best_val_loss=best_val_loss,
                )

        should_log = step % config.log_every == 0 or step == config.max_steps
        if should_log:
            _append_metrics(metrics_path, step=step, metrics=last_metrics)
            print(_format_metrics(step, last_metrics))

        should_save = step % config.save_every == 0 or step == config.max_steps
        if should_save:
            save_stage1_vae_checkpoint(
                path=config.output_dir / "vae_latest.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                config=config,
                metrics=last_metrics,
                best_val_loss=best_val_loss,
            )

    if not (config.output_dir / "vae_best.pt").exists():
        save_stage1_vae_checkpoint(
            path=config.output_dir / "vae_best.pt",
            model=model,
            optimizer=optimizer,
            step=step,
            config=config,
            metrics=last_metrics,
            best_val_loss=best_val_loss,
        )

    return Stage1VAEResult(
        steps=step,
        best_val_loss=best_val_loss,
        last_metrics=last_metrics,
        output_dir=config.output_dir,
    )


def save_stage1_vae_checkpoint(
    path: Path,
    model: KinematicVAE,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Stage1VAEConfig,
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


def load_stage1_vae_checkpoint(
    path: str | Path,
    model: KinematicVAE,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(Path(path), map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    step = int(checkpoint["step"])
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return step, best_val_loss


def parse_args(argv: Optional[Iterable[str]] = None) -> Stage1VAEConfig:
    parser = argparse.ArgumentParser(description="RigFlow4D Stage 1 kinematic VAE training")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/AMASS_RigFlow4D"))
    parser.add_argument("--manifest", type=Path, default=Path("manifest.json"), dest="manifest_path")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/rigflow4d_stage1_tgvae"))
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--velocity-weight", type=float, default=0.1)
    parser.add_argument("--acceleration-weight", type=float, default=0.01)
    parser.add_argument("--bone-length-weight", type=float, default=0.1)
    parser.add_argument("--root-velocity-weight", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--drop-short", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return Stage1VAEConfig(
        data_root=args.data_root,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        beta=args.beta,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        velocity_weight=args.velocity_weight,
        acceleration_weight=args.acceleration_weight,
        bone_length_weight=args.bone_length_weight,
        root_velocity_weight=args.root_velocity_weight,
        val_fraction=args.val_fraction,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        resume_from=args.resume_from,
        drop_short=args.drop_short,
        grad_clip_norm=args.grad_clip_norm,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    config = parse_args(argv)
    result = run_stage1_vae_training(config)
    print(
        f"finished stage1 steps={result.steps} "
        f"best_val_loss={result.best_val_loss:.6f} "
        f"output_dir={result.output_dir}"
    )
    return 0


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _as_tensor(value: object, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _float_metrics(metrics: Mapping[str, Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def _prefix_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def _append_metrics(path: Path, step: int, metrics: Mapping[str, float]) -> None:
    payload = {"step": step, **metrics}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _format_metrics(step: int, metrics: Mapping[str, float]) -> str:
    fields = " ".join(f"{key}={value:.6f}" for key, value in sorted(metrics.items()))
    return f"step={step} {fields}"


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
