from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, Tuple

import numpy as np

from .frame_io import load_frame_source


@dataclass(frozen=True)
class VisualTokenCacheConfig:
    backbone_name: str
    feature_dim: int
    patch_grid: Tuple[int, int]
    has_cls: bool = True
    num_registers: int = 0

    def __post_init__(self) -> None:
        if not self.backbone_name:
            raise ValueError("backbone_name must be non-empty")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if len(self.patch_grid) != 2 or any(int(x) <= 0 for x in self.patch_grid):
            raise ValueError("patch_grid must contain two positive integers")
        if self.num_registers < 0:
            raise ValueError("num_registers must be non-negative")
        object.__setattr__(self, "patch_grid", tuple(int(x) for x in self.patch_grid))


class VisualBackbone(Protocol):
    config: VisualTokenCacheConfig

    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        ...


class DeterministicVisualBackbone:
    def __init__(self, config: VisualTokenCacheConfig) -> None:
        self.config = config

    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        frames = _validate_frames(frames)
        views, frames_count = frames.shape[:2]
        grid_h, grid_w = self.config.patch_grid
        patches = grid_h * grid_w
        pooled = _patch_pool(frames, self.config.patch_grid)
        flat = pooled.reshape(views, frames_count, patches, -1).astype(np.float32)
        base = flat.mean(axis=-1, keepdims=True)
        scales = np.linspace(1.0, 2.0, self.config.feature_dim, dtype=np.float32)
        tokens = base * scales.reshape(1, 1, 1, self.config.feature_dim)
        return tokens.astype(np.float32)


class HuggingFaceVisualBackbone:
    def __init__(
        self,
        model_id: str,
        config: VisualTokenCacheConfig,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        device: str = "cpu",
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be non-empty")
        self.model_id = model_id
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.local_files_only = local_files_only
        self.device = device
        self._processor = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        frames = _validate_frames(frames)
        self._lazy_load()
        try:
            import torch
        except ImportError as exc:
            raise ImportError("HuggingFaceVisualBackbone requires torch at runtime") from exc

        views, frames_count = frames.shape[:2]
        flat_frames = frames.reshape(views * frames_count, *frames.shape[2:])
        inputs = self._processor(
            images=[frame.astype(np.uint8) for frame in flat_frames],
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise ValueError("HuggingFace model output does not expose last_hidden_state")
        tokens = hidden.detach().cpu().numpy().astype(np.float32)
        tokens = _fit_token_count(tokens, expected_tokens=self.config.patch_grid[0] * self.config.patch_grid[1])
        if tokens.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"HuggingFace output feature_dim={tokens.shape[-1]} does not match "
                f"configured feature_dim={self.config.feature_dim}"
            )
        return tokens.reshape(views, frames_count, tokens.shape[1], tokens.shape[2])

    def _lazy_load(self) -> None:
        if self.is_loaded:
            return
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceVisualBackbone requires transformers. Install transformers to use "
                "runtime HuggingFace weight download."
            ) from exc
        kwargs = {
            "cache_dir": str(self.cache_dir) if self.cache_dir is not None else None,
            "local_files_only": self.local_files_only,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        self._processor = AutoImageProcessor.from_pretrained(self.model_id, **kwargs)
        self._model = AutoModel.from_pretrained(self.model_id, **kwargs).to(self.device)
        self._model.eval()


def write_visual_token_cache(
    frames: np.ndarray,
    backbone: VisualBackbone,
    out_path: str | Path,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokens = backbone.encode_frames(frames)
    validate_visual_tokens(tokens, backbone.config)
    np.savez(
        out,
        visual_tokens=tokens.astype(np.float32),
        visual_backbone_name=np.array(backbone.config.backbone_name),
        visual_feature_dim=np.array(backbone.config.feature_dim, dtype=np.int64),
        visual_patch_grid=np.asarray(backbone.config.patch_grid, dtype=np.int64),
        visual_has_cls=np.array(int(backbone.config.has_cls), dtype=np.int64),
        visual_num_registers=np.array(backbone.config.num_registers, dtype=np.int64),
    )
    return out


def write_visual_token_cache_from_source(
    frames_source: str | Path,
    backbone: VisualBackbone,
    out_path: str | Path,
    frames_key: str = "frames",
) -> Path:
    frames = load_frame_source(frames_source, frames_key=frames_key)
    return write_visual_token_cache(frames=frames, backbone=backbone, out_path=out_path)


def inject_visual_cache_into_normalized_npz(
    normalized_npz_path: str | Path,
    visual_cache_path: str | Path,
    out_path: str | Path,
) -> Path:
    normalized_path = Path(normalized_npz_path)
    cache_path = Path(visual_cache_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with np.load(normalized_path, allow_pickle=False) as src:
        payload = {key: np.asarray(src[key]) for key in src.files}
        expected_frames = int(np.asarray(src["positions"]).shape[0])
    with np.load(cache_path, allow_pickle=False) as cache:
        tokens = np.asarray(cache["visual_tokens"], dtype=np.float32)
        if tokens.shape[1] != expected_frames:
            raise ValueError(
                f"visual cache frame count {tokens.shape[1]} must match normalized sample frame "
                f"count {expected_frames}"
            )
        cache_config = VisualTokenCacheConfig(
            backbone_name=str(_scalar(cache["visual_backbone_name"])),
            feature_dim=int(_scalar(cache["visual_feature_dim"])),
            patch_grid=tuple(int(x) for x in np.asarray(cache["visual_patch_grid"]).tolist()),
            has_cls=bool(_scalar(cache["visual_has_cls"])) if "visual_has_cls" in cache else True,
            num_registers=int(_scalar(cache["visual_num_registers"])) if "visual_num_registers" in cache else 0,
        )
        validate_visual_tokens(tokens, cache_config)
        payload.update(
            {
                "visual_tokens": tokens,
                "visual_backbone_name": np.asarray(cache["visual_backbone_name"]),
                "visual_feature_dim": np.asarray(cache["visual_feature_dim"]),
                "visual_patch_grid": np.asarray(cache["visual_patch_grid"]),
                "visual_has_cls": np.asarray(cache["visual_has_cls"]) if "visual_has_cls" in cache else np.array(1),
                "visual_num_registers": np.asarray(cache["visual_num_registers"])
                if "visual_num_registers" in cache
                else np.array(0),
            }
        )
    np.savez(out, **payload)
    return out


def validate_visual_tokens(tokens: np.ndarray, config: VisualTokenCacheConfig) -> None:
    if not isinstance(tokens, np.ndarray):
        raise TypeError("visual_tokens must be a numpy.ndarray")
    if tokens.ndim != 4:
        raise ValueError(f"visual_tokens must have shape [V, T, P, D], got {tokens.shape}")
    if tokens.shape[-1] != config.feature_dim:
        raise ValueError(
            f"visual_tokens feature_dim must match config feature_dim={config.feature_dim}, "
            f"got {tokens.shape}"
        )
    expected_patches = config.patch_grid[0] * config.patch_grid[1]
    if tokens.shape[2] != expected_patches:
        raise ValueError(
            f"visual_tokens patch count must match patch_grid product={expected_patches}, "
            f"got {tokens.shape}"
        )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RigFlow4D visual token cache tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inject_parser = subparsers.add_parser("inject", help="inject a visual cache into normalized npz")
    inject_parser.add_argument("--normalized-npz", required=True, type=Path)
    inject_parser.add_argument("--visual-cache", required=True, type=Path)
    inject_parser.add_argument("--out", required=True, type=Path)

    hf_parser = subparsers.add_parser("write-hf", help="write visual cache with a HuggingFace backbone")
    frames_group = hf_parser.add_mutually_exclusive_group(required=True)
    frames_group.add_argument("--frames", type=Path)
    frames_group.add_argument("--frames-npz", type=Path)
    hf_parser.add_argument("--frames-key", default="frames")
    hf_parser.add_argument("--out", required=True, type=Path)
    hf_parser.add_argument("--hf-model-id", required=True)
    hf_parser.add_argument("--feature-dim", required=True, type=int)
    hf_parser.add_argument("--patch-grid", required=True, nargs=2, type=int)
    hf_parser.add_argument("--cache-dir", type=Path)
    hf_parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "patch_grid", None) is not None:
        args.patch_grid = tuple(args.patch_grid)
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "inject":
        out = inject_visual_cache_into_normalized_npz(
            normalized_npz_path=args.normalized_npz,
            visual_cache_path=args.visual_cache,
            out_path=args.out,
        )
        print(out)
        return 0
    if args.command == "write-hf":
        frames_source = args.frames if args.frames is not None else args.frames_npz
        backbone = HuggingFaceVisualBackbone(
            model_id=args.hf_model_id,
            config=VisualTokenCacheConfig(
                backbone_name=args.hf_model_id,
                feature_dim=args.feature_dim,
                patch_grid=args.patch_grid,
            ),
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
        out = write_visual_token_cache_from_source(
            frames_source=frames_source,
            backbone=backbone,
            out_path=args.out,
            frames_key=args.frames_key,
        )
        print(out)
        return 0
    raise ValueError(f"unknown command: {args.command}")


def _validate_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 5:
        raise ValueError(f"frames must have shape [V, T, H, W, C], got {frames.shape}")
    if frames.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"frames channel dimension must be 1, 3, or 4, got {frames.shape}")
    return frames


def _patch_pool(frames: np.ndarray, patch_grid: Tuple[int, int]) -> np.ndarray:
    views, frames_count, height, width, channels = frames.shape
    grid_h, grid_w = patch_grid
    if height % grid_h != 0 or width % grid_w != 0:
        raise ValueError(
            f"frame height/width {(height, width)} must be divisible by patch_grid {patch_grid}"
        )
    patch_h = height // grid_h
    patch_w = width // grid_w
    patches = frames.reshape(views, frames_count, grid_h, patch_h, grid_w, patch_w, channels)
    return patches.mean(axis=(3, 5))


def _fit_token_count(tokens: np.ndarray, expected_tokens: int) -> np.ndarray:
    if tokens.shape[1] == expected_tokens:
        return tokens
    if tokens.shape[1] > expected_tokens:
        return tokens[:, -expected_tokens:]
    raise ValueError(
        f"HuggingFace output has {tokens.shape[1]} tokens, fewer than expected {expected_tokens}"
    )


def _scalar(value: np.ndarray) -> object:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
