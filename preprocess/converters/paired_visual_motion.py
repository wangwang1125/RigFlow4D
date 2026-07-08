from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping
import warnings

from preprocess.visual.token_cache import (
    DeterministicVisualBackbone,
    HuggingFaceVisualBackbone,
    VisualBackbone,
    VisualTokenCacheConfig,
    inject_visual_cache_into_normalized_npz,
    write_visual_token_cache_from_source,
)


def convert_paired_visual_motion_manifest(
    input_manifest_path: str | Path,
    output_root: str | Path,
    backbone: VisualBackbone,
    frames_key: str = "frames",
    skip_invalid: bool = False,
) -> Path:
    input_manifest = Path(input_manifest_path)
    output_root = Path(output_root)
    input_base = input_manifest.parent
    output_root.mkdir(parents=True, exist_ok=True)
    prepared = _load_prepared_manifest(input_manifest)

    normalized_records: list[dict[str, str]] = []
    for index, record in enumerate(prepared["samples"]):
        try:
            normalized_records.append(
                _convert_record(
                    record=record,
                    index=index,
                    input_base=input_base,
                    output_root=output_root,
                    backbone=backbone,
                    frames_key=frames_key,
                )
            )
        except Exception as exc:
            if not skip_invalid:
                raise
            warnings.warn(
                f"Skipping invalid visual-motion pair {index}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps({"samples": normalized_records}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert paired frame/motion records into RigFlow4D visual-motion samples"
    )
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--frames-key", default="frames")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--hf-model-id")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone-name", default="dinov3_dummy")
    parser.add_argument("--feature-dim", type=int, default=6)
    parser.add_argument("--patch-grid", nargs=2, type=int, default=(2, 2))
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.patch_grid = tuple(args.patch_grid)
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    backbone = _build_backbone(args)
    manifest_path = convert_paired_visual_motion_manifest(
        input_manifest_path=args.input_manifest,
        output_root=args.output_root,
        backbone=backbone,
        frames_key=args.frames_key,
        skip_invalid=args.skip_invalid,
    )
    print(manifest_path)
    return 0


def _convert_record(
    record: Mapping[str, object],
    index: int,
    input_base: Path,
    output_root: Path,
    backbone: VisualBackbone,
    frames_key: str,
) -> dict[str, str]:
    sample_id = str(record.get("sample_id") or f"sample_{index:06d}")
    frames_path = _resolve_required_path(record, "frames", input_base)
    motion_path = _resolve_required_path(record, "motion", input_base)
    out_rel = Path(str(record.get("out") or f"samples/{sample_id}.npz"))
    if out_rel.is_absolute():
        raise ValueError("'out' must be relative to output_root")
    out_path = output_root / out_rel
    cache_path = output_root / "_visual_cache" / out_rel.with_suffix(".visual_cache.npz")

    write_visual_token_cache_from_source(
        frames_source=frames_path,
        backbone=backbone,
        out_path=cache_path,
        frames_key=frames_key,
    )
    inject_visual_cache_into_normalized_npz(
        normalized_npz_path=motion_path,
        visual_cache_path=cache_path,
        out_path=out_path,
    )
    return {
        "sample_id": sample_id,
        "path": out_rel.as_posix(),
        "source_frames": str(frames_path),
        "source_motion": str(motion_path),
    }


def _load_prepared_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("paired visual-motion manifest must contain a 'samples' list")
    for record in samples:
        if not isinstance(record, dict):
            raise ValueError("each paired manifest sample must be an object")
        if "frames" not in record or "motion" not in record:
            raise ValueError("each paired manifest sample must contain 'frames' and 'motion'")
    return {"samples": samples}


def _resolve_required_path(record: Mapping[str, object], key: str, base: Path) -> Path:
    value = record[key]
    path = Path(str(value))
    if not path.is_absolute():
        path = base / path
    return path


def _build_backbone(args: argparse.Namespace) -> VisualBackbone:
    config = VisualTokenCacheConfig(
        backbone_name=args.hf_model_id or args.backbone_name,
        feature_dim=args.feature_dim,
        patch_grid=args.patch_grid,
    )
    if args.hf_model_id:
        return HuggingFaceVisualBackbone(
            model_id=args.hf_model_id,
            config=config,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            device=args.device,
        )
    return DeterministicVisualBackbone(config=config)


if __name__ == "__main__":
    raise SystemExit(main())
