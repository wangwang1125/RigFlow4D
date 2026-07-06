import json

import numpy as np
import pytest

from data.adapters.normalized_npz import NormalizedNpzAdapter
from preprocess.visual.token_cache import (
    DeterministicVisualBackbone,
    HuggingFaceVisualBackbone,
    VisualTokenCacheConfig,
    inject_visual_cache_into_normalized_npz,
    parse_args,
    validate_visual_tokens,
    write_visual_token_cache,
)


def write_normalized_npz(path, frames=3, joints=4):
    np.savez(
        path,
        dataset_name=np.array("unit"),
        input_type=np.array("video"),
        source_label_type=np.array("motion_only"),
        camera_mode=np.array("unknown"),
        parents=np.array([-1, 0, 1, 2], dtype=np.int64)[:joints],
        rest_offsets=np.zeros((joints, 3), dtype=np.float32),
        joint_names=np.array([f"joint_{i}" for i in range(joints)]),
        chain_ids=np.arange(joints, dtype=np.int64),
        chain_coordinates=np.linspace(0.0, 1.0, joints, dtype=np.float32),
        positions=np.zeros((frames, joints, 3), dtype=np.float32),
        local_rotations_6d=np.zeros((frames, joints, 6), dtype=np.float32),
        root_translation=np.zeros((frames, 3), dtype=np.float32),
    )


def write_manifest(path, sample_path):
    path.write_text(
        json.dumps({"samples": [{"sample_id": sample_path.stem, "path": sample_path.name}]}),
        encoding="utf-8",
    )


def test_deterministic_backbone_encodes_dense_tokens():
    frames = np.arange(2 * 3 * 4 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 4, 3)
    backbone = DeterministicVisualBackbone(
        config=VisualTokenCacheConfig(
            backbone_name="dinov3_dummy",
            feature_dim=5,
            patch_grid=(2, 2),
        )
    )

    tokens = backbone.encode_frames(frames)

    assert tokens.shape == (2, 3, 4, 5)
    assert np.isfinite(tokens).all()


def test_huggingface_backbone_is_lazy_and_records_runtime_download_options(tmp_path):
    backbone = HuggingFaceVisualBackbone(
        model_id="example/dinov3-test",
        config=VisualTokenCacheConfig(
            backbone_name="dinov3_hf",
            feature_dim=8,
            patch_grid=(2, 2),
        ),
        cache_dir=tmp_path / "hf_cache",
        local_files_only=True,
    )

    assert backbone.model_id == "example/dinov3-test"
    assert backbone.cache_dir == tmp_path / "hf_cache"
    assert backbone.local_files_only is True
    assert backbone.is_loaded is False


def test_write_visual_token_cache_writes_expected_keys(tmp_path):
    frames = np.ones((1, 3, 4, 4, 3), dtype=np.float32)
    out_path = tmp_path / "visual_cache.npz"
    backbone = DeterministicVisualBackbone(
        config=VisualTokenCacheConfig(
            backbone_name="dinov3_dummy",
            feature_dim=6,
            patch_grid=(2, 2),
            has_cls=False,
            num_registers=4,
        )
    )

    write_visual_token_cache(frames=frames, backbone=backbone, out_path=out_path)

    with np.load(out_path, allow_pickle=False) as cache:
        assert cache["visual_tokens"].shape == (1, 3, 4, 6)
        assert cache["visual_backbone_name"].item() == "dinov3_dummy"
        assert cache["visual_feature_dim"].item() == 6
        np.testing.assert_array_equal(cache["visual_patch_grid"], np.array([2, 2]))
        assert cache["visual_has_cls"].item() == 0
        assert cache["visual_num_registers"].item() == 4


def test_validate_visual_tokens_rejects_wrong_shape():
    config = VisualTokenCacheConfig(
        backbone_name="dinov3_dummy",
        feature_dim=6,
        patch_grid=(2, 2),
    )

    with pytest.raises(ValueError, match="visual_tokens"):
        validate_visual_tokens(np.zeros((3, 4, 6), dtype=np.float32), config)

    with pytest.raises(ValueError, match="feature_dim"):
        validate_visual_tokens(np.zeros((1, 3, 4, 5), dtype=np.float32), config)


def test_inject_visual_cache_into_normalized_npz_loads_with_adapter(tmp_path):
    sample_path = tmp_path / "sample.npz"
    cache_path = tmp_path / "visual_cache.npz"
    out_path = tmp_path / "sample_with_visual.npz"
    manifest_path = tmp_path / "manifest.json"
    write_normalized_npz(sample_path, frames=3)
    write_manifest(manifest_path, out_path)
    frames = np.ones((2, 3, 4, 4, 3), dtype=np.float32)
    backbone = DeterministicVisualBackbone(
        config=VisualTokenCacheConfig(
            backbone_name="dinov3_dummy",
            feature_dim=6,
            patch_grid=(2, 2),
        )
    )
    write_visual_token_cache(frames=frames, backbone=backbone, out_path=cache_path)

    inject_visual_cache_into_normalized_npz(
        normalized_npz_path=sample_path,
        visual_cache_path=cache_path,
        out_path=out_path,
    )
    adapter = NormalizedNpzAdapter(root=tmp_path, manifest_path=manifest_path)
    sample = adapter[0]

    assert sample.visual.backbone_name == "dinov3_dummy"
    assert sample.visual.tokens.shape == (2, 3, 4, 6)
    assert sample.visual.patch_grid == (2, 2)


def test_inject_visual_cache_rejects_frame_mismatch(tmp_path):
    sample_path = tmp_path / "sample.npz"
    cache_path = tmp_path / "visual_cache.npz"
    write_normalized_npz(sample_path, frames=3)
    backbone = DeterministicVisualBackbone(
        config=VisualTokenCacheConfig(
            backbone_name="dinov3_dummy",
            feature_dim=6,
            patch_grid=(2, 2),
        )
    )
    write_visual_token_cache(
        frames=np.ones((1, 2, 4, 4, 3), dtype=np.float32),
        backbone=backbone,
        out_path=cache_path,
    )

    with pytest.raises(ValueError, match="frame"):
        inject_visual_cache_into_normalized_npz(
            normalized_npz_path=sample_path,
            visual_cache_path=cache_path,
            out_path=tmp_path / "bad.npz",
        )


def test_parse_args_builds_visual_cache_config(tmp_path):
    args = parse_args(
        [
            "inject",
            "--normalized-npz",
            str(tmp_path / "sample.npz"),
            "--visual-cache",
            str(tmp_path / "cache.npz"),
            "--out",
            str(tmp_path / "out.npz"),
        ]
    )

    assert args.command == "inject"
    assert args.normalized_npz == tmp_path / "sample.npz"
    assert args.visual_cache == tmp_path / "cache.npz"
    assert args.out == tmp_path / "out.npz"


def test_parse_args_builds_huggingface_write_command(tmp_path):
    args = parse_args(
        [
            "write-hf",
            "--frames-npz",
            str(tmp_path / "frames.npz"),
            "--frames-key",
            "frames",
            "--out",
            str(tmp_path / "cache.npz"),
            "--hf-model-id",
            "example/dinov3-test",
            "--feature-dim",
            "8",
            "--patch-grid",
            "2",
            "2",
            "--cache-dir",
            str(tmp_path / "hf_cache"),
            "--local-files-only",
        ]
    )

    assert args.command == "write-hf"
    assert args.frames_npz == tmp_path / "frames.npz"
    assert args.frames_key == "frames"
    assert args.out == tmp_path / "cache.npz"
    assert args.hf_model_id == "example/dinov3-test"
    assert args.feature_dim == 8
    assert args.patch_grid == (2, 2)
    assert args.cache_dir == tmp_path / "hf_cache"
    assert args.local_files_only is True
