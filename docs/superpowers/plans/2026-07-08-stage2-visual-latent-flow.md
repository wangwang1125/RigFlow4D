# Stage 2 Visual Latent Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first visual-conditioned Stage 2 training path from frame inputs and DINOv3-style token caches to the existing latent flow trainer.

**Architecture:** Add a small visual frame reader, extend the existing token cache CLI so it can consume real frame sources, add a paired visual-motion converter for BEDLAM-style prepared manifests, and make Stage 2 infer visual feature dimensions from normalized samples. Keep tests deterministic and keep real DINOv3/HuggingFace loading lazy.

**Tech Stack:** Python 3, NumPy, Pillow for image loading, PyTorch for the existing trainer, pytest.

---

## File Structure

- Create `preprocess/visual/frame_io.py`: read `.npz` frame tensors and image-frame directories into `[V,T,H,W,C]` arrays.
- Modify `preprocess/visual/__init__.py`: export frame reader APIs.
- Modify `preprocess/visual/token_cache.py`: allow `write-hf` to read either `.npz` frames or image-frame directories.
- Create `preprocess/converters/paired_visual_motion.py`: pair prepared frame sources with normalized or raw motion `.npz` samples and inject visual caches.
- Modify `train/rigflow4d_latent_refiner.py`: infer `visual_dim` from the first sample when visual tokens exist.
- Add `tests/test_visual_frame_io.py`: unit tests for frame source loading.
- Add `tests/test_paired_visual_motion_converter.py`: converter tests using deterministic frame and motion fixtures.
- Modify `tests/test_visual_token_cache.py`: CLI tests for frame source arguments.
- Modify `tests/test_latent_refiner_training.py`: visual dimension inference test.

---

### Task 1: Visual Frame Reader

**Files:**
- Create: `preprocess/visual/frame_io.py`
- Modify: `preprocess/visual/__init__.py`
- Test: `tests/test_visual_frame_io.py`

- [x] **Step 1: Write failing tests**

Tests should cover:

- `.npz` source with `frames [V,T,H,W,C]`.
- single-view image directory sorted by filename.
- multiview image directory with view subdirectories sorted by name.
- mismatched multiview frame counts raise `ValueError`.

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visual_frame_io.py -v`

Expected: FAIL because `preprocess.visual.frame_io` does not exist.

- [x] **Step 3: Implement frame reader**

Implement:

```python
def load_frame_source(path: str | Path, frames_key: str = "frames") -> np.ndarray:
    ...
```

Return `uint8` frames shaped `[V,T,H,W,C]`.

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_visual_frame_io.py -v`

Expected: PASS.

---

### Task 2: Token Cache CLI Reads Frame Sources

**Files:**
- Modify: `preprocess/visual/token_cache.py`
- Test: `tests/test_visual_token_cache.py`

- [x] **Step 1: Write failing tests**

Add tests showing `parse_args(["write-hf", "--frames", ...])` accepts a generic
frame source and that the deterministic writer can load frames through the new
reader without requiring HuggingFace imports.

- [x] **Step 2: Run focused test**

Run: `python -m pytest tests/test_visual_token_cache.py::test_parse_args_builds_hf_writer_with_frame_source -v`

Expected: FAIL because `--frames` is not supported.

- [x] **Step 3: Implement minimal CLI change**

Keep `--frames-npz` compatibility, add `--frames`, and route both through
`load_frame_source`.

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_visual_token_cache.py tests/test_visual_frame_io.py -v`

Expected: PASS.

---

### Task 3: Paired Visual-Motion Converter

**Files:**
- Create: `preprocess/converters/paired_visual_motion.py`
- Test: `tests/test_paired_visual_motion_converter.py`

- [x] **Step 1: Write failing tests**

Test a prepared manifest:

```json
{
  "samples": [
    {
      "frames": "raw/seq_001/frames",
      "motion": "raw/seq_001/motion.npz",
      "out": "samples/seq_001.npz"
    }
  ]
}
```

The converter should load a normalized motion `.npz`, write a visual cache using
the deterministic test backbone, inject the visual fields, and emit a normalized
manifest loadable by `NormalizedNpzAdapter`.

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paired_visual_motion_converter.py -v`

Expected: FAIL because the converter module does not exist.

- [x] **Step 3: Implement converter**

Implement a Python API and CLI:

```python
convert_paired_visual_motion_manifest(
    input_manifest_path,
    output_root,
    backbone,
    frames_key="frames",
    skip_invalid=False,
)
```

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_paired_visual_motion_converter.py tests/test_schema.py -v`

Expected: PASS.

---

### Task 4: Stage 2 Visual Dimension Inference

**Files:**
- Modify: `train/rigflow4d_latent_refiner.py`
- Test: `tests/test_latent_refiner_training.py`

- [x] **Step 1: Write failing test**

Add a test where the config uses `visual_dim=None` and the first dataset sample
has `visual_tokens[..., D]`; building the latent refiner should use `D`.

- [x] **Step 2: Run focused test**

Run: `python -m pytest tests/test_latent_refiner_training.py::test_stage2_config_infers_visual_dim_from_dataset -v`

Expected: FAIL because `visual_dim` currently must be positive.

- [x] **Step 3: Implement inference**

Allow `visual_dim: int | None = None`. When building dataloaders or the model,
inspect the first adapter sample and set the resolved dimension. Preserve
motion-only behavior by falling back to `1`.

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_latent_refiner_training.py tests/test_condition_encoder.py -v`

Expected: PASS.

---

### Task 5: Integrated Verification And Commit

**Files:**
- All files touched above.

- [x] **Step 1: Run focused Stage 2.1 tests**

Run:

```bash
python -m pytest \
  tests/test_visual_frame_io.py \
  tests/test_visual_token_cache.py \
  tests/test_paired_visual_motion_converter.py \
  tests/test_latent_refiner_training.py \
  tests/test_schema.py \
  tests/test_condition_encoder.py \
  -v
```

Expected: PASS.

- [x] **Step 2: Review git diff**

Run: `git diff --stat` and `git diff --check`.

Expected: no whitespace errors and no unrelated `RUN.md` staging.

- [x] **Step 3: Commit**

Commit only implementation files and tests:

```bash
git add preprocess/visual/frame_io.py preprocess/visual/__init__.py preprocess/visual/token_cache.py preprocess/converters/paired_visual_motion.py tests/test_visual_frame_io.py tests/test_visual_token_cache.py tests/test_paired_visual_motion_converter.py tests/test_latent_refiner_training.py train/rigflow4d_latent_refiner.py docs/superpowers/plans/2026-07-08-stage2-visual-latent-flow.md
git commit -m "feat: add visual-conditioned stage2 data path"
```

Expected: commit succeeds.
