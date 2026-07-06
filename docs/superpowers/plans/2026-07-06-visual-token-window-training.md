# Visual Token Window Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry DINOv3-style cached visual tokens from normalized `.npz` samples through motion windows, collated batches, torch conversion, and latent-refiner smoke training.

**Architecture:** Extend `MotionWindow` with optional `visual_tokens [V, W, P, D]` and `visual_mask [V, W, P]`. `MotionWindowDataset` slices the visual cache with the same time window as motion data, `collate_motion_windows` pads optional visual data, and `train_latent_refiner_step` passes visual tensors into `RigFlowLatentRefiner`.

**Tech Stack:** Python 3, NumPy, PyTorch, pytest.

---

## File Structure

- Modify `RigFlow4D/data/window_dataset.py`: add optional visual window fields and collation.
- Modify `RigFlow4D/train/rigflow4d_latent_refiner.py`: convert visual tokens/masks to torch and pass them into the refiner.
- Modify `RigFlow4D/tests/test_schema.py`: add window/collate tests for visual tokens.
- Modify `RigFlow4D/tests/test_latent_refiner_training.py`: add smoke training test with visual tokens.
- Modify `RigFlow4D/RUN.md`: document visual-token-aware smoke training.

---

### Task 1: Visual Token Windowing

**Files:**
- Test: `RigFlow4D/tests/test_schema.py`
- Modify: `RigFlow4D/data/window_dataset.py`

- [x] **Step 1: Write failing visual window tests**

Tests should assert:

- `MotionWindowDataset` slices `sample.visual.tokens [V,T,P,D]` to `[V,W,P,D]`.
- padded visual frames beyond sequence length have `visual_mask=False`.
- `collate_motion_windows` returns `visual_tokens [B,V,W,P,D]` and `visual_mask [B,V,W,P]` when any item has visual tokens.

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schema.py::test_motion_window_dataset_slices_visual_tokens -v`

Expected: FAIL because `MotionWindow` has no `visual_tokens` field.

- [x] **Step 3: Implement visual windowing and collation**

Keep visual fields optional so motion-only samples remain unchanged.

---

### Task 2: Visual-Aware Smoke Training

**Files:**
- Test: `RigFlow4D/tests/test_latent_refiner_training.py`
- Modify: `RigFlow4D/train/rigflow4d_latent_refiner.py`

- [x] **Step 1: Write failing smoke training test**

Test should assert:

- normalized `.npz` with `visual_tokens` appears in collated batch.
- `motion_batch_to_torch` converts `visual_tokens` and `visual_mask`.
- `train_latent_refiner_step` accepts visual fields and returns finite loss.

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_latent_refiner_training.py::test_train_latent_refiner_step_uses_visual_tokens -v`

Expected: FAIL because visual tensors are not converted/passed.

- [x] **Step 3: Implement visual-aware training step**

Pass `visual_tokens` and `visual_mask` from torch batch into `RigFlowLatentRefiner`.

---

## Self-Review

- Spec coverage: This plan carries cached visual tokens through the existing training smoke path. It does not implement DINOv3 extraction or online image loading.
- Placeholder scan: No placeholders remain.
- Type consistency: Uses `visual_tokens` and `visual_mask` consistently across dataset, collate, torch conversion, and training step.
