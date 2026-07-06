# RigFlow4D Latent Refiner Smoke Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first smoke training entry for `RigFlowLatentRefiner`, proving a normalized `.npz` dataset can be windowed, batched, and optimized for a few conditional flow steps.

**Architecture:** Build a small train module that creates `NormalizedNpzAdapter`, wraps it with `MotionWindowDataset`, converts collated numpy batches to torch tensors, constructs `RigFlowLatentRefiner`, and runs a bounded number of optimizer steps. Keep the CLI thin around tested Python functions.

**Tech Stack:** Python 3, PyTorch, pytest.

---

## File Structure

- Create `RigFlow4D/train/rigflow4d_latent_refiner.py`: config/result dataclasses, dataloader builder, model builder, batch conversion, single-step training, smoke-run function, and CLI.
- Create `RigFlow4D/tests/test_latent_refiner_training.py`: tests for dataloader construction, tensor conversion, finite smoke losses, parameter update, and CLI argument parsing.
- Modify `RigFlow4D/RUN.md`: document the smoke training command.

---

### Task 1: Smoke Training Entry

**Files:**
- Test: `RigFlow4D/tests/test_latent_refiner_training.py`
- Create: `RigFlow4D/train/rigflow4d_latent_refiner.py`
- Modify: `RigFlow4D/RUN.md`

- [x] **Step 1: Write failing tests**

Tests should assert:

- `build_motion_dataloader(config)` creates a non-empty dataloader from a temporary normalized `.npz` manifest.
- `motion_batch_to_torch(...)` converts numpy arrays to torch tensors with float and bool dtypes.
- `train_latent_refiner_step(...)` returns a finite scalar loss and updates flow parameters.
- `run_latent_refiner_smoke_training(config)` returns `max_steps` finite losses.
- `parse_args(...)` maps CLI arguments into config fields.

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_latent_refiner_training.py -v`

Expected: FAIL with `ModuleNotFoundError` for `train.rigflow4d_latent_refiner`.

- [x] **Step 3: Implement minimal training module**

The script should support motion-only smoke training:

```text
NormalizedNpzAdapter
  -> MotionWindowDataset
  -> collate_motion_windows
  -> torch motion_batch
  -> RigFlowLatentRefiner
  -> optimizer step
```

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_schema.py tests/test_kinematic_vae.py tests/test_latent_flow.py tests/test_condition_encoder.py tests/test_latent_refiner.py tests/test_latent_refiner_training.py -v`

Expected: all tests pass.

---

## Self-Review

- Spec coverage: This plan covers a smoke training path only. It does not implement distributed training, checkpointing, validation metrics, DINOv3 extraction, or experiment config files.
- Placeholder scan: No placeholders remain.
- Type consistency: Uses `LatentRefinerSmokeConfig`, `LatentRefinerSmokeResult`, `motion_batch`, and `loss_history` consistently.
