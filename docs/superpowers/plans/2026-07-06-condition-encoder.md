# RigFlow4D Condition Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact condition encoder that turns DINOv3-style visual tokens, optional camera hints, target rig features, and pose seed features into the dense vector consumed by latent flow matching.

**Architecture:** Pool each modality with masks, project each pooled summary to a shared hidden dimension, concatenate modality summaries, and fuse them into `condition [B, Dcond]`. Missing modalities use learned missing tokens so image/video, calibrated/uncalibrated, and motion-only stages share the same interface.

**Tech Stack:** Python 3, PyTorch, pytest.

---

## File Structure

- Create `RigFlow4D/models/rigflow4d/condition_encoder.py`: condition input/output dataclass, masked pooling helper, and `RigFlowConditionEncoder`.
- Modify `RigFlow4D/models/rigflow4d/__init__.py`: export condition encoder APIs.
- Create `RigFlow4D/tests/test_condition_encoder.py`: tests for shape, optional camera, mask behavior, export, and latent flow integration.
- Modify `RigFlow4D/RUN.md`: document the condition encoder contract.

---

### Task 1: Condition Encoder

**Files:**
- Test: `RigFlow4D/tests/test_condition_encoder.py`
- Create: `RigFlow4D/models/rigflow4d/condition_encoder.py`
- Modify: `RigFlow4D/models/rigflow4d/__init__.py`
- Modify: `RigFlow4D/RUN.md`

- [x] **Step 1: Write failing tests**

Tests should assert:

- `masked_mean` respects binary masks.
- `RigFlowConditionEncoder.forward(...)` returns `condition [B, condition_dim]`.
- `camera_features=None` is accepted and still returns a valid condition vector.
- wrong visual feature dimension raises `ValueError`.
- exported `RigFlowConditionEncoder` can feed `LatentFlowMatcher`.

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_condition_encoder.py -v`

Expected: FAIL with `ModuleNotFoundError` for `models.rigflow4d.condition_encoder`.

- [x] **Step 3: Implement minimal condition encoder**

The encoder should pool:

```text
visual_tokens   [B, V, T, P, Dv] -> [B, H]
camera_features [B, V, Dc] or [B, V, T, Dc] -> [B, H], optional
rig_features    [B, J, Dr] -> [B, H], optional
pose_seed       [B, T, J, Dp] -> [B, H], optional
```

Then fuse:

```text
concat([visual, camera, rig, pose]) -> condition [B, Dcond]
```

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_schema.py tests/test_kinematic_vae.py tests/test_latent_flow.py tests/test_condition_encoder.py -v`

Expected: all tests pass.

---

## Self-Review

- Spec coverage: This plan covers the condition vector contract only. It does not download DINOv3, implement visual preprocessing, or attach the encoder to a full training loop.
- Placeholder scan: No placeholders remain.
- Type consistency: Uses `visual_tokens`, `camera_features`, `rig_features`, `pose_seed`, and `condition` consistently.
