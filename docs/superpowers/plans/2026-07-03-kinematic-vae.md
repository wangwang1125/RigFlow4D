# RigFlow4D Kinematic VAE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first trainable kinematic latent model that maps motion windows into `z_kin` and reconstructs rig-native positions and local 6D rotations.

**Architecture:** Start with a compact mask-aware MLP VAE. It pools valid time/joint features into a latent code and decodes fixed-shape reconstructions; later we can replace internals with temporal transformers while preserving the public API.

**Tech Stack:** Python 3, PyTorch, pytest.

---

## File Structure

- Create `RigFlow4D/models/__init__.py`: package marker.
- Create `RigFlow4D/models/rigflow4d/__init__.py`: exports.
- Create `RigFlow4D/models/rigflow4d/kinematic_vae.py`: `KinematicVAE`, output dataclass, and `kinematic_vae_loss`.
- Create `RigFlow4D/tests/test_kinematic_vae.py`: forward, loss, and training-step tests.
- Modify `RigFlow4D/RUN.md`: document the VAE entry point.

---

### Task 1: Kinematic VAE

**Files:**
- Test: `RigFlow4D/tests/test_kinematic_vae.py`
- Create: `RigFlow4D/models/rigflow4d/kinematic_vae.py`

- [ ] **Step 1: Write failing tests**

Tests should assert:

- `KinematicVAE.forward(batch)` returns `positions`, `local_rotations_6d`, `mu`, `logvar`, and `z` with expected shapes.
- `kinematic_vae_loss(output, batch)` returns finite scalar terms.
- one optimizer step updates parameters without errors.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_kinematic_vae.py -v`

Expected: FAIL with `ModuleNotFoundError` for the new model package.

- [ ] **Step 3: Implement minimal model and loss**

The model should accept a batch dict with:

```text
positions [B, T, J, 3]
local_rotations_6d [B, T, J, 6]
time_mask [B, T]
joint_mask [B, J]
```

Return reconstruction tensors matching input shapes and latent tensors `[B, latent_dim]`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_kinematic_vae.py tests/test_schema.py -v`

Expected: all tests pass.

---

## Self-Review

- Spec coverage: Implements the first VAE latent contract for Stage 6, not flow matching yet.
- Placeholder scan: No placeholders remain.
- Type consistency: Uses the batch keys emitted by `collate_motion_windows`.
