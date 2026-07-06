# RigFlow4D Latent Refiner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the kinematic VAE, condition encoder, and latent flow matcher into the first trainable RigFlow4D refinement module.

**Architecture:** Encode a motion batch into a target latent `z1` with `KinematicVAE`, encode observation/context tensors into `condition [B, Dcond]` with `RigFlowConditionEncoder`, sample or accept source latent `z0`, and compute conditional flow matching loss. The default detaches `z1` so early flow training can use a stable VAE target before later end-to-end joint optimization.

**Tech Stack:** Python 3, PyTorch, pytest.

---

## File Structure

- Create `RigFlow4D/models/rigflow4d/latent_refiner.py`: integrated latent refinement module and output dataclass.
- Modify `RigFlow4D/models/rigflow4d/__init__.py`: export the refiner APIs.
- Create `RigFlow4D/tests/test_latent_refiner.py`: tests for loss, missing modalities, detach behavior, training step, and export.
- Modify `RigFlow4D/RUN.md`: document the first trainable latent refinement path.

---

### Task 1: Latent Refiner

**Files:**
- Test: `RigFlow4D/tests/test_latent_refiner.py`
- Create: `RigFlow4D/models/rigflow4d/latent_refiner.py`
- Modify: `RigFlow4D/models/rigflow4d/__init__.py`
- Modify: `RigFlow4D/RUN.md`

- [x] **Step 1: Write failing tests**

Tests should assert:

- `RigFlowLatentRefiner.forward(...)` returns finite `loss`, `flow_loss`, latent pair shapes, and `condition [B, Dcond]`.
- the refiner accepts motion-only batches with missing visual/camera/rig/pose conditions.
- with default `detach_vae_target=True`, flow loss backpropagation does not populate VAE parameter gradients.
- one optimizer step updates trainable refiner parameters.
- `RigFlowLatentRefiner` is exported from `models.rigflow4d`.

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_latent_refiner.py -v`

Expected: FAIL with `ModuleNotFoundError` for `models.rigflow4d.latent_refiner`.

- [x] **Step 3: Implement minimal latent refiner**

The module should compose existing components:

```text
motion_batch -> KinematicVAE -> z1 = mu
condition inputs -> RigFlowConditionEncoder -> condition
z0, z1, condition -> LatentFlowMatcher loss
```

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_schema.py tests/test_kinematic_vae.py tests/test_latent_flow.py tests/test_condition_encoder.py tests/test_latent_refiner.py -v`

Expected: all tests pass.

---

## Self-Review

- Spec coverage: This plan covers the first integrated flow-training path only. It does not implement a CLI training script, checkpointing, DINOv3 extraction, or pose/rotation decoding heads.
- Placeholder scan: No placeholders remain.
- Type consistency: Uses `motion_batch`, `condition`, `z0`, `z1`, `flow_loss`, and `detach_vae_target` consistently.
