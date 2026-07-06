# RigFlow4D Latent Flow Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first conditional flow matching module for kinematic latents, so `z_kin` from the VAE can be refined through a learnable latent velocity field.

**Architecture:** Implement a compact MLP velocity predictor over `(z_t, t, condition)`. Keep condition generic as a dense vector so later modules can plug in visual tokens, rig queries, camera relation, and deterministic pose seed summaries without changing the flow API.

**Tech Stack:** Python 3, PyTorch, pytest.

---

## File Structure

- Create `RigFlow4D/models/rigflow4d/latent_flow.py`: flow matcher, training pair sampler, and loss helper.
- Modify `RigFlow4D/models/rigflow4d/__init__.py`: export flow APIs.
- Create `RigFlow4D/tests/test_latent_flow.py`: shape, target velocity, loss, and training-step tests.
- Modify `RigFlow4D/RUN.md`: document the flow matching entry point.

---

### Task 1: Latent Flow Matcher

**Files:**
- Test: `RigFlow4D/tests/test_latent_flow.py`
- Create: `RigFlow4D/models/rigflow4d/latent_flow.py`

- [x] **Step 1: Write failing tests**

Tests should assert:

- `sample_latent_flow_pair(z0, z1, t)` returns `z_t=(1-t)z0+t z1` and target velocity `z1-z0`.
- `LatentFlowMatcher.forward(z_t, t, condition)` returns `[B, latent_dim]`.
- `latent_flow_matching_loss(...)` returns finite scalar terms.
- one optimizer step updates parameters.

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_latent_flow.py -v`

Expected: FAIL with `ModuleNotFoundError` for the new latent flow module.

- [x] **Step 3: Implement minimal flow module**

`LatentFlowMatcher(latent_dim, condition_dim, hidden_dim)` should concatenate:

```text
z_t [B, Dz]
t_features [B, 4]
condition [B, Dc]
```

and predict velocity `[B, Dz]`.

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_schema.py tests/test_kinematic_vae.py tests/test_latent_flow.py -v`

Expected: all tests pass.

---

## Self-Review

- Spec coverage: Implements latent-space conditional flow matching only. Delta P, contact, uncertainty, and visual/rig conditioning heads are intentionally later tasks.
- Placeholder scan: No placeholders remain.
- Type consistency: Uses `latent_dim`, `condition_dim`, `z_t`, `target_velocity`, and `condition` consistently.
