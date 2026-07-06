# Visual Token Cache Writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first DINOv3-style visual token cache writer and normalized-sample injection utility, without requiring real DINOv3 weights yet.

**Architecture:** Define a small visual backbone interface that encodes frames into dense tokens, write cache `.npz` files with `visual_tokens [V,T,P,D]` plus backbone metadata, and provide an injector that copies a normalized motion `.npz` while adding those visual fields. A deterministic test backbone stands in for real DINOv3 until model loading is implemented.

**Tech Stack:** Python 3, NumPy, pytest.

---

## File Structure

- Create `RigFlow4D/preprocess/visual/__init__.py`: export cache writer APIs.
- Create `RigFlow4D/preprocess/visual/token_cache.py`: metadata dataclass, deterministic backbone, cache writer, normalized sample injector, and CLI args.
- Create `RigFlow4D/tests/test_visual_token_cache.py`: tests for cache writing, validation, injection, adapter loading, and CLI args.
- Modify `RigFlow4D/RUN.md`: document the cache writer contract.

---

### Task 1: Visual Token Cache Writer

**Files:**
- Test: `RigFlow4D/tests/test_visual_token_cache.py`
- Create: `RigFlow4D/preprocess/visual/token_cache.py`
- Create: `RigFlow4D/preprocess/visual/__init__.py`
- Modify: `RigFlow4D/RUN.md`

- [x] **Step 1: Write failing tests**

Tests should assert:

- deterministic backbone encodes `frames [V,T,H,W,C]` into `tokens [V,T,P,D]`.
- `write_visual_token_cache(...)` writes cache keys compatible with `VisualTokenCache`.
- wrong token rank or feature dimension raises `ValueError`.
- `inject_visual_cache_into_normalized_npz(...)` produces a sample loadable by `NormalizedNpzAdapter`.
- `parse_args(...)` maps CLI options.

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_visual_token_cache.py -v`

Expected: FAIL with `ModuleNotFoundError` for `preprocess.visual.token_cache`.

- [x] **Step 3: Implement minimal cache writer**

Cache files should contain:

```text
visual_tokens
visual_backbone_name
visual_feature_dim
visual_patch_grid
visual_has_cls
visual_num_registers
```

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_visual_token_cache.py tests/test_schema.py tests/test_latent_refiner_training.py -v`

Expected: all tests pass.

---

## Self-Review

- Spec coverage: This plan covers cache format and injection only. It does not load real DINOv3 weights or read images/videos from disk.
- Placeholder scan: No placeholders remain.
- Type consistency: Uses `visual_tokens`, `visual_backbone_name`, `visual_feature_dim`, and `visual_patch_grid` consistently.
