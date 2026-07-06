# Raw Motion Capture Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first AMASS/AIST++-style raw `.npz` converter that turns `poses/trans` or `smpl_poses/smpl_trans` files into RigFlow4D normalized motion datasets.

**Architecture:** Parse raw axis-angle motion into a parsed motion `.npz` contract, synthesize approximate joint positions with a default SMPL-like 24-joint rest skeleton when no SMPL model is available, then reuse `convert_motion_npz_directory` to write final normalized `.npz` files and `manifest.json`.

**Tech Stack:** Python 3, NumPy, pytest.

---

## File Structure

- Create `RigFlow4D/preprocess/converters/raw_motion_capture.py`: raw parser, default skeleton, simple FK, directory conversion, and CLI.
- Modify `RigFlow4D/preprocess/converters/__init__.py`: export converter APIs.
- Create `RigFlow4D/tests/test_raw_motion_capture_converter.py`: tests for AMASS-like and AIST++-like keys, FK shape, normalized adapter load, and CLI args.
- Modify `RigFlow4D/RUN.md`: document the raw motion conversion path.

---

### Task 1: Raw AMASS/AIST++ Converter

**Files:**
- Test: `RigFlow4D/tests/test_raw_motion_capture_converter.py`
- Create: `RigFlow4D/preprocess/converters/raw_motion_capture.py`
- Modify: `RigFlow4D/preprocess/converters/__init__.py`
- Modify: `RigFlow4D/RUN.md`

- [x] **Step 1: Write failing tests**

Tests should assert:

- AMASS-like `poses/trans` is parsed into `[T, 24, 3]` `local_axis_angle`, `positions`, and `root_translation`.
- AIST++-like `smpl_poses/smpl_trans` is parsed the same way.
- invalid pose dimensions raise `ValueError`.
- `convert_raw_motion_capture_directory(...)` writes a normalized manifest loadable by `NormalizedNpzAdapter`.
- `parse_args(...)` maps CLI arguments.

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_raw_motion_capture_converter.py -v`

Expected: FAIL with `ModuleNotFoundError` for `preprocess.converters.raw_motion_capture`.

- [x] **Step 3: Implement minimal raw converter**

The converter should output parsed motion `.npz` files containing:

```text
parents
rest_offsets
joint_names
chain_ids
chain_coordinates
positions
root_translation
local_axis_angle
```

Then call `convert_motion_npz_directory(...)` to produce final normalized files.

- [x] **Step 4: Run tests**

Run: `python -m pytest tests/test_schema.py tests/test_raw_motion_capture_converter.py -v`

Expected: all tests pass.

---

## Self-Review

- Spec coverage: This plan covers first motion-only parsing. It does not use SMPL/SMPL-X model assets, so generated positions are approximate FK positions for pipeline bootstrapping.
- Placeholder scan: No placeholders remain.
- Type consistency: Uses `poses`, `trans`, `local_axis_angle`, `positions`, and `root_translation` consistently.
