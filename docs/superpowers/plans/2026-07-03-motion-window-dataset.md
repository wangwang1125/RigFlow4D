# RigFlow4D Motion Window Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a training-ready motion window layer that slices validated `RigFlowSample` sequences into fixed temporal windows and pads variable joint counts during collation.

**Architecture:** Keep this layer framework-neutral with NumPy arrays. `MotionWindowDataset` handles temporal slicing and time masks; `collate_motion_windows` pads joints to the batch maximum and emits joint masks.

**Tech Stack:** Python 3, dataclasses, numpy, pytest.

---

## File Structure

- Create `RigFlow4D/data/window_dataset.py`: window item dataclass, dataset wrapper, and collate helper.
- Modify `RigFlow4D/data/__init__.py`: export the new API.
- Modify `RigFlow4D/tests/test_schema.py`: add tests for strided windows and joint padding.
- Modify `RigFlow4D/RUN.md`: document sequence -> window -> batch flow.

---

### Task 1: Motion Window Dataset

**Files:**
- Test: `RigFlow4D/tests/test_schema.py`
- Create: `RigFlow4D/data/window_dataset.py`
- Modify: `RigFlow4D/data/__init__.py`

- [ ] **Step 1: Write failing tests**

Add tests that:

- build one 5-frame sample with `window_size=3`, `stride=2`;
- expect two windows: frames `0:3` and `2:5`;
- build two samples with different joint counts;
- collate them into padded `[B, W, Jmax, C]` arrays with `joint_mask`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schema.py::test_motion_window_dataset_creates_strided_windows -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'data.window_dataset'`.

- [ ] **Step 3: Implement minimal windowing**

`MotionWindowDataset(adapter, window_size, stride, drop_short=False)` should return window items with `positions`, `local_rotations_6d`, `root_translation`, `time_mask`, `joint_mask`, `source_sample_index`, and `start`.

- [ ] **Step 4: Implement collate helper**

`collate_motion_windows(items)` should pad joint dimension to max joints in batch and stack arrays.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_schema.py -v`

Expected: all tests pass.

---

## Self-Review

- Spec coverage: This covers fixed temporal windows and variable joint padding for training batches.
- Placeholder scan: No placeholders remain.
- Type consistency: Uses `RigFlowSample` arrays and keeps NumPy outputs.
