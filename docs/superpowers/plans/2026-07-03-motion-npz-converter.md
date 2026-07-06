# RigFlow4D Motion NPZ Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a motion-only converter that writes parsed AMASS/AIST++-style motion arrays into RigFlow4D normalized `.npz` cache files and a manifest.

**Architecture:** Keep this converter independent from raw AMASS/AIST++ parsing. It accepts an already-decoded motion `.npz` with positions, rig metadata, root translation, and either 6D rotations or axis-angle rotations. It writes the normalized cache consumed by `NormalizedNpzAdapter`.

**Tech Stack:** Python 3, pathlib, json, numpy, pytest.

---

## File Structure

- Create `RigFlow4D/preprocess/converters/__init__.py`: converter exports.
- Create `RigFlow4D/preprocess/converters/motion_npz.py`: conversion logic and axis-angle to 6D helper.
- Modify `RigFlow4D/tests/test_schema.py`: add converter tests using temporary source/output directories.
- Modify `RigFlow4D/RUN.md`: document the motion-only converter.

---

### Task 1: Motion NPZ Converter

**Files:**
- Test: `RigFlow4D/tests/test_schema.py`
- Create: `RigFlow4D/preprocess/converters/motion_npz.py`
- Create: `RigFlow4D/preprocess/converters/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
from preprocess.converters.motion_npz import convert_motion_npz_directory


def test_motion_npz_converter_writes_normalized_manifest(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    write_motion_source_npz(src / "motion_0001.npz")

    manifest_path = convert_motion_npz_directory(src, out, dataset_name="amass")

    adapter = NormalizedNpzAdapter(root=out, manifest_path=manifest_path)
    sample = adapter[0]
    assert sample.dataset_name == "amass"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_schema.py::test_motion_npz_converter_writes_normalized_manifest -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'preprocess.converters'`.

- [ ] **Step 3: Implement converter**

Implement `convert_motion_npz_directory(input_dir, output_dir, dataset_name, source_label_type="motion_only")`:

- read all `.npz` files under `input_dir`;
- require rig metadata and `positions`;
- accept `local_rotations_6d` directly or convert `local_axis_angle` `[T, J, 3]` to 6D;
- default missing `root_translation` to zeros `[T, 3]`;
- write normalized `.npz` files and `manifest.json`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_schema.py -v`

Expected: all tests pass.

---

## Self-Review

- Spec coverage: This implements the cache-writing bridge for motion-only data. It does not parse raw AMASS SMPL parameters or AIST++ pickle internals.
- Placeholder scan: No placeholders remain.
- Type consistency: Output matches `NormalizedNpzAdapter` and `RigFlowSample`.
