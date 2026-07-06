# RigFlow4D Normalized NPZ Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small dataset adapter that reads RigFlow4D normalized `.npz` cache files into validated `RigFlowSample` objects.

**Architecture:** Keep raw dataset parsing separate from training-time loading. Dataset-specific converters will write normalized `.npz` files; `NormalizedNpzAdapter` reads those files through a manifest and emits one schema-validated sample at a time.

**Tech Stack:** Python 3, pathlib, json, numpy, pytest.

---

## File Structure

- Create `RigFlow4D/data/adapters/__init__.py`: adapter exports.
- Create `RigFlow4D/data/adapters/base.py`: minimal dataset adapter protocol.
- Create `RigFlow4D/data/adapters/normalized_npz.py`: manifest reader and `.npz` to `RigFlowSample` conversion.
- Modify `RigFlow4D/data/dataset_registry.py`: add `create_default_registry()` with `normalized_npz`.
- Modify `RigFlow4D/data/__init__.py`: export adapter registry factory.
- Modify `RigFlow4D/tests/test_schema.py`: add adapter tests using temporary normalized `.npz` files.
- Modify `RigFlow4D/RUN.md`: document normalized cache path.

---

### Task 1: Normalized NPZ Adapter

**Files:**
- Test: `RigFlow4D/tests/test_schema.py`
- Create: `RigFlow4D/data/adapters/base.py`
- Create: `RigFlow4D/data/adapters/normalized_npz.py`
- Create: `RigFlow4D/data/adapters/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
from data.adapters.normalized_npz import NormalizedNpzAdapter


def write_normalized_npz(path):
    np.savez(
        path,
        dataset_name=np.array("unit"),
        input_type=np.array("video"),
        source_label_type=np.array("rig_native"),
        camera_mode=np.array("unknown"),
        parents=np.array([-1, 0, 1], dtype=np.int64),
        rest_offsets=np.zeros((3, 3), dtype=np.float32),
        joint_names=np.array(["root", "spine", "head"]),
        chain_ids=np.array([0, 0, 1], dtype=np.int64),
        chain_coordinates=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        positions=np.zeros((2, 3, 3), dtype=np.float32),
        local_rotations_6d=np.zeros((2, 3, 6), dtype=np.float32),
        root_translation=np.zeros((2, 3), dtype=np.float32),
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schema.py::test_normalized_npz_adapter_loads_sample -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'data.adapters'`.

- [ ] **Step 3: Implement adapter**

`NormalizedNpzAdapter(root, manifest_path)` should:

- read a JSON manifest with `{"samples": [{"sample_id": "...", "path": "relative.npz"}]}`;
- implement `__len__`;
- implement `__getitem__(idx)` and return a validated `RigFlowSample`;
- support optional visual token and camera arrays if keys exist.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_schema.py -v`

Expected: all tests pass.

---

### Task 2: Default Registry

**Files:**
- Test: `RigFlow4D/tests/test_schema.py`
- Modify: `RigFlow4D/data/dataset_registry.py`
- Modify: `RigFlow4D/data/__init__.py`

- [ ] **Step 1: Write failing registry test**

```python
from data import create_default_registry


def test_default_registry_builds_normalized_npz_adapter(tmp_path):
    registry = create_default_registry()
    assert "normalized_npz" in registry.names()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_schema.py::test_default_registry_builds_normalized_npz_adapter -v`

Expected: FAIL because `create_default_registry` is missing.

- [ ] **Step 3: Implement default registry**

Register `normalized_npz` to `NormalizedNpzAdapter`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_schema.py -v`

Expected: all tests pass.

---

## Self-Review

- Spec coverage: This plan implements the first stable cache adapter, not real dataset parsers.
- Placeholder scan: No placeholders remain.
- Type consistency: Adapter returns `RigFlowSample` and uses the registry API already introduced.
