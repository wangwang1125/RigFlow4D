# RigFlow4D Dataset Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first RigFlow4D data foundation: a typed sample schema, camera-optional validation, and a dataset adapter registry.

**Architecture:** Keep the schema independent from PyTorch datasets so raw adapters for AMASS, BEDLAM, MVHumanNet++, HuMMan, and evaluation sets can all normalize into one contract. Use lightweight dataclasses and numpy-based validation first; loaders can wrap this later.

**Tech Stack:** Python 3, dataclasses, enum, numpy, pytest.

---

## File Structure

- Create `RigFlow4D/data/schema.py`: typed dataclasses, input/source enums, shape validators, and `RigFlowSample.validate()`.
- Create `RigFlow4D/data/dataset_registry.py`: adapter registry mapping dataset names to constructors.
- Create `RigFlow4D/data/__init__.py`: public imports for new RigFlow4D data contracts.
- Create `RigFlow4D/tests/test_schema.py`: TDD tests for minimal valid samples, camera optional behavior, invalid shapes, and registry behavior.

---

### Task 1: Schema Validation Contract

**Files:**
- Test: `RigFlow4D/tests/test_schema.py`
- Create: `RigFlow4D/data/schema.py`
- Create: `RigFlow4D/data/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from data.schema import (
    CameraMode,
    InputType,
    RigDefinition,
    RigFlowSample,
    SourceLabelType,
    VisualTokenCache,
)


def make_rig():
    return RigDefinition(
        parents=np.array([-1, 0, 1], dtype=np.int64),
        rest_offsets=np.zeros((3, 3), dtype=np.float32),
        joint_names=("root", "spine", "head"),
        chain_ids=np.array([0, 0, 1], dtype=np.int64),
        chain_coordinates=np.array([0.0, 0.5, 1.0], dtype=np.float32),
    )


def test_video_sample_accepts_missing_camera_parameters():
    sample = RigFlowSample(
        dataset_name="unit",
        input_type=InputType.VIDEO,
        source_label_type=SourceLabelType.RIG_NATIVE,
        camera_mode=CameraMode.UNKNOWN,
        rig=make_rig(),
        positions=np.zeros((2, 3, 3), dtype=np.float32),
        local_rotations_6d=np.zeros((2, 3, 6), dtype=np.float32),
        root_translation=np.zeros((2, 3), dtype=np.float32),
        visual=VisualTokenCache(
            tokens=np.zeros((1, 2, 4, 8), dtype=np.float32),
            backbone_name="dinov3_vitl16",
            feature_dim=8,
            patch_grid=(2, 2),
        ),
    )

    sample.validate()


def test_rejects_position_joint_count_mismatch():
    sample = RigFlowSample(
        dataset_name="unit",
        input_type=InputType.VIDEO,
        source_label_type=SourceLabelType.RIG_NATIVE,
        camera_mode=CameraMode.UNKNOWN,
        rig=make_rig(),
        positions=np.zeros((2, 4, 3), dtype=np.float32),
        local_rotations_6d=np.zeros((2, 3, 6), dtype=np.float32),
        root_translation=np.zeros((2, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="positions"):
        sample.validate()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schema.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'data.schema'`.

- [ ] **Step 3: Implement minimal schema**

Create dataclasses with `validate()` methods that check:

- `positions` is `[T, J, 3]`.
- `local_rotations_6d` is `[T, J, 6]`.
- `root_translation` is `[T, 3]`.
- rig fields share joint count `J`.
- visual tokens are `[V, T, P, D]` and `D == feature_dim`.
- camera arrays are optional when `camera_mode == UNKNOWN`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_schema.py -v`

Expected: 2 passed.

---

### Task 2: Camera-Optional Validation

**Files:**
- Modify: `RigFlow4D/tests/test_schema.py`
- Modify: `RigFlow4D/data/schema.py`

- [ ] **Step 1: Add failing camera tests**

```python
def test_calibrated_camera_requires_intrinsics_and_extrinsics():
    sample = RigFlowSample(
        dataset_name="unit",
        input_type=InputType.MULTIVIEW_VIDEO,
        source_label_type=SourceLabelType.SMPLX,
        camera_mode=CameraMode.CALIBRATED,
        rig=make_rig(),
        positions=np.zeros((2, 3, 3), dtype=np.float32),
        local_rotations_6d=np.zeros((2, 3, 6), dtype=np.float32),
        root_translation=np.zeros((2, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="calibrated"):
        sample.validate()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_schema.py::test_calibrated_camera_requires_intrinsics_and_extrinsics -v`

Expected: FAIL because calibrated camera validation is missing.

- [ ] **Step 3: Add camera validation**

Require `camera_intrinsics` and `camera_extrinsics` for `CameraMode.CALIBRATED`. Allow partial/noisy hints for `WEAK_CALIBRATED`. Allow missing arrays for `UNKNOWN`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_schema.py -v`

Expected: 3 passed.

---

### Task 3: Dataset Adapter Registry

**Files:**
- Modify: `RigFlow4D/tests/test_schema.py`
- Create: `RigFlow4D/data/dataset_registry.py`
- Modify: `RigFlow4D/data/__init__.py`

- [ ] **Step 1: Add failing registry test**

```python
from data.dataset_registry import DatasetAdapterRegistry


def test_dataset_registry_registers_and_builds_adapter():
    registry = DatasetAdapterRegistry()

    class DummyAdapter:
        def __init__(self, root, split):
            self.root = root
            self.split = split

    registry.register("dummy", DummyAdapter)
    adapter = registry.build("dummy", root="data_root", split="train")

    assert isinstance(adapter, DummyAdapter)
    assert adapter.root == "data_root"
    assert adapter.split == "train"
```

- [ ] **Step 2: Run registry test to verify it fails**

Run: `python -m pytest tests/test_schema.py::test_dataset_registry_registers_and_builds_adapter -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'data.dataset_registry'`.

- [ ] **Step 3: Implement registry**

Implement `DatasetAdapterRegistry.register(name, cls)` and `DatasetAdapterRegistry.build(name, **kwargs)`. Reject duplicate names and unknown names with clear `ValueError` messages.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_schema.py -v`

Expected: all tests pass.

---

### Task 4: Documentation Hook

**Files:**
- Modify: `RigFlow4D/RUN.md`
- Modify: `RigFlow4D/docs/RigFlow4D_plan.md`

- [ ] **Step 1: Add schema reference text**

Document that all dataset adapters must emit `RigFlowSample` and call `sample.validate()` before caching or batching.

- [ ] **Step 2: Run markdown/path sanity checks**

Run: `python -m pytest tests/test_schema.py -v`

Expected: all tests still pass.

---

## Self-Review

- Spec coverage: This plan covers the Stage 0.5 dataset adapter foundation, camera-optional metadata, and unified sample schema. It does not implement real AMASS/BEDLAM/MVHumanNet++ parsers; those should be separate tasks after this contract is stable.
- Placeholder scan: No task uses TBD/TODO placeholders.
- Type consistency: All tasks use `RigFlowSample`, `RigDefinition`, `VisualTokenCache`, `InputType`, `CameraMode`, and `SourceLabelType` consistently.
