from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from data.schema import RigFlowSample


@dataclass(frozen=True)
class MotionWindow:
    positions: np.ndarray
    local_rotations_6d: np.ndarray
    root_translation: np.ndarray
    parents: np.ndarray
    time_mask: np.ndarray
    joint_mask: np.ndarray
    source_sample_index: int
    start: int
    dataset_name: str
    visual_tokens: np.ndarray | None = None
    visual_mask: np.ndarray | None = None


class MotionWindowDataset:
    def __init__(
        self,
        adapter: Any,
        window_size: int,
        stride: int,
        drop_short: bool = False,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.adapter = adapter
        self.window_size = window_size
        self.stride = stride
        self.drop_short = drop_short
        self.windows = self._build_index()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> MotionWindow:
        sample_index, start = self.windows[index]
        sample = self.adapter[sample_index]
        sample.validate()
        end = min(start + self.window_size, sample.positions.shape[0])
        length = end - start
        joint_count = sample.rig.num_joints

        positions = np.zeros((self.window_size, joint_count, 3), dtype=sample.positions.dtype)
        rotations = np.zeros((self.window_size, joint_count, 6), dtype=sample.local_rotations_6d.dtype)
        root_translation = np.zeros((self.window_size, 3), dtype=sample.root_translation.dtype)
        time_mask = np.zeros((self.window_size,), dtype=bool)
        visual_tokens = None
        visual_mask = None

        positions[:length] = sample.positions[start:end]
        rotations[:length] = sample.local_rotations_6d[start:end]
        root_translation[:length] = sample.root_translation[start:end]
        time_mask[:length] = True
        if sample.visual is not None:
            source_tokens = sample.visual.tokens
            views, _, patches, feature_dim = source_tokens.shape
            visual_tokens = np.zeros(
                (views, self.window_size, patches, feature_dim),
                dtype=source_tokens.dtype,
            )
            visual_mask = np.zeros((views, self.window_size, patches), dtype=bool)
            visual_tokens[:, :length] = source_tokens[:, start:end]
            visual_mask[:, :length] = True

        return MotionWindow(
            positions=positions,
            local_rotations_6d=rotations,
            root_translation=root_translation,
            parents=sample.rig.parents.astype(np.int64, copy=False),
            time_mask=time_mask,
            joint_mask=np.ones((joint_count,), dtype=bool),
            source_sample_index=sample_index,
            start=start,
            dataset_name=sample.dataset_name,
            visual_tokens=visual_tokens,
            visual_mask=visual_mask,
        )

    def _build_index(self) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        for sample_index in range(len(self.adapter)):
            sample: RigFlowSample = self.adapter[sample_index]
            frames = sample.positions.shape[0]
            if frames < self.window_size:
                if not self.drop_short:
                    windows.append((sample_index, 0))
                continue
            for start in range(0, frames - self.window_size + 1, self.stride):
                windows.append((sample_index, start))
        return windows


def collate_motion_windows(items: Sequence[MotionWindow]) -> dict[str, np.ndarray]:
    if not items:
        raise ValueError("items must be non-empty")
    batch_size = len(items)
    window_size = items[0].positions.shape[0]
    max_joints = max(item.positions.shape[1] for item in items)

    positions = np.zeros((batch_size, window_size, max_joints, 3), dtype=items[0].positions.dtype)
    rotations = np.zeros((batch_size, window_size, max_joints, 6), dtype=items[0].local_rotations_6d.dtype)
    root_translation = np.zeros((batch_size, window_size, 3), dtype=items[0].root_translation.dtype)
    parents = np.full((batch_size, max_joints), fill_value=-1, dtype=np.int64)
    time_mask = np.zeros((batch_size, window_size), dtype=bool)
    joint_mask = np.zeros((batch_size, max_joints), dtype=bool)
    source_sample_index = np.zeros((batch_size,), dtype=np.int64)
    start = np.zeros((batch_size,), dtype=np.int64)
    visual_items = [item for item in items if item.visual_tokens is not None]
    visual_tokens = None
    visual_mask = None
    if visual_items:
        feature_dim = visual_items[0].visual_tokens.shape[-1]
        visual_dtype = visual_items[0].visual_tokens.dtype
        max_views = max(item.visual_tokens.shape[0] for item in visual_items)
        max_patches = max(item.visual_tokens.shape[2] for item in visual_items)
        visual_tokens = np.zeros(
            (batch_size, max_views, window_size, max_patches, feature_dim),
            dtype=visual_dtype,
        )
        visual_mask = np.zeros((batch_size, max_views, window_size, max_patches), dtype=bool)

    for batch_index, item in enumerate(items):
        joints = item.positions.shape[1]
        positions[batch_index, :, :joints] = item.positions
        rotations[batch_index, :, :joints] = item.local_rotations_6d
        root_translation[batch_index] = item.root_translation
        parents[batch_index, :joints] = item.parents
        time_mask[batch_index] = item.time_mask
        joint_mask[batch_index, :joints] = item.joint_mask
        source_sample_index[batch_index] = item.source_sample_index
        start[batch_index] = item.start
        if item.visual_tokens is not None:
            if item.visual_tokens.shape[-1] != visual_tokens.shape[-1]:
                raise ValueError("all visual token feature dimensions must match within a batch")
            views = item.visual_tokens.shape[0]
            patches = item.visual_tokens.shape[2]
            visual_tokens[batch_index, :views, :, :patches] = item.visual_tokens
            visual_mask[batch_index, :views, :, :patches] = item.visual_mask

    batch = {
        "positions": positions,
        "local_rotations_6d": rotations,
        "root_translation": root_translation,
        "parents": parents,
        "time_mask": time_mask,
        "joint_mask": joint_mask,
        "source_sample_index": source_sample_index,
        "start": start,
    }
    if visual_tokens is not None:
        batch["visual_tokens"] = visual_tokens
        batch["visual_mask"] = visual_mask
    return batch
