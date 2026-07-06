from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from data.adapters.base import BaseDatasetAdapter
from data.schema import (
    CameraMode,
    InputType,
    RigDefinition,
    RigFlowSample,
    SourceLabelType,
    VisualTokenCache,
)


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _optional_array(npz: Mapping[str, Any], key: str) -> Optional[np.ndarray]:
    if key not in npz:
        return None
    return np.asarray(npz[key])


def _optional_scalar(npz: Mapping[str, Any], key: str) -> Optional[Any]:
    if key not in npz:
        return None
    return _scalar(npz[key])


class NormalizedNpzAdapter(BaseDatasetAdapter):
    def __init__(self, root: str | Path, manifest_path: str | Path) -> None:
        self.root = Path(root)
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_absolute():
            self.manifest_path = self.root / self.manifest_path
        self.samples = self._load_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> RigFlowSample:
        record = self.samples[index]
        path = Path(record["path"])
        if not path.is_absolute():
            path = self.root / path
        sample = self._read_npz(path)
        sample.validate()
        return sample

    @staticmethod
    def _load_manifest(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        samples = manifest.get("samples")
        if not isinstance(samples, list):
            raise ValueError("normalized npz manifest must contain a 'samples' list")
        for record in samples:
            if not isinstance(record, dict) or "path" not in record:
                raise ValueError("each manifest sample must be an object with a 'path' field")
        return samples

    @staticmethod
    def _read_npz(path: Path) -> RigFlowSample:
        with np.load(path, allow_pickle=False) as npz:
            rig = RigDefinition(
                parents=np.asarray(npz["parents"]),
                rest_offsets=np.asarray(npz["rest_offsets"]),
                joint_names=tuple(str(x) for x in np.asarray(npz["joint_names"]).tolist()),
                chain_ids=np.asarray(npz["chain_ids"]),
                chain_coordinates=np.asarray(npz["chain_coordinates"]),
            )
            visual = NormalizedNpzAdapter._read_visual(npz)
            return RigFlowSample(
                dataset_name=str(_scalar(npz["dataset_name"])),
                input_type=InputType(str(_scalar(npz["input_type"]))),
                source_label_type=SourceLabelType(str(_scalar(npz["source_label_type"]))),
                camera_mode=CameraMode(str(_scalar(npz["camera_mode"]))),
                rig=rig,
                positions=np.asarray(npz["positions"]),
                local_rotations_6d=np.asarray(npz["local_rotations_6d"]),
                root_translation=np.asarray(npz["root_translation"]),
                visual=visual,
                camera_intrinsics=_optional_array(npz, "camera_intrinsics"),
                camera_extrinsics=_optional_array(npz, "camera_extrinsics"),
                camera_valid_mask=_optional_array(npz, "camera_valid_mask"),
                view_ids=_optional_array(npz, "view_ids"),
                frame_times=_optional_array(npz, "frame_times"),
                observed_time_mask=_optional_array(npz, "observed_time_mask"),
                contact_labels=_optional_array(npz, "contact_labels"),
            )

    @staticmethod
    def _read_visual(npz: Mapping[str, Any]) -> Optional[VisualTokenCache]:
        if "visual_tokens" not in npz:
            return None
        patch_grid = _optional_array(npz, "visual_patch_grid")
        return VisualTokenCache(
            tokens=np.asarray(npz["visual_tokens"]),
            backbone_name=str(_optional_scalar(npz, "visual_backbone_name") or ""),
            feature_dim=int(_optional_scalar(npz, "visual_feature_dim")),
            patch_grid=tuple(int(x) for x in patch_grid.tolist()) if patch_grid is not None else None,
            has_cls=bool(_optional_scalar(npz, "visual_has_cls") if "visual_has_cls" in npz else True),
            num_registers=int(_optional_scalar(npz, "visual_num_registers") or 0),
        )
