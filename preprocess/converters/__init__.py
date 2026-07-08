from importlib import import_module
from typing import Any

__all__ = [
    "RawMotionCaptureConfig",
    "axis_angle_to_rot6d",
    "convert_motion_npz_directory",
    "convert_paired_visual_motion_manifest",
    "convert_raw_motion_capture_directory",
    "parse_raw_motion_capture_npz",
]

_EXPORT_MODULES = {
    "axis_angle_to_rot6d": ".motion_npz",
    "convert_motion_npz_directory": ".motion_npz",
    "convert_paired_visual_motion_manifest": ".paired_visual_motion",
    "RawMotionCaptureConfig": ".raw_motion_capture",
    "convert_raw_motion_capture_directory": ".raw_motion_capture",
    "parse_raw_motion_capture_npz": ".raw_motion_capture",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name], package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
