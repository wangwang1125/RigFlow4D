from __future__ import annotations

from typing import Any, Callable, Dict


class DatasetAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, adapter_cls: Callable[..., Any]) -> None:
        if not name:
            raise ValueError("dataset adapter name must be non-empty")
        if name in self._adapters:
            raise ValueError(f"dataset adapter '{name}' is already registered")
        self._adapters[name] = adapter_cls

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self._adapters:
            known = ", ".join(sorted(self._adapters)) or "<none>"
            raise ValueError(f"Unknown dataset adapter '{name}'. Registered adapters: {known}")
        return self._adapters[name](**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


def create_default_registry() -> DatasetAdapterRegistry:
    from data.adapters.normalized_npz import NormalizedNpzAdapter

    registry = DatasetAdapterRegistry()
    registry.register("normalized_npz", NormalizedNpzAdapter)
    return registry
