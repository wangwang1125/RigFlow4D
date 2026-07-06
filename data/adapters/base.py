from __future__ import annotations

from abc import ABC, abstractmethod

from data.schema import RigFlowSample


class BaseDatasetAdapter(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> RigFlowSample:
        raise NotImplementedError

    def iter_samples(self):
        for index in range(len(self)):
            yield self[index]
