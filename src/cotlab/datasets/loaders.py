"""Dataset loaders for CoT research.

Supports JSON format with standardized structure:
{
    "id": "unique_id",
    "input": { ... },
    "output": { ... },
    "metadata": { ... }
}
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..core.registry import Registry


@dataclass
class Sample:
    """A single data sample."""

    idx: int
    text: str
    label: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {"idx": self.idx, "text": self.text, "label": self.label, "metadata": self.metadata}


class BaseDataset(ABC):
    """Abstract base class for datasets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample:
        """Get a sample by index."""
        ...

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over samples."""
        for i in range(len(self)):
            yield self[i]

    def sample(self, n: int, seed: int = 42) -> List[Sample]:
        """Random sample of n items."""
        import random

        random.seed(seed)
        indices = random.sample(range(len(self)), min(n, len(self)))
        return [self[i] for i in indices]


class JSONDataset(BaseDataset):
    """Base class for JSON-based datasets."""

    def __init__(self, name: str, path: str, **kwargs):
        self._name = name
        self.path = Path(path)
        self._samples: List[Sample] = []
        self._load()

    def _load(self):
        """Load samples from JSON file."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            sample = self._parse_item(idx, item)
            self._samples.append(sample)

    @abstractmethod
    def _parse_item(self, idx: int, item: Dict[str, Any]) -> Sample:
        """Parse a single JSON item into a Sample. Override in subclasses."""
        ...

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


@Registry.register_dataset("radiology")
class RadiologyDataset(JSONDataset):
    """Radiology reports dataset for pathological fracture detection."""

    def __init__(
        self,
        name: str = "radiology",
        path: str = "data/radiology.json",
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)

    def _parse_item(self, idx: int, item: Dict[str, Any]) -> Sample:
        return Sample(
            idx=idx,
            text=item["input"]["report"],
            label=item["output"]["pathological_fracture"],
            metadata=item.get("metadata", {}),
        )


@Registry.register_dataset("pediatrics")
class PediatricsDataset(JSONDataset):
    """Pediatrics clinical scenarios dataset."""

    def __init__(
        self,
        name: str = "pediatrics",
        path: str = "data/pediatrics.json",
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)

    def _parse_item(self, idx: int, item: Dict[str, Any]) -> Sample:
        return Sample(
            idx=idx,
            text=item["input"]["scenario"],
            label=item["output"]["diagnosis"],
            metadata=item.get("metadata", {}),
        )


@Registry.register_dataset("synthetic")
class SyntheticMedicalDataset(JSONDataset):
    """Synthetic medical QA dataset."""

    def __init__(
        self,
        name: str = "synthetic",
        path: str = "data/synthetic.json",
        repeat: int = 1,
        **kwargs,
    ):
        self.repeat = repeat
        super().__init__(name, path, **kwargs)

    def _load(self):
        """Load samples with optional repeat."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for r in range(self.repeat):
            for idx, item in enumerate(data):
                sample = self._parse_item(r * len(data) + idx, item)
                self._samples.append(sample)

    def _parse_item(self, idx: int, item: Dict[str, Any]) -> Sample:
        return Sample(
            idx=idx,
            text=item["input"]["scenario"],
            label=item["output"]["diagnosis"],
            metadata=item.get("metadata", {}),
        )


@Registry.register_dataset("patching_pairs")
class PatchingPairsDataset(JSONDataset):
    """Clean/corrupted pairs for activation patching experiments."""

    def __init__(
        self,
        name: str = "patching_pairs",
        path: str = "data/patching_pairs.json",
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)

    def _parse_item(self, idx: int, item: Dict[str, Any]) -> Sample:
        return Sample(
            idx=idx,
            text=item["clean"]["input"],
            label=item["clean"]["output"],
            metadata={
                "corrupted_prompt": item["corrupted"]["input"],
                "clean_answer": item["clean"]["output"],
                "corrupted_answer": item["corrupted"]["output"],
                "category": item.get("metadata", {}).get("category", "general"),
            },
        )
