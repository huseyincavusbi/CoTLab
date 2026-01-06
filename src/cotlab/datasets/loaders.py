"""Dataset loaders for CoT research.

Supports JSON format with standardized structure:
{
    "id": "unique_id",
    "input": { ... },
    "output": { ... },
    "metadata": { ... }
}

Also supports CSV format with automatic detection based on file extension.

Datasets can specify compatible prompts via get_compatible_prompts() for
restricting which prompt strategies can be used.
"""

import csv
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

    def get_compatible_prompts(self) -> Optional[List[str]]:
        """
        Return list of compatible prompt names, or None if compatible with all.

        Override this in specialized datasets to restrict usage.
        """
        return None


class JSONDataset(BaseDataset):
    """Base class for JSON/CSV-based datasets with automatic format detection."""

    def __init__(self, name: str, path: str, **kwargs):
        self._name = name
        self.path = Path(path)
        self._samples: List[Sample] = []
        self._load()

    def _load(self):
        """Load samples from JSON or CSV file based on extension."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        if self.path.suffix.lower() == ".csv":
            self._load_csv()
        else:
            self._load_json()

    def _load_json(self):
        """Load samples from JSON file."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in enumerate(data):
            sample = self._parse_item(idx, item)
            self._samples.append(sample)

    def _load_csv(self):
        """Load samples from CSV file."""
        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                sample = self._parse_csv_row(idx, row)
                self._samples.append(sample)

    @abstractmethod
    def _parse_item(self, idx: int, item: Dict[str, Any]) -> Sample:
        """Parse a single JSON item into a Sample. Override in subclasses."""
        ...

    def _parse_csv_row(self, idx: int, row: Dict[str, Any]) -> Sample:
        """Parse a single CSV row into a Sample. Override in subclasses for custom CSV handling."""
        # Default implementation - subclasses should override for specific CSV formats
        raise NotImplementedError(
            f"CSV loading not implemented for {self.__class__.__name__}. "
            "Override _parse_csv_row() or use a JSON file."
        )

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

    def _parse_csv_row(self, idx: int, row: Dict[str, Any]) -> Sample:
        # CSV columns: Synthetic (text), Flag (label)
        return Sample(
            idx=idx,
            text=row.get("Synthetic", ""),
            label=row.get("Flag", ""),
            metadata={},
        )

    def get_compatible_prompts(self) -> list[str]:
        """
        Radiology dataset only works with radiology prompt.

        This dataset is for pathological fracture detection and should
        NOT be used with general medical QA prompts.
        """
        return ["radiology"]


@Registry.register_dataset("cardiology")
class CardiologyDataset(JSONDataset):
    """Cardiology reports dataset for congenital heart defect detection."""

    def __init__(
        self,
        name: str = "cardiology",
        path: str = "data/cardiology.json",
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)

    def _parse_item(self, idx: int, item: Dict[str, Any]) -> Sample:
        return Sample(
            idx=idx,
            text=item["input"]["report"],
            label=item["output"]["congenital_heart_defect"],
            metadata=item.get("metadata", {}),
        )

    def get_compatible_prompts(self) -> list[str]:
        """
        Cardiology dataset only works with cardiology prompt.

        This dataset is for congenital heart defect detection and should
        NOT be used with general medical QA prompts.
        """
        return ["cardiology"]


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

    def _parse_csv_row(self, idx: int, row: Dict[str, Any]) -> Sample:
        # CSV columns: Scenario, Diagnosis, Age_Group, Category
        return Sample(
            idx=idx,
            text=row.get("Scenario", ""),
            label=row.get("Diagnosis", ""),
            metadata={
                "age_group": row.get("Age_Group", ""),
                "category": row.get("Category", ""),
            },
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
        """Load samples with optional repeat. Supports both JSON and CSV."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        if self.path.suffix.lower() == ".csv":
            self._load_csv_with_repeat()
        else:
            self._load_json_with_repeat()

    def _load_json_with_repeat(self):
        """Load JSON samples with optional repeat."""
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for r in range(self.repeat):
            for idx, item in enumerate(data):
                sample = self._parse_item(r * len(data) + idx, item)
                self._samples.append(sample)

    def _load_csv_with_repeat(self):
        """Load CSV samples with optional repeat."""
        with open(self.path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for r in range(self.repeat):
            for idx, row in enumerate(rows):
                sample = self._parse_csv_row(r * len(rows) + idx, row)
                self._samples.append(sample)

    def _parse_item(self, idx: int, item: Dict[str, Any]) -> Sample:
        return Sample(
            idx=idx,
            text=item["input"]["scenario"],
            label=item["output"]["diagnosis"],
            metadata=item.get("metadata", {}),
        )

    def _parse_csv_row(self, idx: int, row: Dict[str, Any]) -> Sample:
        # CSV columns: Scenario, Expected_Answer, Reasoning_Keywords
        return Sample(
            idx=idx,
            text=row.get("Scenario", ""),
            label=row.get("Expected_Answer", ""),
            metadata={
                "reasoning_keywords": row.get("Reasoning_Keywords", ""),
            },
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

    def _parse_csv_row(self, idx: int, row: Dict[str, Any]) -> Sample:
        # CSV columns: Clean_Prompt, Corrupted_Prompt, Clean_Answer, Corrupted_Answer, Category
        return Sample(
            idx=idx,
            text=row.get("Clean_Prompt", ""),
            label=row.get("Clean_Answer", ""),
            metadata={
                "corrupted_prompt": row.get("Corrupted_Prompt", ""),
                "clean_answer": row.get("Clean_Answer", ""),
                "corrupted_answer": row.get("Corrupted_Answer", ""),
                "category": row.get("Category", "general"),
            },
        )


@Registry.register_dataset("tutorial")
class TutorialDataset(JSONDataset):
    """Simple Q&A dataset for tutorials and demos."""

    def __init__(
        self,
        name: str = "tutorial",
        path: str = "data/tutorial.json",
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)

    def _parse_item(self, idx: int, item: Dict[str, Any]) -> Sample:
        return Sample(
            idx=idx,
            text=item["text"],
            label=item["label"],
            metadata={},
        )


@Registry.register_dataset("probing_diagnosis")
class ProbingDiagnosisDataset(JSONDataset):
    """
    Dataset for probing experiments testing diagnosis encoding.

    Each sample has:
    - question: Medical scenario
    - diagnosis: Correct diagnosis (label)
    - category: Medical specialty
    - confounders: Alternative diagnoses
    - key_features: Important clinical features
    """

    def __init__(
        self,
        name: str = "probing_diagnosis",
        path: str = "data/probing_diagnosis.json",
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)

    def _parse_item(self, idx: int, item: Dict[str, Any]) -> Sample:
        return Sample(
            idx=idx,
            text=item["input"]["question"],
            label=item["output"]["diagnosis"],
            metadata={
                "category": item.get("metadata", {}).get("category", "general"),
                "difficulty": item.get("metadata", {}).get("difficulty", "medium"),
                "confounders": item.get("metadata", {}).get("confounders", []),
                "key_features": item.get("metadata", {}).get("key_features", []),
            },
        )
