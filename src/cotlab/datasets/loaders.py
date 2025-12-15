"""Dataset loaders for CoT research."""

import csv
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


@Registry.register_dataset("radiology")
class RadiologyDataset(BaseDataset):
    """
    Dataset from CSV file with radiology reports.

    Uses existing Files/Radiology_Synthetic_Data.csv
    """

    def __init__(
        self,
        name: str = "radiology",
        path: str = "Files/Radiology_Synthetic_Data.csv",
        text_column: str = "Synthetic",
        label_column: str = "Flag",
        **kwargs,
    ):
        self._name = name
        self.path = Path(path)
        self.text_column = text_column
        self.label_column = label_column

        self._samples: List[Sample] = []
        self._load()

    def _load(self):
        """Load samples from CSV."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                text = row.get(self.text_column, "")
                label = row.get(self.label_column)

                # Parse boolean labels
                if label in ("TRUE", "True", "true", "1"):
                    label = True
                elif label in ("FALSE", "False", "false", "0"):
                    label = False

                self._samples.append(
                    Sample(
                        idx=idx, text=text.strip(), label=label, metadata={"source": str(self.path)}
                    )
                )

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


@Registry.register_dataset("pediatrics")
class PediatricsDataset(BaseDataset):
    """
    Dataset from CSV file with pediatrics clinical scenarios.

    Uses data/Pediatrics_Synthetic_Data.csv with 100 general pediatrics cases.
    """

    def __init__(
        self,
        name: str = "pediatrics",
        path: str = "data/Pediatrics_Synthetic_Data.csv",
        text_column: str = "Scenario",
        label_column: str = "Diagnosis",
        **kwargs,
    ):
        self._name = name
        self.path = Path(path)
        self.text_column = text_column
        self.label_column = label_column

        self._samples: List[Sample] = []
        self._load()

    def _load(self):
        """Load samples from CSV."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                text = row.get(self.text_column, "")
                label = row.get(self.label_column, "")
                age_group = row.get("Age_Group", "")
                category = row.get("Category", "")

                self._samples.append(
                    Sample(
                        idx=idx,
                        text=text.strip(),
                        label=label.strip(),
                        metadata={
                            "age_group": age_group,
                            "category": category,
                            "source": str(self.path),
                        },
                    )
                )

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


@Registry.register_dataset("synthetic")
class SyntheticMedicalDataset(BaseDataset):
    """
    Synthetic dataset for testing CoT experiments.

    Loads medical scenarios from data/Synthetic_Medical_Data.csv.
    """

    def __init__(
        self,
        name: str = "synthetic",
        path: str = "data/Synthetic_Medical_Data.csv",
        repeat: int = 1,
        **kwargs,
    ):
        self._name = name
        self.path = Path(path)
        self.repeat = repeat
        self._samples: List[Sample] = []
        self._load()

    def _load(self):
        """Load samples from CSV."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        scenarios = []
        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenarios.append(row)

        for r in range(self.repeat):
            for idx, scenario in enumerate(scenarios):
                keywords = scenario.get("Reasoning_Keywords", "")
                self._samples.append(
                    Sample(
                        idx=r * len(scenarios) + idx,
                        text=scenario.get("Scenario", ""),
                        label=scenario.get("Expected_Answer", ""),
                        metadata={
                            "reasoning_keywords": keywords.split(",") if keywords else [],
                            "expected_answer": scenario.get("Expected_Answer", ""),
                        },
                    )
                )

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


@Registry.register_dataset("patching_pairs")
class PatchingPairsDataset(BaseDataset):
    """
    Dataset of clean/corrupted prompt pairs for activation patching.

    Loads pairs from data/Patching_Pairs_Data.csv.
    """

    def __init__(
        self,
        name: str = "patching_pairs",
        path: str = "data/Patching_Pairs_Data.csv",
        **kwargs,
    ):
        self._name = name
        self.path = Path(path)
        self._samples: List[Sample] = []
        self._load()

    def _load(self):
        """Load samples from CSV."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                self._samples.append(
                    Sample(
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
                )

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]
