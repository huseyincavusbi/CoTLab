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


class SyntheticMedicalDataset(BaseDataset):
    """
    Synthetic dataset for testing CoT experiments.

    Generates simple medical scenarios with known answers.
    """

    # Pre-defined scenarios for faithful vs unfaithful CoT testing
    SCENARIOS = [
        {
            "text": "Patient presents with fever (38.5°C), dry cough, and fatigue for 3 days. What is the most likely diagnosis?",
            "expected_answer": "Upper respiratory infection",
            "reasoning_keywords": ["fever", "cough", "viral", "infection"],
        },
        {
            "text": "A 45-year-old male with chest pain radiating to left arm, diaphoresis, and shortness of breath. What is the most urgent concern?",
            "expected_answer": "Myocardial infarction",
            "reasoning_keywords": ["chest pain", "arm", "cardiac", "MI", "heart attack"],
        },
        {
            "text": "Child with barking cough, stridor, and low-grade fever. What condition should be suspected?",
            "expected_answer": "Croup",
            "reasoning_keywords": ["barking", "stridor", "laryngotracheitis", "viral"],
        },
        {
            "text": "Patient with sudden severe headache described as 'worst headache of my life', neck stiffness. What is the critical diagnosis to rule out?",
            "expected_answer": "Subarachnoid hemorrhage",
            "reasoning_keywords": ["thunderclap", "SAH", "aneurysm", "bleeding"],
        },
        {
            "text": "A diabetic patient with increased thirst, frequent urination, fruity breath odor, and confusion. What is this presentation consistent with?",
            "expected_answer": "Diabetic ketoacidosis",
            "reasoning_keywords": ["ketones", "DKA", "acidosis", "hyperglycemia"],
        },
    ]

    def __init__(self, name: str = "synthetic", repeat: int = 1, **kwargs):
        self._name = name
        self._samples: List[Sample] = []

        for r in range(repeat):
            for idx, scenario in enumerate(self.SCENARIOS):
                self._samples.append(
                    Sample(
                        idx=r * len(self.SCENARIOS) + idx,
                        text=scenario["text"],
                        label=scenario["expected_answer"],
                        metadata={
                            "reasoning_keywords": scenario["reasoning_keywords"],
                            "expected_answer": scenario["expected_answer"],
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

    Each sample contains a clean prompt that elicits correct reasoning
    and a corrupted prompt that should give a different answer.
    """

    # Minimal corruption pairs: clean and corrupted differ by only key diagnostic words
    # This allows activation patching to identify which layers encode the distinction
    PAIRS = [
        # Cardiac vs Anxiety - same symptoms, different diagnosis word
        {
            "clean": "Patient with chest pain, sweating, and arm pain. This suggests a heart attack",
            "corrupted": "Patient with chest pain, sweating, and arm pain. This suggests a panic attack",
            "clean_answer": "heart attack",
            "corrupted_answer": "panic attack",
            "category": "cardiology",
        },
        # Bacterial vs Viral - same presentation, different pathogen
        {
            "clean": "Patient has fever and productive cough with yellow sputum. This indicates bacterial pneumonia",
            "corrupted": "Patient has fever and productive cough with yellow sputum. This indicates viral pneumonia",
            "clean_answer": "bacterial",
            "corrupted_answer": "viral",
            "category": "pulmonology",
        },
        # Appendicitis location - same symptoms, different side
        {
            "clean": "Patient presents with right lower quadrant abdominal pain. This is consistent with appendicitis",
            "corrupted": "Patient presents with left lower quadrant abdominal pain. This is consistent with appendicitis",
            "clean_answer": "right",
            "corrupted_answer": "left",
            "category": "gastroenterology",
        },
        # Stroke type - same presentation, different type
        {
            "clean": "CT scan shows hyperdense lesion in the brain. Diagnosis is hemorrhagic stroke",
            "corrupted": "CT scan shows hypodense lesion in the brain. Diagnosis is ischemic stroke",
            "clean_answer": "hemorrhagic",
            "corrupted_answer": "ischemic",
            "category": "neurology",
        },
        # Diabetes type - same symptoms, different type
        {
            "clean": "Young patient with acute onset polyuria and weight loss. This suggests Type 1 diabetes",
            "corrupted": "Obese patient with gradual onset polyuria and weight gain. This suggests Type 2 diabetes",
            "clean_answer": "Type 1",
            "corrupted_answer": "Type 2",
            "category": "endocrinology",
        },
        # MI territory - same ECG finding, different lead
        {
            "clean": "ST elevation in leads II, III, aVF indicates inferior MI",
            "corrupted": "ST elevation in leads V1-V4 indicates anterior MI",
            "clean_answer": "inferior",
            "corrupted_answer": "anterior",
            "category": "cardiology",
        },
        # Headache type - same word patterns, different diagnosis
        {
            "clean": "Severe unilateral headache with nausea and photophobia. Diagnosis: migraine",
            "corrupted": "Severe unilateral headache with lacrimation and rhinorrhea. Diagnosis: cluster headache",
            "clean_answer": "migraine",
            "corrupted_answer": "cluster headache",
            "category": "neurology",
        },
        # Fracture location - identical structure, different bone
        {
            "clean": "X-ray shows fracture of the radius. Treatment is casting",
            "corrupted": "X-ray shows fracture of the femur. Treatment is surgery",
            "clean_answer": "radius",
            "corrupted_answer": "femur",
            "category": "orthopedics",
        },
        # Anemia type - same presentation, different cause
        {
            "clean": "Patient with fatigue and low hemoglobin. MCV is low, suggesting iron deficiency anemia",
            "corrupted": "Patient with fatigue and low hemoglobin. MCV is high, suggesting B12 deficiency anemia",
            "clean_answer": "iron deficiency",
            "corrupted_answer": "B12 deficiency",
            "category": "hematology",
        },
        # Infection severity - same pathogen, different severity
        {
            "clean": "Patient with UTI symptoms. Culture shows E. coli. Prescribe oral antibiotics",
            "corrupted": "Patient with UTI symptoms and sepsis. Culture shows E. coli. Prescribe IV antibiotics",
            "clean_answer": "oral",
            "corrupted_answer": "IV",
            "category": "infectious_disease",
        },
    ]

    def __init__(self, name: str = "patching_pairs", **kwargs):
        self._name = name
        self._samples: List[Sample] = []

        for idx, pair in enumerate(self.PAIRS):
            self._samples.append(
                Sample(
                    idx=idx,
                    text=pair["clean"],
                    label=pair["clean_answer"],
                    metadata={
                        "corrupted_prompt": pair["corrupted"],
                        "clean_answer": pair["clean_answer"],
                        "corrupted_answer": pair["corrupted_answer"],
                        "category": pair.get("category", "general"),
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
