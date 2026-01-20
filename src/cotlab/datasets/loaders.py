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
        self.path = self._resolve_path_from_registry(name, path)
        self._samples: List[Sample] = []
        self._load()

    def _resolve_path_from_registry(self, name: str, default_path: str) -> Path:
        """Resolve path from data/datasets.yaml registry or fallback to default."""
        import yaml
        from huggingface_hub import hf_hub_download

        # Locate registry relative to this file (src/cotlab/datasets/loaders.py -> root/data/datasets.yaml)
        # root is 3 levels up from this file's directory: src/cotlab/datasets -> src/cotlab -> src -> root
        root_dir = Path(__file__).parent.parent.parent.parent
        registry_path = root_dir / "data/datasets.yaml"

        if not registry_path.exists():
            # Fallback: try CWD
            registry_path = Path("data/datasets.yaml")
            if not registry_path.exists():
                return Path(default_path)

        try:
            with open(registry_path, "r") as f:
                config = yaml.safe_load(f)

            # 1. Get Repo ID
            ds_config = config.get("datasets", {}).get(name, {})
            repo_id = ds_config.get("repo_id", config.get("default", {}).get("repo_id"))

            if not repo_id:
                return Path(default_path)

            # 2. Determine Filename
            # If explicit path in registry, use it.
            # Otherwise, infer from default locally-styled path (e.g. data/radiology.json -> radiology.json)
            filename = ds_config.get("path")
            if not filename:
                # heuristic: strip 'data/' prefix if present to map to HF root
                p = Path(default_path)
                if "data" in p.parts:
                    # e.g. data/foo -> foo, data/tcga/foo -> tcga/foo
                    try:
                        filename = str(p.relative_to("data"))
                    except ValueError:
                        filename = p.name
                else:
                    filename = p.name

            # 3. Download
            try:
                cached_path = hf_hub_download(
                    repo_id=repo_id, filename=filename, repo_type="dataset"
                )
                return Path(cached_path)
            except Exception as e:
                print(
                    f"Warning: Failed to download {name} ({filename}) from HF repo {repo_id}: {e}"
                )
                pass
        except Exception as e:
            print(f"Warning: Failed to load registry: {e}")

        return Path(default_path)

    def _load(self):
        """Load samples from JSON or CSV file based on extension."""
        if not self.path.exists():
            # If resolved path doesn't exist, try resolving as relative to cwd
            if Path(self.path.name).exists():
                self.path = Path(self.path.name)
            else:
                # One last attempt: maybe it's in data/
                p = Path("data") / self.path.name
                if p.exists():
                    self.path = p
                else:
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


@Registry.register_dataset("neurology")
class NeurologyDataset(JSONDataset):
    """Neurology reports dataset for neurological abnormality detection."""

    def __init__(
        self,
        name: str = "neurology",
        path: str = "data/neurology.json",
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)

    def _parse_item(self, idx: int, item: Dict[str, Any]) -> Sample:
        return Sample(
            idx=idx,
            text=item["input"]["report"],
            label=item["output"]["neurological_abnormality"],
            metadata=item.get("metadata", {}),
        )

    def get_compatible_prompts(self) -> list[str]:
        """Neurology dataset only works with neurology prompt."""
        return ["neurology"]


@Registry.register_dataset("oncology")
class OncologyDataset(JSONDataset):
    """Oncology reports dataset for malignancy detection."""

    def __init__(
        self,
        name: str = "oncology",
        path: str = "data/oncology.json",
        **kwargs,
    ):
        super().__init__(name, path, **kwargs)

    def _parse_item(self, idx: int, item: Dict[str, Any]) -> Sample:
        return Sample(
            idx=idx,
            text=item["input"]["report"],
            label=item["output"]["malignancy"],
            metadata=item.get("metadata", {}),
        )

    def get_compatible_prompts(self) -> list[str]:
        """Oncology dataset only works with oncology prompt."""
        return ["oncology"]


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


@Registry.register_dataset("histopathology")
class HistopathologyDataset(BaseDataset):
    """Histopathology report quality evaluation dataset.

    Loads HARE human evaluation annotations and expands each row
    into 4 samples (one per model output with its human score).

    Labels: 0 (poor), 1 (partial), 2 (good)
    """

    def __init__(
        self,
        name: str = "histopathology",
        path: str = "data/histopathology.tsv",
        **kwargs,
    ):
        self._name = name
        self.path = Path(path)
        self._samples: List[Sample] = []
        self._load()

    def _load(self):
        """Load TSV and expand to 4 samples per row."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            sample_idx = 0
            for row_idx, row in enumerate(reader):
                ground_truth = row.get("ground_truth", "")
                # Expand 4 model outputs per row
                for model_num in range(4):
                    report_col = str(model_num)
                    score_col = f"Scoring {model_num}"
                    report_text = row.get(report_col, "")
                    score = row.get(score_col, "")

                    # Skip if missing data
                    if not report_text or score == "":
                        continue

                    try:
                        label = int(float(score))
                    except (ValueError, TypeError):
                        continue

                    self._samples.append(
                        Sample(
                            idx=sample_idx,
                            text=report_text,
                            label=label,
                            metadata={
                                "ground_truth": ground_truth,
                                "model_id": model_num,
                                "row_id": row_idx,
                            },
                        )
                    )
                    sample_idx += 1

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]

    def get_compatible_prompts(self) -> list[str]:
        """Histopathology dataset works with histopathology prompt."""
        return ["histopathology"]


@Registry.register_dataset("tcga")
class TCGADataset(BaseDataset):
    """TCGA Cancer Type Classification dataset.

    Input: Pathology report text
    Output: Cancer type (e.g., BRCA, LUAD)

    Implements official 15% stratified test split using patient IDs.
    """

    def __init__(
        self,
        name: str = "tcga",
        repo_id: Optional[str] = None,
        reports_filename: str = "tcga/TCGA_Reports.csv",
        labels_filename: str = "tcga/tcga_patient_to_cancer_type.csv",
        split: str = "test",  # train, test, or all
        random_seed: int = 0,
        **kwargs,
    ):
        self._name = name
        self.repo_id = repo_id
        self.reports_filename = reports_filename
        self.labels_filename = labels_filename
        self.split = split
        self.random_seed = random_seed
        self._samples: List[Sample] = []
        self._load()

    def _load(self):
        import random

        import yaml
        from huggingface_hub import hf_hub_download

        # Resolve Repo ID: Config > Registry > None
        repo_id = self.repo_id
        if not repo_id:
            registry_path = Path("data/datasets.yaml")
            if registry_path.exists():
                try:
                    with open(registry_path, "r") as f:
                        config = yaml.safe_load(f)
                    # Check explicit "tcga" entry or default
                    ds_config = config.get("datasets", {}).get("tcga", {})
                    repo_id = ds_config.get("repo_id", config.get("default", {}).get("repo_id"))
                except Exception as e:
                    print(f"Warning: Failed to load registry for TCGA: {e}")

        # Download or get cached files
        if repo_id:
            try:
                reports_path = hf_hub_download(
                    repo_id=repo_id, filename=self.reports_filename, repo_type="dataset"
                )
                labels_path = hf_hub_download(
                    repo_id=repo_id, filename=self.labels_filename, repo_type="dataset"
                )
                self.path = Path(reports_path)
                self.labels_path = Path(labels_path)
            except Exception as e:
                raise FileNotFoundError(f"Failed to download from HF repo {repo_id}: {e}")
        else:
            # Fallback for legacy initialization (if used without config)
            # This path logic needs to be robust if arguments provided differently
            self.path = Path(self.reports_filename)
            self.labels_path = Path(self.labels_filename)

        if not self.path.exists():
            raise FileNotFoundError(f"Reports not found: {self.path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {self.labels_path}")

        # 1. Load Labels
        labels = {}
        with open(self.labels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    labels[row[0]] = row[1]

        # 2. Load Reports and Link
        data_by_class = {}  # ctype -> list of (patient_id, text)

        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid_full = row.get("patient_filename", "")
                text = row.get("text", "")

                # Extract patient ID (TCGA-XX-YYYY)
                if len(pid_full) < 12:
                    continue
                pid = pid_full[:12]

                if pid in labels:
                    ctype = labels[pid]
                    if ctype not in data_by_class:
                        data_by_class[ctype] = []
                    data_by_class[ctype].append((pid, text))

        # 3. Stratified Split
        final_samples = []

        # Sort classes for deterministic order before shuffling
        sorted_classes = sorted(data_by_class.keys())

        for ctype in sorted_classes:
            items = data_by_class[ctype]

            # Deterministic shuffle
            random.Random(self.random_seed).shuffle(items)

            # Split index (15% test)
            n_total = len(items)
            n_test = int(n_total * 0.15)

            if self.split == "all":
                selected = items
            elif self.split == "test":
                # Test set is the first 15%
                selected = items[:n_test]
            elif self.split == "train":
                selected = items[n_test:]
            else:
                selected = []

            # Add to samples
            for pid, text in selected:
                final_samples.append((pid, text, ctype))

        # 4. Create Sample objects
        for i, (pid, text, ctype) in enumerate(final_samples):
            self._samples.append(
                Sample(
                    idx=i,
                    text=text,
                    label=ctype,
                    metadata={"patient_id": pid, "cancer_type": ctype},
                )
            )

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]

    def get_compatible_prompts(self) -> list[str]:
        return ["tcga"]


@Registry.register_dataset("medqa")
class MedQADataset(BaseDataset):
    """MedQA USMLE-style 4-option MCQ dataset.

    Format: JSONL with fields:
    - question: Clinical vignette
    - options: Dict {"A": "...", "B": "...", "C": "...", "D": "..."}
    - answer_idx: Correct answer letter (A/B/C/D)
    - meta_info: USMLE step (step1, step2, step3)
    """

    def __init__(
        self,
        name: str = "medqa",
        repo_id: Optional[str] = None,
        filename: str = "medqa/test.jsonl",
        split: str = "test",
        **kwargs,
    ):
        self._name = name
        self.repo_id = repo_id
        self.filename = filename
        self.split = split
        self._samples: List[Sample] = []
        self._load()

    def _load(self):
        import json

        import yaml
        from huggingface_hub import hf_hub_download

        # Resolve repo_id from registry if not provided
        repo_id = self.repo_id
        if not repo_id:
            root_dir = Path(__file__).parent.parent.parent.parent
            registry_path = root_dir / "data/datasets.yaml"
            if registry_path.exists():
                try:
                    with open(registry_path, "r") as f:
                        config = yaml.safe_load(f)
                    ds_config = config.get("datasets", {}).get("medqa", {})
                    repo_id = ds_config.get("repo_id", config.get("default", {}).get("repo_id"))
                except Exception as e:
                    print(f"Warning: Failed to load registry for MedQA: {e}")

        if not repo_id:
            raise ValueError("No repo_id found for MedQA dataset")

        # Download from HF
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=self.filename,
                repo_type="dataset",
            )
        except Exception as e:
            raise FileNotFoundError(f"Failed to download MedQA from {repo_id}: {e}")

        # Parse JSONL
        with open(local_path, "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())

                # Format question with options
                question = data["question"]
                options = data["options"]
                formatted_options = "\n".join(
                    f"{key}) {val}" for key, val in sorted(options.items())
                )
                text = f"{question}\n\n{formatted_options}"

                # Answer is the letter (A, B, C, D)
                label = data["answer_idx"]

                self._samples.append(
                    Sample(
                        idx=i,
                        text=text,
                        label=label,
                        metadata={
                            "step": data.get("meta_info", "unknown"),
                            "answer_text": data.get("answer", ""),
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

    def get_compatible_prompts(self) -> list[str]:
        return ["mcq", "medqa"]
