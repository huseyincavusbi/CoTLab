"""Tests for dataset loaders."""

import json

import pytest

from cotlab.datasets import (
    HistopathologyDataset,
    MARCDataset,
    MedQADataset,
    MMLUMedicalDataset,
    OncologyDataset,
    PatchingPairsDataset,
    PubHealthBenchDataset,
    PubMedQADataset,
    RadiologyDataset,
    Sample,
    SyntheticMedicalDataset,
    TCGADataset,
)


class TestSample:
    """Tests for Sample dataclass."""

    def test_creation(self):
        sample = Sample(idx=0, text="Test text", label="positive")
        assert sample.idx == 0
        assert sample.text == "Test text"
        assert sample.label == "positive"

    def test_metadata_default(self):
        sample = Sample(idx=0, text="Test")
        assert sample.metadata == {}

    def test_to_dict(self):
        sample = Sample(idx=1, text="Text", label="A", metadata={"key": "value"})
        d = sample.to_dict()
        assert d["idx"] == 1
        assert d["label"] == "A"
        assert d["metadata"]["key"] == "value"


class TestSyntheticMedicalDataset:
    """Tests for SyntheticMedicalDataset."""

    @pytest.fixture
    def synthetic_dataset(self, tmp_path, monkeypatch):
        samples = [
            {
                "input": {"scenario": "Scenario A"},
                "output": {"diagnosis": "Dx A"},
                "metadata": {"reasoning_keywords": "fever"},
            },
            {
                "input": {"scenario": "Scenario B"},
                "output": {"diagnosis": "Dx B"},
                "metadata": {"reasoning_keywords": "cough"},
            },
            {
                "input": {"scenario": "Scenario C"},
                "output": {"diagnosis": "Dx C"},
                "metadata": {"reasoning_keywords": "pain"},
            },
        ]
        path = tmp_path / "synthetic.json"
        path.write_text(json.dumps(samples), encoding="utf-8")

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))
        return SyntheticMedicalDataset(), len(samples)

    def test_creation(self, synthetic_dataset):
        dataset, _ = synthetic_dataset
        assert dataset.name == "synthetic"
        assert len(dataset) > 0

    def test_has_expected_samples(self, synthetic_dataset):
        dataset, base_len = synthetic_dataset
        assert len(dataset) == base_len

    def test_repeat_multiplies_samples(self, synthetic_dataset):
        _, base_len = synthetic_dataset
        dataset = SyntheticMedicalDataset(repeat=3)
        assert len(dataset) == base_len * 3

    def test_getitem(self, synthetic_dataset):
        dataset, _ = synthetic_dataset
        sample = dataset[0]
        assert isinstance(sample, Sample)
        assert len(sample.text) > 0

    def test_samples_have_metadata(self, synthetic_dataset):
        dataset, _ = synthetic_dataset
        sample = dataset[0]
        assert "reasoning_keywords" in sample.metadata

    def test_iteration(self, synthetic_dataset):
        dataset, _ = synthetic_dataset
        samples = list(dataset)
        assert len(samples) == len(dataset)

    def test_sample_method(self, synthetic_dataset):
        dataset, _ = synthetic_dataset
        sampled = dataset.sample(n=3, seed=42)
        assert len(sampled) == 3


class TestPatchingPairsDataset:
    """Tests for PatchingPairsDataset."""

    @pytest.fixture
    def patching_pairs_dataset(self, tmp_path, monkeypatch):
        samples = [
            {
                "clean": {"input": "Clean prompt A", "output": "A"},
                "corrupted": {"input": "Corrupted prompt A", "output": "B"},
                "metadata": {"category": "test"},
            },
            {
                "clean": {"input": "Clean prompt B", "output": "C"},
                "corrupted": {"input": "Corrupted prompt B", "output": "D"},
                "metadata": {"category": "test"},
            },
            {
                "clean": {"input": "Clean prompt C", "output": "E"},
                "corrupted": {"input": "Corrupted prompt C", "output": "F"},
                "metadata": {"category": "test"},
            },
        ]
        path = tmp_path / "patching_pairs.json"
        path.write_text(json.dumps(samples), encoding="utf-8")

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))
        return PatchingPairsDataset()

    def test_creation(self, patching_pairs_dataset):
        dataset = patching_pairs_dataset
        assert dataset.name == "patching_pairs"

    def test_has_pairs(self, patching_pairs_dataset):
        dataset = patching_pairs_dataset
        assert len(dataset) >= 3

    def test_samples_have_corrupted_prompt(self, patching_pairs_dataset):
        dataset = patching_pairs_dataset
        sample = dataset[0]
        assert "corrupted_prompt" in sample.metadata
        assert len(sample.metadata["corrupted_prompt"]) > 0

    def test_clean_and_corrupted_different(self, patching_pairs_dataset):
        dataset = patching_pairs_dataset
        sample = dataset[0]
        clean = sample.text
        corrupted = sample.metadata["corrupted_prompt"]
        assert clean != corrupted

    def test_expected_answers_present(self, patching_pairs_dataset):
        dataset = patching_pairs_dataset
        sample = dataset[0]
        assert "clean_answer" in sample.metadata
        assert "corrupted_answer" in sample.metadata


class TestMedQADataset:
    """Tests for MedQADataset with mocked download."""

    def test_loads_sample(self, monkeypatch, tmp_path):
        sample = {
            "question": "What is the diagnosis?",
            "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
            "answer_idx": "A",
            "meta_info": "step1",
            "answer": "A",
        }
        path = tmp_path / "medqa.jsonl"
        path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))
        dataset = MedQADataset(repo_id="dummy", filename="medqa/test.jsonl")

        assert len(dataset) == 1
        assert dataset[0].label == "A"
        assert "A)" in dataset[0].text


class TestPubMedQADataset:
    """Tests for PubMedQADataset with mocked download."""

    def test_loads_sample(self, monkeypatch, tmp_path):
        sample = {
            "question": "Is this true?",
            "context": "Some abstract text.",
            "answer": "yes",
            "pmid": "123",
        }
        path = tmp_path / "pubmedqa.jsonl"
        path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))
        dataset = PubMedQADataset(repo_id="dummy", filename="pubmedqa/test.jsonl")

        assert len(dataset) == 1
        assert dataset[0].label == "yes"
        assert "Question:" in dataset[0].text


class TestHistopathologyDataset:
    """Tests for HistopathologyDataset with a local TSV."""

    def test_loads_samples(self, tmp_path, monkeypatch):
        tsv = tmp_path / "histopathology.tsv"
        tsv.write_text(
            "ground_truth\t0\tScoring 0\t1\tScoring 1\t2\tScoring 2\t3\tScoring 3\n"
            "GT report\tReport A\t2\t\t\t\t\t\n",
            encoding="utf-8",
        )

        import huggingface_hub

        calls = {}

        def _fake_hf_hub_download(**kwargs):
            calls.update(kwargs)
            return str(tsv)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)

        dataset = HistopathologyDataset(repo_id="dummy")
        assert len(dataset) == 1
        assert dataset[0].label == 2
        assert dataset[0].metadata["ground_truth"] == "GT report"
        assert calls.get("repo_id") == "dummy"
        assert calls.get("repo_type") == "dataset"


class TestMMLUMedicalDataset:
    """Tests for MMLUMedicalDataset with mocked download."""

    def test_loads_sample(self, monkeypatch, tmp_path):
        sample = {
            "question": "Which organ filters blood?",
            "choices": ["Heart", "Liver", "Kidney", "Lung"],
            "answer": 2,
            "subject": "anatomy",
        }
        path = tmp_path / "mmlu.jsonl"
        path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))
        dataset = MMLUMedicalDataset(repo_id="dummy", filename="mmlu/medical_test.jsonl")

        assert len(dataset) == 1
        assert dataset[0].label == "C"
        assert "A)" in dataset[0].text
        assert dataset[0].metadata["subject"] == "anatomy"


class TestTCGADataset:
    """Tests for TCGADataset with mocked CSV files."""

    def test_loads_samples_all_split(self, monkeypatch, tmp_path):
        reports_path = tmp_path / "TCGA_Reports.csv"
        labels_path = tmp_path / "tcga_patient_to_cancer_type.csv"

        reports_path.write_text(
            "patient_filename,text\n"
            "TCGA-AB-1234_report.txt,Report text A\n"
            "TCGA-CD-5678_report.txt,Report text B\n",
            encoding="utf-8",
        )
        labels_path.write_text(
            "patient_id,cancer_type\nTCGA-AB-1234,BRCA\nTCGA-CD-5678,LUAD\n",
            encoding="utf-8",
        )

        import huggingface_hub

        def _fake_download(**kwargs):
            filename = kwargs.get("filename", "")
            if "Reports" in filename:
                return str(reports_path)
            return str(labels_path)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

        dataset = TCGADataset(repo_id="dummy", split="all")
        assert len(dataset) == 2
        assert dataset[0].metadata["patient_id"].startswith("TCGA-")
        assert dataset[0].metadata["cancer_type"] in {"BRCA", "LUAD"}


class TestRadiologyDataset:
    """Tests for RadiologyDataset JSON parsing."""

    def test_loads_sample(self, monkeypatch, tmp_path):
        sample = [
            {
                "input": {"report": "Radiology report text"},
                "output": {"pathological_fracture": True},
                "metadata": {"case_id": "R1"},
            }
        ]
        path = tmp_path / "radiology.json"
        path.write_text(json.dumps(sample), encoding="utf-8")

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))
        dataset = RadiologyDataset(path="data/radiology.json")

        assert len(dataset) == 1
        assert dataset[0].label is True
        assert dataset[0].metadata["case_id"] == "R1"


class TestOncologyDataset:
    """Tests for OncologyDataset JSON parsing."""

    def test_loads_sample(self, monkeypatch, tmp_path):
        sample = [
            {
                "input": {"report": "Oncology report text"},
                "output": {"malignancy": False},
                "metadata": {"case_id": "O1"},
            }
        ]
        path = tmp_path / "oncology.json"
        path.write_text(json.dumps(sample), encoding="utf-8")

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))
        dataset = OncologyDataset(path="data/oncology.json")

        assert len(dataset) == 1
        assert dataset[0].label is False
        assert dataset[0].metadata["case_id"] == "O1"


class TestCardiologyDataset:
    """Tests for CardiologyDataset JSON parsing."""

    def test_loads_sample(self, monkeypatch, tmp_path):
        sample = [
            {
                "input": {"report": "Cardiology report text"},
                "output": {"congenital_heart_defect": True},
                "metadata": {"case_id": "C1"},
            }
        ]
        path = tmp_path / "cardiology.json"
        path.write_text(json.dumps(sample), encoding="utf-8")

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))
        from cotlab.datasets import CardiologyDataset

        dataset = CardiologyDataset(path="data/cardiology.json")

        assert len(dataset) == 1
        assert dataset[0].label is True
        assert dataset[0].metadata["case_id"] == "C1"


class TestNeurologyDataset:
    """Tests for NeurologyDataset JSON parsing."""

    def test_loads_sample(self, monkeypatch, tmp_path):
        sample = [
            {
                "input": {"report": "Neurology report text"},
                "output": {"neurological_abnormality": False},
                "metadata": {"case_id": "N1"},
            }
        ]
        path = tmp_path / "neurology.json"
        path.write_text(json.dumps(sample), encoding="utf-8")

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))
        from cotlab.datasets import NeurologyDataset

        dataset = NeurologyDataset(path="data/neurology.json")

        assert len(dataset) == 1
        assert dataset[0].label is False
        assert dataset[0].metadata["case_id"] == "N1"


class TestPediatricsDataset:
    """Tests for PediatricsDataset JSON parsing."""

    def test_loads_sample(self, monkeypatch, tmp_path):
        sample = [
            {
                "input": {"scenario": "Peds scenario"},
                "output": {"diagnosis": "Dx"},
                "metadata": {"case_id": "P1"},
            }
        ]
        path = tmp_path / "pediatrics.json"
        path.write_text(json.dumps(sample), encoding="utf-8")

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))
        from cotlab.datasets import PediatricsDataset

        dataset = PediatricsDataset(path="data/pediatrics.json")

        assert len(dataset) == 1
        assert dataset[0].label == "Dx"
        assert dataset[0].metadata["case_id"] == "P1"


class TestProbingDiagnosisDataset:
    """Tests for ProbingDiagnosisDataset JSON parsing."""

    def test_loads_sample(self, monkeypatch, tmp_path):
        sample = [
            {
                "input": {"question": "Case question"},
                "output": {"diagnosis": "Dx"},
                "metadata": {"category": "cardio", "difficulty": "easy"},
            }
        ]
        path = tmp_path / "probing_diagnosis.json"
        path.write_text(json.dumps(sample), encoding="utf-8")

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))
        from cotlab.datasets import ProbingDiagnosisDataset

        dataset = ProbingDiagnosisDataset(path="data/probing_diagnosis.json")

        assert len(dataset) == 1
        assert dataset[0].label == "Dx"
        assert dataset[0].metadata["category"] == "cardio"


class TestPubHealthBenchDataset:
    """Tests for PubHealthBenchDataset with mocked parquet."""

    def test_loads_sample(self, monkeypatch, tmp_path):
        pyarrow = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        data = {
            "question": ["What is the guidance?"],
            "options": [["Opt A", "Opt B"]],
            "options_formatted": ["A. Opt A\nB. Opt B"],
            "answer_index": [0],
            "answer": ["A"],
            "category": ["general"],
        }
        table = pyarrow.table(data)
        path = tmp_path / "pubhealthbench.parquet"
        pq.write_table(table, path)

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))
        dataset = PubHealthBenchDataset(
            repo_id="dummy",
            filename="pubhealthbench/test-00000-of-00001.parquet",
            split="test",
        )

        assert len(dataset) == 1
        assert dataset[0].label == "A"
        assert "A. Opt A" in dataset[0].text


class TestMARCDataset:
    """Tests for MARCDataset with mocked parquet."""

    def test_loads_sample(self, monkeypatch, tmp_path):
        pyarrow = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        data = {
            "question_id": ["ID001"],
            "question": ["What is the best next step?"],
            "options": [
                {
                    "A": "Option A",
                    "B": "Option B",
                    "C": "",
                    "D": "",
                    "E": "Option E",
                    "F": "",
                    "G": "",
                }
            ],
            "answer": ["E"],
            "src": ["emergency"],
        }
        table = pyarrow.table(data)
        path = tmp_path / "m_arc.parquet"
        pq.write_table(table, path)

        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(path))
        dataset = MARCDataset(
            repo_id="dummy",
            filename="m_arc/test-00000-of-00001.parquet",
            split="test",
        )

        assert len(dataset) == 1
        assert dataset[0].label == "E"
        assert "A) Option A" in dataset[0].text
        assert "E) Option E" in dataset[0].text
