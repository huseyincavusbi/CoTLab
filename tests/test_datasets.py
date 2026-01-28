"""Tests for dataset loaders."""

import json

from cotlab.datasets import (
    HistopathologyDataset,
    MedQADataset,
    MMLUMedicalDataset,
    OncologyDataset,
    PatchingPairsDataset,
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

    def test_creation(self):
        dataset = SyntheticMedicalDataset()
        assert dataset.name == "synthetic"
        assert len(dataset) > 0

    def test_has_expected_samples(self):
        dataset = SyntheticMedicalDataset()
        # Should have 100 scenarios from CSV
        assert len(dataset) == 100

    def test_repeat_multiplies_samples(self):
        dataset = SyntheticMedicalDataset(repeat=3)
        assert len(dataset) == 300

    def test_getitem(self):
        dataset = SyntheticMedicalDataset()
        sample = dataset[0]
        assert isinstance(sample, Sample)
        assert len(sample.text) > 0

    def test_samples_have_metadata(self):
        dataset = SyntheticMedicalDataset()
        sample = dataset[0]
        assert "reasoning_keywords" in sample.metadata

    def test_iteration(self):
        dataset = SyntheticMedicalDataset()
        samples = list(dataset)
        assert len(samples) == len(dataset)

    def test_sample_method(self):
        dataset = SyntheticMedicalDataset(repeat=2)
        sampled = dataset.sample(n=3, seed=42)
        assert len(sampled) == 3


class TestPatchingPairsDataset:
    """Tests for PatchingPairsDataset."""

    def test_creation(self):
        dataset = PatchingPairsDataset()
        assert dataset.name == "patching_pairs"

    def test_has_pairs(self):
        dataset = PatchingPairsDataset()
        assert len(dataset) >= 3

    def test_samples_have_corrupted_prompt(self):
        dataset = PatchingPairsDataset()
        sample = dataset[0]
        assert "corrupted_prompt" in sample.metadata
        assert len(sample.metadata["corrupted_prompt"]) > 0

    def test_clean_and_corrupted_different(self):
        dataset = PatchingPairsDataset()
        sample = dataset[0]
        clean = sample.text
        corrupted = sample.metadata["corrupted_prompt"]
        assert clean != corrupted

    def test_expected_answers_present(self):
        dataset = PatchingPairsDataset()
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
