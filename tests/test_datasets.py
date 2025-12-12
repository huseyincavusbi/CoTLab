"""Tests for dataset loaders."""

from cotlab.datasets import (
    PatchingPairsDataset,
    Sample,
    SyntheticMedicalDataset,
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
        # Should have the pre-defined scenarios
        assert len(dataset) == 5

    def test_repeat_multiplies_samples(self):
        dataset = SyntheticMedicalDataset(repeat=3)
        assert len(dataset) == 15

    def test_getitem(self):
        dataset = SyntheticMedicalDataset()
        sample = dataset[0]
        assert isinstance(sample, Sample)
        assert len(sample.text) > 0

    def test_samples_have_metadata(self):
        dataset = SyntheticMedicalDataset()
        sample = dataset[0]
        assert "reasoning_keywords" in sample.metadata
        assert "expected_answer" in sample.metadata

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
