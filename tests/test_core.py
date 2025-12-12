"""Tests for core base classes."""

import json
import tempfile
from pathlib import Path

from cotlab.core.base import (
    ExperimentResult,
    GenerationOutput,
)


class TestGenerationOutput:
    """Tests for GenerationOutput dataclass."""

    def test_creation(self):
        output = GenerationOutput(text="Hello world", tokens=[1, 2, 3], logprobs=[0.1, 0.2, 0.3])
        assert output.text == "Hello world"
        assert output.tokens == [1, 2, 3]
        assert output.logprobs == [0.1, 0.2, 0.3]

    def test_creation_without_logprobs(self):
        output = GenerationOutput(text="Test", tokens=[1])
        assert output.logprobs is None

    def test_repr(self):
        output = GenerationOutput(text="Short text", tokens=[1, 2])
        repr_str = repr(output)
        assert "GenerationOutput" in repr_str


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_creation(self):
        result = ExperimentResult(
            experiment_name="test_exp",
            model_name="test_model",
            prompt_strategy="cot",
            metrics={"accuracy": 0.95},
            raw_outputs=[{"text": "output1"}],
        )
        assert result.experiment_name == "test_exp"
        assert result.metrics["accuracy"] == 0.95

    def test_to_json(self):
        result = ExperimentResult(
            experiment_name="test",
            model_name="model",
            prompt_strategy="direct",
            metrics={"score": 1.0},
        )
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["experiment_name"] == "test"
        assert parsed["metrics"]["score"] == 1.0

    def test_save_and_load(self):
        result = ExperimentResult(
            experiment_name="save_test",
            model_name="model",
            prompt_strategy="cot",
            metrics={"value": 42},
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result.save(f.name)
            loaded = ExperimentResult.load(f.name)

        assert loaded.experiment_name == "save_test"
        assert loaded.metrics["value"] == 42

        # Cleanup
        Path(f.name).unlink()

    def test_timestamp_auto_generated(self):
        result = ExperimentResult(
            experiment_name="ts_test", model_name="model", prompt_strategy="cot", metrics={}
        )
        assert result.timestamp is not None
        assert len(result.timestamp) > 0
