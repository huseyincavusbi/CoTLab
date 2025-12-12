"""Base classes and data structures for the CoT research framework."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class GenerationOutput:
    """Standard output format for model generation."""

    text: str
    tokens: List[int]
    logprobs: Optional[List[float]] = None

    def __repr__(self) -> str:
        return f"GenerationOutput(text={self.text[:50]}..., tokens={len(self.tokens)})"


@dataclass
class ExperimentResult:
    """JSON-serializable experiment result."""

    experiment_name: str
    model_name: str
    prompt_strategy: str
    metrics: Dict[str, Any]
    raw_outputs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.__dict__, indent=2, default=str)

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "ExperimentResult":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class BasePromptStrategy(ABC):
    """Abstract base class for prompt construction strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        ...

    @abstractmethod
    def build_prompt(self, input_data: Dict[str, Any]) -> str:
        """
        Build a prompt from input data.

        Args:
            input_data: Dictionary with at least 'question' key

        Returns:
            Formatted prompt string
        """
        ...

    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse model response to extract answer and reasoning.

        Args:
            response: Raw model output

        Returns:
            Dictionary with 'answer', 'reasoning', and any other extracted fields
        """
        ...

    def get_system_message(self) -> Optional[str]:
        """Return system message if applicable."""
        return None


class BaseExperiment(ABC):
    """Abstract base class for experiments."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Experiment name for logging."""
        ...

    @abstractmethod
    def run(
        self,
        backend: "InferenceBackend",
        dataset: Any,
        prompt_strategy: BasePromptStrategy,
        **kwargs,
    ) -> ExperimentResult:
        """
        Run the experiment.

        Args:
            backend: Inference backend (vLLM or Transformers)
            dataset: Dataset to run on
            prompt_strategy: How to construct prompts

        Returns:
            ExperimentResult with metrics and outputs
        """
        ...

    def validate_backend(self, backend: "InferenceBackend") -> None:
        """Check if backend supports required features."""
        pass
