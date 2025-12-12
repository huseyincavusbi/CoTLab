"""Abstract base class for inference backends."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..core.base import GenerationOutput


class InferenceBackend(ABC):
    """Abstract interface for model inference backends."""

    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> None:
        """
        Load model into memory.

        Args:
            model_name: HuggingFace model name or path
            **kwargs: Additional model loading arguments
        """
        ...

    @abstractmethod
    def generate(
        self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, **kwargs
    ) -> GenerationOutput:
        """
        Generate text from a single prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            GenerationOutput with text and tokens
        """
        ...

    @abstractmethod
    def generate_batch(
        self, prompts: List[str], max_new_tokens: int = 512, temperature: float = 0.7, **kwargs
    ) -> List[GenerationOutput]:
        """
        Generate text from multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            List of GenerationOutput
        """
        ...

    @property
    @abstractmethod
    def supports_activations(self) -> bool:
        """Whether this backend supports activation extraction."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> Optional[str]:
        """Currently loaded model name."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Free GPU memory and unload model."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
