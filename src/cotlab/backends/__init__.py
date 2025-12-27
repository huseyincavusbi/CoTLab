"""Inference backends module."""

from .base import InferenceBackend
from .transformers_backend import TransformersBackend

# vLLM is optional (requires pip install cotlab[cuda])
try:
    from .vllm_backend import VLLMBackend
except ImportError:
    VLLMBackend = None  # type: ignore

__all__ = [
    "InferenceBackend",
    "VLLMBackend",
    "TransformersBackend",
]
