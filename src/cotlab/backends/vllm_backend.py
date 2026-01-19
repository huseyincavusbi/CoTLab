"""vLLM backend for high-throughput inference."""

import os
from typing import List, Optional

import torch

from ..core.base import GenerationOutput
from ..core.registry import Registry
from .base import InferenceBackend

# Fix CUDA forking issue: vLLM's V1 engine uses multiprocessing, but when
# Hydra/PyTorch have already initialized CUDA, forking fails. Setting spawn
# method creates fresh Python processes instead.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@Registry.register_backend("vllm")
class VLLMBackend(InferenceBackend):
    """
    High-throughput inference backend using vLLM.

    Best for:
    - Large-scale experiments (1000+ samples)
    - Batch inference
    - When activation access is not needed

    Note:
        Does NOT support activation extraction or patching.
    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        max_model_len: int | None = None,
        trust_remote_code: bool = True,
        quantization: str | None = None,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        limit_mm_per_prompt: dict | str | None = None,
        **kwargs,
    ):
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self._model = None
        self._model_name: Optional[str] = None

    def load_model(self, model_name: str, **kwargs) -> None:
        """Load model with vLLM."""
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install vllm")

        # Only pass max_model_len if it's explicitly set
        llm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            **kwargs,
        }
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len

        if self.quantization is not None:
            llm_kwargs["quantization"] = self.quantization

        if self.limit_mm_per_prompt is not None:
            llm_kwargs["limit_mm_per_prompt"] = self.limit_mm_per_prompt

        print(f"DEBUG: VLLM args: {llm_kwargs}")
        self._model = LLM(**llm_kwargs)
        self._model_name = model_name

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> GenerationOutput:
        """Generate from a single prompt."""
        outputs = self.generate_batch([prompt], max_new_tokens, temperature, top_p, **kwargs)
        return outputs[0]

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[GenerationOutput]:
        """Generate from multiple prompts efficiently."""
        from vllm import SamplingParams

        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens, temperature=temperature, top_p=top_p, **kwargs
        )

        outputs = self._model.generate(prompts, sampling_params)

        return [
            GenerationOutput(
                text=output.outputs[0].text,
                tokens=list(output.outputs[0].token_ids),
                logprobs=None,  # Can be enabled in SamplingParams if needed
            )
            for output in outputs
        ]

    @property
    def supports_activations(self) -> bool:
        """vLLM optimizes away intermediate activations."""
        return False

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
