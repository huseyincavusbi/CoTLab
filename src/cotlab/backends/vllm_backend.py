"""vLLM backend for high-throughput inference.

Supports multiple platforms:
- CUDA (NVIDIA GPUs) - standard vLLM
- ROCm (AMD GPUs) - via ROCm Docker or HIP
- Metal (Apple Silicon) - via vllm-metal plugin
"""

import os
import platform as plat
from typing import List, Optional

import torch

from ..core.base import GenerationOutput
from ..core.registry import Registry
from .base import InferenceBackend

# Fix CUDA forking issue: vLLM's V1 engine uses multiprocessing, but when
# Hydra/PyTorch have already initialized CUDA, forking fails. Setting spawn
# method creates fresh Python processes instead.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _is_apple_silicon() -> bool:
    """Detect if running on Apple Silicon."""
    return plat.system() == "Darwin" and plat.processor() == "arm"


def _detect_platform() -> str:
    """Detect the GPU platform.

    Returns:
        "metal" for Apple Silicon
        "cuda" for NVIDIA/AMD GPUs (ROCm uses CUDA-compatible API)
        "cpu" if no GPU available
    """
    if _is_apple_silicon():
        return "metal"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@Registry.register_backend("vllm")
class VLLMBackend(InferenceBackend):
    """
    High-throughput inference backend using vLLM.

    Best for:
    - Large-scale experiments (1000+ samples)
    - Batch inference
    - When activation access is not needed

    Platforms:
    - CUDA: Standard vLLM (pip install vllm)
    - ROCm: vLLM in ROCm Docker container
    - Metal: vLLM-Metal plugin (install via curl script)

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
        platform: str = "auto",
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

        # Platform detection
        self._platform = _detect_platform() if platform == "auto" else platform
        self._setup_platform_env()

    def _setup_platform_env(self) -> None:
        """Configure environment variables for the detected platform."""
        if self._platform == "metal":
            # vLLM-Metal configuration
            os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", str(self.gpu_memory_utilization))
            os.environ.setdefault("VLLM_METAL_USE_MLX", "1")
            os.environ.setdefault("VLLM_METAL_BLOCK_SIZE", "16")
            print("  Platform: Apple Silicon (Metal/MLX)")
        elif self._platform == "cuda":
            print("  Platform: CUDA")
        else:
            print(f"  Platform: {self._platform}")

    def load_model(self, model_name: str, **kwargs) -> None:
        """Load model with vLLM."""
        try:
            from vllm import LLM
        except ImportError:
            if self._platform == "metal":
                raise ImportError(
                    "vLLM not found. On Apple Silicon, install vllm-metal:\n"
                    "curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash"
                )
            raise ImportError("vLLM not installed. Run: pip install vllm")

        # Build LLM kwargs
        llm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "enforce_eager": self.enforce_eager,
            **kwargs,
        }

        # gpu_memory_utilization only applies to CUDA/ROCm
        if self._platform != "metal":
            llm_kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization

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
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> GenerationOutput:
        """Generate from a single prompt."""
        outputs = self.generate_batch(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            **kwargs,
        )
        return outputs[0]

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[GenerationOutput]:
        """Generate from multiple prompts efficiently."""
        from vllm import SamplingParams

        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prompts = self._apply_system_prompt(prompts, system_prompt)

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

    @staticmethod
    def _apply_system_prompt(prompts: List[str], system_prompt: Optional[str]) -> List[str]:
        if not system_prompt:
            return prompts
        system_prompt = system_prompt.strip()
        if not system_prompt:
            return prompts
        return [f"{system_prompt}\n\n{prompt}" for prompt in prompts]

    @property
    def platform(self) -> str:
        """Return the detected platform (cuda, metal, cpu)."""
        return self._platform

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

        # Platform-specific cleanup
        if self._platform != "metal" and torch.cuda.is_available():
            torch.cuda.empty_cache()
