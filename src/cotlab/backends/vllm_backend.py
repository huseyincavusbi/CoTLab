"""vLLM backend for high-throughput inference."""

from typing import List, Optional
import torch

from .base import InferenceBackend
from ..core.base import GenerationOutput
from ..core.registry import Registry


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
        max_model_len: int = 4096,
        trust_remote_code: bool = True,
        **kwargs
    ):
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._model_name: Optional[str] = None
    
    def load_model(self, model_name: str, **kwargs) -> None:
        """Load model with vLLM."""
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vLLM not installed. Run: pip install vllm")
        
        self._model = LLM(
            model=model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
            **kwargs
        )
        self._model_name = model_name
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
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
        **kwargs
    ) -> List[GenerationOutput]:
        """Generate from multiple prompts efficiently."""
        from vllm import SamplingParams
        
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        outputs = self._model.generate(prompts, sampling_params)
        
        return [
            GenerationOutput(
                text=output.outputs[0].text,
                tokens=list(output.outputs[0].token_ids),
                logprobs=None  # Can be enabled in SamplingParams if needed
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
