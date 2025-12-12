"""Transformers backend with activation hook support."""

import os
from typing import List, Optional, Tuple

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core.base import GenerationOutput
from ..core.registry import Registry
from ..patching.cache import ActivationCache
from ..patching.hooks import HookManager
from .base import InferenceBackend

# Load .env file
load_dotenv()


@Registry.register_backend("transformers")
class TransformersBackend(InferenceBackend):
    """
    HuggingFace Transformers backend with full activation access.

    Best for:
    - Activation patching experiments
    - Mechanistic interpretability
    - When you need access to intermediate states

    Features:
        - Full activation extraction via forward hooks
        - Activation patching support
        - Layer-wise caching
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "bfloat16",
        enable_hooks: bool = True,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.enable_hooks = enable_hooks
        self.trust_remote_code = trust_remote_code

        self._model = None
        self._tokenizer = None
        self._model_name: Optional[str] = None
        self._hook_manager: Optional[HookManager] = None

    def load_model(self, model_name: str, **kwargs) -> None:
        """Load model with HuggingFace Transformers."""
        # Get HF token from environment
        hf_token = os.getenv("HF_TOKEN")

        # Print device info
        print(f"  Device: {self.device}")
        print(f"  Dtype: {self.dtype}")
        print("  Cache: ~/.cache/huggingface (HF default)")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=self.trust_remote_code, token=hf_token
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
            token=hf_token,
            **kwargs,
        )
        self._model.eval()
        self._model_name = model_name

        if self.enable_hooks:
            self._hook_manager = HookManager(self._model)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> GenerationOutput:
        """Generate from a single prompt."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode only the new tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return GenerationOutput(text=text, tokens=generated_tokens.tolist(), logprobs=None)

    def generate_batch(
        self, prompts: List[str], max_new_tokens: int = 512, temperature: float = 0.7, **kwargs
    ) -> List[GenerationOutput]:
        """Generate from multiple prompts (sequential for simplicity)."""
        return [self.generate(prompt, max_new_tokens, temperature, **kwargs) for prompt in prompts]

    def generate_with_cache(
        self, prompt: str, layers: Optional[List[int]] = None, max_new_tokens: int = 512, **kwargs
    ) -> Tuple[GenerationOutput, ActivationCache]:
        """
        Generate while caching activations for patching experiments.

        Args:
            prompt: Input prompt
            layers: Which layers to cache (None = all)
            max_new_tokens: Max tokens to generate

        Returns:
            Tuple of (GenerationOutput, ActivationCache)
        """
        if self._hook_manager is None:
            raise RuntimeError("Hooks not enabled. Set enable_hooks=True.")

        cache = ActivationCache()
        self._hook_manager.register_cache_hooks(cache, layers=layers)

        try:
            output = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        finally:
            self._hook_manager.remove_all_hooks()

        return output, cache

    def forward_with_cache(
        self, prompt: str, layers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, ActivationCache]:
        """
        Run forward pass (no generation) and cache activations.

        Useful for patching experiments where you need activations
        without the autoregressive generation loop.

        Args:
            prompt: Input prompt
            layers: Which layers to cache

        Returns:
            Tuple of (logits, ActivationCache)
        """
        if self._hook_manager is None:
            raise RuntimeError("Hooks not enabled. Set enable_hooks=True.")

        cache = ActivationCache()
        self._hook_manager.register_cache_hooks(cache, layers=layers)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        try:
            with torch.no_grad():
                outputs = self._model(**inputs)
        finally:
            self._hook_manager.remove_all_hooks()

        return outputs.logits, cache

    @property
    def supports_activations(self) -> bool:
        return True

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def hook_manager(self) -> Optional[HookManager]:
        return self._hook_manager

    @property
    def num_layers(self) -> int:
        """Get number of transformer layers."""
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        return getattr(self._model.config, "num_hidden_layers", None) or getattr(
            self._model.config, "num_layers", 0
        )

    def unload(self) -> None:
        """Free GPU memory."""
        if self._hook_manager is not None:
            self._hook_manager.remove_all_hooks()
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        torch.cuda.empty_cache()
