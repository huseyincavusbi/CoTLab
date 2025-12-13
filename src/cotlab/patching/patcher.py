"""Activation patcher for causal intervention experiments."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .cache import ActivationCache
from .hooks import HookManager


@dataclass
class PatchingResult:
    """Result from a patching operation."""

    output_text: str
    patched_layers: List[int]
    patched_positions: Optional[List[int]]
    original_answer: Optional[str] = None
    patched_answer: Optional[str] = None

    @property
    def answer_changed(self) -> bool:
        """Check if patching changed the answer."""
        if self.original_answer is None or self.patched_answer is None:
            return False
        return self.original_answer.strip() != self.patched_answer.strip()


@dataclass
class ForwardPatchResult:
    """Result from forward-only patching (no generation)."""

    layer_idx: int
    clean_logits: torch.Tensor
    corrupted_logits: torch.Tensor
    patched_logits: torch.Tensor
    effect_score: float  # How much patching recovered clean behavior

    @property
    def kl_from_clean(self) -> float:
        """KL divergence of patched from clean distribution."""
        clean_probs = torch.softmax(self.clean_logits[:, -1, :], dim=-1)
        patched_probs = torch.softmax(self.patched_logits[:, -1, :], dim=-1)
        return torch.nn.functional.kl_div(
            patched_probs.log(), clean_probs, reduction="batchmean"
        ).item()

    @property
    def kl_from_corrupted(self) -> float:
        """KL divergence of patched from corrupted distribution."""
        corrupted_probs = torch.softmax(self.corrupted_logits[:, -1, :], dim=-1)
        patched_probs = torch.softmax(self.patched_logits[:, -1, :], dim=-1)
        return torch.nn.functional.kl_div(
            patched_probs.log(), corrupted_probs, reduction="batchmean"
        ).item()


class ActivationPatcher:
    """
    Perform activation patching interventions for causal analysis.

    Activation patching is a technique to test causal importance of
    specific model components by replacing activations from one run
    with activations from another run.

    Example:
        >>> patcher = ActivationPatcher(backend)
        >>> clean_out, clean_cache = backend.generate_with_cache(clean_prompt)
        >>> results = patcher.patch_run(
        ...     corrupted_prompt,
        ...     source_cache=clean_cache,
        ...     target_layers=[5, 10, 15]
        ... )
    """

    def __init__(self, backend: "TransformersBackend"):
        """
        Initialize patcher with a backend.

        Args:
            backend: TransformersBackend with hook support
        """
        if not backend.supports_activations:
            raise ValueError("Backend does not support activation access")

        self.backend = backend

    @property
    def hook_manager(self) -> HookManager:
        return self.backend.hook_manager

    def patch_run(
        self,
        prompt: str,
        source_cache: ActivationCache,
        target_layers: List[int],
        token_positions: Optional[List[int]] = None,
        **gen_kwargs,
    ) -> PatchingResult:
        """
        Run generation while patching activations from source_cache.

        This is the core patching operation: run the model on `prompt`
        but replace activations at `target_layers` with values from
        `source_cache` (typically from a "clean" run).

        Args:
            prompt: Input prompt (typically the "corrupted" version)
            source_cache: Activations from another run to patch in
            target_layers: Which layers to patch
            token_positions: Which token positions to patch (None = all)
            **gen_kwargs: Additional generation arguments

        Returns:
            PatchingResult with output and metadata
        """
        # Register patching hooks
        for layer_idx in target_layers:
            source_activation = source_cache.get(layer_idx)
            if source_activation is None:
                raise ValueError(f"Layer {layer_idx} not in source cache")

            self.hook_manager.register_patch_hook(layer_idx, source_activation, token_positions)

        try:
            output = self.backend.generate(prompt, **gen_kwargs)
        finally:
            self.hook_manager.remove_all_hooks()

        return PatchingResult(
            output_text=output.text, patched_layers=target_layers, patched_positions=token_positions
        )

    def sweep_layers(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        layers: Optional[List[int]] = None,
        **gen_kwargs,
    ) -> Dict[int, PatchingResult]:
        """
        Sweep patching across layers to find causal importance.

        For each layer, patches that layer's activations from the clean
        run into the corrupted run and measures the effect.

        Args:
            clean_prompt: Prompt that elicits desired behavior
            corrupted_prompt: Prompt that elicits different behavior
            layers: Which layers to sweep (None = all)
            **gen_kwargs: Generation arguments

        Returns:
            Dict mapping layer_idx -> ForwardPatchResult
        """
        # Get clean logits and activations
        clean_logits, clean_cache = self.backend.forward_with_cache(clean_prompt, layers=layers)

        # Get corrupted logits (no patching)
        corrupted_logits, _ = self.backend.forward_with_cache(corrupted_prompt, layers=layers)

        target_layers = layers if layers is not None else clean_cache.layers
        results = {}

        for layer_idx in target_layers:
            # Run forward pass with patching on this layer
            patched_logits = self._forward_with_patch(corrupted_prompt, clean_cache, layer_idx)

            # Compute effect score (how much patching moves toward clean)
            effect = self._compute_logit_effect(clean_logits, corrupted_logits, patched_logits)

            results[layer_idx] = ForwardPatchResult(
                layer_idx=layer_idx,
                clean_logits=clean_logits.cpu(),
                corrupted_logits=corrupted_logits.cpu(),
                patched_logits=patched_logits.cpu(),
                effect_score=effect,
            )

        return results

    def _forward_with_patch(
        self,
        prompt: str,
        source_cache: ActivationCache,
        layer_idx: int,
    ) -> torch.Tensor:
        """Run a single forward pass with patching at specified layer's residual stream."""
        source_activation = source_cache.get(layer_idx)
        if source_activation is None:
            raise ValueError(f"Layer {layer_idx} not in source cache")

        # Use residual patch hook for safer patching
        self.hook_manager.register_residual_patch_hook(layer_idx, source_activation, None)

        try:
            logits, _ = self.backend.forward_with_cache(prompt, layers=[])
        finally:
            self.hook_manager.remove_all_hooks()

        return logits

    def _compute_logit_effect(
        self,
        clean_logits: torch.Tensor,
        corrupted_logits: torch.Tensor,
        patched_logits: torch.Tensor,
    ) -> float:
        """
        Compute how much patching moves logits from corrupted toward clean.

        Returns value from 0 (no effect) to 1 (full recovery).
        """
        # Use last token logits
        clean_last = clean_logits[:, -1, :]
        corrupted_last = corrupted_logits[:, -1, :]
        patched_last = patched_logits[:, -1, :]

        # Compute distances
        clean_to_corrupted = torch.norm(clean_last - corrupted_last).item()
        patched_to_corrupted = torch.norm(patched_last - corrupted_last).item()

        if clean_to_corrupted == 0:
            return 0.0

        # Effect = how much we moved toward clean (normalized)
        effect = 1.0 - (patched_to_corrupted / clean_to_corrupted)
        return max(0.0, min(1.0, effect))

    def sweep_positions(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        layer_idx: int,
        positions: Optional[List[int]] = None,
        **gen_kwargs,
    ) -> Dict[int, PatchingResult]:
        """
        Sweep patching across token positions at a fixed layer.

        Args:
            clean_prompt: Prompt that elicits desired behavior
            corrupted_prompt: Prompt that elicits different behavior
            layer_idx: Which layer to patch
            positions: Which positions to sweep (None = all)
            **gen_kwargs: Generation arguments

        Returns:
            Dict mapping position -> PatchingResult
        """
        # Get clean activations
        _, clean_cache = self.backend.generate_with_cache(
            clean_prompt, layers=[layer_idx], **gen_kwargs
        )

        # Get corrupted baseline
        corrupted_output = self.backend.generate(corrupted_prompt, **gen_kwargs)

        # Determine positions to sweep
        clean_act = clean_cache.get(layer_idx)
        target_positions = positions if positions is not None else list(range(clean_act.shape[1]))

        results = {}

        for pos in target_positions:
            result = self.patch_run(
                corrupted_prompt,
                clean_cache,
                target_layers=[layer_idx],
                token_positions=[pos],
                **gen_kwargs,
            )
            result.original_answer = corrupted_output.text
            result.patched_answer = result.output_text
            results[pos] = result

        return results

    def ablate_layer(
        self, prompt: str, layer_idx: int, ablation_type: str = "zero", **gen_kwargs
    ) -> PatchingResult:
        """
        Ablate a layer by zeroing or replacing with mean.

        Args:
            prompt: Input prompt
            layer_idx: Which layer to ablate
            ablation_type: "zero" or "mean"
            **gen_kwargs: Generation arguments

        Returns:
            PatchingResult from ablated run
        """

        def zero_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
                return (torch.zeros_like(hidden_states),) + rest
            return torch.zeros_like(output)

        def mean_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
                mean_val = hidden_states.mean(dim=(0, 1), keepdim=True)
                return (mean_val.expand_as(hidden_states),) + rest
            mean_val = output.mean(dim=(0, 1), keepdim=True)
            return mean_val.expand_as(output)

        hook = zero_hook if ablation_type == "zero" else mean_hook
        self.hook_manager.register_forward_hook(layer_idx, hook)

        try:
            output = self.backend.generate(prompt, **gen_kwargs)
        finally:
            self.hook_manager.remove_all_hooks()

        return PatchingResult(
            output_text=output.text, patched_layers=[layer_idx], patched_positions=None
        )
