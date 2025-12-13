"""PyTorch forward hook utilities for activation extraction and patching."""

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


class HookManager:
    """
    Manage PyTorch forward hooks for activation extraction and patching.

    This provides a clean interface for:
    - Registering hooks on specific layers
    - Caching activations during forward pass
    - Patching activations with custom values
    - Cleanup of all registered hooks

    Example:
        >>> manager = HookManager(model)
        >>> cache = ActivationCache()
        >>> manager.register_cache_hooks(cache, layers=[0, 5, 10])
        >>> output = model(input_ids)  # Activations now in cache
        >>> manager.remove_all_hooks()
    """

    def __init__(self, model: nn.Module):
        """
        Initialize hook manager for a model.

        Args:
            model: The transformer model to hook into
        """
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._layer_modules = self._build_layer_mapping()

    # Known layer paths by model_type from HF config
    LAYER_PATHS = {
        # GPT-2 family
        "gpt2": "transformer.h",
        "gpt_neo": "transformer.h",
        "gpt_neox": "gpt_neox.layers",
        # Llama/Mistral family
        "llama": "model.layers",
        "mistral": "model.layers",
        "mixtral": "model.layers",
        "phi": "model.layers",
        "phi3": "model.layers",
        "qwen2": "model.layers",
        # Gemma family
        "gemma": "model.layers",
        "gemma2": "model.layers",
        "gemma3": "model.layers",
        "gemma3_text": "model.layers",
        # BERT/RoBERTa (encoder)
        "bert": "bert.encoder.layer",
        "roberta": "roberta.encoder.layer",
        # T5/BART (encoder-decoder)
        "t5": "decoder.block",
        "bart": "model.decoder.layers",
        # Falcon
        "falcon": "transformer.h",
        # OPT
        "opt": "model.decoder.layers",
        # Bloom
        "bloom": "transformer.h",
    }

    # Safe residual stream hook points (post-layer normalization)
    # These are the output of the final norm in each layer block
    RESIDUAL_HOOK_POINTS = {
        # GPT-2: hook ln_2 (after attention, before MLP residual add)
        "gpt2": "ln_2",
        "gpt_neo": "ln_2",
        "gpt_neox": "post_attention_layernorm",
        # Llama/Mistral: hook post_attention_layernorm
        "llama": "post_attention_layernorm",
        "mistral": "post_attention_layernorm",
        "mixtral": "post_attention_layernorm",
        "phi": "post_attention_layernorm",
        "phi3": "post_attention_layernorm",
        "qwen2": "post_attention_layernorm",
        # Gemma: hook post_feedforward_layernorm (after entire layer)
        "gemma": "post_feedforward_layernorm",
        "gemma2": "post_feedforward_layernorm",
        "gemma3": "post_feedforward_layernorm",
        "gemma3_text": "post_feedforward_layernorm",
        # Falcon
        "falcon": "ln_attn",
        # OPT
        "opt": "self_attn_layer_norm",
        # Bloom
        "bloom": "input_layernorm",
    }

    # Attention output projection modules for head-level patching
    # These modules take concatenated head outputs and project back to hidden dim
    ATTENTION_OUTPUT_POINTS = {
        # GPT-2: attn.c_proj (output of all heads concatenated)
        "gpt2": "attn.c_proj",
        "gpt_neo": "attn.out_proj",
        "gpt_neox": "attention.dense",
        # Llama/Mistral: self_attn.o_proj
        "llama": "self_attn.o_proj",
        "mistral": "self_attn.o_proj",
        "mixtral": "self_attn.o_proj",
        "phi": "self_attn.dense",
        "phi3": "self_attn.o_proj",
        "qwen2": "self_attn.o_proj",
        # Gemma: self_attn.o_proj
        "gemma": "self_attn.o_proj",
        "gemma2": "self_attn.o_proj",
        "gemma3": "self_attn.o_proj",
        "gemma3_text": "self_attn.o_proj",
        # Falcon
        "falcon": "self_attention.dense",
        # OPT
        "opt": "self_attn.out_proj",
        # Bloom
        "bloom": "self_attention.dense",
    }

    def _build_layer_mapping(self) -> Dict[int, nn.Module]:
        """
        Auto-detect transformer layers using HF config model_type.

        Uses known layer paths for common architectures, with fallback
        to regex-based auto-detection for unknown models.
        """
        # Try to get model_type from config
        model_type = getattr(self.model.config, "model_type", None)
        layer_path = self.LAYER_PATHS.get(model_type)

        if layer_path:
            # Try known path first
            layers = self._get_layers_from_path(layer_path)
            if layers:
                return layers
            # If known path fails (e.g., multimodal model), try auto-detect

        # Fallback: auto-detect using regex
        return self._auto_detect_layers()

    def _get_layers_from_path(self, layer_path: str) -> Dict[int, nn.Module]:
        """Get layers from a known path like 'transformer.h' or 'model.layers'."""
        layers = {}

        for name, module in self.model.named_modules():
            # Match pattern: layer_path.{number}
            if name.startswith(layer_path + "."):
                suffix = name[len(layer_path) + 1 :]
                # Only match direct children (no dots in suffix)
                if "." not in suffix and suffix.isdigit():
                    layers[int(suffix)] = module

        return layers

    def _auto_detect_layers(self) -> Dict[int, nn.Module]:
        """Fallback: auto-detect layers using regex pattern matching."""
        import re

        layers = {}
        layer_priority = {}
        layer_pattern = re.compile(r"^(.+?)\.(\d+)$")

        for name, module in self.model.named_modules():
            match = layer_pattern.match(name)
            if not match:
                continue

            prefix = match.group(1)
            layer_idx = int(match.group(2))

            # Skip sublayers (modules with numbered children)
            has_numbered_children = any(c.isdigit() for c, _ in module.named_children())
            if has_numbered_children:
                continue

            # Prioritize language model layers
            if "language_model" in prefix:
                priority = 4
            elif "layers" in prefix or "h" in prefix:
                priority = 1 if ("vision" in prefix or "encoder" in prefix) else 3
            else:
                priority = 2

            if layer_idx not in layer_priority or priority > layer_priority[layer_idx]:
                layers[layer_idx] = module
                layer_priority[layer_idx] = priority

        return layers

    @property
    def num_layers(self) -> int:
        """Number of hookable layers."""
        return len(self._layer_modules)

    @property
    def available_layers(self) -> List[int]:
        """List of available layer indices."""
        return sorted(self._layer_modules.keys())

    def get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get the module for a specific layer."""
        if layer_idx not in self._layer_modules:
            raise ValueError(f"Layer {layer_idx} not found. Available: {self.available_layers}")
        return self._layer_modules[layer_idx]

    def get_residual_module(self, layer_idx: int) -> nn.Module:
        """
        Get the residual stream hook point for a layer.

        Returns the post-layer normalization module (e.g., ln_2 for GPT-2,
        post_feedforward_layernorm for Gemma3) which is safer for patching
        than the full layer block.
        """
        layer_module = self.get_layer_module(layer_idx)
        model_type = getattr(self.model.config, "model_type", None)
        residual_name = self.RESIDUAL_HOOK_POINTS.get(model_type)

        if residual_name:
            # Try to get the specific residual hook point
            if hasattr(layer_module, residual_name):
                return getattr(layer_module, residual_name)

        # Fallback: try common names
        for name in [
            "post_feedforward_layernorm",
            "post_attention_layernorm",
            "ln_2",
            "layer_norm",
        ]:
            if hasattr(layer_module, name):
                return getattr(layer_module, name)

        # Last resort: return the layer itself
        return layer_module

    def get_attention_output_module(self, layer_idx: int) -> nn.Module:
        """
        Get the attention output projection module for a layer.

        This is where individual head outputs are concatenated and projected.
        Used for head-level patching interventions.
        """
        layer_module = self.get_layer_module(layer_idx)
        model_type = getattr(self.model.config, "model_type", None)
        attn_path = self.ATTENTION_OUTPUT_POINTS.get(model_type)

        if attn_path:
            # Navigate nested path like "self_attn.o_proj"
            parts = attn_path.split(".")
            module = layer_module
            for part in parts:
                if hasattr(module, part):
                    module = getattr(module, part)
                else:
                    break
            else:
                return module

        # Fallback: try common attention output names
        for attn_name in ["self_attn", "attn", "attention"]:
            if hasattr(layer_module, attn_name):
                attn = getattr(layer_module, attn_name)
                for proj_name in ["o_proj", "c_proj", "out_proj", "dense"]:
                    if hasattr(attn, proj_name):
                        return getattr(attn, proj_name)

        raise ValueError(f"Could not find attention output module for layer {layer_idx}")

    def register_forward_hook(
        self, layer_idx: int, hook_fn: Callable[[nn.Module, Any, Any], Optional[Any]]
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Register a forward hook on a specific layer.

        Args:
            layer_idx: Index of the transformer layer
            hook_fn: Hook function with signature (module, input, output) -> output

        Returns:
            Handle that can be used to remove the hook
        """
        module = self.get_layer_module(layer_idx)
        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)
        return handle

    def register_cache_hooks(
        self, cache: "ActivationCache", layers: Optional[List[int]] = None, detach: bool = True
    ) -> None:
        """
        Register hooks to cache activations from specified layers.

        Args:
            cache: ActivationCache to store activations in
            layers: Which layers to cache (None = all)
            detach: Whether to detach tensors from computation graph
        """
        target_layers = layers if layers is not None else self.available_layers

        for layer_idx in target_layers:

            def make_hook(idx: int):
                def hook(module, input, output):
                    # output is typically a tuple (hidden_states, ...)
                    if isinstance(output, tuple):
                        activation = output[0]
                    else:
                        activation = output

                    if detach:
                        activation = activation.detach().clone()

                    cache.store(idx, activation)
                    return output

                return hook

            self.register_forward_hook(layer_idx, make_hook(layer_idx))

    def register_patch_hook(
        self,
        layer_idx: int,
        source_activation: torch.Tensor,
        token_positions: Optional[List[int]] = None,
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Register a hook that patches activations with source values.

        Args:
            layer_idx: Layer to patch
            source_activation: Activation tensor to patch in
            token_positions: Which positions to patch (None = all)

        Returns:
            Hook handle
        """

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = ()

            # Skip patching during autoregressive decoding (seq_len=1)
            current_seq_len = hidden_states.shape[1]
            if current_seq_len == 1:
                if rest:
                    return (hidden_states,) + rest
                return hidden_states

            # Create new tensor for patching to avoid in-place operations
            # that can break model internals
            source_seq_len = source_activation.shape[1]
            target_seq_len = hidden_states.shape[1]

            if token_positions is None:
                # Replace with source activations (truncated/padded as needed)
                if target_seq_len <= source_seq_len:
                    # Use source directly (truncated if needed)
                    patched = source_activation[:, :target_seq_len, :].contiguous()
                else:
                    # Need to pad: copy source then keep remaining from original
                    patched = torch.cat(
                        [source_activation, hidden_states[:, source_seq_len:, :]], dim=1
                    )
            else:
                # Patch specific positions by building new tensor
                patched = hidden_states.clone()
                for pos in token_positions:
                    if pos < target_seq_len and pos < source_seq_len:
                        patched[:, pos : pos + 1, :] = source_activation[:, pos : pos + 1, :]

            if rest:
                return (patched,) + rest
            return patched

        return self.register_forward_hook(layer_idx, patch_hook)

    def register_residual_cache_hooks(
        self, cache: "ActivationCache", layers: Optional[List[int]] = None, detach: bool = True
    ) -> None:
        """
        Register hooks to cache activations from residual stream (post-layer norm).

        This is safer than caching from the full layer block as it captures
        the clean residual stream without internal layer state.
        """
        target_layers = layers if layers is not None else self.available_layers

        for layer_idx in target_layers:
            residual_module = self.get_residual_module(layer_idx)

            def make_hook(idx: int):
                def hook(module, input, output):
                    activation = output
                    if detach:
                        activation = activation.detach().clone()
                    cache.store(idx, activation)
                    return output

                return hook

            handle = residual_module.register_forward_hook(make_hook(layer_idx))
            self.handles.append(handle)

    def register_residual_patch_hook(
        self,
        layer_idx: int,
        source_activation: torch.Tensor,
        token_positions: Optional[List[int]] = None,
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Register a patch hook on the residual stream (post-layer norm).

        This is safer than patching the full layer block as it only modifies
        the output of the normalization layer without affecting internal state.
        """
        residual_module = self.get_residual_module(layer_idx)

        def patch_hook(module, input, output):
            # Residual modules typically output a tensor directly (not tuple)
            hidden_states = output

            # Skip single-token decoding
            if hidden_states.shape[1] == 1:
                return hidden_states

            # Match shapes
            target_len = hidden_states.shape[1]
            source_len = source_activation.shape[1]
            min_len = min(target_len, source_len)

            if token_positions is None:
                # Patch overlapping positions
                patched = hidden_states.clone()
                patched[:, :min_len, :] = source_activation[:, :min_len, :]
            else:
                patched = hidden_states.clone()
                for pos in token_positions:
                    if pos < target_len and pos < source_len:
                        patched[:, pos : pos + 1, :] = source_activation[:, pos : pos + 1, :]

            return patched

        handle = residual_module.register_forward_hook(patch_hook)
        self.handles.append(handle)
        return handle

    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_all_hooks()
