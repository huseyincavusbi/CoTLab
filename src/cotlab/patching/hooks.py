"""PyTorch forward hook utilities for activation extraction and patching."""

from typing import Callable, Dict, List, Optional, Any
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
    
    def _build_layer_mapping(self) -> Dict[int, nn.Module]:
        """
        Build mapping from layer indices to modules.
        
        Supports Gemma/Llama-style architecture with model.layers[i]
        """
        layers = {}
        
        # Try common layer naming patterns
        for name, module in self.model.named_modules():
            # Pattern 1: model.layers.X (Gemma, Llama)
            if ".layers." in name:
                # Extract layer number from path like "model.layers.5.mlp"
                parts = name.split(".")
                try:
                    layer_idx = int(parts[parts.index("layers") + 1])
                    # Store the layer block itself
                    if name.endswith(f".layers.{layer_idx}"):
                        layers[layer_idx] = module
                except (ValueError, IndexError):
                    continue
        
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
            raise ValueError(
                f"Layer {layer_idx} not found. Available: {self.available_layers}"
            )
        return self._layer_modules[layer_idx]
    
    def register_forward_hook(
        self,
        layer_idx: int,
        hook_fn: Callable[[nn.Module, Any, Any], Optional[Any]]
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
        self,
        cache: "ActivationCache",
        layers: Optional[List[int]] = None,
        detach: bool = True
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
        token_positions: Optional[List[int]] = None
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
            
            patched = hidden_states.clone()
            
            if token_positions is None:
                # Patch all positions up to min length
                min_len = min(patched.shape[1], source_activation.shape[1])
                patched[:, :min_len, :] = source_activation[:, :min_len, :]
            else:
                # Patch specific positions
                for pos in token_positions:
                    if pos < patched.shape[1] and pos < source_activation.shape[1]:
                        patched[:, pos, :] = source_activation[:, pos, :]
            
            if rest:
                return (patched,) + rest
            return patched
        
        return self.register_forward_hook(layer_idx, patch_hook)
    
    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_all_hooks()
