"""Activation cache for storing and accessing layer outputs."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator
import torch


@dataclass
class ActivationCache:
    """
    Store activations from model forward passes.
    
    This provides a clean interface for:
    - Storing activations by layer index
    - Retrieving full or sliced activations
    - Managing GPU memory
    
    Example:
        >>> cache = ActivationCache()
        >>> cache.store(0, layer_0_activations)
        >>> cache.store(5, layer_5_activations)
        >>> act = cache.get(0)  # Shape: [batch, seq_len, hidden_dim]
        >>> sliced = cache.slice_tokens(0, (10, 20))  # Tokens 10-20
    """
    
    _cache: Dict[int, torch.Tensor] = field(default_factory=dict)
    _metadata: Dict[str, any] = field(default_factory=dict)
    
    def store(self, layer_idx: int, activation: torch.Tensor) -> None:
        """
        Store activation for a layer.
        
        Args:
            layer_idx: Layer index
            activation: Activation tensor, typically [batch, seq_len, hidden_dim]
        """
        self._cache[layer_idx] = activation
    
    def get(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieve activation for a layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Activation tensor or None if not cached
        """
        return self._cache.get(layer_idx)
    
    def __getitem__(self, layer_idx: int) -> torch.Tensor:
        """Allow dict-like access: cache[0]"""
        if layer_idx not in self._cache:
            raise KeyError(f"Layer {layer_idx} not in cache. Available: {self.layers}")
        return self._cache[layer_idx]
    
    def __contains__(self, layer_idx: int) -> bool:
        """Check if layer is cached: 0 in cache"""
        return layer_idx in self._cache
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over cached layer indices."""
        return iter(sorted(self._cache.keys()))
    
    def __len__(self) -> int:
        """Number of cached layers."""
        return len(self._cache)
    
    @property
    def layers(self) -> List[int]:
        """List of cached layer indices, sorted."""
        return sorted(self._cache.keys())
    
    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of cached activations (assumed uniform)."""
        if not self._cache:
            return None
        first = next(iter(self._cache.values()))
        return tuple(first.shape)
    
    def get_all(self) -> Dict[int, torch.Tensor]:
        """Get all cached activations."""
        return self._cache.copy()
    
    def slice_tokens(
        self,
        layer_idx: int,
        token_range: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Get activations for specific token positions.
        
        Args:
            layer_idx: Layer index
            token_range: (start, end) tuple, end is exclusive
            
        Returns:
            Sliced activation tensor
        """
        activation = self._cache[layer_idx]
        start, end = token_range
        return activation[:, start:end, :]
    
    def slice_positions(
        self,
        layer_idx: int,
        positions: List[int]
    ) -> torch.Tensor:
        """
        Get activations for specific non-contiguous positions.
        
        Args:
            layer_idx: Layer index
            positions: List of token positions
            
        Returns:
            Activation tensor with shape [batch, len(positions), hidden_dim]
        """
        activation = self._cache[layer_idx]
        return activation[:, positions, :]
    
    def clear(self) -> None:
        """Clear all cached activations to free memory."""
        self._cache.clear()
        torch.cuda.empty_cache()
    
    def to_device(self, device: str) -> "ActivationCache":
        """Move all cached activations to a device."""
        new_cache = ActivationCache()
        for layer_idx, activation in self._cache.items():
            new_cache.store(layer_idx, activation.to(device))
        return new_cache
    
    def detach(self) -> "ActivationCache":
        """Create a new cache with all activations detached."""
        new_cache = ActivationCache()
        for layer_idx, activation in self._cache.items():
            new_cache.store(layer_idx, activation.detach().clone())
        return new_cache
    
    def set_metadata(self, key: str, value: any) -> None:
        """Store metadata about the cache."""
        self._metadata[key] = value
    
    def get_metadata(self, key: str) -> any:
        """Retrieve metadata."""
        return self._metadata.get(key)
