"""Tests for activation patching module."""

import pytest
import torch

from cotlab.patching import (
    ActivationCache,
    InterventionType,
    Intervention,
    PatchingExperimentSpec,
)


class TestActivationCache:
    """Tests for ActivationCache."""
    
    def test_store_and_get(self):
        cache = ActivationCache()
        tensor = torch.randn(1, 10, 512)
        cache.store(0, tensor)
        
        retrieved = cache.get(0)
        assert retrieved is not None
        assert torch.equal(retrieved, tensor)
    
    def test_get_missing_layer(self):
        cache = ActivationCache()
        assert cache.get(99) is None
    
    def test_getitem(self):
        cache = ActivationCache()
        tensor = torch.randn(1, 5, 256)
        cache.store(5, tensor)
        
        assert torch.equal(cache[5], tensor)
    
    def test_getitem_missing_raises(self):
        cache = ActivationCache()
        with pytest.raises(KeyError):
            _ = cache[99]
    
    def test_contains(self):
        cache = ActivationCache()
        cache.store(3, torch.randn(1, 5, 64))
        
        assert 3 in cache
        assert 99 not in cache
    
    def test_layers_property(self):
        cache = ActivationCache()
        cache.store(5, torch.randn(1, 1, 1))
        cache.store(2, torch.randn(1, 1, 1))
        cache.store(8, torch.randn(1, 1, 1))
        
        assert cache.layers == [2, 5, 8]  # Sorted
    
    def test_len(self):
        cache = ActivationCache()
        assert len(cache) == 0
        
        cache.store(0, torch.randn(1, 1, 1))
        cache.store(1, torch.randn(1, 1, 1))
        assert len(cache) == 2
    
    def test_slice_tokens(self):
        cache = ActivationCache()
        tensor = torch.randn(1, 20, 512)
        cache.store(0, tensor)
        
        sliced = cache.slice_tokens(0, (5, 10))
        assert sliced.shape == (1, 5, 512)
    
    def test_clear(self):
        cache = ActivationCache()
        cache.store(0, torch.randn(1, 1, 1))
        cache.store(1, torch.randn(1, 1, 1))
        
        cache.clear()
        assert len(cache) == 0
    
    def test_metadata(self):
        cache = ActivationCache()
        cache.set_metadata("prompt", "test prompt")
        assert cache.get_metadata("prompt") == "test prompt"


class TestIntervention:
    """Tests for Intervention dataclass."""
    
    def test_creation(self):
        intervention = Intervention(
            type=InterventionType.PATCH,
            layers=[5, 10, 15]
        )
        assert intervention.type == InterventionType.PATCH
        assert len(intervention.layers) == 3
    
    def test_with_positions(self):
        intervention = Intervention(
            type=InterventionType.ZERO,
            layers=[0],
            token_positions=[1, 2, 3]
        )
        assert intervention.token_positions == [1, 2, 3]
    
    def test_repr(self):
        intervention = Intervention(
            type=InterventionType.NOISE,
            layers=[1, 2]
        )
        repr_str = repr(intervention)
        assert "NOISE" in repr_str
        assert "1, 2" in repr_str


class TestPatchingExperimentSpec:
    """Tests for PatchingExperimentSpec."""
    
    def test_creation(self):
        spec = PatchingExperimentSpec(
            clean_prompt="Clean text",
            corrupted_prompt="Corrupted text"
        )
        assert spec.clean_prompt == "Clean text"
        assert spec.corrupted_prompt == "Corrupted text"
    
    def test_add_intervention_builder(self):
        spec = PatchingExperimentSpec(
            clean_prompt="A",
            corrupted_prompt="B"
        )
        
        spec.add_intervention(InterventionType.PATCH, [0, 5])
        spec.add_intervention(InterventionType.ZERO, [10])
        
        assert len(spec.interventions) == 2
    
    def test_with_expected_answers(self):
        spec = PatchingExperimentSpec(
            clean_prompt="What is 2+2?",
            corrupted_prompt="What is 2-2?",
            expected_clean_answer="4",
            expected_corrupted_answer="0"
        )
        assert spec.expected_clean_answer == "4"
