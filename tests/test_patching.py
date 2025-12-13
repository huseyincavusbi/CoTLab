"""Tests for activation patching module."""

import pytest
import torch

from cotlab.patching import (
    ActivationCache,
    Intervention,
    InterventionType,
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
        intervention = Intervention(type=InterventionType.PATCH, layers=[5, 10, 15])
        assert intervention.type == InterventionType.PATCH
        assert len(intervention.layers) == 3

    def test_with_positions(self):
        intervention = Intervention(
            type=InterventionType.ZERO, layers=[0], token_positions=[1, 2, 3]
        )
        assert intervention.token_positions == [1, 2, 3]

    def test_repr(self):
        intervention = Intervention(type=InterventionType.NOISE, layers=[1, 2])
        repr_str = repr(intervention)
        assert "NOISE" in repr_str
        assert "1, 2" in repr_str


class TestPatchingExperimentSpec:
    """Tests for PatchingExperimentSpec."""

    def test_creation(self):
        spec = PatchingExperimentSpec(clean_prompt="Clean text", corrupted_prompt="Corrupted text")
        assert spec.clean_prompt == "Clean text"
        assert spec.corrupted_prompt == "Corrupted text"

    def test_add_intervention_builder(self):
        spec = PatchingExperimentSpec(clean_prompt="A", corrupted_prompt="B")

        spec.add_intervention(InterventionType.PATCH, [0, 5])
        spec.add_intervention(InterventionType.ZERO, [10])

        assert len(spec.interventions) == 2

    def test_with_expected_answers(self):
        spec = PatchingExperimentSpec(
            clean_prompt="What is 2+2?",
            corrupted_prompt="What is 2-2?",
            expected_clean_answer="4",
            expected_corrupted_answer="0",
        )
        assert spec.expected_clean_answer == "4"


class TestHookManager:
    """Tests for HookManager layer detection and residual hooks."""

    @pytest.fixture
    def mock_gpt2_model(self):
        """Create a mock GPT-2-like model structure."""
        import torch.nn as nn

        class MockLayerNorm(nn.Module):
            def forward(self, x):
                return x

        class MockBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln_1 = MockLayerNorm()
                self.ln_2 = MockLayerNorm()

            def forward(self, x):
                return x

        class MockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.h = nn.ModuleList([MockBlock() for _ in range(3)])

            def forward(self, x):
                return x

        class MockGPT2(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = MockTransformer()
                self.config = type("Config", (), {"model_type": "gpt2"})()

            def forward(self, x):
                return x

        return MockGPT2()

    @pytest.fixture
    def mock_gemma_model(self):
        """Create a mock Gemma-like model structure."""
        import torch.nn as nn

        class MockRMSNorm(nn.Module):
            def forward(self, x):
                return x

        class MockDecoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = MockRMSNorm()
                self.post_attention_layernorm = MockRMSNorm()
                self.post_feedforward_layernorm = MockRMSNorm()

            def forward(self, x):
                return x

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([MockDecoderLayer() for _ in range(4)])

            def forward(self, x):
                return x

        class MockGemma(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MockModel()
                self.config = type("Config", (), {"model_type": "gemma3_text"})()

            def forward(self, x):
                return x

        return MockGemma()

    def test_gpt2_layer_detection(self, mock_gpt2_model):
        """Test that HookManager correctly detects GPT-2 layers."""
        from cotlab.patching import HookManager

        hook_manager = HookManager(mock_gpt2_model)

        assert hook_manager.num_layers == 3
        assert hook_manager.available_layers == [0, 1, 2]

    def test_gemma_layer_detection(self, mock_gemma_model):
        """Test that HookManager correctly detects Gemma layers."""
        from cotlab.patching import HookManager

        hook_manager = HookManager(mock_gemma_model)

        assert hook_manager.num_layers == 4
        assert hook_manager.available_layers == [0, 1, 2, 3]

    def test_gpt2_residual_module(self, mock_gpt2_model):
        """Test that GPT-2 residual module returns ln_2."""
        from cotlab.patching import HookManager

        hook_manager = HookManager(mock_gpt2_model)

        residual = hook_manager.get_residual_module(0)
        # Should return ln_2 for GPT-2
        assert residual == mock_gpt2_model.transformer.h[0].ln_2

    def test_gemma_residual_module(self, mock_gemma_model):
        """Test that Gemma residual module returns post_feedforward_layernorm."""
        from cotlab.patching import HookManager

        hook_manager = HookManager(mock_gemma_model)

        residual = hook_manager.get_residual_module(0)
        # Should return post_feedforward_layernorm for Gemma
        assert residual == mock_gemma_model.model.layers[0].post_feedforward_layernorm

    def test_get_layer_module(self, mock_gpt2_model):
        """Test getting layer module by index."""
        from cotlab.patching import HookManager

        hook_manager = HookManager(mock_gpt2_model)

        layer0 = hook_manager.get_layer_module(0)
        layer2 = hook_manager.get_layer_module(2)

        assert layer0 == mock_gpt2_model.transformer.h[0]
        assert layer2 == mock_gpt2_model.transformer.h[2]

    def test_get_layer_module_invalid(self, mock_gpt2_model):
        """Test that invalid layer index raises error."""
        from cotlab.patching import HookManager

        hook_manager = HookManager(mock_gpt2_model)

        with pytest.raises(ValueError, match="Layer 99 not found"):
            hook_manager.get_layer_module(99)

    def test_register_and_remove_hooks(self, mock_gpt2_model):
        """Test hook registration and removal."""
        from cotlab.patching import HookManager

        hook_manager = HookManager(mock_gpt2_model)

        def dummy_hook(module, input, output):
            return output

        _handle = hook_manager.register_forward_hook(0, dummy_hook)
        assert len(hook_manager.handles) == 1

        hook_manager.remove_all_hooks()
        assert len(hook_manager.handles) == 0

    def test_layer_paths_mapping(self):
        """Test that LAYER_PATHS contains expected model types."""
        from cotlab.patching import HookManager

        assert "gpt2" in HookManager.LAYER_PATHS
        assert "gemma3" in HookManager.LAYER_PATHS
        assert "gemma2" in HookManager.LAYER_PATHS  # Gemma family

        assert HookManager.LAYER_PATHS["gpt2"] == "transformer.h"
        assert HookManager.LAYER_PATHS["gemma3"] == "model.layers"

    def test_residual_hook_points_mapping(self):
        """Test that RESIDUAL_HOOK_POINTS contains expected model types."""
        from cotlab.patching import HookManager

        assert "gpt2" in HookManager.RESIDUAL_HOOK_POINTS
        assert "gemma3" in HookManager.RESIDUAL_HOOK_POINTS

        assert HookManager.RESIDUAL_HOOK_POINTS["gpt2"] == "ln_2"
        assert HookManager.RESIDUAL_HOOK_POINTS["gemma3"] == "post_feedforward_layernorm"
