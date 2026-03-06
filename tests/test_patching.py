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

    @pytest.fixture
    def mock_gemma_attn_model(self):
        """Create a mock Gemma-like model with attention output projection."""
        import torch.nn as nn

        class MockSelfAttn(nn.Module):
            def __init__(self, hidden_size: int = 8):
                super().__init__()
                self.o_proj = nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                return self.o_proj(x)

        class MockRMSNorm(nn.Module):
            def forward(self, x):
                return x

        class MockDecoderLayer(nn.Module):
            def __init__(self, hidden_size: int = 8):
                super().__init__()
                self.self_attn = MockSelfAttn(hidden_size)
                self.post_attention_layernorm = MockRMSNorm()
                self.post_feedforward_layernorm = MockRMSNorm()

            def forward(self, x):
                return self.self_attn(x)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([MockDecoderLayer() for _ in range(2)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class MockGemma(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MockModel()
                self.config = type("Config", (), {"model_type": "gemma3_text"})()

            def forward(self, x):
                return self.model(x)

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

    def test_gemma_attention_output_module(self, mock_gemma_attn_model):
        """Test that Gemma attention output module returns o_proj."""
        from cotlab.patching import HookManager

        hook_manager = HookManager(mock_gemma_attn_model)
        attn_out = hook_manager.get_attention_output_module(0)
        assert attn_out == mock_gemma_attn_model.model.layers[0].self_attn.o_proj

    def test_register_attention_cache_hooks(self, mock_gemma_attn_model):
        """Test caching attention output projections via hooks."""
        from cotlab.patching import ActivationCache, HookManager

        hook_manager = HookManager(mock_gemma_attn_model)
        cache = ActivationCache()
        hook_manager.register_attention_cache_hooks(cache, layers=[0])

        dummy = torch.randn(1, 2, 8)
        _ = mock_gemma_attn_model(dummy)
        hook_manager.remove_all_hooks()

        cached = cache.get(0)
        assert cached is not None
        assert cached.shape == dummy.shape

    def test_multi_head_patch_hook(self, mock_gemma_attn_model):
        """Test that multi-head patching overrides specified head outputs."""
        from cotlab.patching import HookManager

        hook_manager = HookManager(mock_gemma_attn_model)
        head_dim = 2  # hidden=8, heads=4

        source = torch.zeros(1, 3, 8)
        # Patch head 1 (positions 2:4) at last token
        source[:, -1, 2:4] = 5.0

        hook_manager.register_multi_head_patch_hook(1, [1], source, head_dim)
        dummy = torch.randn(1, 3, 8)
        out = mock_gemma_attn_model(dummy)
        hook_manager.remove_all_hooks()

        assert torch.allclose(out[:, -1, 2:4], source[:, -1, 2:4])

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


class TestActivationPatcherHeadInfo:
    """Tests for head metadata parsing in ActivationPatcher."""

    def test_get_head_info_from_config(self):
        """Ensure head count and head dim are derived from model config."""
        from cotlab.patching.patcher import ActivationPatcher

        class DummyModel:
            def __init__(self):
                self.config = type(
                    "Config",
                    (),
                    {"num_attention_heads": 4, "hidden_size": 16, "model_type": "gpt2"},
                )()

        class DummyBackend:
            supports_activations = True

            def __init__(self):
                self.model = DummyModel()
                self.hook_manager = None

        patcher = ActivationPatcher(DummyBackend())
        num_heads, head_dim = patcher._get_head_info()
        assert num_heads == 4
        assert head_dim == 4


class TestTokenGroupContrast:
    """Tests for _tag_tokens() and token_group_contrast config in ActivationPatchingExperiment."""

    @pytest.fixture
    def exp(self):
        """Minimal ActivationPatchingExperiment in token_group_contrast mode."""
        from cotlab.experiments.activation_patching import ActivationPatchingExperiment

        return ActivationPatchingExperiment(
            patching_mode="token_group_contrast",
            token_group_contrast_layer=3,
        )

    class MockTokenizer:
        """Minimal tokenizer stub: maps each integer token id to a string via a vocab."""

        def __init__(self, vocab: dict):
            # vocab: {token_id: string}
            self._vocab = vocab

        def decode(self, ids):
            return "".join(self._vocab.get(i, "?") for i in ids)

    def _make_input_ids(self, strings: list, vocab_inv: dict) -> torch.Tensor:
        """Build an input_ids tensor from a list of token strings using inverted vocab."""
        ids = [vocab_inv[s] for s in strings]
        return torch.tensor(ids, dtype=torch.long)

    def test_tag_tokens_three_groups_exhaustive(self, exp):
        """Every token must fall into exactly one group; groups cover all positions."""
        # Build a mini vocab covering a typical MCQ structure.
        vocab = {
            0: "A",  # question token
            1: " patient",
            2: " has",
            3: " fever",
            4: ".",
            5: "\n",  # delimiter
            6: "Options",  # delimiter
            7: ":",  # delimiter
            8: "A.",  # delimiter (answer label)
            9: " Malaria",  # choice text
            10: "\n",  # delimiter
            11: "B.",  # delimiter
            12: " Typhoid",  # choice text
        }

        tokenizer = self.MockTokenizer(vocab)
        # Sequence: ["A", " patient", " has", " fever", ".", "\n", "Options", ":", "A.", " Malaria", "\n", "B.", " Typhoid"]
        tokens_raw = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        input_ids = torch.tensor(tokens_raw, dtype=torch.long)

        groups = exp._tag_tokens(input_ids, tokenizer, {})

        all_positions = sorted(
            groups.get("delimiter", []) + groups.get("choice", []) + groups.get("content", [])
        )
        # Every position must appear exactly once across the 3 primary groups.
        assert all_positions == list(range(len(tokens_raw))), (
            f"Positions not fully covered: got {all_positions}"
        )

    def test_tag_tokens_delimiter_detection(self, exp):
        """\\n and 'Options' are classified as delimiters."""
        vocab = {0: "\n", 1: "Options", 2: " chest", 3: " pain"}
        tokenizer = self.MockTokenizer(vocab)
        input_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        groups = exp._tag_tokens(input_ids, tokenizer, {})

        assert 0 in groups["delimiter"], "\\n should be a delimiter"
        assert 1 in groups["delimiter"], "'Options' should be a delimiter"
        # Non-delimiter tokens must exist in either content OR choice.
        non_delimiters = groups.get("content", []) + groups.get("choice", [])
        assert len(non_delimiters) > 0, "Expected some non-delimiter tokens (content or choice)"

    def test_tag_tokens_medqa_entity_split(self, exp):
        """When metamap_phrases is present, content tokens split into entity and stem."""
        # Vocab: simple words
        vocab = {0: "Patient", 1: " has", 2: " STEMI", 3: " and", 4: " fever"}

        tokenizer = self.MockTokenizer(vocab)
        input_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

        metadata = {"metamap_phrases": ["STEMI"]}
        groups = exp._tag_tokens(input_ids, tokenizer, metadata)

        # After entity split, no plain "content" group — only entity + stem.
        assert "entity" in groups, "entity group missing after metamap split"
        assert "stem" in groups, "stem group missing after metamap split"
        assert len(groups["content"]) == 0, "content should be empty after entity split"

    def test_init_token_group_contrast_mode(self, exp):
        """token_group_contrast is a valid mode and params are stored correctly."""
        assert exp.patching_mode == "token_group_contrast"
        assert exp.token_group_contrast_layer == 3
        assert exp.token_group_mode == "all"

    def test_invalid_mode_raises(self):
        """Passing an unknown patching_mode raises ValueError."""
        from cotlab.experiments.activation_patching import ActivationPatchingExperiment

        with pytest.raises(ValueError, match="patching_mode must be one of"):
            ActivationPatchingExperiment(patching_mode="invalid_mode")

    def test_token_group_contrast_in_valid_modes(self):
        """token_group_contrast appears in VALID_MODES."""
        from cotlab.experiments.activation_patching import ActivationPatchingExperiment

        assert "token_group_contrast" in ActivationPatchingExperiment.VALID_MODES
