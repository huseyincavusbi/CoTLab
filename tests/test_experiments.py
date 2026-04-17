"""Tests for new experiment classes."""

import math

import numpy as np
import torch

from cotlab.experiments import (
    ActivationCompareExperiment,
    ActivationPatchingExperiment,
    AttentionAnalysisExperiment,
    CompositeShiftDetectorExperiment,
    ConfabulationAnalysisExperiment,
    EntropyNeuronOverlapExperiment,
    FullLayerPatchingExperiment,
    MultiHeadPatchingExperiment,
    ResidualNormOODExperiment,
    SAEFeatureNeuronOverlapExperiment,
    SteeringVectorsExperiment,
    SycophancyHeadsExperiment,
)


class TestSycophancyHeadsExperiment:
    """Tests for SycophancyHeadsExperiment."""

    def test_init_defaults(self):
        """Test default initialization."""
        exp = SycophancyHeadsExperiment()
        assert exp.name == "sycophancy_heads"
        assert exp.search_layers is None  # Auto-detected at runtime
        assert exp.suggested_diagnosis == "anxiety"

    def test_init_custom_layers(self):
        """Test custom search layers."""
        exp = SycophancyHeadsExperiment(search_layers=[10, 11, 12])
        assert exp.search_layers == [10, 11, 12]

    def test_name_property(self):
        """Test name property."""
        exp = SycophancyHeadsExperiment(name="custom_name")
        assert exp.name == "custom_name"


class TestActivationCompareExperiment:
    """Tests for ActivationCompareExperiment."""

    def test_init_defaults(self):
        """Test default initialization."""
        exp = ActivationCompareExperiment()
        assert exp.name == "activation_compare"
        assert exp.num_samples is None
        assert exp.pooling == "last_token"

    def test_name_property(self):
        """Test name property."""
        exp = ActivationCompareExperiment(name="custom_compare")
        assert exp.name == "custom_compare"


class TestActivationPatchingExperiment:
    """Tests for ActivationPatchingExperiment head config parsing."""

    def test_head_indices_apply_to_layers(self):
        """Head indices should expand to all layers."""
        from omegaconf import OmegaConf

        exp = ActivationPatchingExperiment(patching={"head_indices": OmegaConf.create([0, 1])})
        targets = exp._resolve_head_targets([1, 2])
        assert targets == {1: [0, 1], 2: [0, 1]}

    def test_target_heads_mapping(self):
        """Target heads dict should map to specific layers only."""
        from omegaconf import OmegaConf

        exp = ActivationPatchingExperiment(
            patching={"target_heads": OmegaConf.create({"1": [0, 2], "3": [1]})}
        )
        targets = exp._resolve_head_targets([1, 2, 3])
        assert targets == {1: [0, 2], 3: [1]}

    def test_head_config_conflict(self):
        """Using head_indices and target_heads together should raise."""
        exp = ActivationPatchingExperiment(patching={"head_indices": [0], "target_heads": {1: [0]}})
        try:
            exp._resolve_head_targets([1])
        except ValueError as exc:
            assert "either target_heads or head_indices" in str(exc)
        else:
            raise AssertionError("Expected ValueError for conflicting head config")


class TestAttentionAnalysisExperiment:
    """Tests for AttentionAnalysisExperiment defaults and helpers."""

    def test_init_defaults(self):
        """Default config should match experiment defaults."""
        exp = AttentionAnalysisExperiment()
        assert exp.name == "attention_analysis"
        assert exp.target_layers == [55, 56, 57, 58, 59, 60]
        assert exp.all_layers is False
        assert exp.force_eager_reload is True
        assert exp.num_samples is None
        assert exp.last_k_tokens == 16
        assert exp.max_input_tokens == 1024
        assert exp.analyze_generated_tokens is False
        assert exp.generated_max_new_tokens == 16
        assert exp.generated_do_sample is False

    def test_entropy_computation(self):
        """Entropy helper should return a finite float."""
        import torch

        exp = AttentionAnalysisExperiment()
        attn = torch.tensor([0.5, 0.5])
        entropy = exp._compute_entropy(attn)
        assert isinstance(entropy, float)
        assert entropy > 0.0


class TestMultiHeadPatchingExperiment:
    """Tests for MultiHeadPatchingExperiment."""

    def test_init_defaults(self):
        """Test default initialization."""
        exp = MultiHeadPatchingExperiment()
        assert exp.name == "multi_head_patching"
        # Default top heads from sycophancy sweep
        assert len(exp.top_heads) == 5
        assert (20, 2) in exp.top_heads

    def test_init_custom_heads(self):
        """Test custom top heads."""
        custom_heads = [[15, 3], [18, 5]]
        exp = MultiHeadPatchingExperiment(top_heads=custom_heads)
        assert exp.top_heads == [(15, 3), (18, 5)]

    def test_name_property(self):
        """Test name property."""
        exp = MultiHeadPatchingExperiment(name="custom_multi")
        assert exp.name == "custom_multi"


class TestFullLayerPatchingExperiment:
    """Tests for FullLayerPatchingExperiment."""

    def test_init_defaults(self):
        """Test default initialization."""
        exp = FullLayerPatchingExperiment()
        assert exp.name == "full_layer_patching"
        assert exp.target_layers is None  # Auto-detected at runtime
        assert exp.suggested_diagnosis == "anxiety"

    def test_init_custom_layers(self):
        """Test custom target layers."""
        exp = FullLayerPatchingExperiment(target_layers=[5, 10, 15])
        assert exp.target_layers == [5, 10, 15]

    def test_name_property(self):
        """Test name property."""
        exp = FullLayerPatchingExperiment(name="full_layer_custom")
        assert exp.name == "full_layer_custom"


class TestSteeringVectorsExperiment:
    """Tests for SteeringVectorsExperiment."""

    def test_init_defaults(self):
        """Test default initialization."""
        exp = SteeringVectorsExperiment()
        assert exp.name == "steering_vectors"
        assert exp.target_layers is None  # None = auto-detect all layers
        assert exp.steering_strengths == [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    def test_init_custom_strengths(self):
        """Test custom steering strengths."""
        custom_strengths = [-1.0, 0.0, 1.0]
        exp = SteeringVectorsExperiment(steering_strengths=custom_strengths)
        assert exp.steering_strengths == custom_strengths

    def test_init_custom_layers(self):
        """Test custom target layers."""
        exp = SteeringVectorsExperiment(target_layers=[10, 15, 20])
        assert exp.target_layers == [10, 15, 20]

    def test_name_property(self):
        """Test name property."""
        exp = SteeringVectorsExperiment(name="steering_custom")
        assert exp.name == "steering_custom"


class TestExperimentImports:
    """Test that all experiments can be imported."""

    def test_import_sycophancy_heads(self):
        """Test SycophancyHeadsExperiment import."""
        from cotlab.experiments import SycophancyHeadsExperiment

        assert SycophancyHeadsExperiment is not None

    def test_import_multi_head_patching(self):
        """Test MultiHeadPatchingExperiment import."""
        from cotlab.experiments import MultiHeadPatchingExperiment

        assert MultiHeadPatchingExperiment is not None

    def test_import_full_layer_patching(self):
        """Test FullLayerPatchingExperiment import."""
        from cotlab.experiments import FullLayerPatchingExperiment

        assert FullLayerPatchingExperiment is not None

    def test_import_steering_vectors(self):
        """Test SteeringVectorsExperiment import."""
        from cotlab.experiments import SteeringVectorsExperiment

        assert SteeringVectorsExperiment is not None

    def test_import_cot_heads(self):
        """Test CoTHeadsExperiment import."""
        from cotlab.experiments import CoTHeadsExperiment

        assert CoTHeadsExperiment is not None

    def test_import_logit_lens(self):
        """Test LogitLensExperiment import."""
        from cotlab.experiments import LogitLensExperiment

        assert LogitLensExperiment is not None


class TestCoTHeadsExperiment:
    """Tests for CoTHeadsExperiment."""

    def test_init_defaults(self):
        """Test default initialization."""
        from cotlab.experiments import CoTHeadsExperiment

        exp = CoTHeadsExperiment()
        assert exp.name == "cot_heads"
        assert exp.search_layers is None  # Auto-detected at runtime

    def test_init_custom_layers(self):
        """Test custom search layers."""
        from cotlab.experiments import CoTHeadsExperiment

        exp = CoTHeadsExperiment(search_layers=[5, 10, 15])
        assert exp.search_layers == [5, 10, 15]


class TestLogitLensExperiment:
    """Tests for LogitLensExperiment."""

    def test_init_defaults(self):
        """Test default initialization."""
        from cotlab.experiments import LogitLensExperiment

        exp = LogitLensExperiment()
        assert exp.name == "logit_lens"
        assert exp.target_layers is None  # Auto-detected at runtime
        assert exp.top_k == 10

    def test_init_custom_layers(self):
        """Test custom target layers."""
        from cotlab.experiments import LogitLensExperiment

        exp = LogitLensExperiment(target_layers=[0, 10, 20])
        assert exp.target_layers == [0, 10, 20]

    def test_init_custom_top_k(self):
        """Test custom top_k."""
        from cotlab.experiments import LogitLensExperiment

        exp = LogitLensExperiment(top_k=10)
        assert exp.top_k == 10


class TestResidualNormOODExperiment:
    """Tests for ResidualNormOODExperiment."""

    def test_init_defaults(self):
        exp = ResidualNormOODExperiment()
        assert exp.name == "residual_norm_ood"
        assert exp._target_layer_config is None
        assert exp.num_samples is None
        assert exp.seed == 42
        assert exp.max_input_tokens == 1024
        assert exp.answer_cue == "\n\nAnswer:"
        assert exp.threshold_percentile_step == 5

    def test_init_custom(self):
        exp = ResidualNormOODExperiment(
            name="custom_ood",
            target_layer=30,
            num_samples=100,
            seed=7,
            threshold_percentile_step=10,
        )
        assert exp.name == "custom_ood"
        assert exp._target_layer_config == 30
        assert exp.num_samples == 100
        assert exp.seed == 7
        assert exp.threshold_percentile_step == 10

    def test_compute_norm(self):
        exp = ResidualNormOODExperiment()
        hidden = torch.tensor([3.0, 4.0])
        assert abs(exp._compute_norm(hidden) - 5.0) < 1e-5

    def test_compute_logit_entropy_uniform(self):
        exp = ResidualNormOODExperiment()
        # Uniform distribution over 4 letters → max entropy = log(4)
        logits = torch.zeros(10)
        letter_ids = [0, 1, 2, 3]
        entropy = exp._compute_logit_entropy(logits, letter_ids)
        assert abs(entropy - math.log(4)) < 1e-4

    def test_compute_logit_entropy_peaked(self):
        exp = ResidualNormOODExperiment()
        # Very peaked distribution → low entropy
        logits = torch.tensor([100.0, 0.0, 0.0, 0.0])
        entropy = exp._compute_logit_entropy(logits, [0, 1, 2, 3])
        assert entropy < 0.01

    def test_compute_logit_entropy_empty_ids(self):
        exp = ResidualNormOODExperiment()
        entropy = exp._compute_logit_entropy(torch.zeros(10), [])
        assert math.isnan(entropy)

    def test_find_threshold_all_same_label(self):
        exp = ResidualNormOODExperiment()
        norms = [1.0, 2.0, 3.0]
        labels = [True, True, True]
        tau, ba = exp._find_threshold(norms, labels)
        # Cannot compute balanced accuracy with one class → returns mean, 0
        assert ba == 0.0

    def test_find_threshold_separable(self):
        exp = ResidualNormOODExperiment()
        norms = [1.0, 1.1, 5.0, 5.1]
        labels = [False, False, True, True]
        tau, ba = exp._find_threshold(norms, labels)
        assert ba > 0.5
        assert 1.0 < tau < 5.1


class TestCompositeShiftDetectorExperiment:
    """Tests for CompositeShiftDetectorExperiment."""

    def test_init_defaults(self):
        exp = CompositeShiftDetectorExperiment()
        assert exp.name == "composite_shift_detector"
        assert exp._norm_layer_config is None
        assert exp.attn_layer == 3
        assert exp.num_samples is None
        assert exp.calibration_fraction == 0.3
        assert exp.window_size == 20
        assert exp.num_bins == 5
        assert exp.seed == 42
        assert exp.max_input_tokens == 1024
        assert exp.answer_cue == "\n\nAnswer:"

    def test_init_custom(self):
        exp = CompositeShiftDetectorExperiment(
            name="my_detector",
            norm_layer=10,
            attn_layer=5,
            calibration_fraction=0.5,
            window_size=10,
            num_bins=3,
        )
        assert exp.name == "my_detector"
        assert exp._norm_layer_config == 10
        assert exp.attn_layer == 5
        assert exp.calibration_fraction == 0.5
        assert exp.window_size == 10
        assert exp.num_bins == 3

    def test_mahalanobis_identity(self):
        exp = CompositeShiftDetectorExperiment()
        mu = np.array([0.0, 0.0])
        prec = np.eye(2)
        d = exp._mahalanobis(np.array([3.0, 4.0]), mu, prec)
        assert abs(d - 5.0) < 1e-5

    def test_mahalanobis_zero_at_mean(self):
        exp = CompositeShiftDetectorExperiment()
        mu = np.array([1.0, 2.0])
        prec = np.eye(2)
        d = exp._mahalanobis(mu, mu, prec)
        assert d == 0.0

    def test_fit_mahalanobis_returns_shapes(self):
        exp = CompositeShiftDetectorExperiment()
        features = np.random.default_rng(0).normal(size=(20, 2))
        mu, prec = exp._fit_mahalanobis(features)
        assert mu.shape == (2,)
        assert prec.shape == (2, 2)

    def test_spearman_perfect_positive(self):
        exp = CompositeShiftDetectorExperiment()
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        rho, p = exp._spearman(x, x)
        assert abs(rho - 1.0) < 1e-6

    def test_spearman_too_short(self):
        exp = CompositeShiftDetectorExperiment()
        rho, p = exp._spearman([1.0], [1.0])
        assert math.isnan(rho) and math.isnan(p)

    def test_bin_accuracy_count(self):
        exp = CompositeShiftDetectorExperiment()
        scores = list(range(20))
        labels = [i % 2 == 0 for i in range(20)]
        bins = exp._bin_accuracy(scores, labels)
        assert len(bins) == 5
        assert all(b["n"] > 0 for b in bins)

    def test_bin_accuracy_empty(self):
        exp = CompositeShiftDetectorExperiment()
        assert exp._bin_accuracy([], []) == []

    def test_rolling_accuracy_length(self):
        exp = CompositeShiftDetectorExperiment(window_size=3)
        scores = [0.1, 0.5, 0.3, 0.8, 0.2]
        labels = [True, False, True, False, True]
        roll_score, roll_acc = exp._rolling_accuracy(scores, labels)
        # expected windows: len(scores) - window_size + 1 = 3
        assert len(roll_score) == 3
        assert len(roll_acc) == 3
        assert all(0.0 <= a <= 1.0 for a in roll_acc)


class TestNewExperimentImports:
    """Smoke-test that newly added experiments are importable."""

    def test_import_residual_norm_ood(self):
        from cotlab.experiments import ResidualNormOODExperiment

        assert ResidualNormOODExperiment is not None

    def test_import_composite_shift_detector(self):
        from cotlab.experiments import CompositeShiftDetectorExperiment

        assert CompositeShiftDetectorExperiment is not None


class TestConfabulationAnalysisExperiment:
    """Tests for ConfabulationAnalysisExperiment."""

    def test_init_defaults(self):
        """Test default initialization."""
        exp = ConfabulationAnalysisExperiment()
        assert exp.name == "confabulation_analysis"
        assert exp.probe_path is None
        assert exp.ood_dataset_path is None
        assert exp.num_samples == 50
        assert exp.conf_high == 13.0
        assert exp.conf_low == 10.0
        assert exp.seed == 42

    def test_init_custom_params(self):
        """Test custom parameters."""
        exp = ConfabulationAnalysisExperiment(
            probe_path="test.json",
            ood_dataset_path="ood.jsonl",
            num_samples=100,
            conf_high=15.0,
            conf_low=8.0,
        )
        assert exp.probe_path == "test.json"
        assert exp.ood_dataset_path == "ood.jsonl"
        assert exp.num_samples == 100
        assert exp.conf_high == 15.0
        assert exp.conf_low == 8.0

    def test_name_property(self):
        """Test name property."""
        exp = ConfabulationAnalysisExperiment(name="custom_confab")
        assert exp.name == "custom_confab"

    def test_compute_h_score(self):
        """Test H-Score computation."""
        exp = ConfabulationAnalysisExperiment()
        features = np.array([0.5, 0.3, 0.2])
        weights = np.array([1.0, -0.5, 0.8])
        h_score = exp._compute_h_score(features, weights)
        assert isinstance(h_score, float)
        assert 0.0 <= h_score <= 1.0


class TestEntropyNeuronOverlapExperiment:
    """Tests for EntropyNeuronOverlapExperiment."""

    def test_init_defaults(self):
        """Test default initialization."""
        exp = EntropyNeuronOverlapExperiment()
        assert exp.name == "entropy_neuron_overlap"
        assert exp.probe_path is None
        assert exp.percentile == 99.0

    def test_init_custom_params(self):
        """Test custom parameters."""
        exp = EntropyNeuronOverlapExperiment(
            probe_path="test.json",
            percentile=95.0,
        )
        assert exp.probe_path == "test.json"
        assert exp.percentile == 95.0

    def test_name_property(self):
        """Test name property."""
        exp = EntropyNeuronOverlapExperiment(name="custom_entropy")
        assert exp.name == "custom_entropy"

    def test_compute_overlap(self):
        """Test overlap computation."""
        exp = EntropyNeuronOverlapExperiment()
        h_neurons = [(1, 10), (2, 20), (3, 30)]
        entropy_neurons = [(2, 20), (4, 40), (5, 50)]

        overlap_metrics = exp._compute_overlap(h_neurons, entropy_neurons)

        assert overlap_metrics["h_neurons_count"] == 3
        assert overlap_metrics["entropy_neurons_count"] == 3
        assert overlap_metrics["overlap_count"] == 1
        assert overlap_metrics["jaccard_index"] == 1.0 / 5.0  # 1 overlap, 5 union
        assert overlap_metrics["overlap_pct_of_h"] == 1.0 / 3.0
        assert overlap_metrics["overlap_pct_of_entropy"] == 1.0 / 3.0


class TestConfabulationAndEntropyImports:
    """Test that new experiments can be imported."""

    def test_import_confabulation_analysis(self):
        """Test ConfabulationAnalysisExperiment import."""
        from cotlab.experiments import ConfabulationAnalysisExperiment

        assert ConfabulationAnalysisExperiment is not None

    def test_import_entropy_neuron_overlap(self):
        """Test EntropyNeuronOverlapExperiment import."""
        from cotlab.experiments import EntropyNeuronOverlapExperiment

        assert EntropyNeuronOverlapExperiment is not None

    def test_import_sae_feature_neuron_overlap(self):
        """Test SAEFeatureNeuronOverlapExperiment import."""

        assert SAEFeatureNeuronOverlapExperiment is not None


class TestSAEFeatureNeuronOverlapExperiment:
    """Tests for SAEFeatureNeuronOverlapExperiment."""

    def test_init_defaults(self):
        """Test default initialization."""

        exp = SAEFeatureNeuronOverlapExperiment()
        assert exp.name == "sae_feature_neuron_overlap"
        assert exp.sae_repo_id == "google/gemma-scope-2b-pt-res"
        assert exp.sae_layer == 9
        assert exp.sae_feature_id == 1000
        assert exp.sae_width == "16k"
        assert exp.top_k_neurons == 50

    def test_init_custom_params(self):
        """Test custom parameters."""

        exp = SAEFeatureNeuronOverlapExperiment(
            sae_repo_id="google/gemma-scope-2-27b-it",
            sae_layer=60,
            sae_feature_id=14443,
            sae_width="262k",
            top_k_neurons=100,
        )
        assert exp.sae_repo_id == "google/gemma-scope-2-27b-it"
        assert exp.sae_layer == 60
        assert exp.sae_feature_id == 14443
        assert exp.sae_width == "262k"
        assert exp.top_k_neurons == 100

    def test_name_property(self):
        """Test name property."""

        exp = SAEFeatureNeuronOverlapExperiment(name="custom_sae")
        assert exp.name == "custom_sae"

    def test_compute_overlap(self):
        """Test overlap computation."""

        exp = SAEFeatureNeuronOverlapExperiment()
        h_neurons = [(9, 10), (9, 20), (9, 30)]
        sae_neurons = [(9, 20), (9, 40), (9, 50)]

        overlap_metrics = exp._compute_overlap(h_neurons, sae_neurons)

        assert overlap_metrics["h_neurons_count"] == 3
        assert overlap_metrics["sae_neurons_count"] == 3
        assert overlap_metrics["overlap_count"] == 1
        assert overlap_metrics["jaccard_index"] == 1.0 / 5.0
        assert overlap_metrics["overlap_pct_of_h"] == 1.0 / 3.0
        assert overlap_metrics["overlap_pct_of_sae"] == 1.0 / 3.0
        assert overlap_metrics["overlap_neurons"] == [(9, 20)]
