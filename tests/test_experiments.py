"""Tests for new experiment classes."""

from cotlab.experiments import (
    FullLayerPatchingExperiment,
    MultiHeadPatchingExperiment,
    SteeringVectorsExperiment,
    SycophancyHeadsExperiment,
)


class TestSycophancyHeadsExperiment:
    """Tests for SycophancyHeadsExperiment."""

    def test_init_defaults(self):
        """Test default initialization."""
        exp = SycophancyHeadsExperiment()
        assert exp.name == "sycophancy_heads"
        assert exp.search_layers == list(range(16, 26))
        assert exp.suggested_diagnosis == "anxiety"

    def test_init_custom_layers(self):
        """Test custom search layers."""
        exp = SycophancyHeadsExperiment(search_layers=[10, 11, 12])
        assert exp.search_layers == [10, 11, 12]

    def test_name_property(self):
        """Test name property."""
        exp = SycophancyHeadsExperiment(name="custom_name")
        assert exp.name == "custom_name"


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
        assert exp.target_layers == [20, 22, 17, 16]
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
        assert exp.target_layer == 20
        assert exp.steering_strengths == [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    def test_init_custom_strengths(self):
        """Test custom steering strengths."""
        custom_strengths = [-1.0, 0.0, 1.0]
        exp = SteeringVectorsExperiment(steering_strengths=custom_strengths)
        assert exp.steering_strengths == custom_strengths

    def test_init_custom_layer(self):
        """Test custom target layer."""
        exp = SteeringVectorsExperiment(target_layer=15)
        assert exp.target_layer == 15

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
        assert len(exp.search_layers) == 20  # layers 10-29

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
        assert exp.target_layers == [0, 5, 10, 15, 20, 25, 30, 33]
        assert exp.top_k == 5

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
