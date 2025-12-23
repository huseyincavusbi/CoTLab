
import pytest

def test_config_loading():
    """Test that we can load the base configuration structure."""
    from cotlab.core.config import Config, ExperimentConfig

    # Just verify we can instantiate the config classes
    conf = Config()
    # Check types instead of hardcoded values where possible to be less brittle
    assert isinstance(conf.seed, int)
    assert isinstance(conf.verbose, bool)

    # Test that we can create an experiment config with required fields
    exp_conf = ExperimentConfig(name="test", _target_="some.target")
    assert exp_conf.name == "test"
    assert exp_conf._target_ == "some.target"
