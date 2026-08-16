"""Probe loading in confabulation_analysis.

Covers every load path (canonical weights/neurons, fit.weights, sibling
.safetensors, uniform fallback, missing data) and the standardized H-Score.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cotlab.experiments.confabulation_analysis import ConfabulationAnalysisExperiment


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


# ---------------------------------------------------------------------------
# Load paths
# ---------------------------------------------------------------------------


def test_canonical_format_weights_only(tmp_path):
    path = _write(
        tmp_path,
        "probe.json",
        {"weights": [1.0, 2.0], "neurons": [{"layer": 5, "index": 10}, {"layer": 8, "index": 20}]},
    )
    w, n, s = ConfabulationAnalysisExperiment(probe_path=path)._load_probe()
    assert list(w) == [1.0, 2.0]
    assert n == [(5, 10), (8, 20)]
    assert s is None


def test_canonical_format_with_stats(tmp_path):
    path = _write(
        tmp_path,
        "probe.json",
        {
            "weights": [1.5],
            "neurons": [{"layer": 5, "index": 10}],
            "intercept": 0.25,
            "feature_stats": {"mean": [0.1], "std": [2.0]},
        },
    )
    w, n, s = ConfabulationAnalysisExperiment(probe_path=path)._load_probe()
    assert list(w) == [1.5]
    assert n == [(5, 10)]
    assert s == {"mean": np.array([0.1]), "std": np.array([2.0]), "intercept": 0.25}


def test_fit_weights_in_json(tmp_path):
    path = _write(
        tmp_path,
        "probe.json",
        {
            "fit": {
                "h_neurons": [[19, 4479]],
                "weights": [2.5],
                "intercept": -0.05,
                "feature_stats": {"mean": [0.0], "std": [1.0]},
            }
        },
    )
    w, n, s = ConfabulationAnalysisExperiment(probe_path=path)._load_probe()
    assert list(w) == [2.5]
    assert n == [(19, 4479)]
    assert s["intercept"] == -0.05


def test_fit_weights_from_sibling_safetensors(tmp_path):
    """hprobes save() writes the coefs to the sibling .safetensors only."""
    from safetensors.torch import save_file

    dim = 2048
    flat = [19 * dim + 4479, 100, 2000]
    tensors = {
        "clf_coef": torch.tensor([2.5, 1.0, -3.0]),
        "clf_intercept": torch.tensor([0.5]),
        "top_k_idx": torch.tensor(flat, dtype=torch.long),
        "col_mean": torch.tensor([0.1, 0.2, 0.3]),
        "col_std": torch.tensor([1.5, 2.0, 1.0]),
    }
    save_file(tensors, str(tmp_path / "probe.safetensors"))
    path = _write(
        tmp_path,
        "probe.json",
        {"fit": {"h_neurons": [[19, 4479]]}, "metadata": {"intermediate_dim": dim}},
    )
    w, n, s = ConfabulationAnalysisExperiment(probe_path=path)._load_probe()
    assert list(w) == [2.5]
    assert n == [(19, 4479)]
    assert s["intercept"] == pytest.approx(0.5)
    assert s["mean"][0] == pytest.approx(0.1, abs=1e-6)
    assert s["std"][0] == pytest.approx(1.5, abs=1e-6)


def test_safetensors_missing_neuron_gets_zero_weight(tmp_path):
    from safetensors.torch import save_file

    dim = 2048
    tensors = {
        "clf_coef": torch.tensor([1.0]),
        "top_k_idx": torch.tensor([19 * dim + 4479], dtype=torch.long),
        "col_mean": torch.tensor([0.0]),
        "col_std": torch.tensor([1.0]),
    }
    save_file(tensors, str(tmp_path / "probe.safetensors"))
    path = _write(
        tmp_path,
        "probe.json",
        {"fit": {"h_neurons": [[19, 4479], [0, 1]]}, "metadata": {"intermediate_dim": dim}},
    )
    w, n, _ = ConfabulationAnalysisExperiment(probe_path=path)._load_probe()
    assert list(w) == [1.0, 0.0]


def test_uniform_fallback_warns(tmp_path, capsys):
    path = _write(tmp_path, "probe.json", {"fit": {"h_neurons": [[5, 10]]}})
    w, n, s = ConfabulationAnalysisExperiment(probe_path=path)._load_probe()
    assert list(w) == [1.0]
    assert s is None
    assert "WARNING" in capsys.readouterr().out


def test_missing_data_raises(tmp_path):
    path = _write(tmp_path, "probe.json", {"nonsense": True})
    with pytest.raises(ValueError, match="missing weights/neurons"):
        ConfabulationAnalysisExperiment(probe_path=path)._load_probe()


def test_missing_probe_path_raises():
    with pytest.raises(ValueError, match="probe_path required"):
        ConfabulationAnalysisExperiment(probe_path=None)._load_probe()


# ---------------------------------------------------------------------------
# H-Score standardization
# ---------------------------------------------------------------------------


def test_h_score_plain():
    exp = ConfabulationAnalysisExperiment()
    features = np.array([2.0, 1.0])
    weights = np.array([0.5, -0.5])
    expected = 1.0 / (1.0 + np.exp(-0.5))
    assert exp._compute_h_score(features, weights) == pytest.approx(expected)


def test_h_score_standardized_matches_hand_computation():
    exp = ConfabulationAnalysisExperiment()
    features = np.array([2.0, 1.0])
    weights = np.array([0.5, -0.5])
    stats = {"mean": np.array([1.0, 0.5]), "std": np.array([2.0, 0.5]), "intercept": 0.25}
    x = (features - stats["mean"]) / (stats["std"] + 1e-8)
    expected = 1.0 / (1.0 + np.exp(-(float(np.dot(weights, x)) + 0.25)))
    assert exp._compute_h_score(features, weights, stats) == pytest.approx(expected)


def test_h_score_standardized_matches_hprobes_formula():
    """The exact formula hprobes uses: sigmoid(coef·(x-mean)/(std+1e-8) + intercept)."""
    exp = ConfabulationAnalysisExperiment()
    rng = np.random.default_rng(0)
    features = rng.normal(size=5)
    weights = rng.normal(size=5)
    stats = {
        "mean": rng.normal(size=5),
        "std": np.abs(rng.normal(size=5)) + 0.1,
        "intercept": 0.7,
    }
    x_norm = (features - stats["mean"]) / (stats["std"] + 1e-8)
    logit = float(np.dot(weights, x_norm)) + stats["intercept"]
    expected = 1.0 / (1.0 + np.exp(-logit))
    assert exp._compute_h_score(features, weights, stats) == pytest.approx(expected)
