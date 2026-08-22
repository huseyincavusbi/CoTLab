"""Unit tests for the probe-confidence correlation experiment."""

import json

import pytest
import torch

from cotlab.experiments.probe_confidence import ProbeConfidenceExperiment


@pytest.fixture
def exp():
    return ProbeConfidenceExperiment(bootstrap_iters=0)


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def test_rejects_negative_bootstrap_iters():
    with pytest.raises(ValueError, match="bootstrap_iters"):
        ProbeConfidenceExperiment(bootstrap_iters=-1)


def test_run_requires_probe_path(exp):
    class _HM:
        num_layers = 1

    class _Backend:
        hook_manager = _HM()

    samples = [{"prompt": "a", "response": "b", "label": True}]
    with pytest.raises(ValueError, match="probe_path"):
        exp.run(backend=_Backend(), dataset=samples)


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------


def test_spearman_perfect_inverse_and_ties():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert ProbeConfidenceExperiment._spearman(a, a * 2) == pytest.approx(1.0)
    assert ProbeConfidenceExperiment._spearman(a, -a) == pytest.approx(-1.0)
    tied = torch.tensor([1.0, 1.0, 2.0, 3.0])
    assert -1 <= ProbeConfidenceExperiment._spearman(tied, a) <= 1


def test_rankdata_averages_ties():
    # 0-based average ranks; shift-invariant, hence identical Spearman
    r = ProbeConfidenceExperiment._rankdata(torch.tensor([3.0, 1.0, 2.0, 2.0]))
    assert r.tolist() == [3.0, 0.0, 1.5, 1.5]


def test_auroc_known_case():
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([True, True, False, False])
    assert ProbeConfidenceExperiment._auroc(scores, labels) == pytest.approx(1.0)
    assert ProbeConfidenceExperiment._auroc(scores.flip(0), labels) == pytest.approx(0.0)


def test_auroc_nan_when_single_class():
    s = torch.tensor([0.1, 0.2])
    y = torch.tensor([True, True])
    assert torch.isnan(torch.tensor(ProbeConfidenceExperiment._auroc(s, y)))


def test_partial_spearman_removes_length_confound():
    torch.manual_seed(0)
    n = 200
    length = torch.randn(n)
    entropy = length * 0.9 + 0.1 * torch.randn(n)
    score = torch.randn(n)  # independent of both
    raw = ProbeConfidenceExperiment._spearman(score, entropy)
    part = ProbeConfidenceExperiment._partial_spearman_length(score, entropy, length)
    # controlling for length should not create correlation out of nothing
    assert abs(part) < 0.35 and abs(raw) < 0.35


# ---------------------------------------------------------------------------
# probe loading and scoring
# ---------------------------------------------------------------------------


def test_load_probe_dense_json(exp, tmp_path):
    p = tmp_path / "probe.json"
    p.write_text(
        json.dumps(
            {
                "fit": {
                    "weights": [1.0, -1.0],
                    "intercept": 0.5,
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                }
            }
        )
    )
    exp.probe_path = str(p)
    w = exp._load_probe_weights()
    assert w["top_idx"] is None
    assert w["intercept"] == 0.5


def test_score_features_matches_manual_sigmoid(exp):
    probe = {
        "coef": torch.tensor([2.0]),
        "intercept": 0.0,
        "mean": torch.tensor([1.0]),
        "std": torch.tensor([1.0]),
        "top_idx": None,
    }
    x = torch.tensor([2.0])  # z = (2-1)/1 = 1 -> sigmoid(2*1) ~ 0.881
    s = exp._score_features(x, probe)
    assert s == pytest.approx(float(torch.sigmoid(torch.tensor(2.0))), abs=1e-6)


def test_score_features_applies_top_k_selection(exp):
    probe = {
        "coef": torch.tensor([5.0, 0.0]),
        "intercept": 0.0,
        "mean": torch.tensor([0.0, 0.0]),
        "std": torch.tensor([1.0, 1.0]),
        "top_idx": torch.tensor([0]),
    }
    x = torch.tensor([1.0, 99.0])  # second dim must be ignored via top_idx
    s = exp._score_features(x, probe)
    assert s == pytest.approx(float(torch.sigmoid(torch.tensor(5.0))), abs=1e-6)


def test_load_probe_requires_path(exp):
    with pytest.raises(ValueError, match="probe_path"):
        exp._load_probe_weights()


def test_load_probe_missing_weights(exp, tmp_path):
    p = tmp_path / "probe.json"
    p.write_text(json.dumps({"something": 1}))
    exp.probe_path = str(p)
    with pytest.raises(ValueError, match="no usable probe weights"):
        exp._load_probe_weights()
