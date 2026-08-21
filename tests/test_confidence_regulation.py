"""Unit tests for the confidence-regulation experiment (weight-space math)."""

import json

import pytest
import torch
from torch import nn

from cotlab.experiments.confidence_regulation import ConfidenceRegulationExperiment


@pytest.fixture
def exp():
    return ConfidenceRegulationExperiment(mode="identify", seed=0)


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        ConfidenceRegulationExperiment(mode="bogus")


def test_rejects_unknown_selection():
    with pytest.raises(ValueError, match="selection"):
        ConfidenceRegulationExperiment(selection="bogus")


def test_rejects_unknown_mediate_scope():
    with pytest.raises(ValueError, match="mediate_scope"):
        ConfidenceRegulationExperiment(mediate_scope="bogus")


# ---------------------------------------------------------------------------
# token loss
# ---------------------------------------------------------------------------


def test_token_loss_matches_log_softmax():
    torch.manual_seed(0)
    exp = ConfidenceRegulationExperiment()
    logits = torch.randn(4, 7, 11) * 3
    targets = torch.randint(0, 11, (6,))
    got = exp._token_loss(logits[:, :-1], targets)
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    tgt_idx = targets.view(1, -1, 1).expand(4, -1, 1)
    want = -log_probs.gather(-1, tgt_idx).squeeze(-1)
    assert got.shape == (4, 6)
    assert torch.allclose(got, want, atol=1e-5)


# ---------------------------------------------------------------------------
# norm application and calibration
# ---------------------------------------------------------------------------


class _AffineNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.variance_epsilon = 1e-5

    def forward(self, x):
        return nn.functional.layer_norm(
            x, x.shape[-1:], self.weight, self.bias, self.variance_epsilon
        )


class _ToyGemmaRMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        dtype = x.dtype
        scaled = x.to(torch.float32)
        var = scaled.pow(2).mean(-1, keepdim=True)
        scaled = scaled * torch.rsqrt(var + self.variance_epsilon)
        return (scaled * (1.0 + self.weight.float())).to(dtype)


@pytest.mark.parametrize("d,T", [(8, 5)])
def test_apply_norm_frozen_scale_identity_affine(d, T):
    exp = ConfidenceRegulationExperiment()
    mod = _AffineNorm(d)
    cfg = exp._calibrate_norm(mod, torch.randn(2, d))
    x = torch.randn(T, d)
    fresh_scale = (x.var(-1, unbiased=False, keepdim=True) + cfg["eps"]).sqrt().squeeze(-1)
    assert torch.allclose(
        exp._apply_norm(x.unsqueeze(0), cfg, frozen_scale=fresh_scale)[0],
        exp._apply_norm(x.unsqueeze(0), cfg)[0],
        atol=1e-6,
    )


def test_calibrate_norm_detects_affine():
    exp = ConfidenceRegulationExperiment()
    cfg = exp._calibrate_norm(_AffineNorm(8), torch.randn(3, 8))
    assert cfg["gain_mode"] == "affine" and not cfg["is_rms"]


def test_calibrate_norm_detects_gemma():
    exp = ConfidenceRegulationExperiment()
    cfg = exp._calibrate_norm(_ToyGemmaRMSNorm(8), torch.randn(3, 8))
    assert cfg["gain_mode"] == "gemma" and cfg["is_rms"]


# ---------------------------------------------------------------------------
# null-space fraction rho
# ---------------------------------------------------------------------------


def test_rho_separates_top_and_bottom_singular_directions():
    torch.manual_seed(0)
    exp = ConfidenceRegulationExperiment(k_null=2)
    d, m = 16, 6
    w_u = torch.randn(50, d)
    # orthonormalize: use SVD of a random matrix for clean singular vectors
    _, _, vh = torch.linalg.svd(w_u, full_matrices=False)
    v = vh.T  # columns of v are the right singular vectors (descending)
    w_out = torch.zeros(d, m)
    w_out[:, 0] = v[:, 0] * 3.0  # top singular direction -> low rho
    w_out[:, 1] = v[:, -1] * 3.0  # bottom direction -> high rho
    rho, _ = exp._compute_rho(w_u, w_out)
    assert rho[1] > 0.99
    assert rho[0] < 1e-6


# ---------------------------------------------------------------------------
# selection and stats helpers
# ---------------------------------------------------------------------------


def test_select_neurons_top_n_and_percent(exp):
    rho = torch.tensor([0.1, 0.9, 0.5, 0.7])
    exp2 = ConfidenceRegulationExperiment(selection="top_n", top_n=2)
    assert sorted(exp2._select_neurons(rho)) == [1, 3]
    exp3 = ConfidenceRegulationExperiment(selection="top_percent", top_percent=0.5)
    assert len(exp3._select_neurons(rho)) == 2


def test_spearman_perfect_and_inverse():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert ConfidenceRegulationExperiment._spearman(a, a * 2) == pytest.approx(1.0)
    assert ConfidenceRegulationExperiment._spearman(a, -a) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# probe loading
# ---------------------------------------------------------------------------


def test_load_probe_legacy_format(exp, tmp_path):
    p = tmp_path / "probe.json"
    p.write_text(json.dumps({"neurons": [{"layer": 33, "index": 4146}]}))
    assert exp._load_probe_neurons.__self__ is exp
    exp.probe_path = str(p)
    assert exp._load_probe_neurons() == [(33, 4146)]


def test_load_probe_fit_format(exp, tmp_path):
    p = tmp_path / "probe.json"
    p.write_text(json.dumps({"fit": {"h_neurons": [[16, 1], [26, 2]]}}))
    exp.probe_path = str(p)
    assert exp._load_probe_neurons() == [(16, 1), (26, 2)]


def test_load_probe_requires_path(exp):
    with pytest.raises(ValueError, match="probe_path"):
        exp._load_probe_neurons()


def test_load_probe_missing_data(exp, tmp_path):
    p = tmp_path / "probe.json"
    p.write_text(json.dumps({"something": 1}))
    exp.probe_path = str(p)
    with pytest.raises(ValueError, match="missing neurons"):
        exp._load_probe_neurons()
