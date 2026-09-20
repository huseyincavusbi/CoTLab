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


def test_accepts_induction_mode():
    assert ConfidenceRegulationExperiment(mode="induction").mode == "induction"
    assert ConfidenceRegulationExperiment(mode="full").mode == "full"


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
# token-frequency family
# ---------------------------------------------------------------------------


def test_freq_scores_rank_aligned_neuron_first():
    torch.manual_seed(0)
    d, m, vocab = 12, 5, 30
    w_u = torch.randn(vocab, d)
    # build v_freq inside the row space of W_U so an exact write exists
    g = torch.randn(d)
    v_freq = w_u @ g
    w_out = 0.01 * torch.randn(d, m)
    w_out[:, 2] = g / g.norm() * 2.0
    scores = ConfidenceRegulationExperiment._compute_freq_scores(w_u, w_out, v_freq)
    assert abs(scores[2]) == pytest.approx(1.0, abs=0.05)
    assert scores[2].abs() > scores.abs().max() * 0.99


def test_v_freq_is_centered_log_unigram(exp, tmp_path):
    import numpy as np

    p = tmp_path / "unigrams.npy"
    counts = np.array([10.0, 1.0, 1.0, 0.0])
    np.save(p, counts)

    class _Tok:
        @staticmethod
        def __call__(text):
            return {"input_ids": [0]}

    class _Emb:
        weight = torch.zeros(4, 3)

    class _Model:
        get_output_embeddings = staticmethod(lambda: _Emb())

    class _Backend:  # minimal surface for _get_v_freq
        model = _Model()

    exp.unigram_path = str(p)
    v = exp._get_v_freq(_Backend())
    assert v.shape == (4,)
    assert v.mean() == pytest.approx(0.0, abs=1e-6)
    assert v[0] > 0  # most frequent token -> above-mean log prob (paper Eq.: log p_i - mean)
    assert v[3] < 0  # never-seen token -> clamped low -> below mean


def test_v_freq_rejects_wrong_vocab_size(exp, tmp_path):
    import numpy as np

    p = tmp_path / "unigrams.npy"
    np.save(p, np.ones(7))

    class _Emb:
        weight = torch.zeros(4, 3)

    class _Model:
        get_output_embeddings = staticmethod(lambda: _Emb())

    class _Backend:
        model = _Model()

    exp.unigram_path = str(p)
    with pytest.raises(ValueError, match="vocab"):
        exp._get_v_freq(_Backend())


def test_rejects_unknown_family():
    with pytest.raises(ValueError, match="neuron_family"):
        ConfidenceRegulationExperiment(neuron_family="bogus")


def test_rejects_negative_layer():
    with pytest.raises(ValueError, match="layer"):
        ConfidenceRegulationExperiment(layer=-1)


# ---------------------------------------------------------------------------
# layer parameterization
# ---------------------------------------------------------------------------


class _FakeHookManager:
    def __init__(self, num_layers):
        self.num_layers = num_layers

    def get_mlp_down_proj_module(self, layer_idx):
        assert 0 <= layer_idx < self.num_layers
        return _FakeDownProj()


class _FakeDownProj:
    """nn.Linear-style weight (d_model, d_mlp); columns are neurons."""

    def __init__(self, d_model=8, d_mlp=6, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(d_model, d_mlp, generator=g)


def _fake_backend(hook_manager, d_model=8, vocab=50):
    class _Emb:
        weight = torch.zeros(vocab, d_model)

    class _OutEmb:
        weight = torch.randn(vocab, d_model, generator=torch.Generator().manual_seed(0))

    class _Model:
        get_input_embeddings = staticmethod(lambda: _Emb())
        get_output_embeddings = staticmethod(lambda: _OutEmb())

    class _Backend:
        model = _Model()

    b = _Backend()
    b.hook_manager = hook_manager
    b.model_name = "fake"
    return b


def test_resolve_layer_defaults_to_final():
    exp = ConfidenceRegulationExperiment()
    backend = _fake_backend(_FakeHookManager(4))
    assert exp._resolve_layer(backend) == 3
    assert exp._is_final_layer(backend)


def test_resolve_layer_explicit_mid_network():
    exp = ConfidenceRegulationExperiment(layer=1)
    backend = _fake_backend(_FakeHookManager(4))
    assert exp._resolve_layer(backend) == 1
    assert not exp._is_final_layer(backend)


def test_resolve_layer_rejects_out_of_range():
    exp = ConfidenceRegulationExperiment(layer=9)
    backend = _fake_backend(_FakeHookManager(4))
    with pytest.raises(ValueError, match="out of range"):
        exp._resolve_layer(backend)


def test_w_out_at_layer_returns_configured_layer_columns():
    exp = ConfidenceRegulationExperiment()
    hm = _FakeHookManager(4)
    backend = _fake_backend(hm)
    w = exp._get_w_out(backend, 2)
    assert w.shape == (8, 6)  # d_model x d_mlp, columns are neurons


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


# ---------------------------------------------------------------------------
# overlap mode
# ---------------------------------------------------------------------------


def test_overlap_mode_runs_and_reports(tmp_path):
    """Regression: overlap mode used to raise NameError on an undefined
    ``final_layer`` in its summary print, so it never returned a result."""
    p = tmp_path / "probe.json"
    p.write_text(json.dumps({"fit": {"h_neurons": [[3, 0], [3, 2]]}}))
    exp = ConfidenceRegulationExperiment(mode="overlap", probe_path=str(p), seed=0)
    backend = _fake_backend(_FakeHookManager(4))

    result = exp._run_overlap(backend)

    m = result.metrics
    assert m["mode"] == "overlap"
    assert m["h_neurons_total"] == 2
    assert m["h_neurons_in_analysis_layer"] == 2
    assert m["entropy_neuron_count"] >= 1
    assert 0 <= m["overlap_count"] <= 2
    assert 0.0 <= m["jaccard_analysis_layer"] <= 1.0
    assert 0.0 <= m["hypergeom_p"] <= 1.0


def test_rejects_unknown_overlap_layers():
    with pytest.raises(ValueError, match="overlap_layers"):
        ConfidenceRegulationExperiment(overlap_layers="bogus")


def test_hypergeom_sf_matches_exact_enumeration():
    from math import comb

    from cotlab.experiments.confidence_regulation import _hypergeom_sf

    # N=10, K=3, n=4 -> P(X>=2) = [C(3,2)C(7,2) + C(3,3)C(7,1)] / C(10,4)
    want = (comb(3, 2) * comb(7, 2) + comb(3, 3) * comb(7, 1)) / comb(10, 4)
    assert _hypergeom_sf(2, 10, 3, 4) == pytest.approx(want)
    assert _hypergeom_sf(0, 10, 3, 4) == 1.0
    assert _hypergeom_sf(5, 10, 3, 4) == 0.0  # k above the feasible max


def test_overlap_multi_layer_runs_and_reports(tmp_path):
    p = tmp_path / "probe.json"
    p.write_text(json.dumps({"fit": {"h_neurons": [[1, 0], [3, 2]]}}))
    exp = ConfidenceRegulationExperiment(
        mode="overlap", probe_path=str(p), overlap_layers="probe", seed=0
    )
    backend = _fake_backend(_FakeHookManager(4))

    result = exp._run_overlap(backend)

    m = result.metrics
    assert m["overlap_layers"] == "probe"
    assert m["layers_analyzed"] == [1, 3]
    assert len(m["per_layer"]) == 2
    assert m["pooled_h_neurons"] == 2
    assert 0.0 <= m["pooled_hypergeom_p"] <= 1.0
    for row in m["per_layer"]:
        assert 0.0 <= row["jaccard"] <= 1.0
        assert 0.0 <= row["hypergeom_p"] <= 1.0


# ---------------------------------------------------------------------------
# norm_logitvar criterion (paper Fig. 2a)
# ---------------------------------------------------------------------------


def test_select_neurons_norm_logitvar():
    exp = ConfidenceRegulationExperiment(
        selection="norm_logitvar", norm_percentile_min=60.0, logit_var_percentile_max=10.0
    )
    norms = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    logit_vars = torch.tensor([0.05, 0.04, 0.03, 0.02, 0.01])
    # only neuron 4 is both high-norm (top 40%) and low-logit-var (bottom 10%)
    assert exp._select_neurons(torch.zeros(5), norms, logit_vars) == [4]


def test_select_neurons_norm_logitvar_requires_metrics():
    exp = ConfidenceRegulationExperiment(selection="norm_logitvar")
    with pytest.raises(ValueError, match="norm_logitvar"):
        exp._select_neurons(torch.zeros(5))


def _nested_norm_backend(d_model=4, vocab=3):
    """Mimics Gemma 3's wrapper path: model.model.language_model.model.norm."""

    class _Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.full((d_model,), 2.0))

    class _Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = _Norm()

    class _Causal(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Text()

    class _Gemma3(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _Causal()

    class _Emb:
        weight = torch.ones(vocab, d_model)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Gemma3()

        def get_output_embeddings(self):
            return _Emb()

    class _Backend:
        model = _Model()

    return _Backend()


def test_final_norm_gain_resolves_nested_language_model():
    gain = ConfidenceRegulationExperiment._get_final_norm_gain(_nested_norm_backend())
    assert gain is not None
    assert torch.equal(gain, torch.full((4,), 2.0))


def test_get_unembedding_folds_nested_norm_gain():
    exp = ConfidenceRegulationExperiment(fold_final_norm=True)
    w = exp._get_unembedding(_nested_norm_backend())
    assert torch.allclose(w, torch.full((3, 4), 2.0))


def test_rejects_negative_mediate_alpha():
    with pytest.raises(ValueError, match="mediate_alpha"):
        ConfidenceRegulationExperiment(mediate_alpha=-1.0)


def test_distribution_stats_uniform_is_max_entropy():
    exp = ConfidenceRegulationExperiment()
    v = 7
    logits = torch.zeros(2, 3, v)
    stats = exp._distribution_stats(logits)
    assert torch.allclose(
        stats["entropy"], torch.full((2, 3), float(torch.log(torch.tensor(float(v)))))
    )
    assert torch.allclose(stats["max_prob"], torch.full((2, 3), 1.0 / v))
    assert torch.allclose(stats["margin"], torch.zeros(2, 3))


def test_distribution_stats_peaked_entropy_and_argmax():
    exp = ConfidenceRegulationExperiment()
    logits = torch.zeros(1, 2, 5)
    logits[0, 0, 3] = 20.0  # near-one-hot at index 3
    stats = exp._distribution_stats(logits)
    assert stats["argmax"][0, 0].item() == 3
    assert stats["max_prob"][0, 0] > 0.99
    assert stats["entropy"][0, 0] < 0.01
    assert stats["margin"][0, 0] > 0.99


def test_percentile_rank_endpoints():
    from cotlab.experiments.confidence_regulation import _percentile_rank

    pct = _percentile_rank(torch.tensor([10.0, 30.0, 20.0]))
    assert pct[0] == 0.0  # smallest
    assert pct[1].item() == pytest.approx(200.0 / 3)  # largest
    assert pct[2].item() == pytest.approx(100.0 / 3)
