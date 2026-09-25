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


def test_resolve_final_norm_module_nested():
    exp = ConfidenceRegulationExperiment()
    mod = exp._resolve_final_norm_module(_nested_norm_backend())
    assert mod is not None
    assert torch.equal(mod.weight, torch.full((4,), 2.0))


def test_get_unembedding_folds_nested_norm_gain():
    exp = ConfidenceRegulationExperiment(fold_final_norm=True)
    w = exp._get_unembedding(_nested_norm_backend())
    assert torch.allclose(w, torch.full((3, 4), 2.0))


def test_rejects_negative_mediate_alpha():
    with pytest.raises(ValueError, match="mediate_alpha"):
        ConfidenceRegulationExperiment(mediate_alpha=-1.0)


def test_accepts_intervene_mode_and_defaults_alphas():
    exp = ConfidenceRegulationExperiment(mode="intervene")
    assert exp.mode == "intervene"
    assert exp.intervene_alphas == [0.0, 2.0]
    assert ConfidenceRegulationExperiment(intervene_alphas=[1.0]).intervene_alphas == [1.0]


def test_rejects_negative_intervene_alpha():
    with pytest.raises(ValueError, match="intervene_alphas"):
        ConfidenceRegulationExperiment(intervene_alphas=[0.0, -2.0])


def test_group_stats_means_and_empirical_p():
    exp = ConfidenceRegulationExperiment(mode="intervene", seed=0)
    stats = {
        "te": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "d_entropy": torch.tensor([0.01, 0.02, 0.03, 0.04]),
        "abs_d_entropy": torch.tensor([0.01, 0.02, 0.03, 0.04]),
        "d_entropy_rel": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "flip_rate": torch.tensor([0.0, 0.1, 0.2, 0.3]),
        "d_max_prob": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        "entropy_up_frac": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        "d_entropy_pos": torch.arange(8, dtype=torch.float).reshape(4, 2),
        "baseline_entropy": 1.0,
        "baseline_max_prob": 0.5,
        "baseline_margin": 0.2,
        "baseline_accuracy": 0.3,
        "positions": 2,
    }
    groups = {"h_neuron": [0], "random_baseline": [1, 2, 3], "selected": [], "norm_matched": []}
    out = exp._group_stats(stats, [0, 1, 2, 3], groups, seed=0)
    assert out["h_neuron_mean_d_entropy"] == pytest.approx(0.01)
    assert out["random_baseline_mean_d_entropy"] == pytest.approx((0.02 + 0.03 + 0.04) / 3)
    assert out["h_neuron_mean_d_entropy_rel"] == pytest.approx(0.1)
    assert out["h_neuron_mean_flip_rate"] == pytest.approx(0.0)
    assert 0.0 <= out["h_neuron_empirical_p_vs_random"] <= 1.0
    assert len(out["h_neuron_mean_d_entropy_ci"]) == 3


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


def test_distribution_stats_accepts_precomputed_logp():
    exp = ConfidenceRegulationExperiment()
    logits = torch.randn(2, 3, 9)
    logp = torch.log_softmax(logits, dim=-1)
    a = exp._distribution_stats(logits)
    b = exp._distribution_stats(logits, logp=logp)
    for key in ("entropy", "max_prob", "margin", "argmax"):
        assert torch.allclose(a[key].float(), b[key].float())


def test_distribution_stats_peaked_entropy_and_argmax():
    exp = ConfidenceRegulationExperiment()
    logits = torch.zeros(1, 2, 5)
    logits[0, 0, 3] = 20.0  # near-one-hot at index 3
    stats = exp._distribution_stats(logits)
    assert stats["argmax"][0, 0].item() == 3
    assert stats["max_prob"][0, 0] > 0.99
    assert stats["entropy"][0, 0] < 0.01
    assert stats["margin"][0, 0] > 0.99


def test_norm_matched_indices_within_window_and_excludes():
    norms = torch.tensor([1.0, 1.01, 1.02, 5.0, 5.05, 9.0])
    matched = ConfidenceRegulationExperiment._norm_matched_indices(
        norms, [0, 3], window=0.05, exclude=(1,), seed=0
    )
    assert matched == [2, 4]


def test_norm_matched_indices_fallback_nearest_when_pool_empty():
    norms = torch.tensor([1.0, 10.0, 20.0])
    matched = ConfidenceRegulationExperiment._norm_matched_indices(norms, [0], window=0.001, seed=0)
    assert len(matched) == 1
    assert matched[0] in (1, 2)


def test_load_corpus_file_txt(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("hello world\n\nfoo bar\n")
    exp = ConfidenceRegulationExperiment(corpus_path=str(p))
    assert exp._corpus_text_or_default() == "hello world\nfoo bar"


def test_load_corpus_file_jsonl_field(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"question": "Q1", "x": 1}\n{"question": "Q2", "x": 2}\n')
    exp = ConfidenceRegulationExperiment(corpus_path=str(p))
    assert exp._corpus_text_or_default() == "Q1\nQ2"


def test_load_corpus_file_jsonl_default_key(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"text": "T1"}\n')
    exp = ConfidenceRegulationExperiment(corpus_path=str(p))
    assert exp._corpus_text_or_default() == "T1"


def test_load_corpus_file_parquet(tmp_path):
    import pandas as pd

    p = tmp_path / "c.parquet"
    pd.DataFrame({"question": ["Qa", "Qb"]}).to_parquet(p)
    exp = ConfidenceRegulationExperiment(corpus_path=str(p))
    assert exp._corpus_text_or_default() == "Qa\nQb"


def test_load_corpus_file_parquet_prefers_text_over_first_column(tmp_path):
    import pandas as pd

    p = tmp_path / "c.parquet"
    pd.DataFrame({"id": [1, 2], "text": ["Ta", "Tb"]}).to_parquet(p)
    exp = ConfidenceRegulationExperiment(corpus_path=str(p))
    assert exp._corpus_text_or_default() == "Ta\nTb"


def test_corpus_max_rows_caps_deterministic_head(tmp_path):
    import pandas as pd

    p = tmp_path / "c.parquet"
    pd.DataFrame({"question": [f"Q{i}" for i in range(10)]}).to_parquet(p)
    exp = ConfidenceRegulationExperiment(corpus_path=str(p), corpus_max_rows=3)
    assert exp._corpus_text_or_default() == "Q0\nQ1\nQ2"


def test_rejects_bad_corpus_max_rows():
    with pytest.raises(ValueError, match="corpus_max_rows"):
        ConfidenceRegulationExperiment(corpus_max_rows=0)


def test_corpus_text_used_when_no_path():
    exp = ConfidenceRegulationExperiment(corpus_text="explicit")
    assert exp._corpus_text_or_default() == "explicit"


def test_position_selector_specs():
    from cotlab.experiments.confidence_regulation import _position_selector

    assert _position_selector(None, 100) == slice(None)
    assert _position_selector("all", 100) == slice(None)
    assert _position_selector("last:16", 100) == slice(84, 100)
    assert _position_selector("last:999", 100) == slice(0, 100)  # clamped
    assert _position_selector("stride:4", 100) == slice(None, None, 4)


def test_eval_positions_validation():
    for good in ("all", "last:16", "stride:2"):
        assert ConfidenceRegulationExperiment(eval_positions=good).eval_positions == good
    for bad in ("bogus", "last:0", "last:x", "stride:-1"):
        with pytest.raises(ValueError, match="eval_positions"):
            ConfidenceRegulationExperiment(eval_positions=bad)


def test_identify_arrays_skips_logit_vars_for_rho_selection():
    exp = ConfidenceRegulationExperiment(selection="top_n", top_n=2, seed=0)
    backend = _fake_backend(_FakeHookManager(4))
    out = exp._identify_arrays(backend)
    assert out["logit_vars"] is None
    s = out["summary"]
    assert s["selected_mean_logit_var"] is None
    assert s["all_mean_logit_var"] is None
    assert s["pearson_rho_logit_var"] is None
    assert all(d["logit_var"] is None for d in out["detail"])
    assert s["selected_mean_rho"] is not None  # rho still computed


def test_identify_arrays_computes_logit_vars_for_norm_logitvar():
    exp = ConfidenceRegulationExperiment(selection="norm_logitvar", seed=0)
    backend = _fake_backend(_FakeHookManager(4))
    out = exp._identify_arrays(backend)
    assert out["logit_vars"] is not None
    assert out["summary"]["all_mean_logit_var"] is not None


def test_null_basis_cached_and_identical():
    exp = ConfidenceRegulationExperiment(seed=0)
    backend = _fake_backend(_FakeHookManager(4))
    w_u = torch.randn(20, 8, generator=torch.Generator().manual_seed(0))
    w_out = torch.randn(8, 6, generator=torch.Generator().manual_seed(1))

    rho_cached, _ = exp._compute_rho(w_u, w_out, backend)
    cache_obj = backend._v_bottom_cache
    rho_again, _ = exp._compute_rho(w_u, w_out, backend)
    assert backend._v_bottom_cache is cache_obj  # reused, not recomputed
    assert torch.allclose(rho_cached, rho_again)

    rho_nocache, _ = exp._compute_rho(w_u, w_out)  # fresh computation
    assert torch.allclose(rho_cached, rho_nocache)


def test_intervene_persists_per_neuron_and_group_indices(monkeypatch):
    exp = ConfidenceRegulationExperiment(
        mode="intervene",
        layer=0,
        top_n=2,
        random_baseline_count=3,
        intervene_alphas=[0.0, 2.0],
        seed=0,
    )
    backend = _fake_backend(_FakeHookManager(4))
    n = 6
    ident = {
        "score": torch.zeros(n),
        "selected": [0, 1],
        "norms": torch.arange(n).float(),
        "layer": 0,
        "summary": {"d_mlp": n},
    }
    monkeypatch.setattr(exp, "_capture_sequences", lambda b: (ident, [], torch.zeros(n), None))

    def fake_ablate(backend, seqs, act_mean, indices, alpha=0.0):
        m = len(indices)
        return {
            "te": torch.full((m,), 0.1),
            "d_entropy": torch.arange(m).float() * 0.01,
            "abs_d_entropy": torch.arange(m).float() * 0.01,
            "flip_rate": torch.full((m,), 0.02),
            "d_max_prob": torch.zeros(m),
            "entropy_up_frac": torch.full((m,), 0.5),
            "d_entropy_rel": torch.zeros(m),
            "d_max_prob_rel": torch.zeros(m),
            "d_entropy_pos": torch.zeros(m, 4),
            "baseline_entropy": 1.0,
            "baseline_max_prob": 0.5,
            "baseline_margin": 0.2,
            "baseline_accuracy": 0.3,
            "positions": 4,
            "de": torch.zeros(m),
            "mediated": None,
        }

    monkeypatch.setattr(exp, "_ablate_neurons_forward", fake_ablate)
    m = exp._run_intervene(backend).metrics
    assert len(m["ablated_indices"]) == len(m["group_indices"]["random_baseline"]) + len(
        m["group_indices"]["selected"]
    )
    assert len(m["group_indices"]["random_baseline"]) == 3
    keys = {str(i) for i in m["ablated_indices"]}
    for row in m["per_alpha"]:
        assert set(row["d_entropy_by_index"]) == keys
        assert set(row["flip_rate_by_index"]) == keys
        assert set(row["abs_d_entropy_by_index"]) == keys
        assert row["alpha"] in (0.0, 2.0)


def test_empirical_p_bounds_and_add_one():
    exp = ConfidenceRegulationExperiment()
    null = [0.0, 0.0, 0.0, 0.0]
    assert exp._empirical_p([10.0], null) == pytest.approx(1 / 5)  # none exceed
    assert exp._empirical_p([-10.0], null) == pytest.approx(1.0)  # all exceed
    assert exp._empirical_p([0.0], null) == pytest.approx(1.0)  # ties count as >=
    assert exp._empirical_p([], null) != exp._empirical_p([], null)  # NaN


def test_bootstrap_ci_contains_mean_and_orders_bounds():
    exp = ConfidenceRegulationExperiment()
    matrix = torch.randn(3, 50)
    mean, lo, hi = exp._bootstrap_ci(matrix, iters=200, seed=0)
    assert lo <= mean <= hi


def test_percentile_rank_endpoints():
    from cotlab.experiments.confidence_regulation import _percentile_rank

    pct = _percentile_rank(torch.tensor([10.0, 30.0, 20.0]))
    assert pct[0] == 0.0  # smallest
    assert pct[1].item() == pytest.approx(200.0 / 3)  # largest
    assert pct[2].item() == pytest.approx(100.0 / 3)


# ---------------------------------------------------------------------------
# propagated (effective-write) descriptor — config
# ---------------------------------------------------------------------------
#
# Validation gates:
#   G1 Fidelity    reduces to the paper's static rho at the final layer
#   G2 Groundtruth matches the finite-difference propagated write
#   G3 Stability   robust to ridge/corpus, not a spurious pairing
#   G4 Validity    predicts causal function (follow-up experiment; not a test)
#   G5 Generality  holds across dense / SwiGLU / gated architectures
#   G6 Context     rho_prop ranking is stable across corpora


def test_rejects_unknown_descriptor():
    with pytest.raises(ValueError, match="descriptor"):
        ConfidenceRegulationExperiment(descriptor="bogus")


def test_rejects_negative_propagation_ridge():
    with pytest.raises(ValueError, match="propagation_ridge"):
        ConfidenceRegulationExperiment(propagation_ridge=-1.0)


def test_rejects_nonpositive_propagation_sequences():
    with pytest.raises(ValueError, match="propagation_sequences"):
        ConfidenceRegulationExperiment(propagation_sequences=0)


def test_g1_fidelity_operator_identity_at_final_layer():
    torch.manual_seed(0)
    post = torch.randn(200, 8)
    operator = ConfidenceRegulationExperiment._effective_operator(post, post, ridge=0.0)
    assert operator.shape == (8, 8)
    assert torch.allclose(operator, torch.eye(8), atol=1e-5)


def test_g3_stability_ridge_keeps_finite():
    torch.manual_seed(0)
    # fewer positions than dimensions -> rank deficient without ridge
    post = torch.randn(5, 12)
    final = torch.randn(5, 12)
    operator = ConfidenceRegulationExperiment._effective_operator(post, final, ridge=1e-3)
    assert operator.shape == (12, 12)
    assert torch.isfinite(operator).all()


def test_g1_fidelity_null_fraction_matches_static_rho():
    torch.manual_seed(0)
    exp = ConfidenceRegulationExperiment(k_null=2)
    d, m = 16, 6
    w_u = torch.randn(40, d)
    w_out = torch.randn(d, m)
    v_bottom, _ = exp._null_basis(w_u, 2, None)
    rho_static, _ = exp._compute_rho(w_u, w_out)
    assert torch.allclose(exp._null_fraction(v_bottom, w_out), rho_static, atol=1e-6)


class _PropEmb:
    def __init__(self, weight):
        self.weight = weight


class _PropModel:
    def __init__(self, w_u):
        self._w_u = w_u

    def get_output_embeddings(self):
        return _PropEmb(self._w_u)


class _PropBackend:
    def __init__(self, w_u):
        self.model = _PropModel(w_u)


def test_g1_fidelity_propagated_equals_static_synthetic(monkeypatch):
    torch.manual_seed(0)
    exp = ConfidenceRegulationExperiment(k_null=2, descriptor="propagated", propagation_ridge=0.0)
    d, m, vocab = 16, 6, 40
    w_u = torch.randn(vocab, d)
    w_out = torch.randn(d, m)
    backend = _PropBackend(w_u)
    post = torch.randn(120, d)  # final layer: post-layer residual == final-norm input
    monkeypatch.setattr(exp, "_capture_residual_pair", lambda _backend, _layer: (post, post))
    rho_prop, diag = exp._propagated_rho({"w_out": w_out, "layer": 0}, backend)
    rho_static, _ = exp._compute_rho(w_u, w_out)
    assert torch.allclose(rho_prop, rho_static, atol=1e-5)
    assert diag["propagation_positions"] == 120


# ---------------------------------------------------------------------------
# shared real-model helper (config-built tiny models, no download)
# ---------------------------------------------------------------------------


def _tiny_backend(arch="gpt2", seed=0, vocab=64):
    """Config-built tiny model (no download) for hook/finite-difference tests.

    Covers the three canonical norm/MLP variants: ``gpt2`` (LayerNorm/GELU),
    ``llama`` (RMSNorm/SwiGLU), ``gemma`` (RMSNorm/gated + post-FFN norm). The
    text-dependent fake tokenizer makes distinct ``corpus_text`` values produce
    distinct id streams (used by the G6 context check).
    """
    from transformers import (
        Gemma2Config,
        Gemma2ForCausalLM,
        GPT2Config,
        GPT2LMHeadModel,
        LlamaConfig,
        LlamaForCausalLM,
    )

    from cotlab.patching.hooks import HookManager

    class _Tok:
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            ids = [(ord(c) % (vocab - 4)) + 1 for c in text if not c.isspace()]
            return {"input_ids": ((ids or [1]) * 4)[:256]}

    class _Backend:
        def __init__(self, model):
            self.model = model
            self.tokenizer = _Tok()
            self.hook_manager = HookManager(model)
            self.device = "cpu"
            self.model_name = f"tiny-{arch}"

    torch.manual_seed(seed)
    if arch == "gpt2":
        cfg = GPT2Config(
            vocab_size=vocab,
            n_positions=128,
            n_embd=32,
            n_layer=2,
            n_head=2,
            n_inner=64,
            bos_token_id=1,
            eos_token_id=2,
        )
        model = GPT2LMHeadModel(cfg)
    elif arch == "llama":
        cfg = LlamaConfig(
            vocab_size=vocab,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
        model = LlamaForCausalLM(cfg)
    elif arch == "gemma":
        # Gemma 2 (gated MLP + post_feedforward_layernorm) -- exercises the
        # post-FFN-norm write path.
        cfg = Gemma2Config(
            vocab_size=vocab,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
        )
        model = Gemma2ForCausalLM(cfg)
    else:
        raise ValueError(f"unknown arch: {arch}")
    return _Backend(model.eval())


def test_g1_fidelity_propagated_equals_static_real_model():
    backend = _tiny_backend()
    exp = ConfidenceRegulationExperiment(
        mode="identify",
        descriptor="propagated",
        top_n=4,
        # default ridge (1e-3): direct-write architectures keep exact G1 fidelity
        mediate_sequences=2,
        seq_len=64,
        seed=0,
    )
    ident = exp._identify_arrays(backend)
    assert ident["layer"] == backend.hook_manager.num_layers - 1
    rho_prop, _ = exp._propagated_rho(ident, backend)
    assert torch.allclose(rho_prop, ident["rho"], atol=1e-3)


def test_g1_fidelity_entangled_final_layer_uses_estimated_operator(monkeypatch):
    torch.manual_seed(0)
    exp = ConfidenceRegulationExperiment(k_null=2, descriptor="propagated", propagation_ridge=1e-3)
    d, m, vocab = 16, 6, 40
    w_u = torch.randn(vocab, d)
    w_out = torch.randn(d, m)
    backend = _PropBackend(w_u)
    backend.hook_manager = type("_HM", (), {"num_layers": 1})()
    post = torch.randn(120, d)
    final = torch.randn(120, d)
    operator = torch.randn(d, d)

    monkeypatch.setattr(exp, "_capture_residual_pair", lambda _backend, _layer: (post, final))
    monkeypatch.setattr(exp, "_effective_operator", lambda _post, _final, _ridge: operator)
    monkeypatch.setattr(exp, "_has_post_ffn_norm", lambda _backend: True)

    rho_prop, _ = exp._propagated_rho({"w_out": w_out, "layer": 0}, backend)
    v_bottom, _ = exp._null_basis(w_u, 2, None)
    expected = exp._null_fraction(v_bottom, operator @ w_out)
    assert torch.allclose(rho_prop, expected, atol=1e-6)


def test_g2_groundtruth_matches_finite_difference():
    backend = _tiny_backend()
    exp = ConfidenceRegulationExperiment(
        mode="identify", layer=0, top_n=4, propagation_ridge=1e-6, seq_len=64, seed=0
    )
    ident = exp._identify_arrays(backend)
    idx = ident["selected"]
    post, final = exp._capture_residual_pair(backend, 0)
    operator = exp._effective_operator(post, final, exp.propagation_ridge)
    estimated = (operator @ ident["w_out"][:, idx]).T  # (m, d_model)
    measured = exp._finite_difference_writes(backend, 0, idx, eps=1e-2, sequences=1)
    cosine = torch.nn.functional.cosine_similarity(estimated, measured, dim=1)
    assert float(cosine.mean()) > 0.8


def test_g2_groundtruth_recovers_known_map():
    torch.manual_seed(0)
    post = torch.randn(2000, 16)
    linear = torch.randn(16, 16)
    final = post @ linear + 0.01 * torch.randn(2000, 16)
    discovered = ConfidenceRegulationExperiment._effective_operator(post, final, ridge=0.0)
    # J maps column residuals, so final = post @ linear means J = linear.T
    assert (discovered.T - linear).abs().mean() < 0.05


def test_g3_stability_shuffle_control():
    torch.manual_seed(0)
    post = torch.randn(2000, 16)
    linear = torch.randn(16, 16)
    final = post @ linear + 0.01 * torch.randn(2000, 16)
    real = ConfidenceRegulationExperiment._effective_operator(post, final, ridge=0.0)
    shuffled = ConfidenceRegulationExperiment._effective_operator(
        post, final[torch.randperm(2000)], ridge=0.0
    )
    # breaking the pairing must collapse the operator far below the real one
    assert shuffled.abs().mean() < 0.25 * real.abs().mean()


def test_g3_stability_ridge_rank_deficient():
    torch.manual_seed(0)
    post = torch.randn(5, 12)
    final = post @ torch.randn(12, 12)
    operator = ConfidenceRegulationExperiment._effective_operator(post, final, ridge=1e-2)
    assert operator.shape == (12, 12)
    assert torch.isfinite(operator).all()


def test_g3_stability_corpus_size():
    backend = _tiny_backend()
    exp2 = ConfidenceRegulationExperiment(
        layer=0, top_n=4, propagation_ridge=1e-6, mediate_sequences=2, seq_len=64, seed=0
    )
    exp4 = ConfidenceRegulationExperiment(
        layer=0, top_n=4, propagation_ridge=1e-6, mediate_sequences=4, seq_len=64, seed=0
    )
    ident = exp2._identify_arrays(backend)
    rho2, _ = exp2._propagated_rho(ident, backend)
    rho4, _ = exp4._propagated_rho(ident, backend)
    assert exp2._pearson(rho2, rho4) > 0.9


# --- G5 Generality ---------------------------------------------------------


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_g5_generality_fidelity_across_architectures(arch):
    # Without a post-FFN norm the write reaches the residual directly, so the
    # propagated descriptor reduces exactly to static rho at the final layer.
    backend = _tiny_backend(arch)
    exp = ConfidenceRegulationExperiment(
        descriptor="propagated",
        top_n=4,
        # direct-write architectures keep exact final-layer G1 fidelity
        mediate_sequences=2,
        seq_len=64,
        seed=0,
    )
    ident = exp._identify_arrays(backend)
    assert ident["layer"] == backend.hook_manager.num_layers - 1
    rho_prop, _ = exp._propagated_rho(ident, backend)
    assert torch.allclose(rho_prop, ident["rho"], atol=1e-3)


def test_g5_generality_post_ffn_norm_uses_write_point():
    # Gemma-style layers normalize the MLP write before the residual add, so the
    # map is estimated from the down-projection write point (not assumed to be
    # the identity).
    backend = _tiny_backend("gemma")
    exp = ConfidenceRegulationExperiment(
        descriptor="propagated", top_n=4, mediate_sequences=2, seq_len=64, seed=0
    )
    assert exp._has_post_ffn_norm(backend)
    assert exp._write_point_module(backend, 0) is backend.hook_manager.get_mlp_down_proj_module(0)
    ident = exp._identify_arrays(backend)
    rho_prop, _ = exp._propagated_rho(ident, backend)
    assert rho_prop.shape == ident["rho"].shape
    assert torch.isfinite(rho_prop).all()


@pytest.mark.parametrize(
    ("arch", "min_cos"),
    [("gpt2", 0.8), ("llama", 0.5), ("gemma", 0.3)],
)
def test_g5_generality_groundtruth_across_architectures(arch, min_cos):
    # Thresholds loosen with the post-FFN norm: RMSNorm re-scales the perturbed
    # write, so the data-averaged linear map tracks the finite-difference
    # response less tightly than in the write-direct architectures.
    backend = _tiny_backend(arch)
    exp = ConfidenceRegulationExperiment(
        layer=0, top_n=4, propagation_ridge=1e-6, seq_len=64, seed=0
    )
    ident = exp._identify_arrays(backend)
    idx = ident["selected"]
    post, final = exp._capture_residual_pair(backend, 0)
    operator = exp._effective_operator(post, final, exp.propagation_ridge)
    estimated = (operator @ ident["w_out"][:, idx]).T
    measured = exp._finite_difference_writes(backend, 0, idx, eps=1e-2, sequences=1)
    cosine = torch.nn.functional.cosine_similarity(estimated, measured, dim=1)
    assert float(cosine.mean()) > min_cos


# --- G6 Context ------------------------------------------------------------


def test_g6_context_corpus_ranking_stable():
    backend = _tiny_backend("gpt2")
    corpus_a = " ".join("alpha beta gamma delta epsilon zeta eta theta" for _ in range(20))
    corpus_b = " ".join("one two three four five six seven eight nine ten" for _ in range(20))
    exp_a = ConfidenceRegulationExperiment(
        layer=0,
        top_n=6,
        propagation_ridge=1e-6,
        mediate_sequences=4,
        seq_len=64,
        corpus_text=corpus_a,
        seed=0,
    )
    exp_b = ConfidenceRegulationExperiment(
        layer=0,
        top_n=6,
        propagation_ridge=1e-6,
        mediate_sequences=4,
        seq_len=64,
        corpus_text=corpus_b,
        seed=0,
    )
    ident = exp_a._identify_arrays(backend)
    rho_a, _ = exp_a._propagated_rho(ident, backend)
    rho_b, _ = exp_b._propagated_rho(ident, backend)
    assert exp_a._pearson(rho_a, rho_b) > 0.9


def test_random_control_excludes_treated_groups():
    excluded = {0, 1, 2, 3}
    drawn = ConfidenceRegulationExperiment._random_control_indices(
        population=20, exclude=excluded, count=5, seed=0
    )
    assert len(drawn) == 5
    assert not (set(drawn) & excluded)
    assert len(set(drawn)) == len(drawn)
    # deterministic for a fixed seed, and capped by the available pool
    assert drawn == ConfidenceRegulationExperiment._random_control_indices(20, excluded, 5, 0)
    capped = ConfidenceRegulationExperiment._random_control_indices(
        population=6, exclude={0, 1, 2, 3, 4}, count=10, seed=0
    )
    assert capped == [5]
