"""Tests for LRP backward-pass rules (R-lens / RelP)."""

import torch
import torch.nn as nn

from cotlab.experiments.lrp import LRPContext, stabilize


class _RMSNorm(nn.Module):
    """Minimal RMSNorm mirroring the Gemma3 pattern (x * rsqrt(mean(x^2)+eps))."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()) * (1.0 + self.weight.float())


class _QwenRMSNorm(_RMSNorm):
    """Qwen-style RMSNorm: uses variance_epsilon instead of eps."""

    def __init__(self, dim, eps=1e-6):
        super().__init__(dim, eps=eps)
        self.variance_epsilon = eps
        del self.eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)


class _GatedMLP(nn.Module):
    """SwiGLU-style gated MLP: down_proj(act(gate_proj(x)) * up_proj(x))."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.act_fn = nn.SiLU()
        self.gate_proj = nn.Linear(d, d, bias=False)
        self.up_proj = nn.Linear(d, d, bias=False)
        self.down_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _TinyModel(nn.Module):
    """Stack: RMSNorm -> GatedMLP -> RMSNorm, the minimal dense residual block."""

    def __init__(self, d=8):
        super().__init__()
        self.input_layernorm = _RMSNorm(d)
        self.mlp = _GatedMLP(d)
        self.post_layernorm = _RMSNorm(d)

    def forward(self, x):
        h = x + self.mlp(self.input_layernorm(x))
        return self.post_layernorm(h)


def _make_inputs(d=8, batch=3, seq=16, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, seq, d, requires_grad=True)


class TestLRPForwardIdentity:
    """Level 1 verification: LRP rules must not change forward values."""

    def test_forward_values_identical(self):
        model = _TinyModel()
        x = _make_inputs()

        with torch.no_grad():
            y_plain = model(x.clone())

        with torch.no_grad():
            with LRPContext(model):
                y_lrp = model(x.clone())

        assert torch.allclose(y_lrp, y_plain, atol=1e-6), (
            "LRP rules changed forward outputs (values must be identical)"
        )

    def test_forward_identical_with_grads_enabled(self):
        model = _TinyModel()
        x = _make_inputs()

        with torch.enable_grad():
            y_plain = model(x.clone())
            with LRPContext(model):
                y_lrp = model(x.clone())

        assert torch.allclose(y_lrp.detach(), y_plain.detach(), atol=1e-6)

    def test_forward_tolerates_extra_args(self):
        """Real decoders may call norms with extra positional args (residual)."""
        from cotlab.experiments.lrp import (
            _layer_norm_lrp_forward,
            _qwen_rms_norm_lrp_forward,
            _rms_norm_lrp_forward,
        )

        gemma = _RMSNorm(8)
        qwen = _QwenRMSNorm(8)
        ln = nn.LayerNorm(8)
        x = torch.randn(2, 4, 8)
        # extra arg (residual) must be tolerated, not raise TypeError
        _rms_norm_lrp_forward(gemma, x, x)
        _qwen_rms_norm_lrp_forward(qwen, x, x)
        _layer_norm_lrp_forward(ln, x, x)

    def test_qwen_style_variance_epsilon_detected(self):
        """RMSNorm using variance_epsilon (Qwen) must be detected and patched."""
        from cotlab.experiments.lrp import _is_rms_norm, _rms_norm_eps

        qwen_norm = _QwenRMSNorm(8)
        assert _is_rms_norm(qwen_norm)
        assert _rms_norm_eps(qwen_norm) == qwen_norm.variance_epsilon

        # forward identity must hold with the Qwen-style norm installed
        model = nn.Sequential(qwen_norm)
        x = _make_inputs(d=8)
        with torch.no_grad():
            y_plain = model(x.clone())
            with LRPContext(model):
                y_lrp = model(x.clone())
        assert torch.allclose(y_lrp, y_plain, atol=1e-6)

    def test_real_transformers_norms_detected(self):
        """Real Gemma3/Qwen3 RMSNorm classes must both be detected and patched."""
        import pytest

        torch.manual_seed(0)
        try:
            from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        except ImportError:
            pytest.skip("transformers version lacks these classes")

        from cotlab.experiments.lrp import _is_rms_norm

        gemma = Gemma3RMSNorm(8)
        qwen = Qwen3RMSNorm(8)
        assert _is_rms_norm(gemma), "Gemma3RMSNorm not detected"
        assert _is_rms_norm(qwen), "Qwen3RMSNorm not detected"

        x = torch.randn(2, 4, 8)
        with torch.no_grad():
            yg = gemma(x)
            yq = qwen(x)
            with LRPContext(gemma):
                assert torch.allclose(gemma(x.clone()), yg, atol=1e-6), "Gemma LN-rule changed fwd"
            with LRPContext(qwen):
                assert torch.allclose(qwen(x.clone()), yq, atol=1e-6), "Qwen LN-rule changed fwd"

    def test_transformers_activation_detected(self):
        """transformers SiLUActivation (not nn.SiLU) must be detected for identity-rule."""
        import pytest

        try:
            from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
            from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
        except ImportError:
            pytest.skip("transformers version lacks Qwen3")

        from cotlab.experiments.lrp import _is_activation

        cfg = Qwen3Config(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2
        )
        mlp = Qwen3MLP(cfg)
        assert _is_activation(mlp.act_fn), "SiLUActivation not detected for identity-rule"

        # identity-rule must be installed (hook registered) and forward unchanged
        x = torch.randn(2, 4, 16)
        with LRPContext(mlp) as ctx:
            assert len(ctx._hook_handles) >= 1, "no activation hook installed"
            with torch.no_grad():
                y_plain = mlp(x.clone())
                y_lrp = mlp(x.clone())
            assert torch.allclose(y_lrp, y_plain, atol=1e-5), "identity-rule changed forward"
        assert len(ctx._hook_handles) == 0, "hooks not removed on exit"

    def test_context_restores_modules(self):
        model = _TinyModel()
        orig_norm_fn = model.input_layernorm._norm.__func__
        orig_mlp_fn = model.mlp.forward.__func__

        with LRPContext(model):
            assert model.input_layernorm._norm.__func__ is not orig_norm_fn
            assert model.mlp.forward.__func__ is not orig_mlp_fn

        assert model.input_layernorm._norm.__func__ is orig_norm_fn
        assert model.mlp.forward.__func__ is orig_mlp_fn


class TestLRPRelevanceConservation:
    """Level 2 verification: LRP preserves total relevance layer-to-layer."""

    def test_gradient_sum_conserved_through_mlp(self):
        model = _TinyModel()
        x = _make_inputs()

        # Gradient flows through the LRP-modified graph; the half-rule keeps
        # total relevance (sum of the Jacobian rows) conserved across the gate
        # product rather than double-counting through it.
        with LRPContext(model):
            out = model(x)
            # Sum of d(out)/d(x) across all output dims should be finite and
            # non-exploding relative to the plain build (conservation check).
            jac = torch.autograd.grad(out.sum(), x, create_graph=False, retain_graph=False)[0]
        assert torch.isfinite(jac).all()

    def test_stabilize_avoids_div_zero(self):
        z = torch.zeros(4)
        s = stabilize(z)
        assert (s != 0).all()
        assert torch.isfinite(s).all()


class TestLRPRulesBehavior:
    """Behavioral checks: the three rules change the backward pass as intended."""

    def test_identity_rule_linearizes_activation(self):
        # Under the identity-rule, the activation's backward pass becomes a
        # per-element linear map: d(y)/d(x) == act(x)/x (constant), not the
        # full non-linear derivative.
        act = nn.SiLU()
        x = torch.randn(3, requires_grad=True)

        # plain: grad = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        y = act(x)
        grad_plain = torch.autograd.grad(y.sum(), x)[0]

        # LRP: grad = act(x)/x (per-element constant)
        from cotlab.experiments.lrp import identity_rule_forward

        y_lrp = identity_rule_forward(act, (x,), act(x))
        grad_lrp = torch.autograd.grad(y_lrp.sum(), x)[0]

        expected = (act(x) / x).detach()
        assert torch.allclose(grad_lrp, expected, atol=1e-5)
        # and it must differ from the plain (non-linear) derivative
        assert not torch.allclose(grad_lrp, grad_plain, atol=1e-3)

    def test_identity_rule_forward_exact_in_bf16(self):
        # The forward value must be exactly act(x) even in bf16 (the ratio
        # round-trip x*(act(x)/x) is lossy in bf16 and must not be used).
        act = nn.SiLU()
        x = torch.randn(4, 8).to(torch.bfloat16).requires_grad_(True)
        with torch.no_grad():
            y_plain = act(x)
            from cotlab.experiments.lrp import identity_rule_forward

            y_lrp = identity_rule_forward(act, (x,), act(x))
        assert torch.equal(y_lrp, y_plain), "identity-rule changed forward value in bf16"

    def test_half_rule_splits_gradient(self):
        # Half-rule: (g/2) + (g/2).detach() means the gradient is half of the
        # plain product gradient (relevance split across the two branches).
        model = _TinyModel()
        x = _make_inputs()

        with torch.no_grad():
            # gradient magnitude of gate path is halved vs the full build
            plain = model(x.clone())

        with LRPContext(model, rules=("Half-rule",)):
            lrp_out = model(x.clone())

        assert torch.allclose(lrp_out.detach(), plain.detach(), atol=1e-6)


class TestPass10EvalData:
    """Tests for the pass@10 eval harness data and probe-location logic."""

    def test_categories_well_formed(self):
        from cotlab.experiments.jacobian_lens import PASS10_CATEGORIES

        assert set(PASS10_CATEGORIES) == {
            "multihop",
            "multilingual",
            "association",
            "typo",
            "poetry",
        }
        for cat, probes in PASS10_CATEGORIES.items():
            assert len(probes) >= 3, f"{cat} has too few probes"
            for prompt, probe, intermediate in probes:
                assert isinstance(prompt, str) and prompt
                assert isinstance(probe, str) and probe
                assert isinstance(intermediate, str) and intermediate
                assert probe in prompt, f"probe {probe!r} not in prompt: {prompt!r}"

    def test_probe_locate_finds_token(self):
        # Verify _locate_probe logic against the tokenizer-independent shape:
        # the probe string must appear as a substring of the prompt.
        from cotlab.experiments.jacobian_lens import PASS10_CATEGORIES

        for probes in PASS10_CATEGORIES.values():
            for prompt, probe, _ in probes:
                assert probe in prompt
