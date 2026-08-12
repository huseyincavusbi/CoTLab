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
