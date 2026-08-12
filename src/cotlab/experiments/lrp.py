"""LRP backward-pass rules for the R-lens (RelP).

Implements the three Layer-wise Relevance Propagation rules used to fit an
R-lens instead of a standard Jacobian lens, following RelP (arXiv:2508.21258)
and the R-lens post (Blank, Bhatia, Nanda, 2026):

  LN-rule       detach the RMSNorm/LayerNorm normalisation scale so the norm
                becomes linear in its input, preventing relevance collapse.
  Identity-rule detach the non-linear factor of GELU/SiLU so the activation's
                backward pass becomes a per-element linear map.
  Half-rule     split relevance evenly across the two branches of a SwiGLU
                product instead of double-counting through it.

All three rules are forward-pass detach() operations: the forward outputs are
identical to the standard build, only the autograd graph changes. LRPContext
installs the rules on a model's modules and restores them on exit.
"""

from __future__ import annotations

import contextlib
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Rule formulas
# ---------------------------------------------------------------------------


def stabilize(z: torch.Tensor) -> torch.Tensor:
    """Add a tiny bias to avoid division-by-zero in ratio rules."""
    return z + ((z == 0.0).to(z) + z.sign()) * 1e-6


def identity_rule_forward(
    module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor
) -> torch.Tensor:
    """Identity-rule for activation modules: x * (act(x)/x).detach().

    Forward value equals act(x); backward passes the constant factor
    act(x)/x, i.e. a per-element linear map.
    """
    x = inp[0]
    x_s = stabilize(x)
    return x_s * (out / x_s).detach()


# ---------------------------------------------------------------------------
# Module detection
# ---------------------------------------------------------------------------

_ACTIVATION_TYPES = (nn.SiLU, nn.GELU)
_LAYERNORM_TYPES = (nn.LayerNorm,)


def _rms_norm_eps(module: nn.Module) -> Optional[float]:
    """Read the epsilon from an RMSNorm module across naming conventions."""
    for attr in ("eps", "variance_epsilon", "epsilon"):
        if hasattr(module, attr):
            return getattr(module, attr)
    return None


def _is_rms_norm(module: nn.Module) -> bool:
    """Heuristic: RMSNorm modules expose a _norm(x) method plus an eps attr.

    Covers both the `eps` naming (Gemma) and `variance_epsilon` (Qwen).
    """
    return (
        hasattr(module, "_norm")
        and _rms_norm_eps(module) is not None
        and not isinstance(module, nn.LayerNorm)
    )


def _is_layer_norm(module: nn.Module) -> bool:
    return isinstance(module, _LAYERNORM_TYPES)


def _is_activation(module: nn.Module) -> bool:
    return isinstance(module, _ACTIVATION_TYPES)


def _is_gated_mlp(module: nn.Module) -> bool:
    return all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn"))


def _is_residual_norm(module: nn.Module, name: str) -> bool:
    """Residual-stream norms, excluding q/k norms which R-lens does not touch."""
    if "q_norm" in name or "k_norm" in name or "qnorm" in name or "knorm" in name:
        return False
    return _is_rms_norm(module) or _is_layer_norm(module)


# ---------------------------------------------------------------------------
# Patched forward implementations
# ---------------------------------------------------------------------------


def _rms_norm_lrp_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """LN-rule for RMSNorm: detach the rsqrt scale factor.

    Matches the standard RMSNorm._norm pattern (Gemma / Qwen) with the rsqrt
    detached; reads the epsilon from either naming convention.
    """
    eps = _rms_norm_eps(module)
    if eps is None:
        raise AttributeError(f"RMSNorm module {type(module).__name__} has no epsilon attribute")
    scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * scale.detach()


def _layer_norm_lrp_forward(module: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    """LN-rule for LayerNorm: detach the normalisation scale (std)."""
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    scale = torch.rsqrt(var + module.eps)
    x_norm = (x - mean) * scale.detach()
    return x_norm * module.weight + module.bias


def _gated_mlp_lrp_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Identity-rule (on the gate activation) + Half-rule (on the product).

    Assumes the SwiGLU pattern: out = down_proj(act_fn(gate_proj(x)) * up_proj(x)).
    The gate activation's identity-rule is applied implicitly by the forward
    hook registered on the activation module; here we only split the product.
    """
    gate_out = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
    gate_out = (gate_out / 2.0) + (gate_out / 2.0).detach()
    return module.down_proj(gate_out)


# ---------------------------------------------------------------------------
# LRPContext: install / restore rules on a model
# ---------------------------------------------------------------------------


class LRPContext:
    """Context manager that installs LRP backward-pass rules on a model.

    Usage:
        with LRPContext(model, rules=("LN-rule", "Identity-rule", "Half-rule")):
            # forward passes here compute relevance instead of gradients

    The forward values are identical to the standard build; only the autograd
    graph changes. Modules are restored on exit.
    """

    def __init__(
        self,
        model: nn.Module,
        rules: Tuple[str, ...] = ("LN-rule", "Identity-rule", "Half-rule"),
    ):
        self.model = model
        self.rules = rules
        self._norm_handles: List[Tuple[nn.Module, str, object]] = []  # (module, attr, original)
        self._mlp_handles: List[Tuple[nn.Module, object]] = []  # (module, original_forward)
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "LRPContext":
        if "LN-rule" in self.rules:
            self._install_norm_rules()
        if "Identity-rule" in self.rules:
            self._install_activation_hooks()
        if "Half-rule" in self.rules:
            self._install_gated_mlp_rules()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        for module, attr, original in self._norm_handles:
            setattr(module, attr, original)
        self._norm_handles.clear()
        for module, original_forward in self._mlp_handles:
            module.forward = original_forward
        self._mlp_handles.clear()

    # -- installers --------------------------------------------------------

    def _install_norm_rules(self) -> None:
        for name, module in self.model.named_modules():
            if not _is_residual_norm(module, name):
                continue
            if _is_rms_norm(module):
                self._norm_handles.append((module, "_norm", module._norm))
                module._norm = _rms_norm_lrp_forward.__get__(module, type(module))
            elif _is_layer_norm(module):
                self._norm_handles.append((module, "forward", module.forward))
                module.forward = _layer_norm_lrp_forward.__get__(module, type(module))

    def _install_activation_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if _is_activation(module):
                self._hook_handles.append(module.register_forward_hook(identity_rule_forward))

    def _install_gated_mlp_rules(self) -> None:
        for name, module in self.model.named_modules():
            if _is_gated_mlp(module):
                self._mlp_handles.append((module, module.forward))
                module.forward = _gated_mlp_lrp_forward.__get__(module, type(module))


@contextlib.contextmanager
def lrp_context(
    model: nn.Module, rules: Tuple[str, ...] = ("LN-rule", "Identity-rule", "Half-rule")
):
    """Convenience context manager wrapping LRPContext."""
    ctx = LRPContext(model, rules=rules)
    with ctx:
        yield
