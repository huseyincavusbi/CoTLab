"""Shared helpers for the scientific-equivalence test harness.

These utilities encode the layered equivalence contract used to prove that a
speedup does not change the measured quantity:

- Layer 0 (static): weights bit-identical, config diff restricted, eval mode.
- Layer 1 (tensor): per-sample ``torch.testing.assert_close`` over unmasked
  positions only.
- Layer 2 (metrics): scientific outputs (accuracy, AUROC, rankings) stable.
- Layer 3 (RNG): greedy determinism and seeded-sampling reproducibility.

Optimizations are classified EXACT (tensor/static checks only), APPROXIMATE
(recorded tolerance), or SEMANTIC-CHANGE (new metric, config-gated).
"""

from typing import Dict, Optional, Sequence

import torch

# Tolerances (transformers `_test_eager_matches_sdpa_inference` conventions).
# bf16: 1e-2/1e-2, fp16: 5e-3/5e-3, fp32: 1e-4/1e-5. Compare unmasked positions,
# >=80% of batch elements must pass.
TOL_CPU_FP32 = (1e-4, 1e-5)
TOL_BF16 = (1e-2, 1e-2)
TOL_MPS_FP32 = (1e-3, 1e-4)  # MPS eager kernels have higher noise than CPU
TOL_SDPA_LOGITS = (5e-1, 1e-1)  # hprobes empirical eager-vs-sdpa
MIN_PASS_RATIO = 0.8


def assert_close_batched_vs_single(
    batch_out: torch.Tensor,
    single_outs: Sequence[torch.Tensor],
    atol: float,
    rtol: float,
    masks: Optional[Sequence[torch.Tensor]] = None,
    check_stride: bool = False,
) -> None:
    """Assert each batch row equals its sequential single-sample reference.

    Only unmasked positions are compared (NEXT.md tolerances require masking
    padding out of the comparison).
    """
    assert batch_out.shape[0] == len(single_outs), (
        f"batch rows ({batch_out.shape[0]}) != references ({len(single_outs)})"
    )
    passed = 0
    for i, single in enumerate(single_outs):
        row = batch_out[i]
        if masks is not None:
            seq_len = int(masks[i].sum())
            row = row[:seq_len]
            single = single[:seq_len]
        torch.testing.assert_close(row, single, atol=atol, rtol=rtol, check_stride=check_stride)
        passed += 1
    assert passed / len(single_outs) >= MIN_PASS_RATIO


def assert_static_equal(
    ref_model,
    cand_model,
    allowed_config_keys: Optional[set] = None,
) -> None:
    """Layer-0 static check: weights bit-identical, config diff restricted.

    ``allowed_config_keys`` defaults to the attention-implementation keys that
    are permitted to differ between eager and sdpa/flash (hprobes precedent).
    """
    if allowed_config_keys is None:
        allowed_config_keys = {
            "_attn_implementation",
            "_attn_implementation_internal",
            "attn_implementation",
            "torch_dtype",
            "_name_or_path",
        }
    ref_cfg = ref_model.config.to_dict()
    cand_cfg = cand_model.config.to_dict()
    differing = {k for k in ref_cfg if ref_cfg.get(k) != cand_cfg.get(k)}
    unexpected = differing - allowed_config_keys
    assert not unexpected, f"unexpected config differences: {sorted(unexpected)}"

    ref_params = dict(ref_model.named_parameters())
    cand_params = dict(cand_model.named_parameters())
    assert set(ref_params) == set(cand_params), "parameter sets differ"
    for name in ref_params:
        assert torch.equal(ref_params[name], cand_params[name]), f"weights differ for {name}"
    assert not ref_model.training and not cand_model.training, "models must be in eval mode"


def record_equivalence(metadata: Dict, **fields) -> Dict:
    """Fill the ``metadata["equivalence"]`` manifest block (idempotent)."""
    manifest = dict(metadata.get("equivalence", {}))
    manifest.update(fields)
    metadata["equivalence"] = manifest
    return metadata


def equivalence_manifest(
    optimization: str,
    device: str,
    dtype: str,
    attn_implementation: str,
    seed: int,
    batch_size: int,
    do_sample: bool,
    temperature: float,
    passed: bool = True,
    **extra,
) -> Dict:
    """Build a defensibility manifest for ``ExperimentResult.metadata``."""
    return {
        "equivalence": {
            "optimization": optimization,
            "passed": passed,
            "reference": "sequential/eager full forward",
            "device": device,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "seed": seed,
            "batch_size": batch_size,
            "do_sample": do_sample,
            "temperature": temperature,
            **extra,
        }
    }
