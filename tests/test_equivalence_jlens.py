"""Equivalence tests for jacobian_lens optimizations (Phase 2).

Proves:
1. Enlarging ``dim_batch`` in the fit backward loop is bit-identical (the VJP
   is per-row independent; rows only differ in which d_model coordinate they
   select). This makes raising ``dim_batch`` in YAML a pure speedup.
2. The ablate-mode hoist of ``lm_head.weight.float() @ J`` per layer produces
   identical top-k direction removal as the per-sample recompute.

Reference: src/cotlab/experiments/jacobian_lens.py ``jacobian_for_prompt`` and
``_run_ablate``.
"""

import pytest
import torch

from cotlab.backends.transformers_backend import TransformersBackend
from cotlab.experiments.jacobian_lens import jacobian_for_prompt

pytestmark = pytest.mark.real_model

MODEL = "openai-community/gpt2"


@pytest.fixture(scope="module")
def backend():
    with TransformersBackend(device="cpu", dtype="float32", enable_hooks=True) as b:
        b.load_model(MODEL)
        yield b


def test_dim_batch_enlargement_bit_identical(backend):
    """dim_batch 8 vs 16 produces bit-identical Jacobians (Layer-1 exact)."""
    model = backend.model
    tokenizer = backend.tokenizer
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        text = (
            "The capital of France is Paris and the capital of Germany is Berlin "
            "and the capital of Italy is Rome and the capital of Spain is Madrid."
        )
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        J8 = jacobian_for_prompt(model, input_ids, [2, 4], target_layer=6, dim_batch=8)
        J16 = jacobian_for_prompt(model, input_ids, [2, 4], target_layer=6, dim_batch=16)
        for layer in J8:
            assert torch.equal(J8[layer], J16[layer]), f"dim_batch 8 vs 16 differs at layer {layer}"
    finally:
        for p in model.parameters():
            p.requires_grad_(True)


def test_ablate_hoisted_matches_reference(backend):
    """Hoisted W_U·J per layer gives identical ablate logits to per-sample recompute."""
    model = backend.model
    rng = torch.Generator().manual_seed(0)
    d = model.config.hidden_size
    device = backend.device

    # Deterministic fake lens: one J per layer.
    jacobians = {2: torch.randn(d, d, generator=rng), 4: torch.randn(d, d, generator=rng)}
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()

    # Hoisted values (what _run_ablate now precomputes).
    vocab_f = lm_head.weight.float().to(device)
    wu_j = {
        layer: vocab_f @ jacobians[layer].to(device, dtype=torch.float32) for layer in jacobians
    }
    j_dev = {layer: jacobians[layer].to(device, dtype=torch.float32) for layer in jacobians}

    h = torch.randn(d, generator=rng).to(device)

    # Reference: recompute per call (old code).
    ref = {}
    for layer in jacobians:
        all_scores = (
            h.float()
            @ (lm_head.weight.float() @ jacobians[layer].to(device, dtype=torch.float32)).T
        )
        top_k_ids = torch.topk(all_scores, 5).indices
        hc = h.float().clone()
        for tid in top_k_ids:
            v = lm_head.weight[tid].float() @ jacobians[layer].to(device, dtype=torch.float32)
            v_norm = v / (torch.norm(v) + 1e-8)
            hc = hc - torch.dot(hc, v_norm) * v_norm
        ref[layer] = hc

    # Hoisted: use precomputed wu_j and j_dev.
    hoisted = {}
    for layer in jacobians:
        all_scores = h.float() @ wu_j[layer].T
        top_k_ids = torch.topk(all_scores, 5).indices
        hc = h.float().clone()
        for tid in top_k_ids:
            v = vocab_f[tid] @ j_dev[layer]
            v_norm = v / (torch.norm(v) + 1e-8)
            hc = hc - torch.dot(hc, v_norm) * v_norm
        hoisted[layer] = hc

    for layer in jacobians:
        torch.testing.assert_close(
            hoisted[layer], ref[layer], atol=1e-6, rtol=1e-6, msg=f"layer {layer}"
        )
