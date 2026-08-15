"""Equivalence tests for batched head-patching sweeps (Phase 2).

Proves the per-row head-patch hook produces identical logits to the sequential
per-(layer, head) forward, for cot_heads. The batched path runs one forward per
layer with num_heads identical rows, each row patching a different head slice of
the last token. Rows are isolated by the causal attention mask (eval, no
dropout), so each row reproduces its sequential counterpart exactly.

Reference: src/cotlab/experiments/cot_heads.py.
"""

import pytest
import torch

from cotlab.backends.transformers_backend import TransformersBackend

pytestmark = pytest.mark.real_model

MODEL = "openai-community/gpt2"


@pytest.fixture(scope="module")
def backend():
    with TransformersBackend(device="cpu", dtype="float32", enable_hooks=True) as b:
        b.load_model(MODEL)
        yield b


def test_batched_head_patch_matches_sequential(backend):
    """Per-row head patch (batched) equals one forward per (layer, head)."""
    model = backend.model
    tokenizer = backend.tokenizer

    config = model.config
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    prompt = "Patient presents with chest pain. What is the diagnosis?"
    tokens = tokenizer(prompt, return_tensors="pt")

    layer_idx = 3
    attn_module = backend.hook_manager.get_attention_output_module(layer_idx)

    # Source activation: deterministic fixed tensor of the right shape.
    rng = torch.Generator().manual_seed(0)
    src = torch.randn(1, tokens["input_ids"].shape[1], hidden_size, generator=rng)

    # --- Sequential reference: one forward per head ---
    seq_logits = []
    with torch.inference_mode():
        for head in range(num_heads):
            h_start = head * head_dim
            h_end = (head + 1) * head_dim
            handle = attn_module.register_forward_hook(
                lambda m, i, o, _s=src, _a=h_start, _b=h_end: _patch_single(o, _s, _a, _b)
            )
            try:
                logits = model(**tokens).logits
            finally:
                handle.remove()
            seq_logits.append(logits[0])

    # --- Batched: num_heads identical rows, per-row head patch ---
    B = num_heads
    batch_tokens = {k: v.expand(B, -1) for k, v in tokens.items()}
    src_batch = src[:, -1, :].unsqueeze(0).expand(B, -1, -1)  # [B, 1, hidden]

    batched_logits = None
    with torch.inference_mode():
        handle = attn_module.register_forward_hook(
            lambda m, i, o, _s=src_batch, _hd=head_dim: _patch_batch(o, _s, _hd)
        )
        try:
            batched_logits = model(**batch_tokens).logits
        finally:
            handle.remove()

    for head in range(num_heads):
        torch.testing.assert_close(
            batched_logits[head],
            seq_logits[head],
            atol=1e-4,
            rtol=1e-5,
            check_stride=False,
            msg=f"head {head} batched != sequential",
        )
    # Layer-2: argmax (changed flag) must agree exactly.
    batched_tops = torch.argmax(batched_logits[:, -1, :], dim=-1).tolist()
    seq_tops = [torch.argmax(lg[-1]).item() for lg in seq_logits]
    assert batched_tops == seq_tops


def _patch_single(output, src, h_start, h_end):
    patched = output.clone()
    patched[:, -1, h_start:h_end] = src[:, -1, h_start:h_end]
    return patched


def _patch_batch(output, src_batch, head_dim):
    patched = output.clone()
    B = patched.shape[0]
    for b in range(B):
        h_start = b * head_dim
        h_end = (b + 1) * head_dim
        patched[b, -1, h_start:h_end] = src_batch[b, -1, h_start:h_end]
    return patched
