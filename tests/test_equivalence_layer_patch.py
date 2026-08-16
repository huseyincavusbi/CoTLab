"""Equivalence tests for batched layer-patching sweep (Phase 2).

Proves the batched per-row layer patch (one forward per sample, one row per
target layer) produces identical logits to the sequential per-layer forward, for
activation_patching. Each row patches a different layer's residual module output
with the clean last-token vector broadcast over positions; rows are isolated by
the causal attention mask (eval, no dropout).

Reference: src/cotlab/experiments/activation_patching.py
``_forward_patched_batch`` vs ``_forward_patched``.
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


def test_batched_layer_patch_matches_sequential(backend):
    """Per-row layer patch (batched) equals one forward per patched layer."""
    model = backend.model
    tokenizer = backend.tokenizer

    prompt = "Question: Patient has chest pain. What is the diagnosis?\n\nAnswer:"
    tokens = tokenizer(prompt, return_tensors="pt")

    target_layers = [2, 3, 4, 5]
    d = model.config.hidden_size
    rng = torch.Generator().manual_seed(0)
    # Patch vectors: deterministic fixed directions (independent of extraction).
    act_cache = {layer: torch.randn(d, generator=rng) for layer in target_layers}

    # --- Sequential reference: one forward per layer ---
    seq_logits = []
    with torch.inference_mode():
        for layer in target_layers:
            mod = backend.hook_manager.get_residual_module(layer)
            patch_gpu = act_cache[layer].unsqueeze(0).unsqueeze(0)
            handle = mod.register_forward_hook(
                lambda m, i, o, _p=patch_gpu: _patch_all_positions(o, _p)
            )
            try:
                out = model(**tokens)
            finally:
                handle.remove()
            seq_logits.append(out.logits[0, -1])

    # --- Batched: one forward, one row per target layer ---
    B = len(target_layers)
    batch_tokens = {k: v.expand(B, -1) for k, v in tokens.items()}
    row_of_layer = {layer: j for j, layer in enumerate(target_layers)}
    handles = []
    for layer in target_layers:
        mod = backend.hook_manager.get_residual_module(layer)
        patch_gpu = act_cache[layer].unsqueeze(0).unsqueeze(0)
        row = row_of_layer[layer]
        handles.append(
            mod.register_forward_hook(lambda m, i, o, _p=patch_gpu, _r=row: _patch_row(o, _p, _r))
        )
    try:
        with torch.inference_mode():
            out = model(**batch_tokens)
    finally:
        for h in handles:
            h.remove()
    batched_logits = out.logits[:, -1, :]

    for j, layer in enumerate(target_layers):
        torch.testing.assert_close(
            batched_logits[j],
            seq_logits[j],
            atol=1e-4,
            rtol=1e-5,
            check_stride=False,
            msg=f"layer {layer} batched != sequential",
        )
    # Layer-2: argmax must agree exactly.
    batched_tops = torch.argmax(batched_logits, dim=-1).tolist()
    seq_tops = [torch.argmax(lg).item() for lg in seq_logits]
    assert batched_tops == seq_tops


def _patch_all_positions(output, patch_gpu):
    if isinstance(output, tuple):
        patched = list(output)
        patched[0] = patch_gpu.expand_as(output[0])
        return tuple(patched)
    return patch_gpu.expand_as(output)


def _patch_row(output, patch_gpu, row):
    broadcast = patch_gpu[0]  # [1, hidden] for a single row
    if isinstance(output, tuple):
        patched = list(output)
        patched[0] = patched[0].clone()
        patched[0][row] = broadcast.expand_as(patched[0][row])
        return tuple(patched)
    patched = output.clone()
    patched[row] = broadcast.expand_as(patched[row])
    return patched
