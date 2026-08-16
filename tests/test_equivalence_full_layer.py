"""Equivalence tests for batched full-layer patching sweeps (Phase 2).

Proves the batched single-layer and cumulative sweeps in full_layer_cot produce
identical logits/argmax to the sequential per-layer forward. Rows are isolated
by the causal attention mask (eval, no dropout).

Reference: src/cotlab/experiments/full_layer_cot.py ``make_patch_hook``.
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


def test_batched_single_layer_patch_matches_sequential(backend):
    """One row per layer (batched) equals one forward per patched layer."""
    from cotlab.experiments.full_layer_cot import make_patch_hook

    model = backend.model
    tokenizer = backend.tokenizer

    prompt = "Question: Patient has chest pain. What is the diagnosis?\n\nAnswer:"
    tokens = tokenizer(prompt, return_tensors="pt")

    layers = [2, 3, 4]
    d = model.config.hidden_size
    rng = torch.Generator().manual_seed(0)
    cache = {
        layer: torch.randn(1, tokens["input_ids"].shape[1], d, generator=rng) for layer in layers
    }

    # --- Sequential reference ---
    seq_tops = []
    with torch.inference_mode():
        for layer in layers:
            src = cache[layer][:, -1, :].unsqueeze(0)
            mod = backend.hook_manager.get_residual_module(layer)
            handle = mod.register_forward_hook(lambda m, i, o, _s=src: _patch_last(o, _s))
            try:
                out = model(**tokens)
            finally:
                handle.remove()
            seq_tops.append(torch.argmax(out.logits[0, -1]).item())

    # --- Batched: one row per layer ---
    B = len(layers)
    batch_tokens = {k: v.expand(B, -1) for k, v in tokens.items()}
    handles = []
    for row, layer in enumerate(layers):
        src = cache[layer][:, -1, :].unsqueeze(0).expand(B, -1, -1)
        mod = backend.hook_manager.get_residual_module(layer)
        handles.append(mod.register_forward_hook(make_patch_hook(src, row=row)))
    try:
        with torch.inference_mode():
            out = model(**batch_tokens)
    finally:
        for h in handles:
            h.remove()
    batched_tops = torch.argmax(out.logits[:, -1, :], dim=-1).tolist()

    assert batched_tops == seq_tops, f"{batched_tops} != {seq_tops}"


def test_batched_cumulative_patch_matches_sequential(backend):
    """Row k patches prefix layers[0..k]; matches sequential cumulative forwards."""
    from cotlab.experiments.full_layer_cot import make_patch_hook

    model = backend.model
    tokenizer = backend.tokenizer

    prompt = "Question: Patient has chest pain. What is the diagnosis?\n\nAnswer:"
    tokens = tokenizer(prompt, return_tensors="pt")

    layers = [2, 3, 4, 5]
    d = model.config.hidden_size
    rng = torch.Generator().manual_seed(0)
    cache = {
        layer: torch.randn(1, tokens["input_ids"].shape[1], d, generator=rng) for layer in layers
    }

    # --- Sequential reference: patch layers[0..k] for each k ---
    seq_tops = []
    with torch.inference_mode():
        for k in range(1, len(layers) + 1):
            handles = []
            for layer in layers[:k]:
                src = cache[layer][:, -1, :].unsqueeze(0)
                mod = backend.hook_manager.get_residual_module(layer)
                handles.append(
                    mod.register_forward_hook(lambda m, i, o, _s=src: _patch_last(o, _s))
                )
            try:
                out = model(**tokens)
            finally:
                for h in handles:
                    h.remove()
            seq_tops.append(torch.argmax(out.logits[0, -1]).item())

    # --- Batched: row k patches prefix; hook at layer[j] patches rows >= j ---
    B = len(layers)
    batch_tokens = {k: v.expand(B, -1) for k, v in tokens.items()}
    handles = []
    for j, layer in enumerate(layers):
        src = cache[layer][:, -1, :].unsqueeze(0).expand(B, -1, -1)
        mod = backend.hook_manager.get_residual_module(layer)
        handles.append(mod.register_forward_hook(make_patch_hook(src, rows=range(j, B))))
    try:
        with torch.inference_mode():
            out = model(**batch_tokens)
    finally:
        for h in handles:
            h.remove()
    batched_tops = torch.argmax(out.logits[:, -1, :], dim=-1).tolist()

    assert batched_tops == seq_tops, f"{batched_tops} != {seq_tops}"


def _patch_last(output, src):
    patched = output.clone()
    patched[:, -1, :] = src[:, -1, :]
    return patched
