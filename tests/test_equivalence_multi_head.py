"""Equivalence tests for batched multi-head progressive sweeps (Phase 2).

Proves the nested-prefix row hook (one forward with len(sizes) identical rows,
row r patching the prefix heads[:sizes[r]]) produces identical logits to the
sequential per-size forward, for multi_head_cot and multi_head_patching.

Reference: src/cotlab/experiments/multi_head_cot.py and
src/cotlab/experiments/multi_head_patching.py ``make_nested_head_hook``.
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


def test_nested_prefix_batch_matches_sequential(backend):
    """Row r (prefix patch) equals the sequential forward for that prefix."""
    from cotlab.experiments.multi_head_cot import make_nested_head_hook

    model = backend.model
    tokenizer = backend.tokenizer

    config = model.config
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    prompt = "Question: Patient has chest pain. What is the diagnosis?\n\nAnswer:"
    tokens = tokenizer(prompt, return_tensors="pt")

    # Nested head prefixes across two layers (3 heads total -> sizes 1..3).
    layers = [2, 3]
    all_heads = [(layers[0], 0), (layers[0], 1), (layers[1], 0)]
    test_sizes = [1, 2, 3]

    rng = torch.Generator().manual_seed(0)
    cache = {
        layer: torch.randn(1, tokens["input_ids"].shape[1], hidden_size, generator=rng)
        for layer in layers
    }

    # --- Sequential reference: one forward per test size ---
    seq_tops = []
    with torch.inference_mode():
        for size in test_sizes:
            handles = []
            for layer_idx in layers:
                heads = [h for (lay, h) in all_heads[:size] if lay == layer_idx]
                if not heads:
                    continue
                attn = backend.hook_manager.get_attention_output_module(layer_idx)
                src = cache[layer_idx]
                handles.append(attn.register_forward_hook(_patch_single_head(src, heads, head_dim)))
            try:
                out = model(**tokens)
            finally:
                for h in handles:
                    h.remove()
            seq_tops.append(torch.argmax(out.logits[0, -1]).item())

    # --- Batched: one forward, one row per test size ---
    B = len(test_sizes)
    batch_tokens = {k: v.expand(B, -1) for k, v in tokens.items()}
    handles = []
    for layer_idx in layers:
        row_to_heads = {
            r: [h for (lay, h) in all_heads[:size] if lay == layer_idx]
            for r, size in enumerate(test_sizes)
        }
        row_to_heads = {r: hs for r, hs in row_to_heads.items() if hs}
        if not row_to_heads:
            continue
        attn = backend.hook_manager.get_attention_output_module(layer_idx)
        handles.append(
            attn.register_forward_hook(
                make_nested_head_hook(cache[layer_idx], row_to_heads, head_dim)
            )
        )
    try:
        with torch.inference_mode():
            out = model(**batch_tokens)
    finally:
        for h in handles:
            h.remove()
    batched_tops = torch.argmax(out.logits[:, -1, :], dim=-1).tolist()

    assert batched_tops == seq_tops, f"{batched_tops} != {seq_tops}"


def _patch_single_head(src, heads, head_dim):
    def hook(module, inp, output):
        patched = output.clone()
        for h in heads:
            h_start = h * head_dim
            h_end = (h + 1) * head_dim
            patched[:, -1, h_start:h_end] = src[:, -1, h_start:h_end]
        return patched

    return hook
