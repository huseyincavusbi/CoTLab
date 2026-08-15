"""Equivalence test for probing_classifier hook-based hidden capture.

Proves forward hooks on decoder layer modules capture the last generated
token's per-layer hidden state bit-identically to ``outputs.hidden_states[-1]``
(the embedding is index 0, so layer ``l`` is at ``hidden_states[-1][l+1]``).
This lets us drop ``output_hidden_states=True`` (a memory hog that retains
steps x layers x [B, seq, hidden]) without changing the captured features.

Reference: src/cotlab/experiments/probing_classifier.py run().
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


def test_hook_capture_matches_output_hidden_states(backend):
    """Layer-module hooks capture the last-token hidden bit-identically."""
    model = backend.model
    tokenizer = backend.tokenizer

    inputs = tokenizer("The capital of France is", return_tensors="pt")
    layers = [2, 4, 6]

    captured = {}
    handles = []
    for layer in layers:
        mod = backend.hook_manager.get_layer_module(layer)
        handles.append(
            mod.register_forward_hook(
                lambda m, i, o, _l=layer: captured.__setitem__(
                    _l, (o[0] if isinstance(o, tuple) else o)[0, -1].detach()
                )
            )
        )
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        for h in handles:
            h.remove()

    last_step = outputs.hidden_states[-1]
    for layer in layers:
        ref = last_step[layer + 1][0, -1, :]  # +1: embedding is index 0
        got = captured[layer]
        assert torch.equal(ref, got), f"layer {layer} hook != output_hidden_states"
