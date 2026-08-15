"""Equivalence tests for batched intervention hooks (Phase 2).

Proves the per-row batched hook produces identical effects to the sequential
per-variant forward, for the steering_vectors strength sweep. The batched path
replicates the clean prompt across the batch and applies a per-row multiplier,
which is mathematically identical to one forward per strength (attention rows
are isolated by the causal mask; eval mode has no dropout).

Reference: src/cotlab/experiments/steering_vectors.py.
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


def test_batched_strength_sweep_matches_sequential(backend):
    """Per-row strength hook (batched) equals one forward per strength."""
    model = backend.model
    tokenizer = backend.tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = "Patient presents with chest pain. What is the diagnosis?"
    tokens = tokenizer(prompt, return_tensors="pt")

    # Layer to steer: GPT-2 layer 3.
    layer_idx = 3
    residual_module = backend.hook_manager.get_residual_module(layer_idx)

    # Steering vector: a deterministic fixed direction (independent of extraction).
    d = model.config.hidden_size
    rng = torch.Generator().manual_seed(0)
    vector = torch.randn(1, d, generator=rng)

    strengths = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    token_you = tokenizer.encode(" You", add_special_tokens=False)[0]
    token_acute = tokenizer.encode(" Acute", add_special_tokens=False)[0]

    # --- Sequential reference: one forward per strength ---
    seq_effects = []
    with torch.inference_mode():
        for strength in strengths:
            handle = residual_module.register_forward_hook(
                lambda m, i, o, _s=strength: _steer(o, _s, vector)
            )
            try:
                logits = model(**tokens).logits
            finally:
                handle.remove()
            seq_effects.append((logits[0, -1, token_you] - logits[0, -1, token_acute]).item())

    # --- Batched: B identical rows, per-row multiplier hook ---
    B = len(strengths)
    batch_tokens = {k: v.expand(B, -1) for k, v in tokens.items()}
    strengths_t = torch.tensor(strengths).view(B, 1).float()

    batched_effects = []
    with torch.inference_mode():
        hook = residual_module.register_forward_hook(
            lambda m, i, o, _s=strengths_t: _steer_batch(o, _s, vector)
        )
        try:
            batched_logits = model(**batch_tokens).logits
        finally:
            hook.remove()
        effects = batched_logits[:, -1, token_you] - batched_logits[:, -1, token_acute]
        batched_effects = effects.float().cpu().tolist()

    assert len(batched_effects) == len(seq_effects)
    for i, (b, s) in enumerate(zip(batched_effects, seq_effects)):
        assert abs(b - s) < 1e-4, f"strength {strengths[i]}: batched={b} seq={s}"
    # Layer-2: best anti/pro layer and effect_range must agree exactly.
    assert min(batched_effects) == pytest.approx(min(seq_effects), abs=1e-4)
    assert max(batched_effects) == pytest.approx(max(seq_effects), abs=1e-4)


def _steer(output, strength, vector):
    steered = output.clone()
    steered[:, -1, :] = steered[:, -1, :] + strength * vector
    return steered


def _steer_batch(output, strengths, vector):
    steered = output.clone()
    steered[:, -1, :] = steered[:, -1, :] + strengths * vector
    return steered
