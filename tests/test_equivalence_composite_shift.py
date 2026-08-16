"""Equivalence tests for batched sample loops (Phase 2).

Proves the left-padded batched forward (with position_ids remap) in
composite_shift_detector produces identical per-sample features to the
sequential per-sample forward.

Reference: src/cotlab/experiments/composite_shift_detector.py ``_forward_batch``
vs ``_forward``.
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


def test_batched_forward_matches_sequential(backend):
    """Batched forward (left-pad + position_ids) equals per-sample forward."""
    from cotlab.experiments.composite_shift_detector import CompositeShiftDetectorExperiment

    tokenizer = backend.tokenizer

    prompts = [
        "Patient presents with fever and cough.\n\nAnswer:",
        "Patient presents with headache.\n\nAnswer:",
        "Patient presents with rash.\n\nAnswer:",
        "Patient presents with joint pain.\n\nAnswer:",
    ]
    exp = CompositeShiftDetectorExperiment(batch_size=4, max_input_tokens=64)

    # --- Sequential reference: one forward per prompt ---
    seq_logits, seq_norms, seq_ent = [], [], []
    with torch.inference_mode():
        for p in prompts:
            tokens = exp._tokenize(tokenizer, p, backend.device)
            lg, norm, ent = exp._forward(backend, tokens, norm_layer=3)
            seq_logits.append(lg)
            seq_norms.append(norm)
            seq_ent.append(ent)

    # --- Batched: left-pad + position_ids ---
    batch_tokens = exp._tokenize_batch(tokenizer, prompts, backend.device)
    bat_logits, bat_norms, bat_ent = exp._forward_batch(backend, batch_tokens, norm_layer=3)

    for i in range(len(prompts)):
        torch.testing.assert_close(
            bat_logits[i], seq_logits[i], atol=1e-4, rtol=1e-5, msg=f"prompt {i} logits"
        )
        assert abs(bat_norms[i].item() - seq_norms[i]) < 1e-4, f"prompt {i} norm"
        if not torch.isnan(bat_ent[i]) and not torch.isnan(torch.tensor(seq_ent[i])):
            assert abs(bat_ent[i].item() - seq_ent[i]) < 1e-4, f"prompt {i} entropy"
