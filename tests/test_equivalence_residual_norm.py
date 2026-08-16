"""Equivalence tests for residual_norm_ood batched sample loop (Phase 2).

Proves the left-padded batched forward produces per-sample norms/correctness
identical to the sequential path, with entropies within fp32 tolerance
(~1e-5 float accumulation noise from batched kernels).

Reference: src/cotlab/experiments/residual_norm_ood.py ``_forward_batch``.
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
    from cotlab.experiments.residual_norm_ood import ResidualNormOODExperiment

    tokenizer = backend.tokenizer

    prompts = [
        "Patient presents with fever and cough.\n\nAnswer:",
        "Patient presents with headache.\n\nAnswer:",
        "Patient presents with rash.\n\nAnswer:",
        "Patient presents with joint pain.\n\nAnswer:",
    ]
    exp = ResidualNormOODExperiment(batch_size=4, max_input_tokens=64, target_layer=6)

    # --- Sequential reference ---
    seq_logits, seq_hidden = [], []
    with torch.inference_mode():
        for p in prompts:
            tokens = exp._tokenize(tokenizer, p, backend.device)
            lg, hd = exp._forward(backend, tokens, target_layer=6)
            seq_logits.append(lg)
            seq_hidden.append(hd)

    # --- Batched ---
    batch_tokens = exp._tokenize_batch(tokenizer, prompts, backend.device)
    bat_logits, bat_hidden = exp._forward_batch(backend, batch_tokens, target_layer=6)

    for i in range(len(prompts)):
        # Last-token logits close within fp32 tolerance.
        torch.testing.assert_close(
            bat_logits[i], seq_logits[i], atol=1e-4, rtol=1e-5, msg=f"prompt {i} logits"
        )
        # Norm exact (same hidden values).
        assert abs(bat_hidden[i].norm(p=2).item() - seq_hidden[i].norm(p=2).item()) < 1e-5
        # Hidden close within fp32 tolerance.
        torch.testing.assert_close(
            bat_hidden[i], seq_hidden[i], atol=1e-4, rtol=1e-5, msg=f"prompt {i} hidden"
        )
