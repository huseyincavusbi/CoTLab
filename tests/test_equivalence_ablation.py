"""Equivalence tests for batched cot_ablation sweep (Phase 2).

Proves the batched per-row layer ablation (one forward with len(layers)
identical rows, one row per ablated layer) produces identical per-layer logits
to the sequential per-layer forward, for all three ablation types.

Reference: src/cotlab/experiments/cot_ablation.py
``_forward_with_ablations_batch`` vs ``_forward_with_ablation``.
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


@pytest.mark.parametrize("ablation_type", ["zero", "mean", "noise"])
def test_batched_ablation_matches_sequential(backend, ablation_type):
    """Per-row layer ablation (batched) equals one forward per ablated layer."""
    from cotlab.experiments.cot_ablation import CoTAblationExperiment

    prompt = (
        "Question: Patient has chest pain. What is the diagnosis?\n\n"
        "Let me think. The patient has chest pain. The answer is pneumonia."
    )
    layers = [2, 3, 4, 5]
    positions = [3, 4, 5]

    exp = CoTAblationExperiment(ablation_type=ablation_type)

    # Build a baseline cache via the backend path used by the experiment.
    _, cache = backend.forward_with_cache(prompt, layers=list(range(6)))

    # --- Sequential reference: one forward per layer ---
    torch.manual_seed(0)
    seq_logits = []
    for layer in layers:
        out = exp._forward_with_ablation(backend, prompt, cache, layer, positions)
        seq_logits.append(out[0])

    # --- Batched: one forward, one row per layer ---
    torch.manual_seed(0)
    batched = exp._forward_with_ablations_batch(backend, prompt, cache, layers, positions)
    batched_logits = batched[:, -1, :]

    for row, layer in enumerate(layers):
        # Compare last-token logits per row (the value the effect uses).
        torch.testing.assert_close(
            batched_logits[row],
            seq_logits[row][-1],
            atol=1e-4,
            rtol=1e-5,
            check_stride=False,
            msg=f"layer {layer} ({ablation_type}) batched != sequential",
        )
        # Layer-2: argmax must agree exactly.
        assert torch.argmax(batched_logits[row]).item() == torch.argmax(seq_logits[row][-1]).item()
