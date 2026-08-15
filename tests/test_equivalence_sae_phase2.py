"""Equivalence tests for sae_feature_analysis Phase-2 pair batching.

Proves the batched two-condition forward (few_shot + zero_shot in one 2-row
left-padded batch) captures per-row last-token residuals identical to two
sequential forwards.

Reference: src/cotlab/experiments/sae_feature_analysis.py
``_extract_last_residuals_batch``.
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


def test_batched_pair_residuals_match_sequential(backend):
    """Two-row batched last-token residuals equal two sequential forwards."""
    from cotlab.experiments.sae_feature_analysis import SAEFeatureAnalysisExperiment

    tokenizer = backend.tokenizer
    exp = SAEFeatureAnalysisExperiment(max_input_tokens=64)

    prompts = [
        "Patient presents with fever.\n\nAnswer:",
        "Here is an example.\nPatient presents with fever.\n\nAnswer:",
    ]
    layers = [2, 4]

    # --- Sequential: one forward per prompt ---
    seq_resid = {}
    with torch.inference_mode():
        for idx, p in enumerate(prompts):
            tokens = exp._tokenize(tokenizer, p, backend.device)
            resid = exp._extract_residuals(backend, tokens, layers)
            for layer in layers:
                seq_resid.setdefault(layer, {})[idx] = resid[layer][-1]

    # --- Batched: 2 rows, left-pad + position_ids ---
    batch_tokens = exp._tokenize_batch(tokenizer, prompts, backend.device)
    bat_resid = exp._extract_last_residuals_batch(backend, batch_tokens, layers)

    for layer in layers:
        for idx in range(len(prompts)):
            torch.testing.assert_close(
                bat_resid[layer][idx],
                seq_resid[layer][idx],
                atol=1e-4,
                rtol=1e-5,
                msg=f"layer {layer} prompt {idx}",
            )
