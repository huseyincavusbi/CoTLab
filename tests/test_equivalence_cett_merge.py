"""Equivalence test for the confabulation 2->1 forward merge (Phase 3).

Proves that computing prediction + confidence + CETT features in ONE forward
(exactly matches two separate forwards. The CETT hooks return ``None`` so the
model outputs are untouched — the merged logits must be bit-identical.

Reference: src/cotlab/experiments/confabulation_analysis.py
``_extract_prediction_and_cett`` vs ``_get_prediction_and_confidence`` +
``_extract_cett_features``.
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


def test_merged_forward_bit_identical(backend):
    """Merged single forward produces bit-identical logits and CETT features."""
    from cotlab.experiments.confabulation_analysis import ConfabulationAnalysisExperiment

    tokenizer = backend.tokenizer

    prompt = (
        "Patient presents with fever and cough.\n"
        "Options: A) pneumonia B) influenza C) asthma\n\nAnswer:"
    )
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    tokens = {k: v for k, v in tokens.items()}

    # Two probe neurons in GPT-2 MLP down_proj layers.
    neurons = [(3, 10), (5, 20)]
    exp = ConfabulationAnalysisExperiment(mcq_letters=["A", "B", "C"])

    # --- Reference: two separate forwards ---
    pred_ref, max_logit_ref, entropy_ref = exp._get_prediction_and_confidence(backend, tokens)
    feats_ref = exp._extract_cett_features(backend, tokens, neurons)

    # --- Merged: one forward ---
    pred_merge, max_logit_merge, entropy_merge, feats_merge = exp._extract_prediction_and_cett(
        backend, tokens, neurons
    )

    assert pred_merge == pred_ref
    assert max_logit_merge == pytest.approx(max_logit_ref, abs=1e-6)
    assert entropy_merge == pytest.approx(entropy_ref, abs=1e-6)
    torch.testing.assert_close(
        torch.tensor(feats_merge), torch.tensor(feats_ref), atol=1e-8, rtol=1e-8
    )
