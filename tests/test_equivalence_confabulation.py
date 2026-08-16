"""Equivalence test for confabulation_analysis batched sample loop.

Proves the left-padded batched forward (with position_ids remap) produces
per-row prediction/confidence/entropy/CETT features identical to the sequential
per-sample forward.

Reference: src/cotlab/experiments/confabulation_analysis.py
``_extract_prediction_and_cett_batch`` vs ``_extract_prediction_and_cett``.
"""

import numpy as np
import pytest

from cotlab.backends.transformers_backend import TransformersBackend

pytestmark = pytest.mark.real_model

MODEL = "openai-community/gpt2"


@pytest.fixture(scope="module")
def backend():
    with TransformersBackend(device="cpu", dtype="float32", enable_hooks=True) as b:
        b.load_model(MODEL)
        yield b


def test_batched_extraction_matches_sequential(backend):
    """Batched per-row prediction/confidence/CETT equals sequential per-sample."""
    from cotlab.experiments.confabulation_analysis import ConfabulationAnalysisExperiment

    exp = ConfabulationAnalysisExperiment(mcq_letters=["A", "B", "C", "D"], max_input_tokens=64)
    neurons = [(3, 10), (5, 20)]
    prompts = [
        "Patient presents with fever and cough.\nOptions: A) pneumonia B) flu C) asthma D) cold\n\nAnswer:",
        "Patient presents with headache.\nOptions: A) migraine B) tumor C) sinus D) stress\n\nAnswer:",
        "Patient presents with rash.\nOptions: A) eczema B) psoriasis C) allergy D) infection\n\nAnswer:",
    ]

    # --- Sequential reference ---
    seq = []
    for p in prompts:
        tokens = exp._tokenize(backend, p)
        pred, ml, ent, feats = exp._extract_prediction_and_cett(backend, tokens, neurons)
        seq.append((pred, ml, ent, feats.copy()))

    # --- Batched ---
    tokens_b = exp._tokenize_batch(backend, prompts)
    preds, mls, ents, feats_b = exp._extract_prediction_and_cett_batch(
        backend, tokens_b, neurons, len(prompts)
    )

    for i in range(len(prompts)):
        assert preds[i] == seq[i][0], f"row {i} prediction"
        assert abs(mls[i] - seq[i][1]) < 1e-4, f"row {i} max_logit"
        assert abs(ents[i] - seq[i][2]) < 1e-4, f"row {i} entropy"
        assert np.allclose(feats_b[i], seq[i][3], atol=1e-4), f"row {i} features"
