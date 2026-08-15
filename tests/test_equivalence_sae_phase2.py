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


def test_phase1_batched_matches_sequential(backend):
    """Batched Phase-1 vocab probing equals sequential per-term probing."""
    import torch.nn as nn

    from cotlab.experiments.sae_feature_analysis import SAEFeatureAnalysisExperiment

    model = backend.model

    exp = SAEFeatureAnalysisExperiment(batch_size=4, max_input_tokens=64)
    exp.histo_vocab = ["pneumonia", "fever", "cough", "rash", "arthritis"]
    exp.vocab_context_prefix = "Histopathology finding:"
    exp.answer_cue = "\n\nAnswer:"

    d = model.config.hidden_size

    class FakeSAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.w_enc = nn.Parameter(torch.randn(d, 32))
            self.b_dec = nn.Parameter(torch.zeros(d))
            self.b_enc = nn.Parameter(torch.zeros(32))
            self.threshold = nn.Parameter(torch.tensor(0.1))

        @torch.no_grad()
        def encode(self, x):
            h = x - self.b_dec
            pre = h @ self.w_enc + self.b_enc
            return pre * (pre > self.threshold).float()

    torch.manual_seed(0)
    saes = {layer: FakeSAE() for layer in [2, 4, 6]}

    exp.batch_size = 1
    r_seq = exp._probe_vocab(backend, saes, [2, 4, 6])
    exp.batch_size = 4
    r_bat = exp._probe_vocab(backend, saes, [2, 4, 6])

    for layer in [2, 4, 6]:
        assert set(r_seq[layer].keys()) == set(r_bat[layer].keys()), f"layer {layer} keys differ"
        for feat, val in r_seq[layer].items():
            assert abs(val - r_bat[layer][feat]) < 1e-6, f"layer {layer} feat {feat}"
