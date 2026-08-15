"""Equivalence test for batched generation in classification/cot_faithfulness.

Proves the batch_size plumbing produces identical generated responses to the
sequential (batch_size=1) path for greedy decoding, per sample.

Reference: src/cotlab/experiments/classification.py and cot_faithfulness.py.
"""

import pytest

from cotlab.backends.transformers_backend import TransformersBackend

pytestmark = pytest.mark.real_model

MODEL = "openai-community/gpt2"


@pytest.fixture(scope="module")
def backend():
    with TransformersBackend(device="cpu", dtype="float32", enable_hooks=True) as b:
        b.load_model(MODEL)
        yield b


def _tiny_ds(texts):
    from cotlab.datasets.loaders import Sample

    samples = [Sample(idx=i, text=t, label="x") for i, t in enumerate(texts)]

    class TinyDS:
        name = "tiny"

        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            return samples[i]

        def sample(self, n, seed=42):
            import random

            rng = random.Random(seed)
            return [self[i] for i in rng.sample(range(len(self)), min(n, len(self)))]

    return TinyDS()


def test_classification_batched_matches_sequential(backend):
    """Greedy batched generate (batch_size>1) == sequential responses."""
    from cotlab.experiments.classification import ClassificationExperiment
    from cotlab.prompts import SimplePromptStrategy

    texts = [
        "Patient presents with fever and cough.",
        "Patient presents with headache.",
        "Patient presents with rash.",
        "Patient presents with chest pain.",
    ]
    ds = _tiny_ds(texts)
    kw = dict(max_new_tokens=8, temperature=0.0, do_sample=False)
    seq = ClassificationExperiment(batch_size=1).run(backend, ds, SimplePromptStrategy(), **kw)
    bat = ClassificationExperiment(batch_size=2).run(backend, ds, SimplePromptStrategy(), **kw)
    seq_resp = [r["response"] for r in seq.raw_outputs]
    bat_resp = [r["response"] for r in bat.raw_outputs]
    assert len(seq_resp) == len(bat_resp) == len(texts)
    for i, (a, b) in enumerate(zip(seq_resp, bat_resp)):
        assert a == b, f"sample {i}: sequential={a!r} batched={b!r}"


def test_cot_faithfulness_batched_matches_sequential(backend):
    """Greedy batched generate == sequential for cot_faithfulness."""
    from cotlab.experiments.cot_faithfulness import CoTFaithfulnessExperiment
    from cotlab.prompts import ChainOfThoughtStrategy

    texts = [
        "Patient presents with fever and cough.",
        "Patient presents with headache.",
    ]
    ds = _tiny_ds(texts)
    kw = dict(max_new_tokens=8, temperature=0.0, do_sample=False)
    seq = CoTFaithfulnessExperiment(batch_size=1).run(backend, ds, ChainOfThoughtStrategy(), **kw)
    bat = CoTFaithfulnessExperiment(batch_size=2).run(backend, ds, ChainOfThoughtStrategy(), **kw)
    assert seq.metrics.get("cot_direct_agreement") == bat.metrics.get("cot_direct_agreement")
