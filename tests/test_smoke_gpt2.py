"""Real-model smoke and determinism tests on GPT-2 (CPU, float32).

Establishes the real-model equivalence harness baseline for CoTLab. These are
the first tests to exercise a real ``AutoModelForCausalLM`` forward/generate
path, gated behind the ``real_model`` marker. CI runs them in the
``integration-tests`` job which already caches the GPT-2 weights
(``tests.yml``: ``-hf-gpt2`` cache key). All checks are greedy/deterministic so
the RNG layer of the equivalence contract (Layer 3) is provable exactly.
"""

import pytest
import torch

from cotlab.backends.transformers_backend import TransformersBackend

pytestmark = pytest.mark.real_model

MODEL = "openai-community/gpt2"
MAX_NEW_TOKENS = 8


@pytest.fixture(scope="module")
def backend():
    with TransformersBackend(device="cpu", dtype="float32", enable_hooks=True) as b:
        b.load_model(MODEL)
        yield b


def test_generate_greedy_deterministic(backend):
    """Greedy generation is deterministic run-to-run (Layer 3)."""
    kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, do_sample=False)
    out1 = backend.generate("The capital of France is", **kwargs)
    out2 = backend.generate("The capital of France is", **kwargs)
    assert out1.tokens, "expected generated tokens"
    assert out1.tokens == out2.tokens, "greedy generation must be deterministic"


def test_generate_batch_sequential_matches_generate(backend):
    """Sequential generate_batch (batch_size=1) equals per-prompt generate."""
    prompts = ["The capital of France is", "1 + 1 ="]
    kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, do_sample=False)
    batched = backend.generate_batch(prompts, **kwargs)
    assert len(batched) == len(prompts)
    for i, prompt in enumerate(prompts):
        single = backend.generate(prompt, **kwargs)
        assert batched[i].tokens == single.tokens, f"prompt {i} differs"


def test_forward_with_cache_hooks_fire(backend):
    """forward_with_cache returns logits plus a per-layer activation cache."""
    logits, cache = backend.forward_with_cache("The Eiffel Tower is in", layers=[0, 2, 4])
    assert logits.shape[0] == 1
    assert set(cache.layers) == {0, 2, 4}
    for layer in cache.layers:
        act = cache.get(layer)
        assert act.shape[-1] == backend.model.config.hidden_size


def test_forward_with_attention_cache_hooks_fire(backend):
    """forward_with_attention_cache returns logits plus attention-output cache."""
    logits, cache = backend.forward_with_attention_cache("The Eiffel Tower is in", layers=[0, 2])
    assert logits.shape[0] == 1
    assert set(cache.layers) == {0, 2}
    assert cache.get(0).shape[-1] == backend.model.config.hidden_size


def test_inference_mode_bit_identical(backend):
    """torch.inference_mode is numerically identical to torch.no_grad."""
    tokens = backend.tokenizer("The Eiffel Tower is in", return_tensors="pt")
    with torch.no_grad():
        ref = backend.model(**tokens).logits
    with torch.inference_mode():
        cand = backend.model(**tokens).logits
    assert torch.equal(ref, cand), "inference_mode must be bit-identical to no_grad"


def test_inference_mode_cache_hooks_identical(backend):
    """Residual-cache hooks fire under inference_mode and capture identical values."""
    logits_ref, cache_ref = backend.forward_with_cache("The Eiffel Tower is in", layers=[0, 2, 4])
    assert logits_ref.shape[0] == 1
    assert set(cache_ref.layers) == {0, 2, 4}
    # forward_with_cache runs under torch.inference_mode internally; verify the
    # captured activations are the same values the no_grad path would produce.
    tokens = backend.tokenizer("The Eiffel Tower is in", return_tensors="pt")
    handles = []
    captured = {}
    with torch.no_grad():
        for layer in [0, 2, 4]:
            mod = backend.hook_manager.get_residual_module(layer)
            handles.append(
                mod.register_forward_hook(
                    lambda m, i, o, _l=layer: captured.update(
                        {_l: (o[0] if isinstance(o, tuple) else o).detach().clone()}
                    )
                )
            )
        backend.model(**tokens)
    for h in handles:
        h.remove()
    assert set(captured) == set(cache_ref.layers)
    for layer in cache_ref.layers:
        assert torch.equal(cache_ref.get(layer), captured[layer]), (
            f"layer {layer} cache differs between inference_mode and no_grad"
        )
