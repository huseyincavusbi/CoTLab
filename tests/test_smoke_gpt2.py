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


def test_generate_batch_batched_greedy_matches_sequential(backend):
    """Greedy batched generate (batch_size>1) equals sequential, token-exact.

    Left-padding + position_ids remap guarantee each row sees identical
    positional embeddings to its single-sample run (Layer 1 tensor + Layer 3
    token equivalence). This is the scientific contract for batching.
    """
    prompts = [
        "The capital of France is",
        "1 + 1 =",
        "The Eiffel Tower is in",
        "Two plus two equals",
    ]
    kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, do_sample=False)
    sequential = backend.generate_batch(prompts, **kwargs)
    batched = backend.generate_batch(prompts, batch_size=2, **kwargs)
    assert len(sequential) == len(batched) == len(prompts)
    for i, (s, b) in enumerate(zip(sequential, batched)):
        assert s.tokens == b.tokens, f"prompt {i} batched != sequential"


def test_generate_batch_prefill_logits_close(backend):
    """Batched prefill logits match sequential prefill logits at unmasked positions."""
    from equivalence_utils import TOL_CPU_FP32, assert_close_batched_vs_single

    prompts = ["The capital of France is", "1 + 1 =", "The Eiffel Tower is in"]
    orig_side = backend.tokenizer.padding_side
    backend.tokenizer.padding_side = "left"
    if backend.tokenizer.pad_token_id is None:
        backend.tokenizer.pad_token_id = backend.tokenizer.eos_token_id
    try:
        tokens = backend.tokenizer(prompts, return_tensors="pt", padding=True)
        masks = tokens["attention_mask"]
        position_ids = masks.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(masks == 0, 0)
        tokens["position_ids"] = position_ids
        with torch.inference_mode():
            batched_logits = backend.model(**tokens).logits
            seq_logits = []
            for p in prompts:
                single = backend.tokenizer(p, return_tensors="pt")
                seq_logits.append(backend.model(**single).logits[0])
    finally:
        backend.tokenizer.padding_side = orig_side
    atol, rtol = TOL_CPU_FP32
    assert_close_batched_vs_single(batched_logits, seq_logits, atol, rtol, masks=masks)


def test_generate_batch_uneven_chunk(backend):
    """Batched generate handles a trailing chunk (len % batch_size != 0)."""
    prompts = ["The capital of France is", "1 + 1 =", "The Eiffel Tower is in"]
    kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, do_sample=False)
    sequential = backend.generate_batch(prompts, **kwargs)
    batched = backend.generate_batch(prompts, batch_size=4, **kwargs)
    assert len(batched) == 3
    for i, (s, b) in enumerate(zip(sequential, batched)):
        assert s.tokens == b.tokens, f"prompt {i} differs (uneven chunk)"


def test_generate_batch_sampled_reproducible(backend):
    """Sampled batched generation is reproducible for a fixed seed and batch."""
    prompts = ["The capital of France is", "1 + 1 =", "The Eiffel Tower is in"]
    kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, do_sample=True)
    r1 = backend.generate_batch(prompts, batch_size=2, **kwargs)
    r2 = backend.generate_batch(prompts, batch_size=2, **kwargs)
    assert len(r1) == len(r2) == 3
    # do_sample=True draws from the global torch RNG; resetting the seed before
    # each call makes the batched stream reproducible for the identical batch.
    torch.manual_seed(1234)
    s1 = backend.generate_batch(prompts, batch_size=2, **kwargs)
    torch.manual_seed(1234)
    s2 = backend.generate_batch(prompts, batch_size=2, **kwargs)
    for i, (a, b) in enumerate(zip(s1, s2)):
        assert a.tokens == b.tokens, f"sampled batch {i} not reproducible under seed"


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
