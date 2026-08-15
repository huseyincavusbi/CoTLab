"""Equivalence tests for jacobian_lens batched intervention modes (Phase 2).

Proves the batched sample-row intervention (left-pad + position_ids, per-row
hook) reproduces the sequential per-sample forward for steer and swap modes.

Reference: src/cotlab/experiments/jacobian_lens.py ``_run_steer`` / ``_run_swap``.
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


def test_batched_steer_matches_sequential(backend):
    """Batched per-row steer equals one sequential steer forward per sample."""
    from cotlab.experiments.jacobian_lens import JacobianLensExperiment

    model = backend.model
    tokenizer = backend.tokenizer

    prompts = [
        "Question: Patient has chest pain. What is the diagnosis?\n\nAnswer:",
        "Question: Patient has a headache. What is the diagnosis?\n\nAnswer:",
        "Question: Patient has a fever. What is the diagnosis?\n\nAnswer:",
    ]
    exp = JacobianLensExperiment()
    exp.max_input_tokens = 64
    exp.steer_alpha = 2.0

    layer = 2
    rng = torch.Generator().manual_seed(0)
    d = model.config.hidden_size
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    target_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    J = torch.randn(d, d, generator=rng)
    v_t = lm_head.weight[target_id].float() @ J  # [d]

    # --- Sequential reference: one forward per prompt ---
    seq_logits = []
    with torch.inference_mode():
        for p in prompts:
            tokens = exp._tokenize_batch(tokenizer, [p], backend.device)
            block = backend.hook_manager.get_layer_module(layer)
            handle = block.register_forward_hook(
                lambda m, i, o: _steer_hook(o, exp.steer_alpha, v_t)
            )
            try:
                out = model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    position_ids=tokens["position_ids"],
                    output_hidden_states=False,
                    use_cache=False,
                )
            finally:
                handle.remove()
            seq_logits.append(out.logits[0, -1])

    # --- Batched: all prompts as rows, one forward ---
    batch = exp._tokenize_batch(tokenizer, prompts, backend.device)
    block = backend.hook_manager.get_layer_module(layer)
    handle = block.register_forward_hook(lambda m, i, o: _steer_hook(o, exp.steer_alpha, v_t))
    try:
        with torch.inference_mode():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                output_hidden_states=False,
                use_cache=False,
            )
    finally:
        handle.remove()
    bat_logits = out.logits[:, -1, :]

    for i in range(len(prompts)):
        torch.testing.assert_close(
            bat_logits[i], seq_logits[i], atol=1e-4, rtol=1e-5, msg=f"prompt {i}"
        )
        assert torch.argmax(bat_logits[i]).item() == torch.argmax(seq_logits[i]).item()


def _steer_hook(output, alpha, vec):
    if isinstance(output, tuple):
        t, rest = output[0], output[1:]
    else:
        t, rest = output, ()
    t[:, -1, :] = t[:, -1, :] + alpha * vec
    return (t,) + rest if rest else t


def test_batched_swap_matches_sequential(backend):
    """Batched per-row swap equals one sequential swap forward per sample."""
    from cotlab.experiments.jacobian_lens import JacobianLensExperiment

    model = backend.model
    tokenizer = backend.tokenizer

    prompts = [
        "Question: Patient has chest pain. What is the diagnosis?\n\nAnswer:",
        "Question: Patient has a headache. What is the diagnosis?\n\nAnswer:",
        "Question: Patient has a fever. What is the diagnosis?\n\nAnswer:",
    ]
    exp = JacobianLensExperiment()
    exp.max_input_tokens = 64

    layer = 2
    rng = torch.Generator().manual_seed(1)
    d = model.config.hidden_size
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    src_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    tgt_id = tokenizer.encode(" Berlin", add_special_tokens=False)[0]
    J = torch.randn(d, d, generator=rng)
    v_src = (lm_head.weight[src_id].float() @ J).unsqueeze(1)
    v_tgt = (lm_head.weight[tgt_id].float() @ J).unsqueeze(1)
    V = torch.cat([v_src, v_tgt], dim=1)
    device = backend.device
    V_pinv = torch.linalg.pinv(V.T @ V + 1e-6 * torch.eye(2)) @ V.T

    # --- Sequential reference: one forward per prompt ---
    seq_logits = []
    with torch.inference_mode():
        for p in prompts:
            tokens = exp._tokenize_batch(tokenizer, [p], backend.device)
            block = backend.hook_manager.get_layer_module(layer)
            handle = block.register_forward_hook(
                lambda m, i, o: _swap_hook(o, V.to(device), V_pinv.to(device))
            )
            try:
                out = model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    position_ids=tokens["position_ids"],
                    output_hidden_states=False,
                    use_cache=False,
                )
            finally:
                handle.remove()
            seq_logits.append(out.logits[0, -1])

    # --- Batched: all prompts as rows, one forward ---
    batch = exp._tokenize_batch(tokenizer, prompts, backend.device)
    block = backend.hook_manager.get_layer_module(layer)
    handle = block.register_forward_hook(
        lambda m, i, o: _swap_hook(o, V.to(device), V_pinv.to(device))
    )
    try:
        with torch.inference_mode():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                output_hidden_states=False,
                use_cache=False,
            )
    finally:
        handle.remove()
    bat_logits = out.logits[:, -1, :]

    for i in range(len(prompts)):
        torch.testing.assert_close(
            bat_logits[i], seq_logits[i], atol=1e-4, rtol=1e-5, msg=f"prompt {i}"
        )
        assert torch.argmax(bat_logits[i]).item() == torch.argmax(seq_logits[i]).item()


def _swap_hook(output, V, V_pinv):
    if isinstance(output, tuple):
        t, rest = output[0], output[1:]
    else:
        t, rest = output, ()
    h = t[:, -1, :].unsqueeze(-1)  # [B, d, 1]
    B = h.shape[0]
    c = torch.bmm(V_pinv.unsqueeze(0).expand(B, -1, -1), h)  # [B, 2, 1]
    c_swapped = c.clone()
    c_swapped[:, 0], c_swapped[:, 1] = c[:, 1].clone(), c[:, 0].clone()
    h_new = h + torch.bmm(V.unsqueeze(0).expand(B, -1, -1), c_swapped - c)
    t[:, -1, :] = h_new.squeeze(-1)
    return (t,) + rest if rest else t
