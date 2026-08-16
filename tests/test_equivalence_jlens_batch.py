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


def test_batched_ablate_matches_sequential(backend):
    """Batched per-row ablate (top-k projection) equals sequential per-sample."""
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
    exp.ablate_top_n = 5

    layers = [2, 4]
    rng = torch.Generator().manual_seed(2)
    d = model.config.hidden_size
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    device = backend.device
    vocab = lm_head.weight.float().to(device)
    jacobians = {layer: torch.randn(d, d, generator=rng) for layer in layers}
    wu_j = {layer: vocab @ jacobians[layer].to(device, dtype=torch.float32) for layer in layers}
    j_dev = {layer: jacobians[layer].to(device, dtype=torch.float32) for layer in layers}

    # --- Sequential reference: one forward per prompt ---
    seq_logits = []
    with torch.inference_mode():
        for p in prompts:
            tokens = exp._tokenize_batch(tokenizer, [p], backend.device)
            handles = []
            for layer in layers:
                block = backend.hook_manager.get_layer_module(layer)
                handles.append(
                    block.register_forward_hook(
                        lambda m, i, o, _w=wu_j[layer], _j=j_dev[layer], _v=vocab, _k=5: (
                            _ablate_single(o, _w, _j, _v, _k)
                        )
                    )
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
                for h in handles:
                    h.remove()
            seq_logits.append(out.logits[0, -1])

    # --- Batched: all prompts as rows, one forward ---
    batch = exp._tokenize_batch(tokenizer, prompts, backend.device)
    handles = []
    for layer in layers:
        block = backend.hook_manager.get_layer_module(layer)
        handles.append(
            block.register_forward_hook(
                lambda m, i, o, _w=wu_j[layer], _j=j_dev[layer], _v=vocab, _k=5: _ablate_batch(
                    o, _w, _j, _v, _k
                )
            )
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
        for h in handles:
            h.remove()
    bat_logits = out.logits[:, -1, :]

    for i in range(len(prompts)):
        torch.testing.assert_close(
            bat_logits[i], seq_logits[i], atol=1e-4, rtol=1e-5, msg=f"prompt {i}"
        )
        assert torch.argmax(bat_logits[i]).item() == torch.argmax(seq_logits[i]).item()


def _ablate_single(output, wu_j, J, vocab, top_n):
    if isinstance(output, tuple):
        t, rest = output[0], output[1:]
    else:
        t, rest = output, ()
    h = t[0, -1, :].float()
    all_scores = h @ wu_j.T
    top_k_ids = torch.topk(all_scores, top_n).indices
    for tid in top_k_ids:
        v = vocab[tid] @ J
        v_norm = v / (torch.norm(v) + 1e-8)
        h = h - torch.dot(h, v_norm) * v_norm
    t[0, -1, :] = h.to(t.dtype)
    return (t,) + rest if rest else t


def _ablate_batch(output, wu_j, J, vocab, top_n):
    if isinstance(output, tuple):
        t, rest = output[0], output[1:]
    else:
        t, rest = output, ()
    h = t[:, -1, :].float()
    all_scores = h @ wu_j.T
    top_k_ids = torch.topk(all_scores, top_n, dim=-1).indices
    for j in range(top_n):
        v = vocab[top_k_ids[:, j]] @ J
        v_norm = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
        h = h - torch.bmm(v_norm.unsqueeze(1), h.unsqueeze(-1)).squeeze(-1) * v_norm
    t[:, -1, :] = h.to(t.dtype)
    return (t,) + rest if rest else t


def test_batched_decompose_matches_sequential(backend):
    """Batched per-row decompose metrics equal sequential per-sample metrics."""
    from cotlab.experiments.jacobian_lens import JacobianLensExperiment

    model = backend.model
    tokenizer = backend.tokenizer

    prompts = [
        "Question: Patient has chest pain. What is the diagnosis?\n\nAnswer:",
        "Question: Patient has a headache. What is the diagnosis?\n\nAnswer:",
    ]
    exp = JacobianLensExperiment()
    exp.max_input_tokens = 64
    exp.top_k = 5

    layers = [2]
    rng = torch.Generator().manual_seed(3)
    d = model.config.hidden_size
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    device = backend.device
    vocab = lm_head.weight.float().to(device)
    jacobians = {layer: torch.randn(d, d, generator=rng) for layer in layers}
    wu_j = {layer: vocab @ jacobians[layer].to(device, dtype=torch.float32) for layer in layers}
    j_dev = {layer: jacobians[layer].to(device, dtype=torch.float32) for layer in layers}

    # --- Sequential reference: one forward per prompt, per-layer metrics ---
    seq_results = []
    with torch.no_grad():
        for p in prompts:
            tokens = exp._tokenize_batch(tokenizer, [p], backend.device)
            out = model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                position_ids=tokens["position_ids"],
                output_hidden_states=True,
                use_cache=False,
            )
            h = out.hidden_states[layers[0] + 1][0, -1, :].float()
            all_scores = h @ wu_j[layers[0]].T
            _, top_k_ids = torch.topk(all_scores, exp.top_k)
            h_nonj = h.clone()
            for tid in top_k_ids:
                v = vocab[tid] @ j_dev[layers[0]]
                v_norm = v / (torch.norm(v) + 1e-8)
                h_nonj = h_nonj - torch.dot(h_nonj, v_norm) * v_norm
            total_var = torch.var(h).item()
            j_var_frac = (
                float(torch.var(h - h_nonj).item() / total_var) if total_var > 1e-8 else 0.0
            )
            seq_results.append((total_var, j_var_frac))

    # --- Batched: one forward, per-row metrics ---
    batch = exp._tokenize_batch(tokenizer, prompts, backend.device)
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            output_hidden_states=True,
            use_cache=False,
        )
    h = out.hidden_states[layers[0] + 1][:, -1, :].float()
    all_scores = h @ wu_j[layers[0]].T
    _, top_k_ids = torch.topk(all_scores, exp.top_k, dim=-1)
    h_nonj = h.clone()
    for j in range(exp.top_k):
        v = vocab[top_k_ids[:, j]] @ j_dev[layers[0]]
        v_norm = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
        h_nonj = h_nonj - torch.bmm(v_norm.unsqueeze(1), h_nonj.unsqueeze(-1)).squeeze(-1) * v_norm
    total_var = torch.var(h, dim=-1)
    j_var_frac = torch.where(
        total_var > 1e-8, torch.var(h - h_nonj, dim=-1) / total_var, torch.zeros_like(total_var)
    )

    for i in range(len(prompts)):
        assert abs(float(total_var[i].item()) - seq_results[i][0]) < 1e-4, f"prompt {i} total_var"
        assert abs(float(j_var_frac[i].item()) - seq_results[i][1]) < 1e-4, f"prompt {i} j_var_frac"


def test_batched_apply_decode_matches_sequential(backend):
    """Batched stacked-layer decode equals per-layer lens.decode/logit-lens."""
    from cotlab.experiments.jacobian_lens import JacobianLens, JacobianLensExperiment

    model = backend.model
    tokenizer = backend.tokenizer

    exp = JacobianLensExperiment()
    exp.max_input_tokens = 64
    exp.top_k = 5

    layers = [2, 4]
    rng = torch.Generator().manual_seed(4)
    d = model.config.hidden_size
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    norm = model.transformer.ln_f if hasattr(model.transformer, "ln_f") else None
    device = backend.device
    jacobians = {layer: torch.randn(d, d, generator=rng) for layer in layers}
    lens = JacobianLens(jacobians=jacobians, d_model=d)

    prompt = "Question: Patient has chest pain. What is the diagnosis?\n\nAnswer:"
    tokens = exp._tokenize_batch(tokenizer, [prompt], backend.device)

    with torch.inference_mode():
        out = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            position_ids=tokens["position_ids"],
            output_hidden_states=True,
            use_cache=False,
        )
    hs = out.hidden_states

    # --- Sequential: per-layer lens.decode + logit-lens ---
    seq_jl, seq_ll = [], []
    for layer in layers:
        h = hs[layer + 1][0, -1, :].detach().clone()
        with torch.inference_mode():
            jl = lens.decode(h.unsqueeze(0), layer, lm_head, norm)[0]
        seq_jl.append(jl)
        ll_h = h if norm is None else norm(h)
        with torch.inference_mode():
            ll = lm_head(ll_h.unsqueeze(0))[0]
        seq_ll.append(ll)

    # --- Batched: stack h, one norm + lm_head ---
    J_stack = torch.stack([jacobians[layer].to(device, dtype=torch.float32) for layer in layers])
    h_stack = torch.stack([hs[layer + 1][0, -1, :] for layer in layers]).to(device)
    with torch.inference_mode():
        transported = torch.bmm(h_stack.unsqueeze(1), J_stack.transpose(-1, -2)).squeeze(1)
        if norm is not None:
            transported = norm(transported)
        bat_jl = lm_head(transported)
        ll_h = h_stack if norm is None else norm(h_stack)
        bat_ll = lm_head(ll_h)

    for i in range(len(layers)):
        torch.testing.assert_close(bat_jl[i], seq_jl[i], atol=1e-4, rtol=1e-5, msg=f"jl layer {i}")
        torch.testing.assert_close(bat_ll[i], seq_ll[i], atol=1e-4, rtol=1e-5, msg=f"ll layer {i}")
        assert torch.argmax(bat_jl[i]).item() == torch.argmax(seq_jl[i]).item()
