"""Equivalence test for attention_analysis head-entropy vectorization.

Proves the vectorized per-head entropy computation (batched reductions over
the head dimension) produces bit-identical values to the sequential per-head
loop it replaces.

Reference: src/cotlab/experiments/attention_analysis.py ``_analyze_batch``.
"""

import torch


def test_vectorized_head_entropy_matches_loop():
    """Vectorized entropy over the head dim equals the per-head loop."""
    from cotlab.experiments.attention_analysis import AttentionAnalysisExperiment

    exp = AttentionAnalysisExperiment()
    torch.manual_seed(0)
    # (heads, seq_len, seq_len) attention weights summing to 1 over last dim.
    sample_attn = torch.softmax(torch.randn(12, 7, 7), dim=-1)
    last_token_attn = sample_attn[:, -1, :]  # (heads, seq_len)

    # --- Per-head loop (old) ---
    loop_last = [exp._compute_entropy(last_token_attn[h]) for h in range(12)]
    loop_all = [exp._compute_mean_entropy_over_queries(sample_attn[h]) for h in range(12)]

    # --- Vectorized (new) ---
    eps = 1e-10
    p_last = last_token_attn.float()
    vec_last = (-(p_last * torch.log(p_last + eps)).sum(dim=-1)).tolist()
    p_all = sample_attn.float()
    vec_all = (-(p_all * torch.log(p_all + eps)).sum(dim=-1).mean(dim=-1)).tolist()

    assert loop_last == vec_last, "last-token entropies differ"
    assert loop_all == vec_all, "all-token entropies differ"
