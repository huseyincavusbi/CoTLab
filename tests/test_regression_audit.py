"""Regression tests for bugs found in the final scientific-correctness audit.

Covers the three CRITICAL regressions introduced during optimization:
- C1: entropy_neuron_overlap vectorized selection returned GLOBAL indices instead
  of per-layer indices (crash + wrong overlap metrics).
- C2: cot_ablation mean mode vectorization computed the mean once over the
  original clone instead of per position from the progressively-modified tensor.
- C3: FewShotStrategy memoized few_shot toggling (few_shot_contrast collapse).
"""

import numpy as np
import pytest
import torch

from cotlab.prompts.strategies import FewShotStrategy


@pytest.fixture(scope="module")
def backend():
    from cotlab.backends.transformers_backend import TransformersBackend

    with TransformersBackend(device="cpu", dtype="float32", enable_hooks=True) as b:
        b.load_model("openai-community/gpt2")
        yield b


# ---------------------------------------------------------------------------
# C1: entropy_neuron_overlap per-layer index correctness
# ---------------------------------------------------------------------------


@pytest.mark.real_model
def test_entropy_neurons_use_within_layer_indices(backend):
    """Returned (layer, idx) pairs have idx within the layer's width."""
    from cotlab.experiments.entropy_neuron_overlap import EntropyNeuronOverlapExperiment

    exp = EntropyNeuronOverlapExperiment()
    neurons, all_norms = exp._identify_entropy_neurons(backend, 90.0)
    assert len(neurons) > 0
    # Each layer's down_proj width bounds its indices.
    widths = {}
    for layer, idx in neurons:
        widths.setdefault(layer, None)
    for layer in widths:
        w = backend.hook_manager.get_mlp_down_proj_module(layer).weight.data.float()
        widths[layer] = w.shape[1]
    for layer, idx in neurons:
        assert 0 <= idx < widths[layer], f"idx {idx} out of range for layer {layer}"
    # _compute_norm_statistics must not crash.
    stats = exp._compute_norm_statistics([(2, 5)], neurons[:20], all_norms, backend)
    assert stats["h_norm_mean"] > 0


@pytest.mark.real_model
def test_entropy_norms_match_reference(backend):
    """all_norms[global_offset + idx] equals the module's per-column norm."""
    from cotlab.experiments.entropy_neuron_overlap import EntropyNeuronOverlapExperiment

    exp = EntropyNeuronOverlapExperiment()
    neurons, all_norms = exp._identify_entropy_neurons(backend, 90.0)
    # Rebuild the flat layout to recover global offsets (12 identical-width layers
    # in GPT-2, but use per-layer sizes to be robust).
    num_layers = backend.hook_manager.num_layers
    sizes = []
    for layer in range(num_layers):
        w = backend.hook_manager.get_mlp_down_proj_module(layer).weight.data.float()
        sizes.append(w.shape[1])
    starts = np.concatenate([np.array([0]), np.cumsum(sizes)[:-1]])
    for layer, idx in neurons[:10]:
        w = backend.hook_manager.get_mlp_down_proj_module(layer).weight.data.float()
        ref = float(w[:, idx].norm(p=2).item())
        got = float(all_norms[int(starts[layer]) + idx])
        assert abs(ref - got) < 1e-6


# ---------------------------------------------------------------------------
# C2: cot_ablation mean mode reproduces the per-position mean reference
# ---------------------------------------------------------------------------


@pytest.mark.real_model
def test_cot_ablation_mean_matches_main_reference(backend):
    """mean ablation equals the per-position recomputed mean (main behavior)."""
    from cotlab.experiments.cot_ablation import CoTAblationExperiment

    prompt = (
        "Question: Patient has chest pain. What is the diagnosis?\n\n"
        "Let me think. The patient has chest pain. The answer is pneumonia."
    )
    positions = [3, 4, 5]
    exp = CoTAblationExperiment()
    _, cache = backend.forward_with_cache(prompt, layers=list(range(6)))
    src = cache.get(2)

    for atype in ["zero", "mean", "noise"]:
        exp.ablation_type = atype
        torch.manual_seed(7)
        new_act = exp._build_ablated_activation(src, positions)
        # Main reference: per-position loop with recomputed mean.
        torch.manual_seed(7)
        ref = src.clone()
        for pos in positions:
            if pos < ref.shape[1]:
                if atype == "zero":
                    ref[:, pos, :] = 0
                elif atype == "mean":
                    ref[:, pos, :] = ref.mean(dim=1)
                elif atype == "noise":
                    ref[:, pos, :] += torch.randn_like(ref[:, pos, :])
        assert torch.equal(new_act, ref), f"{atype}: not bit-identical to main reference"


# ---------------------------------------------------------------------------
# C3: FewShotStrategy few_shot toggle
# ---------------------------------------------------------------------------


def test_few_shot_strategy_toggle_changes_prompt():
    """Toggling few_shot must change the FewShotStrategy prompt (no stale memo)."""
    s = FewShotStrategy()
    clean = s.build_prompt({"question": "Q"})
    s.few_shot = False
    corrupt = s.build_prompt({"question": "Q"})
    s.few_shot = True
    clean_again = s.build_prompt({"question": "Q"})
    assert clean != corrupt, "few_shot toggle had no effect"
    assert clean == clean_again, "cached examples not idempotent"
