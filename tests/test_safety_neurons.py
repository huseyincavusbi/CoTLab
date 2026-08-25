"""Unit tests for the safety-neurons experiment (no model downloads)."""

import torch
from safetensors.torch import save_file
from torch import nn

from cotlab.core.base import GenerationOutput
from cotlab.experiments.safety_neurons import SafetyNeuronsExperiment

# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def test_rejects_unknown_mode():
    with pytest_match("mode"):
        SafetyNeuronsExperiment(mode="bogus", first_peft_path="x")


def test_rejects_unknown_token_type():
    with pytest_match("token_type"):
        SafetyNeuronsExperiment(token_type="bogus", first_peft_path="x")


def test_rejects_unknown_selection():
    with pytest_match("selection"):
        SafetyNeuronsExperiment(selection="bogus", first_peft_path="x")


def test_rejects_missing_peft_paths():
    with pytest_match("peft_path"):
        SafetyNeuronsExperiment(first_peft_path=None, second_peft_path=None)


def pytest_match(pattern):
    import pytest

    return pytest.raises(ValueError, match=pattern)


# ---------------------------------------------------------------------------
# IA3 loading
# ---------------------------------------------------------------------------


def _save_adapter(tmp_path, tensors):
    d = tmp_path / "adapter"
    d.mkdir()
    save_file(tensors, str(d / "adapter_model.safetensors"))
    return str(d)


def test_load_ia3_keeps_only_down_proj(tmp_path):
    path = _save_adapter(
        tmp_path,
        {
            "base_model.model.layers.0.mlp.down_proj.ia3": torch.ones(4),
            "base_model.model.layers.0.self_attn.k_proj.ia3": torch.ones(4),
        },
    )
    vectors = SafetyNeuronsExperiment._load_ia3_vectors(path)
    assert len(vectors) == 1
    assert any(k.endswith("mlp.down_proj") or k.endswith(".down_proj") for k in vectors)


def test_load_ia3_rejects_duplicate_stem(tmp_path):
    path = _save_adapter(
        tmp_path,
        {
            "layers.0.mlp.down_proj.ia3": torch.ones(4),
            "layers.0.mlp.down_proj.ia3_l": torch.ones(4),
        },
    )
    import pytest

    with pytest.raises(ValueError, match="both"):
        SafetyNeuronsExperiment._load_ia3_vectors(path)


def test_load_ia3_missing_file(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        SafetyNeuronsExperiment._load_ia3_vectors(str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# change scores + ranking
# ---------------------------------------------------------------------------


def test_change_scores_rms_matches_manual():
    first = torch.randn(30, 2, 5)
    second = torch.randn(30, 2, 5)
    got = SafetyNeuronsExperiment._compute_change_scores(first, second)
    want = (first - second).square().mean(dim=0).sqrt()
    assert got.shape == (2, 5)
    assert torch.allclose(got, want)


def test_change_scores_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError, match="mismatch"):
        SafetyNeuronsExperiment._compute_change_scores(torch.zeros(3, 2, 5), torch.zeros(4, 2, 5))


def test_select_neurons_top_percent_orders_by_score():
    exp = SafetyNeuronsExperiment(selection="top_percent", top_percent=0.5, first_peft_path="x")
    scores = torch.tensor([[1.0, 9.0], [3.0, 5.0]])  # top-2 of 4: 9 (L0,I1), 5 (L1,I1)
    assert sorted(exp._select_neurons(scores)) == [(0, 1), (1, 1)]


def test_select_neurons_top_n_respects_cap():
    exp = SafetyNeuronsExperiment(selection="top_n", top_n=2, first_peft_path="x")
    scores = torch.tensor([[1.0, 9.0], [3.0, 5.0]])
    assert sorted(exp._select_neurons(scores)) == [(0, 1), (1, 1)]


# ---------------------------------------------------------------------------
# select masks
# ---------------------------------------------------------------------------


def _exp_token(token_type):
    return SafetyNeuronsExperiment(token_type=token_type, first_peft_path="x")


def test_select_mask_completion_covers_generated_span():
    mask = _exp_token("completion")._select_mask(torch.zeros(7), prompt_len=4)
    assert mask.tolist() == [False] * 4 + [True] * 3


def test_select_mask_prompt_last_single_position():
    mask = _exp_token("prompt_last")._select_mask(torch.zeros(7), prompt_len=4)
    assert mask.tolist() == [False] * 3 + [True] + [False] * 3


def test_select_mask_prompt_all_positions():
    mask = _exp_token("prompt")._select_mask(torch.zeros(7), prompt_len=4)
    assert bool(mask.all())


# ---------------------------------------------------------------------------
# toy model fixtures (real forwards through nn.Linear modules)
# ---------------------------------------------------------------------------


class _ToyDownProj(nn.Module):
    def __init__(self, d_mlp, d_model):
        super().__init__()
        self.linear = nn.Linear(d_mlp, d_model)

    def forward(self, x):
        return self.linear(x)


class _ToyLayer(nn.Module):
    def __init__(self, d_mlp, d_model):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = _ToyDownProj(d_mlp, d_model)
        # kept outside .mlp so IA3 hooks never touch it; restores d_mlp width
        self.up = nn.Linear(d_model, d_mlp)


class _ToyModel(nn.Module):
    """Embed -> per-layer [up -> tanh -> down_proj]; deterministic in ids."""

    def __init__(self, num_layers=3, vocab=20, d_mlp=6, d_model=8, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab, d_model)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab, d_model, generator=g))
        self.layers = nn.ModuleList(_ToyLayer(d_mlp, d_model) for _ in range(num_layers))

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer.mlp.down_proj(torch.tanh(layer.up(x)))
        return x


class _ToyHookManager:
    def __init__(self, model):
        self.model = model
        self.num_layers = len(model.layers)

    def get_mlp_down_proj_module(self, idx):
        return self.model.layers[idx].mlp.down_proj


class _Tok:
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        base = sum(ord(c) for c in text[:3])
        return {"input_ids": [base % 17 + 1, (base * 7) % 17 + 1, (base * 13) % 17 + 1]}


class _Backend:
    def __init__(self, model, generate_tokens=None):
        self.model = model
        self.hook_manager = _ToyHookManager(model)
        self.tokenizer = _Tok()
        self.device = "cpu"
        self.model_name = "toy"
        self._generate_tokens = generate_tokens or [4, 5]

    def generate(self, prompt, max_new_tokens=8, do_sample=False):
        return GenerationOutput(text="gen", tokens=list(self._generate_tokens))


def test_ia3_toggle_round_trip_restores_outputs():
    torch.manual_seed(0)
    model = _ToyModel(seed=3)
    backend = _Backend(model)
    ids = torch.tensor([[1, 2, 3]])
    base_out = backend.model(ids)

    exp = SafetyNeuronsExperiment(first_peft_path="x")
    # scale neuron 0 of every layer hard
    handles = []
    for layer in model.layers:

        def make_hook():
            def hook(mod, inp):
                return (inp[0] * 5.0,) + tuple(inp[1:])

            return hook

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook()))
    scaled_out = backend.model(ids)
    for h in handles:
        h.remove()

    exp._clear_ia3()  # no-op safety
    restored_out = backend.model(ids)
    assert not torch.allclose(base_out, scaled_out)
    assert torch.allclose(base_out, restored_out)


def test_capture_activations_position_aligned_and_layered():
    torch.manual_seed(0)
    model = _ToyModel(num_layers=3, seed=5)
    backend = _Backend(model)
    rows = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 6]])]
    masks = [
        torch.tensor([True, False, True, False]),
        torch.tensor([True, True]),
    ]
    exp = SafetyNeuronsExperiment(first_peft_path="x")
    acts = exp._capture_activations(backend, rows, masks)
    assert acts.shape == (4, 3, 6)  # (selected positions, layers, d_mlp)
    # same inputs -> identical captures (determinism)
    acts_again = exp._capture_activations(backend, rows, masks)
    assert torch.allclose(acts, acts_again)


def test_identify_end_to_end_constructive_ground_truth(tmp_path):
    """IA3-scaled neuron must rank top-1 in the contrast."""
    torch.manual_seed(0)
    model = _ToyModel(num_layers=3, seed=7)

    # build adapter scaling exactly one hidden unit of layer 1's down_proj input
    target_key = "layers.1.mlp.down_proj"
    vec = torch.full((6,), 1.0)
    vec[2] = 40.0  # neuron 2 of layer 1 gets a huge activation change
    adapter_dir = tmp_path / "ad"
    adapter_dir.mkdir()
    save_file({f"{target_key}.ia3": vec}, str(adapter_dir / "adapter_model.safetensors"))

    class _DS:
        def sample(self, n, seed=42):
            return [type("S", (), {"text": "abc"})() for _ in range(2)]

    exp = SafetyNeuronsExperiment(
        mode="identify",
        first_peft_path=None,
        second_peft_path=str(adapter_dir),
        token_type="prompt",
        selection="top_n",
        top_n=3,
    )
    result = exp.run(_Backend(model), dataset=_DS())
    top = result.metadata["selected_neurons"][0]
    assert (top["layer"], top["index"]) == (1, 2)
    assert result.metrics["ia3_modules_second"] == 1
    assert result.metrics["ia3_modules_first"] == 0
