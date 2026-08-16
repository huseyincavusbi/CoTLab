"""vLLM backend seeding and main.py seed dispatch.

vLLM samples in worker processes that do not inherit torch RNG, so the config
seed must be passed explicitly to ``SamplingParams``. vLLM imports are lazy,
which lets us exercise the code path with a fake ``vllm`` module (no GPU).
"""

import sys
import types
from pathlib import Path

import pytest

from cotlab.backends.vllm_backend import VLLMBackend


class _FakeSamplingParams(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _FakeOutput:
    def __init__(self, text):
        self.text = text
        self.token_ids = [1, 2]


class _FakeVllmModel:
    def __init__(self, backend):
        self.backend = backend

    def generate(self, prompts, sampling_params):
        self.backend._last_sampling_params = sampling_params
        return [type("R", (), {"outputs": [_FakeOutput(p)]})() for p in prompts]


@pytest.fixture
def vllm_backend(monkeypatch):
    b = VLLMBackend(model_name="fake-model")
    b._model = _FakeVllmModel(b)
    fake_module = types.ModuleType("vllm")
    fake_module.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_module)
    return b


def test_seed_reaches_sampling_params(vllm_backend):
    out = vllm_backend.generate_batch(["q1", "q2"], seed=42, temperature=0.7)
    assert [o.text for o in out] == ["q1", "q2"]
    assert vllm_backend._last_sampling_params["seed"] == 42
    assert vllm_backend._last_sampling_params["temperature"] == 0.7


def test_seed_omitted_when_none(vllm_backend):
    vllm_backend.generate_batch(["q1"], temperature=0.7)
    assert "seed" not in vllm_backend._last_sampling_params


def test_greedy_unaffected_by_seed(vllm_backend):
    vllm_backend.generate_batch(["q1"], temperature=0.0, seed=7)
    sp = vllm_backend._last_sampling_params
    assert sp["seed"] == 7 and sp["temperature"] == 0.0


def test_main_passes_seed_to_vllm_generate():
    """main.py hands the config seed to vLLM (non-transformers) runs."""
    main_src = Path("src/cotlab/main.py").read_text()
    assert "gen_kwargs[\"seed\"] = int(cfg.seed)" in main_src
    assert 'endswith("TransformersBackend")' in main_src
