"""Tests for backend helpers."""

import torch

from cotlab.backends.transformers_backend import TransformersBackend


def test_normalize_load_kwargs_converts_dtype():
    kwargs = {"bnb_4bit_compute_dtype": "bfloat16"}
    normalized = TransformersBackend._normalize_load_kwargs(kwargs)
    assert normalized["bnb_4bit_compute_dtype"] == torch.bfloat16


def test_normalize_load_kwargs_keeps_dtype():
    kwargs = {"bnb_4bit_compute_dtype": torch.float16}
    normalized = TransformersBackend._normalize_load_kwargs(kwargs)
    assert normalized["bnb_4bit_compute_dtype"] == torch.float16


def test_apply_system_prompt_ignores_empty():
    prompt = "Question?"
    assert TransformersBackend._apply_system_prompt(prompt, None) == prompt
    assert TransformersBackend._apply_system_prompt(prompt, "   ") == prompt


def test_apply_system_prompt_prefixes():
    prompt = "Question?"
    system = "You are a tester."
    expected = "You are a tester.\n\nQuestion?"
    assert TransformersBackend._apply_system_prompt(prompt, system) == expected
