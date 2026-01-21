"""Tests for backend helpers."""

import torch

from cotlab.backends.transformers_backend import TransformersBackend


def test_normalize_load_kwargs_converts_dtype():
    kwargs = {"bnb_4bit_compute_dtype": "bfloat16"}
    normalized = TransformersBackend._normalize_load_kwargs(kwargs)
    assert normalized["bnb_4bit_compute_dtype"] == torch.bfloat16
