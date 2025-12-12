"""Hydra-compatible configuration dataclasses."""

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class BackendConfig:
    """Configuration for inference backend."""

    _target_: str = MISSING
    device: str = "cuda"
    dtype: str = "bfloat16"


@dataclass
class TransformersBackendConfig(BackendConfig):
    """Transformers-specific backend config."""

    _target_: str = "cotlab.backends.TransformersBackend"
    enable_hooks: bool = True
    trust_remote_code: bool = True


@dataclass
class VLLMBackendConfig(BackendConfig):
    """vLLM-specific backend config."""

    _target_: str = "cotlab.backends.VLLMBackend"
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    trust_remote_code: bool = True


@dataclass
class ModelConfig:
    """Configuration for model loading."""

    name: str = MISSING
    variant: str = "4b"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class PromptConfig:
    """Configuration for prompt strategy."""

    _target_: str = MISSING
    name: str = MISSING
    system_role: Optional[str] = None


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    _target_: str = MISSING
    name: str = MISSING
    path: str = MISSING


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    _target_: str = MISSING
    name: str = MISSING
    description: str = ""
    num_samples: int = 100
    tests: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)


@dataclass
class Config:
    """Root configuration."""

    backend: BackendConfig = field(default_factory=BackendConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    seed: int = 42
    verbose: bool = True
    dry_run: bool = False
