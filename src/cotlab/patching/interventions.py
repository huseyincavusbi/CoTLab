"""Intervention types and specifications."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class InterventionType(Enum):
    """Types of activation interventions."""

    PATCH = auto()  # Replace with activations from another run
    ZERO = auto()  # Zero out activations
    NOISE = auto()  # Add Gaussian noise
    MEAN_ABLATE = auto()  # Replace with mean activation
    SCALE = auto()  # Scale activations by factor


@dataclass
class Intervention:
    """Specification for a single activation intervention."""

    type: InterventionType
    layers: List[int]
    token_positions: Optional[List[int]] = None

    # Parameters for specific intervention types
    noise_scale: float = 0.1  # For NOISE type
    scale_factor: float = 0.0  # For SCALE type
    source_cache_key: Optional[str] = None  # For PATCH type

    def __repr__(self) -> str:
        pos_str = f", positions={self.token_positions}" if self.token_positions else ""
        return f"Intervention({self.type.name}, layers={self.layers}{pos_str})"


@dataclass
class PatchingExperimentSpec:
    """Full specification for a patching experiment."""

    clean_prompt: str
    corrupted_prompt: str
    interventions: List[Intervention] = field(default_factory=list)

    expected_clean_answer: Optional[str] = None
    expected_corrupted_answer: Optional[str] = None

    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_intervention(
        self, type: InterventionType, layers: List[int], **kwargs
    ) -> "PatchingExperimentSpec":
        """Add an intervention (builder pattern)."""
        self.interventions.append(Intervention(type=type, layers=layers, **kwargs))
        return self


@dataclass
class LayerImportance:
    """Results from a layer importance sweep."""

    layer_idx: int
    effect_size: float  # How much patching changed the output
    original_output: str
    patched_output: str
    answer_recovered: bool  # Did patching recover the expected answer?

    @property
    def is_important(self) -> bool:
        """Whether this layer appears causally important."""
        return abs(self.effect_size) > 0.1 or self.answer_recovered


@dataclass
class ThoughtAnchor:
    """A token position identified as important for reasoning."""

    position: int
    token: str
    layer_effects: Dict[int, float]  # layer -> effect size

    @property
    def max_effect(self) -> float:
        """Maximum effect across all layers."""
        return max(abs(e) for e in self.layer_effects.values()) if self.layer_effects else 0.0
