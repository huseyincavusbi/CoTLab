"""Activation patching module for causal interventions."""

from .hooks import HookManager
from .cache import ActivationCache
from .patcher import ActivationPatcher, PatchingResult
from .interventions import (
    InterventionType,
    Intervention,
    PatchingExperimentSpec,
    LayerImportance,
    ThoughtAnchor,
)

__all__ = [
    "HookManager",
    "ActivationCache",
    "ActivationPatcher",
    "PatchingResult",
    "InterventionType",
    "Intervention",
    "PatchingExperimentSpec",
    "LayerImportance",
    "ThoughtAnchor",
]
