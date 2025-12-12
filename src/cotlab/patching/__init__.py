"""Activation patching module for causal interventions."""

from .cache import ActivationCache
from .hooks import HookManager
from .interventions import (
    Intervention,
    InterventionType,
    LayerImportance,
    PatchingExperimentSpec,
    ThoughtAnchor,
)
from .patcher import ActivationPatcher, PatchingResult

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
