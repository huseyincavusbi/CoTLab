"""Experiments module."""

from .activation_patching import ActivationPatchingExperiment
from .attention_analysis import AttentionAnalysisExperiment
from .classification import ClassificationExperiment
from .cot_ablation import CoTAblationExperiment
from .cot_faithfulness import CoTFaithfulnessExperiment
from .cot_heads import CoTHeadsExperiment
from .full_layer_cot import FullLayerCoTExperiment
from .full_layer_patching import FullLayerPatchingExperiment
from .logit_lens import LogitLensExperiment
from .multi_head_cot import MultiHeadCoTExperiment
from .multi_head_patching import MultiHeadPatchingExperiment
from .probing_classifier import ProbingClassifierExperiment
from .steering_vectors import SteeringVectorsExperiment
from .sycophancy_heads import SycophancyHeadsExperiment

__all__ = [
    "CoTAblationExperiment",
    "CoTFaithfulnessExperiment",
    "ActivationPatchingExperiment",
    "AttentionAnalysisExperiment",
    "ProbingClassifierExperiment",
    "ClassificationExperiment",
    "SycophancyHeadsExperiment",
    "MultiHeadPatchingExperiment",
    "FullLayerPatchingExperiment",
    "SteeringVectorsExperiment",
    "CoTHeadsExperiment",
    "LogitLensExperiment",
    "MultiHeadCoTExperiment",
    "FullLayerCoTExperiment",
]
