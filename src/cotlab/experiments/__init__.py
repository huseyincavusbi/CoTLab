"""Experiments module."""

from .activation_compare import ActivationCompareExperiment
from .activation_patching import ActivationPatchingExperiment
from .attention_analysis import AttentionAnalysisExperiment
from .classification import ClassificationExperiment
from .composite_shift_detector import CompositeShiftDetectorExperiment
from .confabulation_analysis import ConfabulationAnalysisExperiment
from .cot_ablation import CoTAblationExperiment
from .cot_faithfulness import CoTFaithfulnessExperiment
from .cot_heads import CoTHeadsExperiment
from .entropy_neuron_overlap import EntropyNeuronOverlapExperiment
from .full_layer_cot import FullLayerCoTExperiment
from .full_layer_patching import FullLayerPatchingExperiment
from .h_neuron_analysis import HNeuronAnalysisExperiment
from .logit_lens import LogitLensExperiment
from .multi_head_cot import MultiHeadCoTExperiment
from .multi_head_patching import MultiHeadPatchingExperiment
from .probing_classifier import ProbingClassifierExperiment
from .residual_norm_ood import ResidualNormOODExperiment
from .sae_feature_analysis import SAEFeatureAnalysisExperiment
from .steering_vectors import SteeringVectorsExperiment
from .sycophancy_heads import SycophancyHeadsExperiment

__all__ = [
    "HNeuronAnalysisExperiment",
    "ConfabulationAnalysisExperiment",
    "EntropyNeuronOverlapExperiment",
    "CompositeShiftDetectorExperiment",
    "CoTAblationExperiment",
    "CoTFaithfulnessExperiment",
    "ActivationPatchingExperiment",
    "ActivationCompareExperiment",
    "AttentionAnalysisExperiment",
    "ProbingClassifierExperiment",
    "ClassificationExperiment",
    "ResidualNormOODExperiment",
    "SAEFeatureAnalysisExperiment",
    "SycophancyHeadsExperiment",
    "MultiHeadPatchingExperiment",
    "FullLayerPatchingExperiment",
    "SteeringVectorsExperiment",
    "CoTHeadsExperiment",
    "LogitLensExperiment",
    "MultiHeadCoTExperiment",
    "FullLayerCoTExperiment",
]
