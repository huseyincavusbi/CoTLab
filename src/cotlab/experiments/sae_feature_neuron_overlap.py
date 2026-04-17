"""SAE Feature to H-Neuron Overlap Experiment.

Tests whether SAE features (e.g., histopathology-discriminative features) overlap
with H-Neurons, revealing if domain-specific knowledge and hallucination circuits
share the same neurons.

Background
----------
SAE features decompose the residual stream into interpretable
directions. Some features activate strongly on domain-specific content (e.g.,
histopathology vocabulary). This experiment tests whether the MLP neurons that
contribute most to a given SAE feature overlap with H-Neurons.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..patching.sae import GemmaScopeLayer


@Registry.register_experiment("sae_feature_neuron_overlap")
class SAEFeatureNeuronOverlapExperiment(BaseExperiment):
    """Analyze overlap between SAE feature contributors and H-Neurons."""

    def __init__(
        self,
        name: str = "sae_feature_neuron_overlap",
        description: str = "Test if SAE features overlap with H-Neurons",
        sae_repo_id: str = "google/gemma-scope-2b-pt-res",
        sae_layer: int = 9,
        sae_feature_id: int = 1000,
        sae_width: str = "16k",
        probe_path: Optional[str] = None,
        top_k_neurons: int = 50,
        **kwargs,
    ):
        self._name = name
        self.description = description
        self.sae_repo_id = sae_repo_id
        self.sae_layer = sae_layer
        self.sae_feature_id = sae_feature_id
        self.sae_width = sae_width
        self.probe_path = probe_path
        self.top_k_neurons = top_k_neurons

    @property
    def name(self) -> str:
        return self._name

    def _load_probe(self) -> List[Tuple[int, int]]:
        """Load H-Neuron indices from probe."""
        if not self.probe_path:
            raise ValueError("probe_path required for overlap analysis")

        with open(self.probe_path) as f:
            probe_data = json.load(f)

        # Handle both old format (neurons) and new format (fit.h_neurons)
        if "neurons" in probe_data:
            return [(n["layer"], n["index"]) for n in probe_data["neurons"]]
        elif "fit" in probe_data and "h_neurons" in probe_data["fit"]:
            h_neurons = probe_data["fit"]["h_neurons"]
            # Handle list format [[layer, index], ...] or dict format
            if h_neurons and isinstance(h_neurons[0], list):
                return [(layer, idx) for layer, idx in h_neurons]
            else:
                return [(n["layer"], n["index"]) for n in h_neurons]
        else:
            raise ValueError("Probe file missing neurons data")

    def _identify_sae_contributing_neurons(
        self, backend: InferenceBackend
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Identify MLP neurons that contribute most to the SAE feature.

        Strategy:
        1. Load SAE decoder weights for the feature
        2. The decoder maps from feature space back to residual stream
        3. Compare decoder output with MLP down_proj weights to find contributing neurons
        4. Return top-K neurons by contribution strength
        """
        print(f"Loading SAE for layer {self.sae_layer}, width {self.sae_width}...")

        # Load SAE
        sae = GemmaScopeLayer.from_pretrained(
            repo_id=self.sae_repo_id,
            layer=self.sae_layer,
            width=self.sae_width,
        )

        # Get decoder weights for this feature
        # w_dec shape: [d_sae, d_model]
        # We want the d_model vector for this feature
        decoder_weight = sae.w_dec.data.float()  # [d_sae, d_model]
        feature_vector = decoder_weight[self.sae_feature_id, :].cpu().numpy()  # [d_model]

        print(f"SAE feature #{self.sae_feature_id} decoder vector shape: {feature_vector.shape}")

        # Get MLP down_proj weights at this layer
        # down_proj maps from intermediate_dim to d_model
        # down_proj.weight shape: [d_model, intermediate_dim]
        mlp_down_proj = backend.hook_manager.get_mlp_down_proj_module(self.sae_layer)
        down_proj_weight = (
            mlp_down_proj.weight.data.float().cpu().numpy()
        )  # [d_model, intermediate]

        print(f"MLP down_proj weight shape: {down_proj_weight.shape}")

        # Compute contribution of each MLP neuron to the SAE feature
        # For each neuron j, compute dot product: feature_vector · down_proj[:, j]
        # This measures how much neuron j's output aligns with the SAE feature direction
        contributions = np.abs(down_proj_weight.T @ feature_vector)  # [intermediate_dim]

        # Get top-K neurons
        top_indices = np.argsort(contributions)[-self.top_k_neurons :][::-1]
        top_contributions = contributions[top_indices]

        sae_neurons = [(self.sae_layer, int(idx)) for idx in top_indices]

        return sae_neurons, top_contributions

    def _compute_overlap(
        self,
        h_neurons: List[Tuple[int, int]],
        sae_neurons: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """Compute overlap statistics."""
        h_set = set(h_neurons)
        sae_set = set(sae_neurons)

        overlap = h_set & sae_set
        union = h_set | sae_set

        jaccard = len(overlap) / len(union) if union else 0.0

        return {
            "h_neurons_count": len(h_neurons),
            "sae_neurons_count": len(sae_neurons),
            "overlap_count": len(overlap),
            "jaccard_index": float(jaccard),
            "overlap_pct_of_h": float(len(overlap) / len(h_neurons)) if h_neurons else 0.0,
            "overlap_pct_of_sae": float(len(overlap) / len(sae_neurons)) if sae_neurons else 0.0,
            "overlap_neurons": sorted(list(overlap)),
        }

    def run(
        self,
        backend: InferenceBackend,
        dataset: Any = None,
        prompt_strategy: Any = None,
        **kwargs,
    ) -> ExperimentResult:
        """Run SAE feature to H-Neuron overlap analysis."""

        print(f"Model         : {backend.model_name}")
        print(f"SAE Layer     : {self.sae_layer}")
        print(f"SAE Feature   : #{self.sae_feature_id}")
        print(f"Top-K Neurons : {self.top_k_neurons}")

        # Load H-Neurons from probe
        all_h_neurons = self._load_probe()
        # Filter to target layer
        h_neurons = [(layer, idx) for layer, idx in all_h_neurons if layer == self.sae_layer]

        print(f"H-Neurons (all layers): {len(all_h_neurons)}")
        print(f"H-Neurons (L{self.sae_layer}): {len(h_neurons)}")

        # Identify SAE-contributing neurons
        sae_neurons, contributions = self._identify_sae_contributing_neurons(backend)

        print(f"SAE-contributing neurons: {len(sae_neurons)}")

        # Compute overlap
        overlap_metrics = self._compute_overlap(h_neurons, sae_neurons)

        # Add contribution statistics
        metrics = {
            **overlap_metrics,
            "sae_contribution_mean": float(np.mean(contributions)),
            "sae_contribution_std": float(np.std(contributions)),
            "sae_contribution_max": float(np.max(contributions)),
            "sae_contribution_min": float(np.min(contributions)),
        }

        # Print summary
        print("\n" + "=" * 66)
        print("SAE FEATURE → H-NEURON OVERLAP ANALYSIS")
        print("=" * 66)
        print(f"SAE Layer           : L{self.sae_layer}")
        print(f"SAE Feature         : #{self.sae_feature_id}")
        print(f"Top-K SAE Neurons   : {metrics['sae_neurons_count']}")
        print(f"H-Neurons (L{self.sae_layer})    : {metrics['h_neurons_count']}")
        print(f"Overlap             : {metrics['overlap_count']}")
        print(f"Jaccard Index       : {metrics['jaccard_index']:.4f}")
        print(f"Overlap % of H      : {metrics['overlap_pct_of_h']:.2%}")
        print(f"Overlap % of SAE    : {metrics['overlap_pct_of_sae']:.2%}")
        print()
        print(
            f"SAE Contribution    : {metrics['sae_contribution_mean']:.3f} ± {metrics['sae_contribution_std']:.3f}"
        )
        print(
            f"  Range: [{metrics['sae_contribution_min']:.3f}, {metrics['sae_contribution_max']:.3f}]"
        )

        if metrics["overlap_count"] > 0:
            print(f"\nOverlapping neurons: {metrics['overlap_neurons'][:10]}")
            if len(metrics["overlap_neurons"]) > 10:
                print(f"  ... and {len(metrics['overlap_neurons']) - 10} more")

        print("=" * 66)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="n/a",
            metrics=metrics,
            metadata={
                "probe_path": self.probe_path,
                "sae_layer": self.sae_layer,
                "sae_feature_id": self.sae_feature_id,
                "sae_width": self.sae_width,
                "top_k_neurons": self.top_k_neurons,
            },
        )
