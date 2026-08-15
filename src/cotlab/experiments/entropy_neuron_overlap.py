"""Entropy Neuron Overlap Experiment.

Identifies entropy neurons (high-norm output projection neurons) and compares
with H-Neurons to test if hallucination and uncertainty use separate circuits.

Entropy Neurons
---------------
Neurons with high L2 norm of W_down columns. These modulate output uncertainty
via layer normalization: high-norm neurons have larger impact on the residual
stream and can shift the model toward more peaked or diffuse distributions.

Hypothesis
----------
If H-Neurons and entropy neurons are mechanistically separate, they should have
minimal overlap (low Jaccard index).
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry


@Registry.register_experiment("entropy_neuron_overlap")
class EntropyNeuronOverlapExperiment(BaseExperiment):
    """Identify entropy neurons and compare with H-Neurons."""

    def __init__(
        self,
        name: str = "entropy_neuron_overlap",
        description: str = "Test overlap between H-Neurons and entropy neurons",
        probe_path: Optional[str] = None,
        percentile: float = 99.0,
        **kwargs,
    ):
        self._name = name
        self.description = description
        self.probe_path = probe_path
        self.percentile = percentile

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

    def _identify_entropy_neurons(
        self, backend: InferenceBackend, percentile: float
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Identify high-norm neurons across all layers."""
        num_layers = backend.hook_manager.num_layers

        # Vectorized: collect per-layer column norms, then build (layer, idx)
        # pairs with np.repeat/np.arange instead of Python per-neuron appends.
        layer_norms = []
        for layer in tqdm(range(num_layers), desc="Computing norms"):
            mlp_down_proj = backend.hook_manager.get_mlp_down_proj_module(layer)
            w_down = mlp_down_proj.weight.data.float()
            layer_norms.append(w_down.norm(p=2, dim=0).cpu().numpy())

        all_norms = np.concatenate(layer_norms)
        layer_sizes = np.array([len(n) for n in layer_norms])
        layer_ids = np.repeat(np.arange(num_layers), layer_sizes)
        neuron_info = list(zip(layer_ids.tolist(), np.arange(len(all_norms)).tolist()))

        threshold = np.percentile(all_norms, percentile)

        entropy_neurons = [
            (int(layer_ids[i]), int(i)) for i in np.nonzero(all_norms >= threshold)[0]
        ]

        return entropy_neurons, all_norms

    def _compute_overlap(
        self,
        h_neurons: List[Tuple[int, int]],
        entropy_neurons: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """Compute overlap statistics."""
        h_set = set(h_neurons)
        e_set = set(entropy_neurons)

        overlap = h_set & e_set
        union = h_set | e_set

        jaccard = len(overlap) / len(union) if union else 0.0

        # Layer distribution
        h_layers = [layer for layer, _ in h_neurons]
        e_layers = [layer for layer, _ in entropy_neurons]
        overlap_layers = [layer for layer, _ in overlap]

        return {
            "h_neurons_count": len(h_neurons),
            "entropy_neurons_count": len(entropy_neurons),
            "overlap_count": len(overlap),
            "jaccard_index": float(jaccard),
            "overlap_pct_of_h": float(len(overlap) / len(h_neurons)) if h_neurons else 0.0,
            "overlap_pct_of_entropy": float(len(overlap) / len(entropy_neurons))
            if entropy_neurons
            else 0.0,
            "h_layer_mean": float(np.mean(h_layers)) if h_layers else 0.0,
            "h_layer_std": float(np.std(h_layers)) if h_layers else 0.0,
            "entropy_layer_mean": float(np.mean(e_layers)) if e_layers else 0.0,
            "entropy_layer_std": float(np.std(e_layers)) if e_layers else 0.0,
            "overlap_layer_mean": float(np.mean(overlap_layers)) if overlap_layers else 0.0,
        }

    def _compute_norm_statistics(
        self,
        h_neurons: List[Tuple[int, int]],
        entropy_neurons: List[Tuple[int, int]],
        all_norms: np.ndarray,
        backend: InferenceBackend,
    ) -> Dict[str, float]:
        """Compute norm statistics for each group."""
        # Cache per-layer column norms once; indexing reproduces the per-neuron
        # ``w_down[:, idx].norm(p=2)`` value exactly (same reduction, same dtype).
        col_norms_by_layer: Dict[int, np.ndarray] = {}
        for layer in {layer for layer, _ in h_neurons} | {layer for layer, _ in entropy_neurons}:
            mlp_down_proj = backend.hook_manager.get_mlp_down_proj_module(layer)
            col_norms_by_layer[layer] = (
                mlp_down_proj.weight.data.float().norm(p=2, dim=0).cpu().numpy()
            )

        h_norms = [float(col_norms_by_layer[layer][idx]) for layer, idx in h_neurons]
        e_norms = [float(col_norms_by_layer[layer][idx]) for layer, idx in entropy_neurons]

        return {
            "h_norm_mean": float(np.mean(h_norms)) if h_norms else 0.0,
            "h_norm_std": float(np.std(h_norms)) if h_norms else 0.0,
            "entropy_norm_mean": float(np.mean(e_norms)) if e_norms else 0.0,
            "entropy_norm_std": float(np.std(e_norms)) if e_norms else 0.0,
            "all_norm_mean": float(np.mean(all_norms)),
            "all_norm_std": float(np.std(all_norms)),
            "percentile_threshold": float(np.percentile(all_norms, self.percentile)),
        }

    def run(
        self,
        backend: InferenceBackend,
        dataset: Any = None,
        prompt_strategy: Any = None,
        **kwargs,
    ) -> ExperimentResult:
        """Run entropy neuron overlap analysis."""

        # Load H-Neurons
        h_neurons = self._load_probe()

        print(f"Model         : {backend.model_name}")
        print(f"H-Neurons     : {len(h_neurons)}")
        print(f"Percentile    : {self.percentile}")

        # Identify entropy neurons
        entropy_neurons, all_norms = self._identify_entropy_neurons(backend, self.percentile)

        print(f"Entropy neurons: {len(entropy_neurons)}")

        # Compute overlap
        overlap_metrics = self._compute_overlap(h_neurons, entropy_neurons)
        norm_metrics = self._compute_norm_statistics(h_neurons, entropy_neurons, all_norms, backend)

        metrics = {**overlap_metrics, **norm_metrics}

        # Print summary
        print("\n" + "=" * 66)
        print("ENTROPY NEURON OVERLAP ANALYSIS")
        print("=" * 66)
        print(f"H-Neurons           : {metrics['h_neurons_count']}")
        print(f"Entropy Neurons     : {metrics['entropy_neurons_count']}")
        print(f"Overlap             : {metrics['overlap_count']}")
        print(f"Jaccard Index       : {metrics['jaccard_index']:.4f}")
        print(f"Overlap % of H      : {metrics['overlap_pct_of_h']:.2%}")
        print(f"Overlap % of Entropy: {metrics['overlap_pct_of_entropy']:.2%}")
        print()
        print(f"H-Neuron norm       : {metrics['h_norm_mean']:.3f} ± {metrics['h_norm_std']:.3f}")
        print(
            f"Entropy neuron norm : {metrics['entropy_norm_mean']:.3f} ± {metrics['entropy_norm_std']:.3f}"
        )
        print(
            f"All neurons norm    : {metrics['all_norm_mean']:.3f} ± {metrics['all_norm_std']:.3f}"
        )
        print(f"Threshold (p{self.percentile}): {metrics['percentile_threshold']:.3f}")
        print()
        print(f"H-Neuron layers     : {metrics['h_layer_mean']:.1f} ± {metrics['h_layer_std']:.1f}")
        print(
            f"Entropy layers      : {metrics['entropy_layer_mean']:.1f} ± {metrics['entropy_layer_std']:.1f}"
        )
        print("=" * 66)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="n/a",
            metrics=metrics,
            metadata={
                "probe_path": self.probe_path,
                "percentile": self.percentile,
                "h_neurons": h_neurons,
                "entropy_neurons": entropy_neurons[:100],  # Truncate for size
            },
        )
