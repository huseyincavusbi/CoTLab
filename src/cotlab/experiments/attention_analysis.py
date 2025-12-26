"""Attention Pattern Analysis Experiment.

Extracts attention weights at critical layers (55-60) and computes
attention entropy to understand which tokens each prompt strategy focuses on.

Enhanced to support multiple dataset samples for statistical robustness.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger


@Registry.register_experiment("attention_analysis")
class AttentionAnalysisExperiment(BaseExperiment):
    """
    Analyze attention patterns at critical layers.

    Computes:
    1. Attention entropy per head (higher = more distributed attention)
    2. Top-attended tokens per head
    3. Aggregated statistics across multiple samples
    """

    def __init__(
        self,
        name: str = "attention_analysis",
        description: str = "Analyze attention patterns at critical layers",
        target_layers: Optional[List[int]] = None,
        num_samples: int = 20,
        question: str = "Patient presents with chest pain, sweating, and shortness of breath. What is the diagnosis?",
        **kwargs,
    ):
        self._name = name
        self.description = description
        # Default to layers 55-60 (critical reasoning layers found earlier)
        self._target_layers_config = target_layers or [55, 56, 57, 58, 59, 60]
        self.target_layers = self._target_layers_config
        self.num_samples = num_samples
        self.question = question  # Fallback if no dataset

    @property
    def name(self) -> str:
        return self._name

    def _compute_entropy(self, attn_dist: torch.Tensor) -> float:
        """Compute entropy of attention distribution."""
        eps = 1e-10
        return -torch.sum(attn_dist * torch.log(attn_dist + eps)).item()

    def _analyze_single_sample(
        self,
        model,
        tokenizer,
        prompt: str,
        device: str,
        num_heads: int,
    ) -> Dict[str, Any]:
        """Analyze attention for a single sample."""
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = tokens["input_ids"]

        with torch.no_grad():
            outputs = model(**tokens, output_attentions=True, return_dict=True)

        attentions = outputs.attentions

        if attentions is None or len(attentions) == 0:
            return None

        sample_results = {}

        for layer_idx in self.target_layers:
            if layer_idx >= len(attentions):
                continue

            attn = attentions[layer_idx]  # (batch, heads, seq, seq)
            last_token_attn = attn[0, :, -1, :]  # (heads, seq)

            head_entropies = []
            for h in range(num_heads):
                entropy = self._compute_entropy(last_token_attn[h])
                head_entropies.append(entropy)

            avg_entropy = np.mean(head_entropies)
            min_head = int(np.argmin(head_entropies))

            # Get top-attended tokens for the most focused head
            focused_head_attn = last_token_attn[min_head]
            top_positions = torch.topk(focused_head_attn, k=min(5, input_ids.shape[1]))

            top_tokens = []
            for pos, weight in zip(top_positions.indices.tolist(), top_positions.values.tolist()):
                token_str = tokenizer.decode([input_ids[0, pos]])
                top_tokens.append({"token": token_str, "weight": weight})

            sample_results[layer_idx] = {
                "avg_entropy": avg_entropy,
                "head_entropies": head_entropies,
                "min_entropy": min(head_entropies),
                "max_entropy": max(head_entropies),
                "focused_head": min_head,
                "top_tokens": top_tokens,
            }

        return sample_results

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        num_samples: Optional[int] = None,
        logger: Optional[ExperimentLogger] = None,
    ) -> ExperimentResult:
        """Run attention analysis experiment on multiple samples."""

        tokenizer = backend._tokenizer
        model = backend._model

        # Get model config
        config = model.config
        if hasattr(config, "text_config"):
            config = config.text_config
        num_heads = config.num_attention_heads

        print(f"Model: {backend.model_name}")
        print(f"Attention heads: {num_heads}")
        print(f"Target layers: {self.target_layers}")

        # Set eager attention to enable output_attentions by reloading if necessary
        # We need to check if the model is already using eager attention
        current_attn = getattr(model, "config", None) and getattr(
            model.config, "_attn_implementation", None
        )

        if current_attn != "eager":
            print(f"Current attention implementation: {current_attn}")
            print(
                "Reloading model with attn_implementation='eager' to support output_attentions=True..."
            )
            # We need to preserve the model name before unloading
            model_name = backend.model_name
            backend.unload()
            # Reload with eager attention
            backend.load_model(model_name, attn_implementation="eager")
            model = backend._model
            tokenizer = backend._tokenizer

        # Get samples from dataset
        n_samples = num_samples or self.num_samples
        samples = dataset.sample(n_samples) if n_samples < len(dataset) else list(dataset)
        print(f"\nAnalyzing attention on {len(samples)} samples...")

        # Aggregate statistics across samples
        layer_entropy_stats: Dict[int, List[float]] = defaultdict(list)
        layer_head_entropy_stats: Dict[int, List[List[float]]] = defaultdict(list)
        all_top_tokens: Dict[int, List[str]] = defaultdict(list)

        sample_results = []

        for sample in tqdm(samples, desc="Processing samples"):
            question = sample.text
            prompt = prompt_strategy.build_prompt({"question": question, "text": question})

            result = self._analyze_single_sample(
                model, tokenizer, prompt, backend.device, num_heads
            )

            if result is None:
                print(f"\nWarning: Attention not available for sample {sample.idx}")
                continue

            sample_results.append(
                {
                    "sample_idx": sample.idx,
                    "layer_results": result,
                }
            )

            # Aggregate stats
            for layer_idx, layer_data in result.items():
                layer_entropy_stats[layer_idx].append(layer_data["avg_entropy"])
                layer_head_entropy_stats[layer_idx].append(layer_data["head_entropies"])
                for tok in layer_data["top_tokens"][:3]:  # Top 3 tokens
                    all_top_tokens[layer_idx].append(tok["token"])

        if not sample_results:
            return ExperimentResult(
                experiment_name=self.name,
                model_name=backend.model_name,
                prompt_strategy=prompt_strategy.name
                if hasattr(prompt_strategy, "name")
                else "custom",
                metrics={"error": "attention_not_supported", "num_layers_analyzed": 0},
                raw_outputs=[],
                metadata={"target_layers": self.target_layers},
            )

        # Compute aggregated statistics
        print("\n" + "=" * 70)
        print("ATTENTION ANALYSIS: Aggregated Statistics Across Samples")
        print("=" * 70)
        print(f"{'Layer':<8} | {'Mean Entropy':<14} | {'Std Entropy':<14} | {'Top Tokens'}")
        print("-" * 70)

        aggregated_results = []

        for layer_idx in sorted(layer_entropy_stats.keys()):
            entropies = layer_entropy_stats[layer_idx]
            mean_entropy = np.mean(entropies)
            std_entropy = np.std(entropies)

            # Count most common top tokens
            tokens = all_top_tokens[layer_idx]
            from collections import Counter

            token_counts = Counter(tokens)
            top_3_tokens = token_counts.most_common(5)
            top_tokens_str = ", ".join([f"'{t}'" for t, _ in top_3_tokens[:3]])

            # Aggregate head-level entropies
            head_entropies_all = np.array(
                layer_head_entropy_stats[layer_idx]
            )  # (n_samples, n_heads)
            mean_per_head = np.mean(head_entropies_all, axis=0).tolist()
            std_per_head = np.std(head_entropies_all, axis=0).tolist()

            aggregated_results.append(
                {
                    "layer": layer_idx,
                    "mean_entropy": float(mean_entropy),
                    "std_entropy": float(std_entropy),
                    "mean_per_head": mean_per_head,
                    "std_per_head": std_per_head,
                    "top_tokens": [{"token": t, "count": c} for t, c in top_3_tokens],
                }
            )

            print(
                f"L{layer_idx:<7} | {mean_entropy:<14.4f} | {std_entropy:<14.4f} | {top_tokens_str}"
            )

        print("-" * 70)

        # Overall metrics
        all_mean_entropies = [r["mean_entropy"] for r in aggregated_results]
        overall_mean = np.mean(all_mean_entropies) if all_mean_entropies else 0

        # Find most focused layer
        most_focused_layer = (
            min(aggregated_results, key=lambda x: x["mean_entropy"])["layer"]
            if aggregated_results
            else None
        )
        most_focused_entropy = (
            min(r["mean_entropy"] for r in aggregated_results) if aggregated_results else 0
        )

        metrics = {
            "num_samples_analyzed": len(sample_results),
            "num_layers_analyzed": len(aggregated_results),
            "num_heads": num_heads,
            "overall_mean_entropy": float(overall_mean),
            "most_focused_layer": most_focused_layer,
            "most_focused_entropy": float(most_focused_entropy),
        }

        print(f"\nOverall mean entropy: {overall_mean:.4f}")
        print(f"Most focused layer: L{most_focused_layer} (entropy: {most_focused_entropy:.4f})")

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics=metrics,
            raw_outputs=aggregated_results,
            metadata={
                "target_layers": self.target_layers,
                "num_samples": len(samples),
                "sample_results": sample_results,  # Include per-sample data
            },
        )
