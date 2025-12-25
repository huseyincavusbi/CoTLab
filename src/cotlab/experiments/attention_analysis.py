"""Attention Pattern Analysis Experiment.

Extracts attention weights at critical layers (55-60) and computes
attention entropy to understand which tokens each prompt strategy focuses on.
"""

from typing import Any, List, Optional

import numpy as np
import torch

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
    3. Attention to question vs instruction tokens
    """

    def __init__(
        self,
        name: str = "attention_analysis",
        description: str = "Analyze attention patterns at critical layers",
        target_layers: Optional[List[int]] = None,
        question: str = "Patient presents with chest pain, sweating, and shortness of breath. What is the diagnosis?",
        **kwargs,
    ):
        self._name = name
        self.description = description
        # Default to layers 55-60 (critical reasoning layers found earlier)
        self._target_layers_config = target_layers or [55, 56, 57, 58, 59, 60]
        self.target_layers = self._target_layers_config
        self.question = question

    @property
    def name(self) -> str:
        return self._name

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        logger: Optional[ExperimentLogger] = None,
    ) -> ExperimentResult:
        """Run attention analysis experiment."""

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

        # Build prompt
        prompt = prompt_strategy.build_prompt({"question": self.question})
        tokens = tokenizer(prompt, return_tensors="pt").to(backend.device)
        input_ids = tokens["input_ids"]

        print(f"\nPrompt: {prompt[:100]}...")
        print(f"Token count: {input_ids.shape[1]}")

        # Set eager attention to enable output_attentions
        try:
            model._attn_implementation = "eager"
            for layer in model.modules():
                if hasattr(layer, "_attn_implementation"):
                    layer._attn_implementation = "eager"
        except Exception as e:
            print(f"Warning: Could not set eager attention: {e}")

        # Get attention weights with output_attentions=True
        with torch.no_grad():
            outputs = model(**tokens, output_attentions=True, return_dict=True)

        attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq)

        if attentions is None or len(attentions) == 0:
            print("\nWarning: Model did not return attention weights.")
            print("This model's attention implementation may not support output_attentions=True.")
            return ExperimentResult(
                experiment_name=self.name,
                model_name=backend.model_name,
                prompt_strategy=prompt_strategy.name
                if hasattr(prompt_strategy, "name")
                else "custom",
                metrics={"error": "attention_not_supported", "num_layers_analyzed": 0},
                raw_outputs=[],
                metadata={"target_layers": self.target_layers, "question": self.question},
            )

        results = []
        layer_entropies = {}

        print("\n" + "=" * 60)
        print("ATTENTION ANALYSIS: Entropy per Layer/Head")
        print("=" * 60)
        print(f"{'Layer':<8} | {'Avg Entropy':<12} | {'Min H (head)':<15} | {'Max H (head)':<15}")
        print("-" * 60)

        for layer_idx in self.target_layers:
            if layer_idx >= len(attentions):
                continue

            attn = attentions[layer_idx]  # (batch, heads, seq, seq)

            # Compute entropy for each head at the last token position
            # This is what the model attends to when generating the next token
            last_token_attn = attn[0, :, -1, :]  # (heads, seq)

            head_entropies = []
            for h in range(num_heads):
                # Get attention distribution for this head
                attn_dist = last_token_attn[h]  # (seq,)

                # Compute entropy: H = -sum(p * log(p))
                # Add small epsilon to avoid log(0)
                eps = 1e-10
                entropy = -torch.sum(attn_dist * torch.log(attn_dist + eps)).item()
                head_entropies.append(entropy)

            avg_entropy = np.mean(head_entropies)
            min_entropy = np.min(head_entropies)
            max_entropy = np.max(head_entropies)
            min_head = np.argmin(head_entropies)
            max_head = np.argmax(head_entropies)

            layer_entropies[layer_idx] = {
                "avg_entropy": avg_entropy,
                "head_entropies": head_entropies,
                "min_entropy": min_entropy,
                "max_entropy": max_entropy,
                "min_head": int(min_head),
                "max_head": int(max_head),
            }

            print(
                f"L{layer_idx:<7} | {avg_entropy:<12.4f} | H{min_head} ({min_entropy:.3f}) | H{max_head} ({max_entropy:.3f})"
            )

            # Get top-3 attended positions for lowest entropy head (most focused)
            focused_head_attn = last_token_attn[min_head]
            top_positions = torch.topk(focused_head_attn, k=min(5, input_ids.shape[1]))

            top_tokens = []
            for pos, weight in zip(top_positions.indices.tolist(), top_positions.values.tolist()):
                token_str = tokenizer.decode([input_ids[0, pos]])
                top_tokens.append({"position": pos, "token": token_str, "weight": weight})

            results.append(
                {
                    "layer": layer_idx,
                    "avg_entropy": avg_entropy,
                    "head_entropies": head_entropies,
                    "focused_head": int(min_head),
                    "top_attended_tokens": top_tokens,
                }
            )

        print("-" * 60)

        # Compute overall metrics
        all_entropies = [layer_entropies[layer]["avg_entropy"] for layer in layer_entropies]

        metrics = {
            "num_layers_analyzed": len(results),
            "num_heads": num_heads,
            "overall_avg_entropy": float(np.mean(all_entropies)) if all_entropies else 0,
            "prompt_length": input_ids.shape[1],
        }

        # Find which layer has most focused attention (lowest entropy)
        if layer_entropies:
            most_focused_layer = min(
                layer_entropies.keys(), key=lambda layer: layer_entropies[layer]["avg_entropy"]
            )
            metrics["most_focused_layer"] = most_focused_layer
            metrics["most_focused_entropy"] = layer_entropies[most_focused_layer]["avg_entropy"]

        print(f"\nOverall average entropy: {metrics['overall_avg_entropy']:.4f}")
        if "most_focused_layer" in metrics:
            print(
                f"Most focused layer: L{metrics['most_focused_layer']} (entropy: {metrics['most_focused_entropy']:.4f})"
            )

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics=metrics,
            raw_outputs=results,
            metadata={
                "target_layers": self.target_layers,
                "question": self.question,
            },
        )
