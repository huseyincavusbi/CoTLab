"""CoT Ablation experiment - test if reasoning tokens affect model answers."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, BasePromptStrategy, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger


@dataclass
class AblationResult:
    """Result from a single ablation test."""

    sample_idx: int
    question: str
    # Original CoT response
    cot_response: str
    cot_answer: str
    cot_reasoning: str
    # After ablating reasoning tokens
    ablated_answer: str
    answer_changed: bool
    # Layer-wise ablation effects
    layer_effects: Dict[int, float] = field(default_factory=dict)


@Registry.register_experiment("cot_ablation")
class CoTAblationExperiment(BaseExperiment):
    """
    Test CoT faithfulness by ablating reasoning token activations.

    This experiment:
    1. Generates a CoT response with reasoning
    2. Identifies which tokens are "reasoning" vs "answer"
    3. Ablates (zeros) reasoning token activations at each layer
    4. Measures how much the final answer changes

    If CoT is faithful, ablating reasoning should change the answer.
    If CoT is post-hoc rationalization, ablating shouldn't matter.
    """

    def __init__(
        self,
        name: str = "cot_ablation",
        description: str = "Test if reasoning tokens causally affect model answers",
        num_samples: int = 10,
        ablation_type: str = "zero",  # "zero", "mean", or "noise"
        **kwargs,
    ):
        self._name = name
        self.description = description
        self.num_samples = num_samples
        self.ablation_type = ablation_type

    @property
    def name(self) -> str:
        return self._name

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: BasePromptStrategy,
        num_samples: Optional[int] = None,
        logger: Optional[ExperimentLogger] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Run the CoT ablation experiment."""
        from ..prompts import ChainOfThoughtStrategy

        n_samples = num_samples or self.num_samples
        samples = dataset.sample(n_samples) if n_samples < len(dataset) else list(dataset)

        # Ensure we have a CoT strategy
        cot_strategy = (
            prompt_strategy
            if isinstance(prompt_strategy, ChainOfThoughtStrategy)
            else ChainOfThoughtStrategy()
        )

        results = []
        metrics = {
            "total_samples": 0,
            "answers_changed": 0,
            "answers_unchanged": 0,
            "avg_reasoning_tokens": 0,
            "avg_effect_per_layer": {},
        }

        # Get hook manager for ablation
        hook_manager = backend.hook_manager
        if hook_manager is None:
            raise RuntimeError("Backend must have hooks enabled for ablation experiment")

        num_layers = hook_manager.num_layers
        layer_effects_sum = {i: 0.0 for i in range(num_layers)}

        print(f"Running CoT Ablation on {len(samples)} samples, {num_layers} layers...")

        for sample in tqdm(samples, desc="Processing samples"):
            input_data = {"question": sample.text, "text": sample.text}

            # Step 1: Generate original CoT response
            cot_prompt = cot_strategy.build_prompt(input_data)
            cot_output = backend.generate(cot_prompt, **kwargs)
            cot_parsed = cot_strategy.parse_response(cot_output.text)

            cot_answer = cot_parsed.get("answer", "")
            cot_reasoning = cot_parsed.get("reasoning", "")

            # Step 2: Build full sequence (prompt + response) and cache it
            tokenizer = backend._tokenizer
            full_text = cot_prompt + cot_output.text
            prompt_tokens = len(tokenizer.encode(cot_prompt))

            # Find where reasoning ends in the response
            reasoning_token_count = self._find_reasoning_end(cot_output.text, tokenizer)

            # Reasoning positions are from prompt_len to prompt_len + reasoning_token_count
            # These are the positions we'll ablate
            reasoning_positions = list(range(prompt_tokens, prompt_tokens + reasoning_token_count))

            # Step 3: Get baseline logits on FULL sequence (prompt + response)
            baseline_logits, baseline_cache = backend.forward_with_cache(
                full_text, layers=list(range(num_layers))
            )

            # Step 4: Ablate reasoning tokens at each layer and measure effect
            layer_effects = {}
            for layer_idx in range(num_layers):
                ablated_logits = self._forward_with_ablation(
                    backend,
                    full_text,
                    baseline_cache,
                    layer_idx,
                    reasoning_positions,
                )

                # Measure effect: how much did logits change at the last position?
                baseline_last = baseline_logits[0, -1].float()
                ablated_last = ablated_logits[0, -1].float()

                effect = torch.norm(ablated_last - baseline_last).item()
                layer_effects[layer_idx] = effect
                layer_effects_sum[layer_idx] += effect

            # Step 5: Check if answer changed with full ablation at critical layer
            max_effect_layer = max(layer_effects, key=layer_effects.get)
            ablated_logits = self._forward_with_ablation(
                backend,
                full_text,
                baseline_cache,
                max_effect_layer,
                reasoning_positions,
            )

            # Get ablated vs baseline predictions
            ablated_token = ablated_logits[0, -1].argmax().item()
            baseline_token = baseline_logits[0, -1].argmax().item()
            ablated_answer = tokenizer.decode([ablated_token])

            answer_changed = ablated_token != baseline_token

            # Record results
            result = AblationResult(
                sample_idx=sample.idx,
                question=sample.text,
                cot_response=cot_output.text,
                cot_answer=cot_answer,
                cot_reasoning=cot_reasoning[:200],  # Truncate for storage
                ablated_answer=ablated_answer,
                answer_changed=answer_changed,
                layer_effects=layer_effects,
            )
            results.append(result)

            # Update metrics
            metrics["total_samples"] += 1
            metrics["avg_reasoning_tokens"] += len(reasoning_positions)
            if answer_changed:
                metrics["answers_changed"] += 1
            else:
                metrics["answers_unchanged"] += 1

            if logger:
                logger.log_sample(sample.idx, result.__dict__)

        # Compute final metrics
        n = metrics["total_samples"]
        if n > 0:
            metrics["avg_reasoning_tokens"] /= n
            metrics["change_rate"] = metrics["answers_changed"] / n
            metrics["unchanged_rate"] = metrics["answers_unchanged"] / n

            for layer_idx in range(num_layers):
                metrics["avg_effect_per_layer"][layer_idx] = layer_effects_sum[layer_idx] / n

        # Format layer effects for output
        for layer_idx in range(num_layers):
            metrics[f"layer_{layer_idx}_avg_effect"] = metrics["avg_effect_per_layer"].get(
                layer_idx, 0
            )

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name or "unknown",
            prompt_strategy=cot_strategy.name,
            metrics=metrics,
            raw_outputs=[r.__dict__ for r in results],
            metadata={
                "num_samples": n,
                "num_layers": num_layers,
                "ablation_type": self.ablation_type,
                "description": self.description,
            },
        )

    def _find_reasoning_end(self, response: str, tokenizer) -> int:
        """Find approximate token position where reasoning ends and answer begins."""
        # Look for common answer markers
        markers = ["Final Answer:", "Therefore,", "The answer is", "answer is"]

        for marker in markers:
            pos = response.lower().find(marker.lower())
            if pos > 0:
                # Return token count up to this position
                reasoning_part = response[:pos]
                return len(tokenizer.encode(reasoning_part))

        # Fallback: use 80% of response as reasoning
        total_tokens = len(tokenizer.encode(response))
        return int(total_tokens * 0.8)

    def _forward_with_ablation(
        self,
        backend: InferenceBackend,
        prompt: str,
        cache,
        layer_idx: int,
        positions_to_ablate: List[int],
    ) -> torch.Tensor:
        """Run forward pass with specific positions ablated at a layer."""
        hook_manager = backend.hook_manager
        source_activation = cache.get(layer_idx)

        if source_activation is None:
            raise ValueError(f"Layer {layer_idx} not in cache")

        # Create ablated activation (zero out specified positions)
        ablated_activation = source_activation.clone()
        for pos in positions_to_ablate:
            if pos < ablated_activation.shape[1]:
                if self.ablation_type == "zero":
                    ablated_activation[:, pos, :] = 0
                elif self.ablation_type == "mean":
                    ablated_activation[:, pos, :] = ablated_activation.mean(dim=1)
                elif self.ablation_type == "noise":
                    ablated_activation[:, pos, :] += torch.randn_like(ablated_activation[:, pos, :])

        # Register ablation hook
        hook_manager.register_residual_patch_hook(layer_idx, ablated_activation, None)

        try:
            logits, _ = backend.forward_with_cache(prompt, layers=[])
        finally:
            hook_manager.remove_all_hooks()

        return logits
