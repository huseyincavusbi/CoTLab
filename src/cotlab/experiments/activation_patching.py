"""Activation Patching experiment implementation."""

from typing import Any, Dict

from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, BasePromptStrategy, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger
from ..patching import ActivationPatcher


@Registry.register_experiment("activation_patching")
class ActivationPatchingExperiment(BaseExperiment):
    """
    Layer-wise causal interventions to study CoT importance.

    This experiment uses activation patching to test the causal
    importance of different layers for the model's reasoning:

    1. Run model on "clean" prompt → cache activations
    2. Run model on "corrupted" prompt → get different output
    3. Patch layer L from clean → corrupted run
    4. If output changes toward clean, layer L is causally important

    This helps answer: "Which layers actually use the CoT reasoning?"
    """

    def __init__(
        self,
        name: str = "activation_patching",
        description: str = "",
        patching: Dict[str, Any] = None,
        num_samples: int = 50,
        **kwargs,
    ):
        self._name = name
        self.description = description
        self.patching_config = patching or {}
        self.num_samples = num_samples

        self.sweep_all_layers = self.patching_config.get("sweep_all_layers", True)
        self.target_layers = self.patching_config.get("target_layers", None)

    @property
    def name(self) -> str:
        return self._name

    def validate_backend(self, backend: InferenceBackend) -> None:
        """Ensure backend supports activation patching."""
        if not backend.supports_activations:
            raise ValueError(
                f"Backend {type(backend).__name__} does not support activations. "
                "Use TransformersBackend for activation patching experiments."
            )

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: BasePromptStrategy,
        num_samples: int = None,
        logger: ExperimentLogger = None,
        **kwargs,
    ) -> ExperimentResult:
        """Run the activation patching experiment."""
        self.validate_backend(backend)

        n_samples = num_samples or self.num_samples
        samples = dataset.sample(n_samples) if n_samples < len(dataset) else list(dataset)

        # Create patcher
        patcher = ActivationPatcher(backend)

        # Determine layers to sweep
        if self.target_layers:
            layers = self.target_layers
        else:
            layers = list(range(backend.num_layers))

        results = []
        layer_effects = {layer: [] for layer in layers}

        print(f"Running Activation Patching on {len(samples)} samples, {len(layers)} layers...")

        for sample in tqdm(samples, desc="Processing samples"):
            # Get clean and corrupted prompts
            clean_prompt = sample.text
            corrupted_prompt = sample.metadata.get("corrupted_prompt")

            if corrupted_prompt is None:
                # If no corrupted prompt, create a simple one
                corrupted_prompt = "What is the answer?"

            input_data = {"question": clean_prompt}
            clean_formatted = prompt_strategy.build_prompt(input_data)

            corrupted_input = {"question": corrupted_prompt}
            corrupted_formatted = prompt_strategy.build_prompt(corrupted_input)

            # Get baseline outputs
            clean_output = backend.generate(clean_formatted, **kwargs)
            corrupted_output = backend.generate(corrupted_formatted, **kwargs)

            # Sweep layers
            sweep_results = patcher.sweep_layers(
                clean_prompt=clean_formatted,
                corrupted_prompt=corrupted_formatted,
                layers=layers,
                **kwargs,
            )

            sample_result = {
                "sample_idx": sample.idx,
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                "clean_output": clean_output.text,
                "corrupted_output": corrupted_output.text,
                "layer_results": {},
            }

            for layer_idx, patch_result in sweep_results.items():
                # Measure how much output changed toward clean
                effect = self._compute_effect(
                    clean_output.text, corrupted_output.text, patch_result.output_text
                )

                layer_effects[layer_idx].append(effect)
                sample_result["layer_results"][layer_idx] = {
                    "patched_output": patch_result.output_text,
                    "effect": effect,
                    "answer_changed": patch_result.answer_changed,
                }

            results.append(sample_result)

            if logger:
                logger.log_sample(sample.idx, sample_result)

        # Compute aggregate metrics
        metrics = {
            "num_layers": len(layers),
            "num_samples": len(samples),
        }

        # Average effect per layer
        avg_effects = {}
        for layer_idx, effects in layer_effects.items():
            avg_effect = sum(effects) / len(effects) if effects else 0
            avg_effects[f"layer_{layer_idx}_avg_effect"] = avg_effect

        metrics.update(avg_effects)

        # Find most important layers
        sorted_layers = sorted(
            layer_effects.items(), key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0, reverse=True
        )
        metrics["top_5_layers"] = [layer for layer, _ in sorted_layers[:5]]

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name or "unknown",
            prompt_strategy=prompt_strategy.name,
            metrics=metrics,
            raw_outputs=results,
            metadata={"layers_swept": layers, "description": self.description},
        )

    def _compute_effect(
        self, clean_output: str, corrupted_output: str, patched_output: str
    ) -> float:
        """
        Compute how much patching moved output from corrupted toward clean.

        Returns a value from 0 (no effect) to 1 (full recovery).
        """
        # Simple word overlap metric
        clean_words = set(clean_output.lower().split())
        corrupted_words = set(corrupted_output.lower().split())
        patched_words = set(patched_output.lower().split())

        if clean_words == corrupted_words:
            return 0.0  # No difference to measure

        # How similar is patched to clean vs corrupted?
        clean_overlap = len(patched_words & clean_words)
        corrupted_overlap = len(patched_words & corrupted_words)

        total = clean_overlap + corrupted_overlap
        if total == 0:
            return 0.0

        # Return fraction toward clean
        return clean_overlap / total
