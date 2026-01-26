"""Activation Patching experiment implementation."""

from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core import create_component
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
        variants: Optional[List[Dict[str, Any]]] = None,
        patching: Dict[str, Any] = None,
        num_samples: int = 50,
        seed: int = 42,
        **kwargs,
    ):
        self._name = name
        self.description = description
        self.variants = variants or []
        self.patching_config = patching or {}
        self.num_samples = num_samples
        self.seed = seed

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
        variants = self._normalize_variants(
            variants=self.variants,
            base_dataset=dataset,
            base_prompt=prompt_strategy,
            base_num_samples=n_samples,
            base_seed=self.seed,
        )

        # If no variants provided, use dataset sample with optional corrupted prompt
        if len(variants) == 1:
            samples = (
                dataset.sample(n_samples, seed=self.seed)
                if n_samples < len(dataset)
                else list(dataset)
            )
            clean_variant = variants[0]
            corrupt_variant = None
        else:
            clean_variant = variants[0]
            corrupt_variant = variants[1]

        # Create patcher
        patcher = ActivationPatcher(backend)

        # Determine layers to sweep
        if self.target_layers:
            layers = self.target_layers
        else:
            layers = list(range(backend.num_layers))

        results = []
        layer_effects = {layer: [] for layer in layers}

        if corrupt_variant is None:
            print(f"Running Activation Patching on {len(samples)} samples, {len(layers)} layers...")
        else:
            clean_dataset = clean_variant["dataset"]
            corrupt_dataset = corrupt_variant["dataset"]
            clean_prompt_strategy = clean_variant["prompt"]
            corrupt_prompt_strategy = corrupt_variant["prompt"]
            clean_samples = (
                clean_dataset.sample(clean_variant["num_samples"], seed=clean_variant["seed"])
                if clean_variant["num_samples"] < len(clean_dataset)
                else list(clean_dataset)
            )
            corrupt_samples = (
                corrupt_dataset.sample(corrupt_variant["num_samples"], seed=corrupt_variant["seed"])
                if corrupt_variant["num_samples"] < len(corrupt_dataset)
                else list(corrupt_dataset)
            )
            pair_count = min(len(clean_samples), len(corrupt_samples))
            print(
                "Running Activation Patching across variants: "
                f"{clean_dataset.name}/{clean_prompt_strategy.name} -> "
                f"{corrupt_dataset.name}/{corrupt_prompt_strategy.name} "
                f"({pair_count} paired samples), {len(layers)} layers..."
            )

        if corrupt_variant is None:
            for sample in tqdm(samples, desc="Processing samples"):
                # Get clean and corrupted prompts
                clean_prompt = sample.text
                corrupted_prompt = sample.metadata.get("corrupted_prompt")

                if corrupted_prompt is None:
                    # If no corrupted prompt, create a simple one
                    corrupted_prompt = "What is the answer?"

                clean_formatted = self._build_prompt(
                    prompt_strategy=prompt_strategy,
                    sample_text=clean_prompt,
                    metadata=sample.metadata,
                )
                corrupted_formatted = self._build_prompt(
                    prompt_strategy=prompt_strategy,
                    sample_text=corrupted_prompt,
                    metadata={},
                )

                # Sweep layers using forward-only patching (no generation)
                sweep_results = patcher.sweep_layers(
                    clean_prompt=clean_formatted,
                    corrupted_prompt=corrupted_formatted,
                    layers=layers,
                )

                sample_result = {
                    "sample_idx": sample.idx,
                    "clean_prompt": clean_prompt,
                    "corrupted_prompt": corrupted_prompt,
                    "layer_results": {},
                }

                for layer_idx, patch_result in sweep_results.items():
                    # Use the effect_score computed from logit comparison
                    effect = patch_result.effect_score

                    layer_effects[layer_idx].append(effect)
                    sample_result["layer_results"][layer_idx] = {
                        "effect": effect,
                    }

                results.append(sample_result)

                if logger:
                    logger.log_sample(sample.idx, sample_result)
        else:
            clean_dataset = clean_variant["dataset"]
            corrupt_dataset = corrupt_variant["dataset"]
            clean_prompt_strategy = clean_variant["prompt"]
            corrupt_prompt_strategy = corrupt_variant["prompt"]
            clean_samples = (
                clean_dataset.sample(clean_variant["num_samples"], seed=clean_variant["seed"])
                if clean_variant["num_samples"] < len(clean_dataset)
                else list(clean_dataset)
            )
            corrupt_samples = (
                corrupt_dataset.sample(corrupt_variant["num_samples"], seed=corrupt_variant["seed"])
                if corrupt_variant["num_samples"] < len(corrupt_dataset)
                else list(corrupt_dataset)
            )
            pair_count = min(len(clean_samples), len(corrupt_samples))
            for idx in tqdm(range(pair_count), desc="Processing paired samples"):
                clean_sample = clean_samples[idx]
                corrupt_sample = corrupt_samples[idx]

                clean_formatted = self._build_prompt(
                    prompt_strategy=clean_prompt_strategy,
                    sample_text=clean_sample.text,
                    metadata=clean_sample.metadata,
                )
                corrupted_formatted = self._build_prompt(
                    prompt_strategy=corrupt_prompt_strategy,
                    sample_text=corrupt_sample.text,
                    metadata=corrupt_sample.metadata,
                )

                # Sweep layers using forward-only patching (no generation)
                sweep_results = patcher.sweep_layers(
                    clean_prompt=clean_formatted,
                    corrupted_prompt=corrupted_formatted,
                    layers=layers,
                )

                sample_result = {
                    "clean_sample_idx": clean_sample.idx,
                    "corrupt_sample_idx": corrupt_sample.idx,
                    "clean_dataset": clean_dataset.name,
                    "corrupt_dataset": corrupt_dataset.name,
                    "clean_prompt": clean_sample.text,
                    "corrupted_prompt": corrupt_sample.text,
                    "layer_results": {},
                }

                for layer_idx, patch_result in sweep_results.items():
                    # Use the effect_score computed from logit comparison
                    effect = patch_result.effect_score

                    layer_effects[layer_idx].append(effect)
                    sample_result["layer_results"][layer_idx] = {
                        "effect": effect,
                    }

                results.append(sample_result)

                if logger:
                    logger.log_sample(idx, sample_result)

        # Compute aggregate metrics
        metrics = {
            "num_layers": len(layers),
            "num_samples": len(results),
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

        metadata = {
            "layers_swept": layers,
            "description": self.description,
        }
        if corrupt_variant is None:
            metadata["variants"] = [variants[0]["name"]]
        else:
            metadata["variants"] = [clean_variant["name"], corrupt_variant["name"]]

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name or "unknown",
            prompt_strategy=prompt_strategy.name,
            metrics=metrics,
            raw_outputs=results,
            metadata=metadata,
        )

    def _normalize_variants(
        self,
        *,
        variants: List[Dict[str, Any]],
        base_dataset: BaseDataset,
        base_prompt: BasePromptStrategy,
        base_num_samples: int,
        base_seed: int,
    ) -> List[Dict[str, Any]]:
        if not variants:
            return [
                {
                    "name": "base",
                    "dataset": base_dataset,
                    "prompt": base_prompt,
                    "num_samples": base_num_samples,
                    "seed": base_seed,
                }
            ]

        normalized = []
        variant_list = list(variants) if isinstance(variants, ListConfig) else variants
        for idx, variant in enumerate(variant_list):
            cfg = variant
            if isinstance(variant, DictConfig):
                cfg = OmegaConf.to_container(variant, resolve=True)

            name = cfg.get("name", f"variant_{idx}")
            dataset_cfg = cfg.get("dataset", None)
            prompt_cfg = cfg.get("prompt", None)
            num_samples = cfg.get("num_samples", base_num_samples)
            seed = cfg.get("seed", base_seed)

            resolved_dataset = self._resolve_component(dataset_cfg, base_dataset)
            resolved_prompt = self._resolve_component(prompt_cfg, base_prompt)

            normalized.append(
                {
                    "name": name,
                    "dataset": resolved_dataset,
                    "prompt": resolved_prompt,
                    "num_samples": num_samples,
                    "seed": seed,
                }
            )

        return normalized

    def _resolve_component(self, cfg: Any, base: Any) -> Any:
        if cfg is None:
            return base
        if isinstance(cfg, str) and cfg.lower() in ("base", "default"):
            return base
        if isinstance(cfg, (BaseDataset, BasePromptStrategy)):
            return cfg
        if isinstance(cfg, DictConfig):
            return create_component(cfg)
        if isinstance(cfg, dict):
            return create_component(OmegaConf.create(cfg))
        raise ValueError(f"Unsupported component config type: {type(cfg)}")

    def _build_prompt(
        self, *, prompt_strategy: BasePromptStrategy, sample_text: str, metadata: Dict[str, Any]
    ) -> str:
        prompt_input = {
            "question": sample_text,
            "text": sample_text,
            "report": sample_text,
            "metadata": metadata or {},
        }
        prompt = prompt_strategy.build_prompt(prompt_input)
        system_prompt = self._get_system_prompt(prompt_strategy)
        return self._apply_system_prompt(prompt, system_prompt)

    @staticmethod
    def _get_system_prompt(prompt_strategy: BasePromptStrategy) -> Optional[str]:
        system_prompt = None
        get_system_message = getattr(prompt_strategy, "get_system_message", None)
        if callable(get_system_message):
            system_prompt = get_system_message()
        if system_prompt is None:
            get_system_prompt = getattr(prompt_strategy, "get_system_prompt", None)
            if callable(get_system_prompt):
                system_prompt = get_system_prompt()
        if not system_prompt:
            return None
        stripped = system_prompt.strip()
        return stripped if stripped else None

    @staticmethod
    def _apply_system_prompt(prompt: str, system_prompt: Optional[str]) -> str:
        if not system_prompt:
            return prompt
        return f"{system_prompt}\n\n{prompt}"

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
