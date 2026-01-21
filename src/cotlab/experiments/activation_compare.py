"""Activation comparison experiment for residual stream activations."""

from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core import create_component
from ..core.base import BaseExperiment, BasePromptStrategy, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger


@Registry.register_experiment("activation_compare")
class ActivationCompareExperiment(BaseExperiment):
    """
    Compare mean residual stream activations across multiple runs.

    Each run can override dataset and/or prompt strategy. The experiment
    caches residual stream activations and compares pooled vectors per layer.
    """

    def __init__(
        self,
        name: str = "activation_compare",
        description: str = "Compare residual stream activations across runs",
        variants: Optional[List[Dict[str, Any]]] = None,
        num_samples: int = 20,
        seed: int = 42,
        layers: Optional[List[int]] = None,
        pooling: str = "last_token",
        comparison_mode: str = "pairwise",
        store_per_layer: bool = True,
        log_samples: bool = False,
        **kwargs,
    ):
        self._name = name
        self.description = description
        self.variants = variants or []
        self.num_samples = num_samples
        self.seed = seed
        self.layers = layers
        self.pooling = pooling
        self.comparison_mode = comparison_mode
        self.store_per_layer = store_per_layer
        self.log_samples = log_samples

    @property
    def name(self) -> str:
        return self._name

    def validate_backend(self, backend: InferenceBackend) -> None:
        if not backend.supports_activations:
            raise ValueError(
                f"Backend {type(backend).__name__} does not support activations. "
                "Use TransformersBackend for activation comparison."
            )

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: BasePromptStrategy,
        num_samples: Optional[int] = None,
        logger: Optional[ExperimentLogger] = None,
        **kwargs,
    ) -> ExperimentResult:
        self.validate_backend(backend)

        base_num_samples = num_samples if num_samples is not None else self.num_samples

        if self.layers is None:
            if not hasattr(backend, "num_layers"):
                raise ValueError("layers must be provided when backend has no num_layers")
            layers = list(range(backend.num_layers))
        else:
            layers = list(self.layers)

        variants = self._normalize_variants(
            variants=self.variants,
            base_dataset=dataset,
            base_prompt=prompt_strategy,
            base_num_samples=base_num_samples,
            base_seed=self.seed,
        )

        run_summaries: List[Dict[str, Any]] = []
        run_states: List[Dict[str, Any]] = []

        for idx, variant in enumerate(variants):
            run_name = variant["name"]
            run_dataset = variant["dataset"]
            run_prompt = variant["prompt"]
            run_num_samples = variant["num_samples"]
            run_seed = variant["seed"]

            print(
                f"Collecting activations for {run_name}: "
                f"{run_dataset.name}/{run_prompt.name} ({run_num_samples} samples)"
            )

            layer_means, sample_indices, system_prompt = self._collect_run_means(
                backend=backend,
                dataset=run_dataset,
                prompt_strategy=run_prompt,
                layers=layers,
                num_samples=run_num_samples,
                seed=run_seed,
                pooling=self.pooling,
                logger=logger if self.log_samples else None,
                run_label=run_name,
            )

            layer_indices = [layer for layer in layers if layer in layer_means]
            layer_norms = [float(torch.norm(layer_means[layer]).item()) for layer in layer_indices]

            summary = {
                "name": run_name,
                "dataset": run_dataset.name,
                "prompt": run_prompt.name,
                "num_samples": len(sample_indices),
                "seed": run_seed,
                "pooling": self.pooling,
                "layer_indices": layer_indices,
                "mean_activation_norms": layer_norms,
                "system_prompt": system_prompt,
            }
            run_summaries.append(summary)
            run_states.append(
                {
                    "name": run_name,
                    "layer_means": layer_means,
                    "layer_indices": layer_indices,
                }
            )

            if logger:
                logger.log_sample(idx, {"run_summary": summary})

        comparisons = self._compare_runs(run_states, layers=layers)

        metrics = {
            "num_runs": len(run_states),
            "num_layers": len(layers),
            "comparison_mode": self.comparison_mode,
            "pooling": self.pooling,
            "pair_count": len(comparisons),
        }

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name or "unknown",
            prompt_strategy=prompt_strategy.name,
            metrics=metrics,
            raw_outputs=[{"runs": run_summaries, "comparisons": comparisons}],
            metadata={
                "layers": layers,
                "variants": [v["name"] for v in variants],
                "description": self.description,
            },
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

    def _collect_run_means(
        self,
        *,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: BasePromptStrategy,
        layers: List[int],
        num_samples: int,
        seed: int,
        pooling: str,
        logger: Optional[ExperimentLogger],
        run_label: str,
    ) -> tuple[Dict[int, torch.Tensor], List[int], Optional[str]]:
        samples = self._select_samples(dataset, num_samples=num_samples, seed=seed)

        layer_sums: Dict[int, torch.Tensor] = {}
        layer_counts: Dict[int, int] = {}
        sample_indices: List[int] = []

        system_prompt = self._get_system_prompt(prompt_strategy)

        for sample in tqdm(samples, desc=f"Samples ({run_label})"):
            prompt = prompt_strategy.build_prompt(
                {
                    "question": sample.text,
                    "text": sample.text,
                    "report": sample.text,
                    "metadata": sample.metadata,
                }
            )
            full_prompt = self._apply_system_prompt(prompt, system_prompt)

            _, cache = backend.forward_with_cache(full_prompt, layers=layers)

            for layer_idx in layers:
                activation = cache.get(layer_idx)
                if activation is None:
                    continue
                pooled = self._pool_activation(activation, pooling=pooling)
                if layer_idx not in layer_sums:
                    layer_sums[layer_idx] = torch.zeros_like(pooled)
                    layer_counts[layer_idx] = 0
                layer_sums[layer_idx] += pooled
                layer_counts[layer_idx] += 1

            cache.clear()
            sample_indices.append(sample.idx)

            if logger:
                logger.log_sample(
                    sample.idx,
                    {"run": run_label, "prompt": prompt, "system_prompt": system_prompt},
                )

        layer_means = {
            layer_idx: layer_sums[layer_idx] / max(layer_counts[layer_idx], 1)
            for layer_idx in layer_sums
        }

        return layer_means, sample_indices, system_prompt

    def _pool_activation(self, activation: torch.Tensor, pooling: str) -> torch.Tensor:
        if activation.dim() == 2:
            pooled = activation[0]
        else:
            if pooling == "mean":
                pooled = activation[0].mean(dim=0)
            elif pooling == "last_token":
                pooled = activation[0, -1]
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")
        return pooled.detach().float().cpu()

    def _select_samples(self, dataset: BaseDataset, *, num_samples: int, seed: int) -> List[Any]:
        if num_samples is None or num_samples <= 0 or num_samples >= len(dataset):
            return list(dataset)
        return dataset.sample(num_samples, seed=seed)

    def _compare_runs(self, run_states: List[Dict[str, Any]], *, layers: List[int]) -> List[Dict]:
        comparisons: List[Dict[str, Any]] = []

        if len(run_states) < 2:
            return comparisons

        pairs = []
        if self.comparison_mode == "baseline":
            base = run_states[0]
            pairs = [(base, other) for other in run_states[1:]]
        elif self.comparison_mode == "pairwise":
            for i in range(len(run_states)):
                for j in range(i + 1, len(run_states)):
                    pairs.append((run_states[i], run_states[j]))
        else:
            raise ValueError(f"Unsupported comparison mode: {self.comparison_mode}")

        for run_a, run_b in pairs:
            common_layers = [
                layer
                for layer in layers
                if layer in run_a["layer_means"] and layer in run_b["layer_means"]
            ]
            cosine_values = []
            l2_values = []

            for layer in common_layers:
                vec_a = run_a["layer_means"][layer]
                vec_b = run_b["layer_means"][layer]
                cosine_values.append(self._cosine_similarity(vec_a, vec_b))
                l2_values.append(float(torch.norm(vec_a - vec_b).item()))

            comparison = {
                "run_a": run_a["name"],
                "run_b": run_b["name"],
                "layer_indices": common_layers,
                "mean_cosine_similarity": self._safe_mean(cosine_values),
                "mean_l2_distance": self._safe_mean(l2_values),
            }

            if self.store_per_layer:
                comparison["layer_cosine_similarity"] = cosine_values
                comparison["layer_l2_distance"] = l2_values

            comparisons.append(comparison)

        return comparisons

    @staticmethod
    def _cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
        denom = torch.norm(vec_a) * torch.norm(vec_b)
        if denom.item() == 0:
            return 0.0
        return float(torch.dot(vec_a, vec_b).item() / denom.item())

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

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
