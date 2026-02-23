"""Activation Compare experiment — collect mean residual-stream vectors.

One run = one condition (dataset + prompt settings).
Saves per-layer mean activation vectors to results.json so that two or more
runs can be compared offline with ``scripts/compare_activations.py``.

Design follows logit_lens.py: hooks project inside the callback and move
tensors to CPU immediately to avoid GPU page-faults on long reports.
"""

from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger


@Registry.register_experiment("activation_compare")
class ActivationCompareExperiment(BaseExperiment):
    """
    Collect layer-wise mean residual-stream activations for one condition.

    Forward-passes N samples through the model.  At each layer a lightweight
    hook captures the hidden state at the last token (or mean-pooled across
    all tokens) and moves it to CPU immediately.  After all samples the
    per-layer mean vector is computed and saved to results.json.

    Two saved runs can then be compared with ``scripts/compare_activations.py``
    which computes cosine-similarity and L2-distance profiles per layer.
    """

    def __init__(
        self,
        name: str = "activation_compare",
        description: str = "Collect mean layer activations for representational comparison",
        layer_stride: int = 2,
        num_samples: Optional[int] = None,
        pooling: str = "last_token",  # "last_token" | "mean"
        max_input_tokens: int = 1024,
        seed: int = 42,
        answer_cue: str = "\n\nAnswer:",  # appended so last position mirrors logit_lens
        # Legacy fields kept so old YAML configs don't break
        layers: Optional[List[int]] = None,
        variants: Optional[List[Dict[str, Any]]] = None,
        comparison_mode: str = "pairwise",
        store_per_layer: bool = True,
        log_samples: bool = False,
        **kwargs,
    ):
        self._name = name
        self.description = description
        self.layer_stride = layer_stride
        self.num_samples = num_samples
        self.pooling = pooling
        self.max_input_tokens = max_input_tokens
        self.seed = seed
        self.answer_cue = answer_cue
        # Legacy fields silently ignored — kept for backward compat
        self._layers_legacy = layers
        self._variants_legacy = variants

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_layers(self, backend: InferenceBackend) -> List[int]:
        all_layers = list(range(backend.hook_manager.num_layers))
        return all_layers[:: self.layer_stride]

    def _pool(self, tensor: torch.Tensor) -> torch.Tensor:
        """tensor: [seq_len, hidden_size] → [hidden_size]"""
        if self.pooling == "last_token":
            return tensor[-1]
        elif self.pooling == "mean":
            return tensor.mean(dim=0)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        logger: Optional[ExperimentLogger] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Collect mean residual-stream activations for one dataset condition."""

        target_layers = self._resolve_layers(backend)
        tokenizer = backend._tokenizer

        print(f"Model : {backend.model_name}")
        print(f"Layers ({len(target_layers)}): {target_layers}")
        print(f"Pooling: {self.pooling}")

        if self.num_samples is None:
            samples = list(dataset)
        else:
            samples = dataset.sample(self.num_samples, seed=self.seed)
        n = len(samples)
        print(f"Samples: {n}\n")

        # Accumulators: layer_idx → running sum tensor (float32, CPU)
        layer_sums: Dict[int, torch.Tensor] = {}
        layer_sq_sums: Dict[int, torch.Tensor] = {}
        layer_counts: Dict[int, int] = {}
        processed = 0

        for sample in tqdm(samples, desc="Activation collect"):
            prompt_input = {
                "text": sample.text,
                "question": sample.text,
                "report": sample.text,
                "metadata": sample.metadata or {},
            }
            prompt_str = prompt_strategy.build_prompt(prompt_input) + self.answer_cue

            tokens = tokenizer(
                prompt_str,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(backend.device)

            # Capture hidden state in each hook; move to CPU immediately
            layer_vecs: Dict[int, torch.Tensor] = {}

            def make_hook(layer_idx: int):
                def hook(module, inp, output):
                    tensor = output[0] if isinstance(output, tuple) else output
                    # tensor: [batch, seq_len, hidden]
                    with torch.no_grad():
                        vec = self._pool(tensor[0]).cpu().float()
                    layer_vecs[layer_idx] = vec
                    return output

                return hook

            handles = []
            for layer_idx in target_layers:
                if layer_idx < backend.hook_manager.num_layers:
                    mod = backend.hook_manager.get_residual_module(layer_idx)
                    handles.append(mod.register_forward_hook(make_hook(layer_idx)))

            try:
                with torch.no_grad():
                    backend._model(**tokens)
            except Exception as e:
                tqdm.write(f"  [skip] sample {sample.idx}: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                for h in handles:
                    h.remove()
                continue
            finally:
                for h in handles:
                    h.remove()

            for layer_idx, vec in layer_vecs.items():
                if layer_idx not in layer_sums:
                    layer_sums[layer_idx] = torch.zeros_like(vec)
                    layer_sq_sums[layer_idx] = torch.zeros_like(vec)
                    layer_counts[layer_idx] = 0
                layer_sums[layer_idx] += vec
                layer_sq_sums[layer_idx] += vec**2
                layer_counts[layer_idx] += 1

            del layer_vecs
            torch.cuda.empty_cache()
            processed += 1

        # --- Compute statistics -----------------------------------------
        mean_activations: Dict[int, List[float]] = {}
        activation_norm: Dict[int, float] = {}
        activation_std: Dict[int, float] = {}

        for layer_idx in target_layers:
            cnt = layer_counts.get(layer_idx, 0)
            if cnt == 0:
                continue
            mean_vec = layer_sums[layer_idx] / cnt  # [hidden]
            var_vec = layer_sq_sums[layer_idx] / cnt - mean_vec**2
            std_val = float(var_vec.clamp(min=0).sqrt().mean().item())
            norm_val = float(mean_vec.norm().item())
            mean_activations[layer_idx] = mean_vec.tolist()
            activation_norm[layer_idx] = round(norm_val, 4)
            activation_std[layer_idx] = round(std_val, 6)

        # --- Print summary -----------------------------------------------
        print("\n" + "=" * 60)
        print("ACTIVATION COLLECT SUMMARY")
        print("=" * 60)
        print(f"Processed samples : {processed} / {n}")
        print(f"{'Layer':>6}  {'Norm':>10}  {'Std':>10}")
        print("-" * 32)
        for layer_idx in target_layers:
            if layer_idx in activation_norm:
                print(
                    f"{layer_idx:>6}  {activation_norm[layer_idx]:>10.2f}  {activation_std[layer_idx]:>10.6f}"
                )
        print("=" * 60)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=(
                prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom"
            ),
            metrics={
                "num_samples": processed,
                "pooling": self.pooling,
                "layer_stride": self.layer_stride,
                "activation_norms": activation_norm,
                "activation_std": activation_std,
            },
            raw_outputs={
                "mean_activations_per_layer": mean_activations,
            },
            metadata={
                "target_layers": target_layers,
                "pooling": self.pooling,
                "num_samples": processed,
                "seed": self.seed,
                "answer_cue": self.answer_cue,
            },
        )
