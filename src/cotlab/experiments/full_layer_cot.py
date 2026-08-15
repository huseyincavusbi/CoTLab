"""Full Layer CoT Patching Experiment.

Patches complete layer outputs (attention + MLP) from CoT to Direct prompts
to test if full residual stream transfer affects the answer.
"""

from typing import Any, Dict, List, Optional

import torch

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger
from ..prompts import ChainOfThoughtStrategy, DirectAnswerStrategy


def make_patch_hook(src: torch.Tensor, row: Optional[int] = None, rows=None):
    """Row-aware residual patch hook.

    Patches the last-token residual at a layer. With ``row`` set, only that
    batch row is patched; with ``rows`` (an iterable), every row in it is
    patched (used for cumulative prefixes). ``src`` is [B, 1, hidden].
    """

    def hook(module, input, output):
        patched = output.clone()
        if rows is not None:
            for r in rows:
                patched[r, -1, :] = src[r, -1, :]
        elif row is not None:
            patched[row, -1, :] = src[row, -1, :]
        else:
            patched[:, -1, :] = src[:, -1, :]
        return patched

    return hook


@Registry.register_experiment("full_layer_cot")
class FullLayerCoTExperiment(BaseExperiment):
    """
    Patch complete layer outputs from CoT to Direct prompts.

    Unlike head patching (attention only), this patches the full
    residual stream after each layer, including MLP contributions.
    """

    def __init__(
        self,
        name: str = "full_layer_cot",
        description: str = "Patch full layers from CoT to Direct",
        target_layers: Optional[List[int]] = None,
        question: str = "Patient presents with chest pain, sweating, and shortness of breath. What is the diagnosis?",
        **kwargs,
    ):
        self._name = name
        self.description = description
        # None = auto-detect all layers at runtime
        self._target_layers_config = target_layers
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
        """Run full layer CoT patching experiment."""

        tokenizer = backend._tokenizer
        model = backend._model

        # Auto-detect all layers if not specified
        if self._target_layers_config is None:
            target_layers = list(range(backend.hook_manager.num_layers))
        else:
            target_layers = self._target_layers_config
        self.target_layers = target_layers

        print(f"Model: {backend.model_name}")
        print(f"Target layers: {len(target_layers)} layers")

        # 1. Setup Prompts
        cot_strategy = ChainOfThoughtStrategy()
        direct_strategy = DirectAnswerStrategy()

        cot_prompt = cot_strategy.build_prompt({"question": self.question})
        direct_prompt = direct_strategy.build_prompt({"question": self.question})

        # 2. Get baselines
        direct_tokens = tokenizer(direct_prompt, return_tensors="pt").to(backend.device)
        with torch.inference_mode():
            direct_logits = model(**direct_tokens).logits
        direct_top = torch.argmax(direct_logits[0, -1]).item()
        direct_token = tokenizer.decode([direct_top])
        print(f"\nDirect answer: '{direct_token}'")

        # 3. Cache CoT residual stream
        print("Caching CoT residual stream...")
        cot_cache: Dict[int, torch.Tensor] = {}

        def make_cache_hook(cache_dict: dict, layer_idx: int):
            def hook(module, input, output):
                cache_dict[layer_idx] = output.detach().clone()
                return output

            return hook

        handles = []
        for layer_idx in self.target_layers:
            if layer_idx < backend.hook_manager.num_layers:
                residual_module = backend.hook_manager.get_residual_module(layer_idx)
                h = residual_module.register_forward_hook(make_cache_hook(cot_cache, layer_idx))
                handles.append(h)

        cot_tokens = tokenizer(cot_prompt, return_tensors="pt").to(backend.device)
        with torch.inference_mode():
            cot_logits = model(**cot_tokens).logits

        for h in handles:
            h.remove()

        cot_top = torch.argmax(cot_logits[0, -1]).item()
        cot_token = tokenizer.decode([cot_top])
        print(f"CoT answer: '{cot_token}'")

        # 4. Single layer patching
        print("\n" + "=" * 60)
        print("FULL LAYER COT PATCHING: CoT -> Direct")
        print("=" * 60)
        print(f"{'Layer':<8} | {'Changed?':<10} | {'Top Token':<15}")
        print("-" * 60)

        results = []

        # 4a. Single-layer sweep batched into ONE forward: one row per layer,
        # each row patching a different layer's last-token residual. Rows are
        # isolated by the causal attention mask (eval, no dropout).
        single_layers = sorted(cot_cache.keys())
        if single_layers:
            B = len(single_layers)
            batch_tokens = {k: v.expand(B, -1) for k, v in direct_tokens.items()}
            handles = []
            for row, layer_idx in enumerate(single_layers):
                src = cot_cache[layer_idx][:, -1, :].unsqueeze(0).expand(B, -1, -1)
                residual_module = backend.hook_manager.get_residual_module(layer_idx)
                handles.append(residual_module.register_forward_hook(make_patch_hook(src, row=row)))
            try:
                with torch.inference_mode():
                    patched_logits_batch = model(**batch_tokens).logits
            finally:
                for h in handles:
                    h.remove()

            patched_tops = torch.argmax(patched_logits_batch[:, -1, :], dim=-1).cpu().tolist()
            for row, layer_idx in enumerate(single_layers):
                patched_top = patched_tops[row]
                patched_token = tokenizer.decode([patched_top])
                changed = patched_top != direct_top
                results.append(
                    {
                        "layer": layer_idx,
                        "changed": changed,
                        "patched_token": patched_token,
                    }
                )
                status = "YES" if changed else "no"
                print(f"L{layer_idx:<7} | {status:<10} | {patched_token}")

        # 5. Cumulative patching
        print("\n" + "-" * 60)
        print("CUMULATIVE PATCHING (all layers up to N):")
        print("-" * 60)

        cumulative_results = []

        # 5a. Cumulative sweep batched into ONE forward: row k patches layers
        # target[0..k] (monotone prefixes), so the hook at layer target[j]
        # patches every row >= j. Each row reproduces its sequential forward.
        n_cum = len(self.target_layers)
        if single_layers:
            B = n_cum
            batch_tokens = {k: v.expand(B, -1) for k, v in direct_tokens.items()}
            layer_list = sorted(cot_cache.keys())[:n_cum]
            handles = []
            for j, layer_idx in enumerate(layer_list):
                src = cot_cache[layer_idx][:, -1, :].unsqueeze(0).expand(B, -1, -1)
                residual_module = backend.hook_manager.get_residual_module(layer_idx)
                handles.append(
                    residual_module.register_forward_hook(make_patch_hook(src, rows=range(j, B)))
                )
            try:
                with torch.inference_mode():
                    patched_logits_batch = model(**batch_tokens).logits
            finally:
                for h in handles:
                    h.remove()

            patched_tops = torch.argmax(patched_logits_batch[:, -1, :], dim=-1).cpu().tolist()
            for num_layers in range(1, n_cum + 1):
                row = num_layers - 1
                patched_top = patched_tops[row]
                patched_token = tokenizer.decode([patched_top])
                changed = patched_top != direct_top
                layers_to_patch = layer_list[:num_layers]
                cumulative_results.append(
                    {
                        "num_layers": num_layers,
                        "layers": layers_to_patch,
                        "changed": changed,
                        "patched_token": patched_token,
                    }
                )
                layers_str = ", ".join(f"L{layer}" for layer in layers_to_patch)
                status = "YES" if changed else "no"
                print(f"{layers_str:<25} | {status:<10} | {patched_token}")

        print("-" * 60)

        # Summary
        single_changed = sum(1 for r in results if r["changed"])
        cumulative_changed = sum(1 for r in cumulative_results if r["changed"])

        print(f"\nSingle layers changed: {single_changed}/{len(results)}")
        print(f"Cumulative changed: {cumulative_changed}/{len(cumulative_results)}")

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="cot_vs_direct",
            metrics={
                "direct_top_token": direct_token,
                "cot_top_token": cot_token,
                "single_layers_changed": single_changed,
                "cumulative_changed": cumulative_changed,
            },
            raw_outputs={
                "single_layer": results,
                "cumulative": cumulative_results,
            },
            metadata={"target_layers": self.target_layers},
        )
