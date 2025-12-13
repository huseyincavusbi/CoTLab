"""Full Layer CoT Patching Experiment.

Patches complete layer outputs (attention + MLP) from CoT to Direct prompts
to test if full residual stream transfer affects答案.
"""

from typing import Any, Dict, List, Optional

import torch

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger
from ..prompts import ChainOfThoughtStrategy, DirectAnswerStrategy


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
        # Test key layers across the model
        self.target_layers = target_layers or [5, 10, 15, 20, 25, 30]
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

        print(f"Model: {backend.model_name}")
        print(f"Target layers: {self.target_layers}")

        # 1. Setup Prompts
        cot_strategy = ChainOfThoughtStrategy()
        direct_strategy = DirectAnswerStrategy()

        cot_prompt = cot_strategy.build_prompt({"question": self.question})
        direct_prompt = direct_strategy.build_prompt({"question": self.question})

        # 2. Get baselines
        direct_tokens = tokenizer(direct_prompt, return_tensors="pt").to(backend.device)
        with torch.no_grad():
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
        with torch.no_grad():
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

        for layer_idx in sorted(cot_cache.keys()):
            source_act = cot_cache[layer_idx]

            def make_patch_hook(src):
                def hook(module, input, output):
                    patched = output.clone()
                    patched[:, -1, :] = src[:, -1, :]
                    return patched

                return hook

            residual_module = backend.hook_manager.get_residual_module(layer_idx)
            handle = residual_module.register_forward_hook(make_patch_hook(source_act))

            try:
                with torch.no_grad():
                    patched_logits = model(**direct_tokens).logits

                patched_top = torch.argmax(patched_logits[0, -1]).item()
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

            finally:
                handle.remove()

        # 5. Cumulative patching
        print("\n" + "-" * 60)
        print("CUMULATIVE PATCHING (all layers up to N):")
        print("-" * 60)

        cumulative_results = []

        for num_layers in range(1, len(self.target_layers) + 1):
            layers_to_patch = sorted(cot_cache.keys())[:num_layers]
            handles = []

            for layer_idx in layers_to_patch:
                source_act = cot_cache[layer_idx]

                def make_patch_hook(src):
                    def hook(module, input, output):
                        patched = output.clone()
                        patched[:, -1, :] = src[:, -1, :]
                        return patched

                    return hook

                residual_module = backend.hook_manager.get_residual_module(layer_idx)
                h = residual_module.register_forward_hook(make_patch_hook(source_act))
                handles.append(h)

            try:
                with torch.no_grad():
                    patched_logits = model(**direct_tokens).logits

                patched_top = torch.argmax(patched_logits[0, -1]).item()
                patched_token = tokenizer.decode([patched_top])
                changed = patched_top != direct_top

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

            finally:
                for h in handles:
                    h.remove()

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
