"""Multi-Head CoT Patching Experiment.

Patches multiple CoT attention heads simultaneously to find the
minimal circuit needed to transfer CoT reasoning effects.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger
from ..prompts import ChainOfThoughtStrategy, DirectAnswerStrategy


@Registry.register_experiment("multi_head_cot")
class MultiHeadCoTExperiment(BaseExperiment):
    """
    Patch multiple CoT heads simultaneously.

    Tests if combining heads produces the effect that single heads don't.
    Uses progressive addition: 1 head, 2 heads, 4 heads, 8 heads, etc.
    """

    def __init__(
        self,
        name: str = "multi_head_cot",
        description: str = "Find minimal CoT reasoning circuit",
        target_layers: Optional[List[int]] = None,
        heads_per_layer: Optional[List[int]] = None,
        question: str = "Patient presents with chest pain, sweating, and shortness of breath. What is the diagnosis?",
        **kwargs,
    ):
        self._name = name
        self.description = description
        # None = auto-detect all layers at runtime
        self._target_layers = target_layers
        # Heads to test per layer (auto-detect from model)
        self._heads_per_layer = heads_per_layer
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
        """Run multi-head CoT patching experiment."""

        tokenizer = backend._tokenizer
        model = backend._model

        # Get model config
        config = model.config
        if hasattr(config, "text_config"):
            config = config.text_config
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads

        # Auto-detect all layers if not specified
        if self._target_layers is None:
            self.target_layers = list(range(backend.hook_manager.num_layers))
        else:
            self.target_layers = self._target_layers

        # Auto-detect all heads if not specified
        if self._heads_per_layer is None:
            self.heads_per_layer = list(range(num_heads))
        else:
            self.heads_per_layer = self._heads_per_layer

        print(f"Model: {backend.model_name}")
        print(f"Target layers: {len(self.target_layers)} (all)")
        print(f"Heads per layer: {len(self.heads_per_layer)} (all)")

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

        # 3. Cache CoT attention outputs
        print("Caching CoT attention outputs...")
        cot_attn_cache: Dict[int, torch.Tensor] = {}

        def make_cache_hook(cache_dict: dict, layer_idx: int):
            def hook(module, input, output):
                cache_dict[layer_idx] = output.detach().clone()
                return output

            return hook

        handles = []
        for layer_idx in self.target_layers:
            attn_module = backend.hook_manager.get_attention_output_module(layer_idx)
            h = attn_module.register_forward_hook(make_cache_hook(cot_attn_cache, layer_idx))
            handles.append(h)

        cot_tokens = tokenizer(cot_prompt, return_tensors="pt").to(backend.device)
        with torch.no_grad():
            cot_logits = model(**cot_tokens).logits

        for h in handles:
            h.remove()

        cot_top = torch.argmax(cot_logits[0, -1]).item()
        cot_token = tokenizer.decode([cot_top])
        print(f"CoT answer: '{cot_token}'")

        # 4. Build all (layer, head) combinations
        all_heads: List[Tuple[int, int]] = []
        for layer in self.target_layers:
            for head in self.heads_per_layer:
                all_heads.append((layer, head))

        print(f"Total head combinations: {len(all_heads)}")

        # 5. Progressive patching: 1, 2, 4, 8, 16, all
        test_sizes = [1, 2, 4, 8, 16, len(all_heads)]
        test_sizes = [s for s in test_sizes if s <= len(all_heads)]

        print("\n" + "=" * 60)
        print("MULTI-HEAD COT PATCHING")
        print("=" * 60)
        print(f"{'# Heads':<10} | {'Changed?':<10} | {'Top Token':<15}")
        print("-" * 60)

        results = []

        for num_heads_to_patch in test_sizes:
            heads_to_patch = all_heads[:num_heads_to_patch]

            # Group by layer for efficient patching
            by_layer: Dict[int, List[int]] = {}
            for layer, head in heads_to_patch:
                by_layer.setdefault(layer, []).append(head)

            # Register multi-head patch hooks
            handles = []
            for layer_idx, head_list in by_layer.items():
                h = backend.hook_manager.register_multi_head_patch_hook(
                    layer_idx=layer_idx,
                    head_indices=head_list,
                    source_activation=cot_attn_cache[layer_idx],
                    head_dim=head_dim,
                )
                handles.append(h)

            try:
                with torch.no_grad():
                    patched_logits = model(**direct_tokens).logits

                patched_top = torch.argmax(patched_logits[0, -1]).item()
                patched_token = tokenizer.decode([patched_top])
                changed = patched_top != direct_top

                results.append(
                    {
                        "num_heads": num_heads_to_patch,
                        "changed": changed,
                        "patched_token": patched_token,
                        "heads": [f"L{layer}H{h}" for layer, h in heads_to_patch],
                    }
                )

                status = "YES" if changed else "no"
                print(f"{num_heads_to_patch:<10} | {status:<10} | {patched_token}")

            finally:
                for h in handles:
                    h.remove()

        print("-" * 60)

        # Find minimum heads needed
        min_heads_changed = None
        for r in results:
            if r["changed"]:
                min_heads_changed = r["num_heads"]
                break

        if min_heads_changed:
            print(f"\nMinimum heads to change answer: {min_heads_changed}")
        else:
            print("\nNo combination changed the answer")

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="cot_vs_direct",
            metrics={
                "total_heads_tested": len(all_heads),
                "direct_top_token": direct_token,
                "cot_top_token": cot_token,
                "min_heads_to_change": min_heads_changed,
            },
            raw_outputs=results,
            metadata={
                "target_layers": self.target_layers,
                "heads_per_layer": self.heads_per_layer,
            },
        )
