"""CoT Head Patching Experiment.

Find which attention heads encode Chain-of-Thought reasoning by patching
between CoT and non-CoT prompts.
"""

from typing import Any, Dict, List, Optional

import torch

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger
from ..prompts import ChainOfThoughtStrategy, DirectAnswerStrategy


@Registry.register_experiment("cot_heads")
class CoTHeadsExperiment(BaseExperiment):
    """
    Find attention heads that encode CoT reasoning.

    Patches attention outputs between:
    - Clean: DirectAnswer prompt (no reasoning)
    - Corrupted: CoT prompt (with reasoning)

    Heads that change the answer when patched are "reasoning heads".
    """

    def __init__(
        self,
        name: str = "cot_heads",
        description: str = "Find attention heads encoding CoT reasoning",
        search_layers: Optional[List[int]] = None,
        question: str = "Patient presents with chest pain, sweating, and shortness of breath. What is the diagnosis?",
        **kwargs,
    ):
        self._name = name
        self.description = description
        # None = auto-detect all layers at runtime
        self._search_layers_config = search_layers
        self.search_layers = search_layers  # Will be set in run() if None
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
        """Run CoT head patching experiment."""

        # Auto-detect all layers if not specified
        if self._search_layers_config is None:
            self.search_layers = list(range(backend.hook_manager.num_layers))
            print(f"Auto-detected {len(self.search_layers)} layers")

        tokenizer = backend._tokenizer
        model = backend._model

        # Get model config
        config = model.config
        if hasattr(config, "text_config"):
            config = config.text_config
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads

        print(f"Model: {backend.model_name}")
        print(f"Heads: {num_heads}, Hidden: {hidden_size}, Head dim: {head_dim}")
        print(f"Search layers: {self.search_layers}")

        # 1. Setup Prompts
        cot_strategy = ChainOfThoughtStrategy()
        direct_strategy = DirectAnswerStrategy()

        cot_prompt = cot_strategy.build_prompt({"question": self.question})
        direct_prompt = direct_strategy.build_prompt({"question": self.question})

        print(f"\nCoT prompt length: {len(cot_prompt)} chars")
        print(f"Direct prompt length: {len(direct_prompt)} chars")

        # 2. Get baseline logits for both prompts
        direct_tokens = tokenizer(direct_prompt, return_tensors="pt").to(backend.device)
        with torch.no_grad():
            direct_logits = model(**direct_tokens).logits

        # Get top prediction from direct (no-CoT) prompt
        direct_top = torch.argmax(direct_logits[0, -1]).item()
        direct_token = tokenizer.decode([direct_top])
        print(f"Direct answer starts with: '{direct_token}'")

        # 3. Cache attention outputs from CoT prompt
        print("\nCaching CoT attention outputs...")
        cot_attn_cache: Dict[int, torch.Tensor] = {}

        def make_cache_hook(cache_dict: dict, layer_idx: int):
            def hook(module, input, output):
                cache_dict[layer_idx] = output.detach().clone()
                return output

            return hook

        handles = []
        for layer_idx in self.search_layers:
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
        print(f"CoT answer starts with: '{cot_token}'")

        # 4. Head Patching: Patch CoT heads into Direct run
        print("\n" + "=" * 60)
        print("COT HEAD PATCHING: CoT -> Direct")
        print("=" * 60)
        print(f"{'Layer':<6} | {'Head':<5} | {'Change?':<8} | {'Top Token':<15}")
        print("-" * 60)

        results = []
        changed_heads = []

        for layer_idx in self.search_layers:
            for head_idx in range(num_heads):
                cot_attn = cot_attn_cache[layer_idx]
                head_start = head_idx * head_dim
                head_end = (head_idx + 1) * head_dim

                def make_patch_hook(src, h_start, h_end):
                    def hook(module, input, output):
                        patched = output.clone()
                        patched[:, -1, h_start:h_end] = src[:, -1, h_start:h_end]
                        return patched

                    return hook

                attn_module = backend.hook_manager.get_attention_output_module(layer_idx)
                handle = attn_module.register_forward_hook(
                    make_patch_hook(cot_attn, head_start, head_end)
                )

                try:
                    with torch.no_grad():
                        patched_logits = model(**direct_tokens).logits

                    patched_top = torch.argmax(patched_logits[0, -1]).item()
                    patched_token = tokenizer.decode([patched_top])
                    changed = patched_top != direct_top

                    results.append(
                        {
                            "layer": layer_idx,
                            "head": head_idx,
                            "changed": changed,
                            "patched_token": patched_token,
                        }
                    )

                    if changed:
                        changed_heads.append((layer_idx, head_idx))
                        print(f"L{layer_idx:<5} | H{head_idx:<4} | YES      | {patched_token}")

                finally:
                    handle.remove()

        print("-" * 60)

        # 5. Summary
        print(f"\nTotal heads tested: {len(results)}")
        print(f"Heads that changed answer: {len(changed_heads)}")

        if changed_heads:
            print("\nCOT REASONING HEADS:")
            for layer, head in changed_heads:
                print(f"  Layer {layer}, Head {head}")
        else:
            print("\nNo single head changed the answer (CoT may be distributed)")

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="cot_vs_direct",
            metrics={
                "num_heads_tested": len(results),
                "heads_changed_answer": len(changed_heads),
                "direct_top_token": direct_token,
                "cot_top_token": cot_token,
            },
            raw_outputs=results,
            metadata={
                "search_layers": self.search_layers,
                "changed_heads": [f"L{layer}H{h}" for layer, h in changed_heads],
            },
        )
