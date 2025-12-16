"""Logit Lens Experiment.

Visualize what the model "thinks" at each layer by projecting
intermediate activations through the unembedding matrix.
"""

from typing import Any, Dict, List, Optional

import torch

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger


@Registry.register_experiment("logit_lens")
class LogitLensExperiment(BaseExperiment):
    """
    Logit Lens: Decode intermediate representations.

    At each layer, project the residual stream through the unembedding
    matrix to see what tokens the model would predict at that point.
    This reveals where reasoning emerges across layers.
    """

    def __init__(
        self,
        name: str = "logit_lens",
        description: str = "Visualize layer-by-layer token predictions",
        target_layers: Optional[List[int]] = None,
        top_k: int = 5,
        question: str = "Patient presents with chest pain, sweating, and shortness of breath. What is the diagnosis?",
        **kwargs,
    ):
        self._name = name
        self.description = description
        # None = auto-detect all layers at runtime
        self._target_layers_config = target_layers
        self.target_layers = target_layers  # Will be set in run() if None
        self.top_k = top_k
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
        """Run logit lens experiment."""

        # Auto-detect all layers if not specified
        if self._target_layers_config is None:
            self.target_layers = list(range(backend.hook_manager.num_layers))
            print(f"Auto-detected {len(self.target_layers)} layers")

        tokenizer = backend._tokenizer
        model = backend._model

        # Get unembedding matrix (lm_head)
        if hasattr(model, "lm_head"):
            lm_head = model.lm_head
        else:
            lm_head = model.get_output_embeddings()

        print(f"Model: {backend.model_name}")
        print(f"Target layers: {self.target_layers}")
        print(f"Top-k tokens: {self.top_k}")

        # 1. Build prompt
        prompt = prompt_strategy.build_prompt({"question": self.question})
        tokens = tokenizer(prompt, return_tensors="pt").to(backend.device)
        print(f"\nPrompt: {prompt[:100]}...")
        print(f"Token count: {tokens['input_ids'].shape[1]}")

        # 2. Cache residual stream at each target layer
        residual_cache: Dict[int, torch.Tensor] = {}

        def make_cache_hook(cache_dict: dict, layer_idx: int):
            def hook(module, input, output):
                cache_dict[layer_idx] = output.detach().clone()
                return output

            return hook

        handles = []
        for layer_idx in self.target_layers:
            if layer_idx < backend.hook_manager.num_layers:
                residual_module = backend.hook_manager.get_residual_module(layer_idx)
                h = residual_module.register_forward_hook(
                    make_cache_hook(residual_cache, layer_idx)
                )
                handles.append(h)

        with torch.no_grad():
            final_logits = model(**tokens).logits

        for h in handles:
            h.remove()

        # 3. Apply logit lens at each layer
        print("\n" + "=" * 60)
        print("LOGIT LENS: Layer-by-Layer Predictions")
        print("=" * 60)
        print("(Last token position - what would the model predict here?)")
        print("-" * 60)

        results = []

        for layer_idx in sorted(residual_cache.keys()):
            residual = residual_cache[layer_idx]
            # Handle both 3D [batch, seq, hidden] (Transformer) and 2D [batch, hidden] (Mamba)
            if residual.dim() == 3:
                last_hidden = residual[0, -1, :]  # [hidden]
            elif residual.dim() == 2:
                last_hidden = residual[0, :]  # [hidden] - Mamba already gives last token
            else:
                print(f"Warning: Unexpected tensor shape at layer {layer_idx}: {residual.shape}")
                continue

            # Project through unembedding
            with torch.no_grad():
                logits = lm_head(last_hidden.unsqueeze(0))  # [1, vocab]

            # Get top-k predictions
            probs = torch.softmax(logits[0], dim=-1)
            top_probs, top_ids = torch.topk(probs, self.top_k)

            top_tokens = [tokenizer.decode([tid]) for tid in top_ids.tolist()]
            top_probs_list = top_probs.tolist()

            layer_result = {
                "layer": layer_idx,
                "top_tokens": top_tokens,
                "top_probs": top_probs_list,
            }
            results.append(layer_result)

            # Format output
            token_strs = ", ".join(
                f"'{t}' ({p:.2%})" for t, p in zip(top_tokens[:3], top_probs_list[:3])
            )
            print(f"Layer {layer_idx:>2}: {token_strs}")

        print("-" * 60)

        # Final prediction
        final_top = torch.argmax(final_logits[0, -1]).item()
        final_token = tokenizer.decode([final_top])
        print(f"\nFinal prediction: '{final_token}'")

        # 4. Analyze where final answer emerges
        final_token_id = final_top
        emergence_layer = None
        for result in results:
            top_ids = tokenizer.encode(result["top_tokens"][0])
            if top_ids and top_ids[-1] == final_token_id:
                emergence_layer = result["layer"]
                break

        if emergence_layer is not None:
            print(f"Answer '{final_token}' first appears at layer {emergence_layer}")
        else:
            print("Answer emerges only at final layer or later layers")

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics={
                "final_prediction": final_token,
                "num_layers_analyzed": len(results),
                "emergence_layer": emergence_layer,
            },
            raw_outputs=results,
            metadata={
                "target_layers": self.target_layers,
                "top_k": self.top_k,
            },
        )
