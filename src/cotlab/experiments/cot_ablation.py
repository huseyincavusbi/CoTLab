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
        num_samples: Optional[int] = None,
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

        n_samples = num_samples if num_samples is not None else self.num_samples
        if n_samples is None:
            samples = list(dataset)
        else:
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
            baseline_last = baseline_logits[0, -1].float()

            # Step 4: Ablate reasoning tokens at each layer and measure effect.
            # The L per-layer ablations are mutually independent interventions
            # (each patches exactly one layer's residual), so they batch into a
            # single forward with L identical rows, one row per layer. Rows are
            # isolated by the causal attention mask (eval, no dropout), so each
            # row reproduces its sequential counterpart exactly.
            ablated_logits_batch = self._forward_with_ablations_batch(
                backend,
                full_text,
                baseline_cache,
                list(range(num_layers)),
                reasoning_positions,
            )

            layer_effects = {}
            for layer_idx in range(num_layers):
                ablated_last = ablated_logits_batch[layer_idx, -1].float()
                effect = torch.norm(ablated_last - baseline_last).item()
                layer_effects[layer_idx] = effect
                layer_effects_sum[layer_idx] += effect

            # Step 5: Check if answer changed with full ablation at critical layer.
            max_effect_layer = max(layer_effects, key=layer_effects.get)
            if self.ablation_type == "noise":
                # Noise mode: the original re-runs the forward with a FRESH noise
                # draw (a different realization than the loop), and that fresh
                # realization drives `answer_changed`; the extra draw also keeps
                # the global RNG state aligned with the sequential path. Preserve
                # both by keeping the single re-run forward for noise only.
                ablated_logits = self._forward_with_ablation(
                    backend,
                    full_text,
                    baseline_cache,
                    max_effect_layer,
                    reasoning_positions,
                )
            else:
                # zero/mean: deterministic per layer, so the batched row for
                # max_effect_layer is bit-identical to a fresh re-run.
                ablated_logits = ablated_logits_batch[max_effect_layer].unsqueeze(0)

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

        ablated_activation = self._build_ablated_activation(source_activation, positions_to_ablate)

        # Register ablation hook
        hook_manager.register_residual_patch_hook(layer_idx, ablated_activation, None)

        try:
            logits, _ = backend.forward_with_cache(prompt, layers=[])
        finally:
            hook_manager.remove_all_hooks()

        return logits

    def _build_ablated_activation(
        self, source_activation: torch.Tensor, positions_to_ablate: List[int]
    ) -> torch.Tensor:
        """Zero/mean/noise-ablate reasoning positions in a source residual.

        Mirrors the sequential per-position loop exactly. For ``noise`` the
        draw order matters: it consumes one ``randn_like`` per position per
        call, in the order the caller invokes this. The batched path builds the
        L layers in the same order the sequential loop would, so the RNG
        consumption per layer is identical.
        """
        ablated_activation = source_activation.clone()
        valid = [p for p in positions_to_ablate if p < ablated_activation.shape[1]]
        if not valid:
            return ablated_activation
        if self.ablation_type == "zero":
            ablated_activation[:, valid, :] = 0
        elif self.ablation_type == "mean":
            # The reference recomputes the mean per position from the
            # progressively-modified tensor (each write changes the mean).
            # Must stay sequential to reproduce it exactly.
            for pos in valid:
                ablated_activation[:, pos, :] = ablated_activation.mean(dim=1)
        elif self.ablation_type == "noise":
            # One randn_like over [P, d] draws the same RNG stream in the same
            # order as the per-position loop (verified bit-identical).
            ablated_activation[:, valid, :] += torch.randn_like(ablated_activation[:, valid, :])
        return ablated_activation

    def _forward_with_ablations_batch(
        self,
        backend: InferenceBackend,
        prompt: str,
        cache,
        layers: List[int],
        positions_to_ablate: List[int],
    ) -> torch.Tensor:
        """Ablate each layer in one forward with ``len(layers)`` identical rows.

        Row ``j`` ablated the residual at ``layers[j]``; rows are isolated by
        the causal attention mask (eval, no dropout), so each row reproduces the
        sequential per-layer forward exactly.

        Returns logits [len(layers), seq_len, vocab] at the last token's row.
        """
        B = len(layers)
        if B == 0:
            raise ValueError("layers must be non-empty for a batched ablation forward")

        # Precompute the ablated residual per layer IN ORDER (matches the
        # sequential loop's RNG draw order for noise mode).
        ablated_by_layer = {}
        for layer_idx in layers:
            source = cache.get(layer_idx)
            if source is None:
                raise ValueError(f"Layer {layer_idx} not in cache")
            ablated_by_layer[layer_idx] = self._build_ablated_activation(
                source, positions_to_ablate
            )

        inputs = backend._tokenizer(prompt, return_tensors="pt").to(backend.device)
        batch_tokens = {k: v.expand(B, -1) for k, v in inputs.items()}
        model = backend._model
        model_dtype = next(model.parameters()).dtype

        def make_ablation_hook(layer_idx: int, row: int):
            src = ablated_by_layer[layer_idx].to(dtype=model_dtype, device=backend.device)

            def hook(module, inp, output):
                if isinstance(output, tuple):
                    patched = list(output)
                    patched[0] = patched[0].clone()
                    patched[0][row] = src[0]
                    return tuple(patched)
                patched = output.clone()
                patched[row] = src[0]
                return patched

            return hook

        handles = []
        for row, layer_idx in enumerate(layers):
            mod = backend.hook_manager.get_residual_module(layer_idx)
            handles.append(mod.register_forward_hook(make_ablation_hook(layer_idx, row)))

        try:
            with torch.inference_mode():
                out = model(**batch_tokens)
        finally:
            for h in handles:
                h.remove()

        return out.logits.detach()
