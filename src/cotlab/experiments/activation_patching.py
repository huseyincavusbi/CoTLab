"""Activation Patching experiment — causal intervention via residual-stream replacement.

Two patching modes
------------------
``pairs``  (default — requires PatchingPairsDataset)
    clean   = sample.text
    corrupt = sample.metadata["corrupted_prompt"]
    Answers Q: which layers encode the specific diagnosis/fact?

``few_shot_contrast``  (works with ANY dataset)
    clean   = few-shot prompt of the sample  (prompt_strategy with few_shot=True)
    corrupt = zero-shot prompt of the sample (prompt_strategy with few_shot=False)
    Answers Q: which layers causally drive few-shot's benefit on OOD / non-OOD?
    Use this mode for afrimedqa, medmcqa, cardiology, etc.

Algorithm (logit-recovery metric, one sample):
  1. Forward clean → cache per-layer residuals (CPU).
  2. Forward corrupt → baseline logit at last token.
  3. For each layer L (strided):
       Re-run corrupt with hook replacing layer L's output with cached clean.
       effect(L) = (logit_patched[clean_tok] - logit_corrupt[clean_tok])
                  / (logit_clean[clean_tok]  - logit_corrupt[clean_tok] + ε)
       1 = full recovery, 0 = no effect, negative = made things worse.

Memory safety: activations moved to CPU immediately inside each hook.
"""

from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger


@Registry.register_experiment("activation_patching")
class ActivationPatchingExperiment(BaseExperiment):
    """
    Layer-wise causal activation patching with logit-recovery scoring.

    Supports two patching modes:
    - ``pairs``              PatchingPairsDataset clean/corrupt pairs.
    - ``few_shot_contrast``  Any dataset — few-shot (clean) vs zero-shot (corrupt).
    """

    def __init__(
        self,
        name: str = "activation_patching",
        description: str = "Layer-wise causal activation patching (logit recovery)",
        patching_mode: str = "pairs",  # "pairs" | "few_shot_contrast"
        layer_stride: int = 2,
        num_samples: int = 50,
        max_input_tokens: int = 1024,
        seed: int = 42,
        answer_cue: str = "\n\nAnswer:",
        # Legacy fields kept so old YAML configs don't break
        variants: Optional[List[Dict[str, Any]]] = None,
        patching: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if patching_mode not in ("pairs", "few_shot_contrast"):
            raise ValueError(
                f"patching_mode must be 'pairs' or 'few_shot_contrast', got {patching_mode!r}"
            )
        self._name = name
        self.description = description
        self.patching_mode = patching_mode
        self.layer_stride = layer_stride
        self.num_samples = num_samples
        self.max_input_tokens = max_input_tokens
        self.seed = seed
        self.answer_cue = answer_cue
        self.patching = patching or {}

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_layers(self, backend: InferenceBackend) -> List[int]:
        all_layers = list(range(backend.hook_manager.num_layers))
        return all_layers[:: self.layer_stride]

    def _resolve_head_targets(self, layers: List[int]) -> Dict[int, List[int]]:
        """Resolve optional head-target mapping from `patching` config.

        Supported configs (mutually exclusive):
        - `patching.head_indices`: list of heads to apply to all `layers`
        - `patching.target_heads`: mapping layer -> list of heads
        """
        head_indices = self.patching.get("head_indices")
        target_heads = self.patching.get("target_heads")

        if head_indices is not None and target_heads is not None:
            raise ValueError("Use either target_heads or head_indices, not both.")

        if target_heads is not None:
            resolved: Dict[int, List[int]] = {}
            for layer_key, heads in dict(target_heads).items():
                layer_idx = int(layer_key)
                if layer_idx in layers:
                    resolved[layer_idx] = [int(h) for h in list(heads)]
            return resolved

        if head_indices is not None:
            head_list = [int(h) for h in list(head_indices)]
            return {layer_idx: head_list for layer_idx in layers}

        return {}

    def _answer_token_id(self, tokenizer, label) -> Optional[int]:
        """Return the first token id of the label string (the logit we track)."""
        if label is None:
            return None
        label_str = str(label).strip()
        if not label_str:
            return None
        for prefix in (" ", ""):
            ids = tokenizer.encode(prefix + label_str, add_special_tokens=False)
            if ids:
                return ids[0]
        return None

    def _tokenize(self, tokenizer, text: str, device):
        return tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(device)

    def _forward_with_cache(
        self,
        backend: InferenceBackend,
        tokens,
        target_layers: List[int],
    ) -> tuple:
        """Run a forward pass, caching residual activations (last token, CPU) per layer.

        Returns:
            logits_last  – [vocab_size] float32 CPU tensor at last token position
            act_cache    – dict[layer_idx → [hidden] float32 CPU tensor]
        """
        act_cache: Dict[int, torch.Tensor] = {}

        def make_cache_hook(layer_idx: int):
            def hook(module, inp, output):
                tensor = output[0] if isinstance(output, tuple) else output
                with torch.no_grad():
                    # keep bfloat16 so patching is dtype-compatible with the model
                    act_cache[layer_idx] = tensor[0, -1].detach().cpu()
                return output

            return hook

        handles = [
            backend.hook_manager.get_residual_module(layer_idx).register_forward_hook(
                make_cache_hook(layer_idx)
            )
            for layer_idx in target_layers
            if layer_idx < backend.hook_manager.num_layers
        ]
        try:
            with torch.no_grad():
                out = backend._model(**tokens)
        finally:
            for h in handles:
                h.remove()

        logits_last = out.logits[0, -1].detach().float().cpu()  # float32 for stable arithmetic
        return logits_last, act_cache

    def _forward_patched(
        self,
        backend: InferenceBackend,
        tokens,
        patch_layer: int,
        patch_vec: torch.Tensor,  # CPU [hidden]
    ) -> torch.Tensor:
        """Forward pass replacing layer `patch_layer` output with `patch_vec`.

        Returns [vocab_size] float32 CPU logit vector at last token.
        """
        # cast to model dtype (bfloat16) before injection — avoids dtype mismatch
        model_dtype = next(backend._model.parameters()).dtype
        patch_gpu = patch_vec.to(dtype=model_dtype, device=backend.device)

        def patch_hook(module, inp, output):
            if isinstance(output, tuple):
                patched = list(output)
                patched[0] = patch_gpu.unsqueeze(0).unsqueeze(0).expand_as(output[0])
                return tuple(patched)
            return patch_gpu.unsqueeze(0).unsqueeze(0).expand_as(output)

        mod = backend.hook_manager.get_residual_module(patch_layer)
        handle = mod.register_forward_hook(patch_hook)
        try:
            with torch.no_grad():
                out = backend._model(**tokens)
        finally:
            handle.remove()
            del patch_gpu

        return out.logits[0, -1].detach().float().cpu()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _build_prompt(self, prompt_strategy: Any, text: str, metadata: dict) -> str:
        return (
            prompt_strategy.build_prompt(
                {
                    "text": text,
                    "question": text,
                    "report": text,
                    "metadata": metadata,
                }
            )
            + self.answer_cue
        )

    def _build_prompt_few_shot(
        self, prompt_strategy: Any, text: str, metadata: dict, few_shot: bool
    ) -> str:
        """Build prompt with few_shot toggled — restores original value afterwards."""
        orig = getattr(prompt_strategy, "few_shot", None)
        try:
            if hasattr(prompt_strategy, "few_shot"):
                prompt_strategy.few_shot = few_shot
            return self._build_prompt(prompt_strategy, text, metadata)
        finally:
            if orig is not None:
                prompt_strategy.few_shot = orig

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        logger: Optional[ExperimentLogger] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Run layer-sweep activation patching over paired (clean, corrupted) samples."""

        target_layers = self._resolve_layers(backend)
        tokenizer = backend._tokenizer

        print(f"Model        : {backend.model_name}")
        print(f"Patching mode: {self.patching_mode}")
        print(f"Layers ({len(target_layers)}): {target_layers}")
        print(f"Stride : {self.layer_stride}  |  max_input_tokens: {self.max_input_tokens}")

        samples = dataset.sample(self.num_samples, seed=self.seed)
        n = len(samples)
        print(f"Samples: {n}  (each requires {len(target_layers) + 2} forward passes)\n")

        # Per-layer effect accumulators
        layer_effects: Dict[int, List[float]] = {lid: [] for lid in target_layers}
        per_sample_results: List[Dict] = []
        processed = 0

        for sample in tqdm(samples, desc="Activation patching"):
            clean_tok_id = self._answer_token_id(tokenizer, sample.label)
            if clean_tok_id is None:
                tqdm.write(f"  [skip] sample {sample.idx}: cannot resolve answer token")
                continue

            # ── Build clean / corrupted prompt strings based on mode ──────
            if self.patching_mode == "pairs":
                corrupted_prompt = sample.metadata.get("corrupted_prompt")
                if not corrupted_prompt:
                    tqdm.write(f"  [skip] sample {sample.idx}: no corrupted_prompt in metadata")
                    continue
                clean_str = self._build_prompt(prompt_strategy, sample.text, sample.metadata or {})
                corr_str = self._build_prompt(prompt_strategy, corrupted_prompt, {})
            else:  # few_shot_contrast
                # few-shot = clean  (more context → better answer representation)
                # zero-shot = corrupted
                clean_str = self._build_prompt_few_shot(
                    prompt_strategy, sample.text, sample.metadata or {}, few_shot=True
                )
                corr_str = self._build_prompt_few_shot(
                    prompt_strategy, sample.text, sample.metadata or {}, few_shot=False
                )

            clean_tokens = self._tokenize(tokenizer, clean_str, backend.device)
            corr_tokens = self._tokenize(tokenizer, corr_str, backend.device)

            try:
                # Step 1 — clean forward, cache activations
                logits_clean, act_cache = self._forward_with_cache(
                    backend, clean_tokens, target_layers
                )
                # Step 2 — corrupted baseline (no patching needed, reuse cache run)
                logits_corr, _ = self._forward_with_cache(backend, corr_tokens, [])
            except Exception as exc:
                tqdm.write(f"  [skip] sample {sample.idx} (baseline): {type(exc).__name__}: {exc}")
                torch.cuda.empty_cache()
                continue

            clean_logit = float(logits_clean[clean_tok_id].item())
            corr_logit = float(logits_corr[clean_tok_id].item())
            denom = clean_logit - corr_logit  # may be 0 or negative

            sample_layer_effects: Dict[int, float] = {}

            # Step 3 — patching sweep over layers
            for layer_idx in target_layers:
                if layer_idx not in act_cache:
                    continue
                try:
                    logits_patch = self._forward_patched(
                        backend, corr_tokens, layer_idx, act_cache[layer_idx]
                    )
                except Exception as exc:
                    tqdm.write(f"  [skip] sample {sample.idx} layer {layer_idx}: {exc}")
                    torch.cuda.empty_cache()
                    continue

                patch_logit = float(logits_patch[clean_tok_id].item())
                eps = 1e-6
                if abs(denom) < eps:
                    effect = 0.0
                else:
                    effect = (patch_logit - corr_logit) / denom
                # Clip to [-1, 2] to handle outliers
                effect = max(-1.0, min(2.0, effect))
                layer_effects[layer_idx].append(effect)
                sample_layer_effects[layer_idx] = round(effect, 4)
                torch.cuda.empty_cache()

            per_sample_results.append(
                {
                    "sample_idx": sample.idx,
                    "clean_logit": round(clean_logit, 4),
                    "corrupt_logit": round(corr_logit, 4),
                    "logit_gap": round(denom, 4),
                    "layer_effects": sample_layer_effects,
                }
            )
            processed += 1
            torch.cuda.empty_cache()

        # --- Aggregate --------------------------------------------------
        mean_effects: Dict[int, float] = {}
        for layer_idx in target_layers:
            vals = layer_effects[layer_idx]
            mean_effects[layer_idx] = round(sum(vals) / len(vals), 4) if vals else 0.0

        sorted_by_effect = sorted(mean_effects.items(), key=lambda x: x[1], reverse=True)
        top_5_layers = [lid for lid, _ in sorted_by_effect[:5]]

        # --- Print summary -----------------------------------------------
        print("\n" + "=" * 62)
        print("ACTIVATION PATCHING SUMMARY  (logit-recovery effect)")
        print("=" * 62)
        print(f"Processed samples : {processed} / {n}")
        print(f"Top-5 causal layers: {top_5_layers}")
        print()
        print(f"{'Layer':>6}  {'Mean Effect':>12}  {'N samples':>10}")
        print("-" * 34)
        for layer_idx in target_layers:
            n_val = len(layer_effects[layer_idx])
            print(f"{layer_idx:>6}  {mean_effects[layer_idx]:>12.4f}  {n_val:>10}")
        print("=" * 62)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=(
                prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom"
            ),
            metrics={
                "num_samples": processed,
                "layer_stride": self.layer_stride,
                "mean_effect_per_layer": mean_effects,
                "top_5_causal_layers": top_5_layers,
            },
            raw_outputs={"per_sample": per_sample_results},
            metadata={
                "target_layers": target_layers,
                "layer_stride": self.layer_stride,
                "num_samples": processed,
                "seed": self.seed,
                "answer_cue": self.answer_cue,
            },
        )
