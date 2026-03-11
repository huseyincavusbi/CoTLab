"""SAE Feature Analysis Experiment.

Uses GemmaScope-2 JumpReLU Sparse Autoencoders to answer two linked questions:

  1. Vocabulary probing (phase 1 — always runs)
     Which sparse features at layers L24-L28 activate most strongly on
     histopathological vocabulary (nuclear pleomorphism, mitotic index, etc.)?

  2. Few-shot contrast (phase 2 — runs when ``few_shot_contrast=True``)
     Do those features show significantly higher activation when few-shot
     histopathology exemplars are present in the context vs. absent?

Pipeline
--------
Phase 1:
  For each term in ``histo_vocab``:
    • Build a short diagnostic context: "Histopathology finding: {term}\\n\\nAnalysis:"
    • Forward pass → cache residual stream at each target layer.
    • SAE-encode residuals at the token positions that decode to the term.
    • Accumulate per-feature activations.
  → Rank features by mean activation across all histo terms.
  → Top-K = "histo features" for phase 2.

Phase 2:
  For each dataset sample:
    • Build prompt with few_shot=True  (clean).
    • Build prompt with few_shot=False (corrupt / zero-shot).
    • Forward pass for each → SAE-encode residuals at last-token position.
    • Collect activation of each histo feature under both conditions.
  → Mann-Whitney U test per feature; Bonferroni correction over top_k features.
  → Effect size = (mean_few_shot - mean_zero_shot) / pooled_std.

No SAELens or TransformerLens dependency — SAE weights are loaded directly
from HuggingFace using ``huggingface_hub`` (already in project deps).
"""

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger
from ..patching.sae import GemmaScopeLayer

# Default histopathology vocabulary — overridable via config.
DEFAULT_HISTO_VOCAB: List[str] = [
    "pleomorphism",
    "mitosis",
    "mitotic",
    "necrosis",
    "carcinoma",
    "adenocarcinoma",
    "squamous",
    "lymphocyte",
    "stroma",
    "hyperchromasia",
    "chromatin",
    "tubular",
    "papillary",
    "cribriform",
    "comedonecrosis",
    "angiolymphatic",
    "perineural",
    "Gleason",
    "Ki-67",
    "HER2",
    "invasion",
    "differentiation",
    "microinvasion",
    "dysplasia",
    "atypia",
]


@Registry.register_experiment("sae_feature_analysis")
class SAEFeatureAnalysisExperiment(BaseExperiment):
    """
    GemmaScope-2 SAE feature analysis at target residual-stream layers.

    Combines vocabulary probing (which features respond to histo terms?) with
    an optional few-shot contrast test (do those features activate more when
    histo exemplars are in-context?).
    """

    def __init__(
        self,
        name: str = "sae_feature_analysis",
        description: str = "GemmaScope-2 SAE histo feature identification and few-shot contrast",
        # SAE configuration
        sae_repo_id: str = "google/gemma-scope-2-270m-it",
        sae_site: str = "resid_post_all",
        sae_width: str = "16k",
        sae_l0: str = "small",
        # Layer selection
        target_layers: Optional[List[int]] = None,  # null = [8,9,10,11,12] for 270m
        # Vocabulary probing
        histo_vocab: Optional[List[str]] = None,  # null = DEFAULT_HISTO_VOCAB
        vocab_context_prefix: str = "Histopathology finding:",
        top_k_features: int = 20,
        # Few-shot contrast
        few_shot_contrast: bool = True,
        num_samples: int = 50,
        seed: int = 42,
        max_input_tokens: int = 1024,
        answer_cue: str = "\n\nAnswer:",
        **kwargs,
    ):
        self._name = name
        self.description = description
        self.sae_repo_id = sae_repo_id
        self.sae_site = sae_site
        self.sae_width = sae_width
        self.sae_l0 = sae_l0
        self._target_layers_config = target_layers
        self.histo_vocab = histo_vocab or DEFAULT_HISTO_VOCAB
        self.vocab_context_prefix = vocab_context_prefix
        self.top_k_features = top_k_features
        self.few_shot_contrast = few_shot_contrast
        self.num_samples = num_samples
        self.seed = seed
        self.max_input_tokens = max_input_tokens
        self.answer_cue = answer_cue

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Layer resolution
    # ------------------------------------------------------------------

    def _resolve_layers(self, backend: InferenceBackend) -> List[int]:
        if self._target_layers_config is not None:
            return list(self._target_layers_config)
        # Default: all layers — lets post-hoc analysis focus on any band (e.g. L24-28).
        return list(range(backend.hook_manager.num_layers))

    # ------------------------------------------------------------------
    # SAE loading
    # ------------------------------------------------------------------

    def _load_saes(self, layers: List[int], device: str) -> Dict[int, GemmaScopeLayer]:
        """Download and move SAE weights for each target layer."""
        hf_token = os.getenv("HF_TOKEN")
        saes: Dict[int, GemmaScopeLayer] = {}
        for layer_idx in layers:
            sae = GemmaScopeLayer.from_pretrained(
                repo_id=self.sae_repo_id,
                layer=layer_idx,
                site=self.sae_site,
                width=self.sae_width,
                l0_label=self.sae_l0,
                token=hf_token,
            )
            saes[layer_idx] = sae.to(device)
        return saes

    # ------------------------------------------------------------------
    # Residual stream extraction
    # ------------------------------------------------------------------

    def _extract_residuals(
        self,
        backend: InferenceBackend,
        tokens: Dict[str, torch.Tensor],
        layers: List[int],
    ) -> Dict[int, torch.Tensor]:
        """Single forward pass caching residual stream at each target layer.

        Returns:
            dict: layer_idx → float32 CPU tensor of shape [seq_len, d_model].
        """
        cache: Dict[int, torch.Tensor] = {}

        def make_hook(layer_idx: int):
            def hook(module, inp, output):
                tensor = output[0] if isinstance(output, tuple) else output
                with torch.no_grad():
                    cache[layer_idx] = tensor[0].detach().float().cpu()

            return hook

        handles = []
        for layer_idx in layers:
            if layer_idx < backend.hook_manager.num_layers:
                mod = backend.hook_manager.get_residual_module(layer_idx)
                handles.append(mod.register_forward_hook(make_hook(layer_idx)))

        try:
            with torch.no_grad():
                backend._model(**tokens)
        finally:
            for h in handles:
                h.remove()

        return cache

    # ------------------------------------------------------------------
    # Token-position helpers
    # ------------------------------------------------------------------

    def _term_token_positions(
        self,
        input_ids: torch.Tensor,
        tokenizer,
        term: str,
    ) -> List[int]:
        """Find token positions in input_ids that overlap with ``term``.

        Uses char-offset mapping to align decoded tokens with the term string.
        Returns an empty list if the term is not found in the decoded text.
        """
        full_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
        term_lower = term.lower()
        # Find all char-level occurrences of the term.
        spans: List[Tuple[int, int]] = []
        for m in re.finditer(re.escape(term_lower), full_text.lower()):
            spans.append((m.start(), m.end()))
        if not spans:
            return []

        # Build cumulative char offset per token.
        tok_ranges: List[Tuple[int, int]] = []
        offset = 0
        for tid in input_ids.tolist():
            decoded = tokenizer.decode([tid], skip_special_tokens=False)
            tok_ranges.append((offset, offset + len(decoded)))
            offset += len(decoded)

        positions: List[int] = []
        for i, (tok_start, tok_end) in enumerate(tok_ranges):
            for es, ee in spans:
                if tok_start < ee and tok_end > es:  # overlap
                    positions.append(i)
                    break

        return positions

    def _tokenize(self, tokenizer, text: str, device: str) -> Dict[str, torch.Tensor]:
        return tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(device)

    # ------------------------------------------------------------------
    # Phase 1: vocabulary probing
    # ------------------------------------------------------------------

    def _probe_vocab(
        self,
        backend: InferenceBackend,
        saes: Dict[int, GemmaScopeLayer],
        layers: List[int],
    ) -> Dict[int, Dict[int, float]]:
        """Probe which SAE features activate on histo vocabulary tokens.

        For each term in ``histo_vocab``:
          1. Build a short diagnostic context.
          2. Forward pass → residuals at each target layer.
          3. SAE-encode residuals at the term's token positions.
          4. Accumulate mean activation per feature.

        Returns:
            histo_scores[layer_idx][feature_idx] = mean activation across all terms.
        """
        tokenizer = backend._tokenizer
        device = backend.device

        # Accumulators: layer → feature_idx → list of activation values.
        accum: Dict[int, Dict[int, List[float]]] = {layer: {} for layer in layers}

        print(f"\nPhase 1: vocabulary probing over {len(self.histo_vocab)} terms …")
        for term in tqdm(self.histo_vocab, desc="Vocab probe"):
            prompt = f"{self.vocab_context_prefix} {term}{self.answer_cue}"
            tokens = self._tokenize(tokenizer, prompt, device)
            input_ids = tokens["input_ids"][0]

            term_positions = self._term_token_positions(input_ids, tokenizer, term)
            if not term_positions:
                # Fallback: use all non-special token positions.
                term_positions = list(range(len(input_ids)))

            residuals = self._extract_residuals(backend, tokens, layers)

            for layer_idx, resid in residuals.items():
                # resid: [seq_len, d_model]  (CPU float32)
                sae = saes[layer_idx]
                term_resid = resid[term_positions]  # [n_toks, d_model]
                with torch.no_grad():
                    features = sae.encode(term_resid.to(sae.w_enc.device))
                    # mean over term tokens → [d_sae]
                    mean_acts = features.mean(dim=0).cpu()

                for feat_idx, val in enumerate(mean_acts.tolist()):
                    if val > 0:
                        accum[layer_idx].setdefault(feat_idx, []).append(val)

        # Aggregate: mean activation per feature (0 if never fired).
        histo_scores: Dict[int, Dict[int, float]] = {}
        for layer_idx in layers:
            scores: Dict[int, float] = {}
            for feat_idx, vals in accum[layer_idx].items():
                scores[feat_idx] = sum(vals) / len(vals)
            histo_scores[layer_idx] = scores

        return histo_scores

    # ------------------------------------------------------------------
    # Phase 2: few-shot contrast
    # ------------------------------------------------------------------

    def _build_prompt(self, prompt_strategy: Any, text: str, metadata: dict, few_shot: bool) -> str:
        """Build prompt with few_shot toggled; restore original value after."""
        orig = getattr(prompt_strategy, "few_shot", None)
        try:
            if hasattr(prompt_strategy, "few_shot"):
                prompt_strategy.few_shot = few_shot
            result = prompt_strategy.build_prompt(
                {"text": text, "question": text, "report": text, "metadata": metadata}
            )
            return result + self.answer_cue
        finally:
            if orig is not None:
                prompt_strategy.few_shot = orig

    def _contrast_few_shot(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        saes: Dict[int, GemmaScopeLayer],
        layers: List[int],
        top_features: Dict[int, List[int]],
    ) -> Dict[int, Dict[int, Dict[str, List[float]]]]:
        """Collect per-feature activations under few-shot vs zero-shot conditions.

        Returns:
            contrast[layer_idx][feature_idx] = {
                "few_shot":  [float, ...],   # one value per sample
                "zero_shot": [float, ...],
            }
        """
        tokenizer = backend._tokenizer
        device = backend.device
        samples = dataset.sample(self.num_samples, seed=self.seed)

        # contrast[layer][feature] = {few_shot: [], zero_shot: []}
        contrast: Dict[int, Dict[int, Dict[str, List[float]]]] = {
            layer: {f: {"few_shot": [], "zero_shot": []} for f in top_features[layer]}
            for layer in layers
        }

        print(f"\nPhase 2: few-shot contrast over {len(samples)} samples …")
        for sample in tqdm(samples, desc="Few-shot contrast"):
            for condition, few_shot_flag in [("few_shot", True), ("zero_shot", False)]:
                try:
                    prompt_str = self._build_prompt(
                        prompt_strategy, sample.text, sample.metadata or {}, few_shot_flag
                    )
                except Exception as exc:
                    tqdm.write(f"  [skip] sample {sample.idx} prompt ({condition}): {exc}")
                    continue

                tokens = self._tokenize(tokenizer, prompt_str, device)
                residuals = self._extract_residuals(backend, tokens, layers)

                for layer_idx, resid in residuals.items():
                    # Use last-token residual — position before the answer letter.
                    last_resid = resid[-1].unsqueeze(0)  # [1, d_model]
                    sae = saes[layer_idx]
                    with torch.no_grad():
                        features = sae.encode(last_resid.to(sae.w_enc.device))
                        feat_acts = features[0].cpu()  # [d_sae]

                    for feat_idx in top_features[layer_idx]:
                        val = float(feat_acts[feat_idx].item())
                        contrast[layer_idx][feat_idx][condition].append(val)

                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return contrast

    # ------------------------------------------------------------------
    # Statistical testing
    # ------------------------------------------------------------------

    @staticmethod
    def _mann_whitney(a: List[float], b: List[float]) -> Tuple[float, float, float]:
        """Mann-Whitney U test (two-sided) + effect size.

        Returns:
            (U_statistic, p_value, effect_size)
            effect_size = (mean_a - mean_b) / pooled_std  (Cohen's d approximation)
        """
        if not a or not b:
            return (float("nan"), float("nan"), float("nan"))

        # Try scipy first.
        try:
            from scipy.stats import mannwhitneyu  # noqa: PLC0415

            result = mannwhitneyu(a, b, alternative="two-sided")
            U, p = float(result.statistic), float(result.pvalue)
        except ImportError:
            # Normal approximation for large samples.
            n1, n2 = len(a), len(b)
            combined = sorted(enumerate(a + b), key=lambda t: t[1])
            ranks = [0.0] * (n1 + n2)
            for rank_idx, (orig_idx, _) in enumerate(combined, start=1):
                ranks[orig_idx] = float(rank_idx)
            U = sum(ranks[:n1]) - n1 * (n1 + 1) / 2
            mu = n1 * n2 / 2
            sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12 + 1e-12)
            z = (U - mu) / sigma
            p = float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))

        # Effect size (Cohen's d approximation).
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        var_a = sum((v - mean_a) ** 2 for v in a) / max(len(a) - 1, 1)
        var_b = sum((v - mean_b) ** 2 for v in b) / max(len(b) - 1, 1)
        pooled_std = math.sqrt((var_a + var_b) / 2 + 1e-12)
        effect_size = (mean_a - mean_b) / pooled_std

        return (U, p, effect_size)

    @staticmethod
    def _bonferroni(p_values: List[float]) -> List[float]:
        n = len(p_values)
        return [min(1.0, p * n) for p in p_values]

    # ------------------------------------------------------------------
    # Summary printing
    # ------------------------------------------------------------------

    def _print_vocab_summary(
        self,
        histo_scores: Dict[int, Dict[int, float]],
        top_features: Dict[int, List[int]],
    ) -> None:
        print("\n" + "=" * 70)
        print("PHASE 1 — VOCABULARY PROBING RESULTS")
        print("=" * 70)
        for layer_idx, feat_list in top_features.items():
            scores = histo_scores[layer_idx]
            print(f"\n  Layer {layer_idx}  (top {len(feat_list)} histo features)")
            print(f"  {'Feature':>10}  {'Mean Activation':>16}")
            print("  " + "-" * 30)
            for fid in feat_list:
                print(f"  {fid:>10}  {scores.get(fid, 0.0):>16.4f}")

    def _print_contrast_summary(
        self,
        contrast_stats: Dict[int, List[Dict]],
    ) -> None:
        print("\n" + "=" * 70)
        print("PHASE 2 — FEW-SHOT CONTRAST RESULTS")
        print("=" * 70)
        for layer_idx, stats_list in contrast_stats.items():
            sig = [s for s in stats_list if s["p_bonferroni"] < 0.05]
            print(
                f"\n  Layer {layer_idx}  — {len(sig)}/{len(stats_list)} features significant (p_bonf<0.05)"
            )
            if sig:
                print(
                    f"  {'Feature':>10}  {'Effect':>8}  {'p_raw':>9}  "
                    f"{'p_bonf':>9}  {'mean_fs':>9}  {'mean_zs':>9}"
                )
                print("  " + "-" * 62)
                for s in sorted(sig, key=lambda x: -abs(x["effect_size"])):
                    direction = "↑" if s["effect_size"] > 0 else "↓"
                    print(
                        f"  {s['feature_idx']:>10}  "
                        f"{s['effect_size']:>+7.3f}{direction}  "
                        f"{s['p_raw']:>9.4f}  "
                        f"{s['p_bonferroni']:>9.4f}  "
                        f"{s['mean_few_shot']:>9.4f}  "
                        f"{s['mean_zero_shot']:>9.4f}"
                    )
        print()
        print("  Positive effect = higher activation with few-shot exemplars.")
        print("=" * 70)

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
        """Run SAE feature analysis experiment."""

        layers = self._resolve_layers(backend)

        print(f"Model         : {backend.model_name}")
        print(f"SAE repo      : {self.sae_repo_id}")
        print(f"SAE site      : {self.sae_site}  width={self.sae_width}  l0={self.sae_l0}")
        print(f"Target layers : {layers}")
        print(f"Histo vocab   : {len(self.histo_vocab)} terms")
        print(f"Few-shot contr: {self.few_shot_contrast}")

        # Load SAEs.
        print("\nLoading SAE weights …")
        saes = self._load_saes(layers, backend.device)

        # ── Phase 1: vocabulary probing ────────────────────────────────
        histo_scores = self._probe_vocab(backend, saes, layers)

        # Select top-K features per layer by mean histo activation.
        top_features: Dict[int, List[int]] = {}
        for layer_idx in layers:
            scores = histo_scores[layer_idx]
            ranked = sorted(scores, key=lambda f: -scores[f])
            top_features[layer_idx] = ranked[: self.top_k_features]

        self._print_vocab_summary(histo_scores, top_features)

        # ── Phase 2: few-shot contrast ─────────────────────────────────
        contrast_stats_per_layer: Dict[int, List[Dict]] = {}

        if self.few_shot_contrast and dataset is not None:
            contrast_raw = self._contrast_few_shot(
                backend, dataset, prompt_strategy, saes, layers, top_features
            )

            for layer_idx in layers:
                stats_list: List[Dict] = []
                raw_p: List[float] = []

                for feat_idx in top_features[layer_idx]:
                    fs = contrast_raw[layer_idx][feat_idx]["few_shot"]
                    zs = contrast_raw[layer_idx][feat_idx]["zero_shot"]
                    U, p_raw, effect = self._mann_whitney(fs, zs)
                    raw_p.append(p_raw if not math.isnan(p_raw) else 1.0)
                    stats_list.append(
                        {
                            "feature_idx": feat_idx,
                            "histo_score": round(histo_scores[layer_idx].get(feat_idx, 0.0), 4),
                            "U_statistic": round(U, 2) if not math.isnan(U) else None,
                            "p_raw": round(p_raw, 6) if not math.isnan(p_raw) else None,
                            "effect_size": round(effect, 4) if not math.isnan(effect) else None,
                            "mean_few_shot": round(sum(fs) / len(fs), 4) if fs else None,
                            "mean_zero_shot": round(sum(zs) / len(zs), 4) if zs else None,
                            "n_few_shot": len(fs),
                            "n_zero_shot": len(zs),
                            "p_bonferroni": None,  # filled below
                        }
                    )

                bonf_p = self._bonferroni(raw_p)
                for entry, bp in zip(stats_list, bonf_p):
                    entry["p_bonferroni"] = round(bp, 6)

                contrast_stats_per_layer[layer_idx] = stats_list

            self._print_contrast_summary(contrast_stats_per_layer)

        # ── Build result ───────────────────────────────────────────────
        # Compact metrics: top feature per layer + significant contrast count.
        compact_metrics: Dict[str, Any] = {
            "layers_analysed": layers,
            "sae_repo_id": self.sae_repo_id,
            "top_k_features": self.top_k_features,
        }
        for layer_idx in layers:
            prefix = f"layer_{layer_idx}"
            scores = histo_scores[layer_idx]
            feats = top_features[layer_idx]
            compact_metrics[f"{prefix}_top_feature"] = feats[0] if feats else None
            compact_metrics[f"{prefix}_top_histo_score"] = (
                round(scores[feats[0]], 4) if feats else 0.0
            )
            if layer_idx in contrast_stats_per_layer:
                n_sig = sum(
                    1
                    for s in contrast_stats_per_layer[layer_idx]
                    if s["p_bonferroni"] is not None and s["p_bonferroni"] < 0.05
                )
                compact_metrics[f"{prefix}_n_significant_features"] = n_sig

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=(
                prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom"
            ),
            metrics=compact_metrics,
            raw_outputs={
                "histo_scores_top_features": {
                    str(layer): {
                        str(f): histo_scores[layer].get(f, 0.0) for f in top_features[layer]
                    }
                    for layer in layers
                },
                "contrast_stats": {
                    str(layer): contrast_stats_per_layer.get(layer, []) for layer in layers
                },
            },
            metadata={
                "sae_repo_id": self.sae_repo_id,
                "sae_site": self.sae_site,
                "sae_width": self.sae_width,
                "sae_l0": self.sae_l0,
                "target_layers": layers,
                "histo_vocab": self.histo_vocab,
                "top_k_features": self.top_k_features,
                "few_shot_contrast": self.few_shot_contrast,
                "num_samples": self.num_samples,
                "seed": self.seed,
            },
        )
