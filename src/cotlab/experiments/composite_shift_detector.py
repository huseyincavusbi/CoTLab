"""Composite Distribution Shift Detector Experiment.

Combines two single-pass signals to build a composite OOD / distribution-shift
detector and tests whether that signal *precedes* observable accuracy degradation.

Signals (one forward pass each)
--------------------------------
  1. L61 residual norm  ||h_{L61}[-1]||₂
       Hypothesis: OOD samples → abnormal norms (too high or too low).

  2. L3 attention entropy  -Σ p_head * log(p_head + ε)   averaged over heads
       Entropy of the last-token attention distribution at layer 3.
       Hypothesis: OOD / uncertain samples → diffuse attention → higher entropy.

Composite anomaly score
-----------------------
Mahalanobis distance from an in-distribution baseline:

    d_M(x) = sqrt( (x - μ)ᵀ Σ⁻¹ (x - μ) )

where x = [norm, attn_entropy], μ and Σ are estimated on a calibration split
(``calibration_fraction`` of the data, taken from the first N samples).

Degradation analysis
--------------------
Samples are sorted by ascending composite score (most in-distribution first).
A rolling window of ``window_size`` consecutive sorted samples yields a
rolling accuracy curve.  Spearman ρ between composite score and *inversely*
ordered accuracy measures whether high anomaly score precedes accuracy drop.

Additionally, samples are binned into ``num_bins`` quantile groups and mean
accuracy per bin is reported.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger


@Registry.register_experiment("composite_shift_detector")
class CompositeShiftDetectorExperiment(BaseExperiment):
    """Composite L61-norm + L3-entropy distribution shift detector.

    Single forward pass with ``output_attentions=True`` extracts both signals.
    Mahalanobis distance from a calibration baseline is used as the composite
    anomaly score.  Spearman correlation between anomaly score and accuracy
    over sorted windows tests the "precedes degradation" hypothesis.
    """

    def __init__(
        self,
        name: str = "composite_shift_detector",
        description: str = "L61 norm + L3 entropy Mahalanobis OOD detector",
        norm_layer: Optional[int] = None,  # null = last transformer block
        attn_layer: int = 3,  # L3 attention
        num_samples: Optional[int] = None,
        calibration_fraction: float = 0.3,  # fraction used to fit μ, Σ
        window_size: int = 20,  # rolling accuracy window
        num_bins: int = 5,  # quantile bins for accuracy report
        seed: int = 42,
        max_input_tokens: int = 1024,
        answer_cue: str = "\n\nAnswer:",
        mcq_letters: Optional[List[str]] = None,
        **kwargs,
    ):
        self._name = name
        self.description = description
        self._norm_layer_config = norm_layer
        self.attn_layer = attn_layer
        self.num_samples = num_samples
        self.calibration_fraction = calibration_fraction
        self.window_size = window_size
        self.num_bins = num_bins
        self.seed = seed
        self.max_input_tokens = max_input_tokens
        self.answer_cue = answer_cue
        self.mcq_letters = mcq_letters or list("ABCDEFGHIJ")

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_norm_layer(self, backend: InferenceBackend) -> int:
        if self._norm_layer_config is not None:
            return int(self._norm_layer_config)
        return backend.hook_manager.num_layers - 1

    def _tokenize(self, tokenizer, text: str, device: str) -> Dict[str, torch.Tensor]:
        return tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(device)

    def _answer_letter_token_ids(self, tokenizer) -> List[int]:
        ids = set()
        for letter in self.mcq_letters:
            for prefix in (" ", "", "\n"):
                encoded = tokenizer.encode(prefix + letter, add_special_tokens=False)
                if encoded:
                    ids.add(encoded[-1])
        return sorted(ids)

    def _answer_token_id(self, tokenizer, label) -> Optional[int]:
        if label is None:
            return None
        label_str = str(label).strip()
        for prefix in (" ", ""):
            ids = tokenizer.encode(prefix + label_str, add_special_tokens=False)
            if ids:
                return ids[0]
        return None

    # ------------------------------------------------------------------
    # Single forward pass: residual norm + L3 attention entropy
    # ------------------------------------------------------------------

    def _forward(
        self,
        backend: InferenceBackend,
        tokens: Dict[str, torch.Tensor],
        norm_layer: int,
    ) -> Tuple[torch.Tensor, float, float]:
        """Single forward pass with attention output.

        Returns:
            last_logits  : float32 CPU tensor [vocab_size]
            l2_norm      : scalar float
            attn_entropy : scalar float — mean last-token entropy across heads at L3
        """
        hidden_store: Dict[str, torch.Tensor] = {}

        def hook(module, inp, output):
            tensor = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                hidden_store["h"] = tensor[0, -1].detach().float().cpu()

        mod = backend.hook_manager.get_residual_module(norm_layer)
        handle = mod.register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = backend._model(
                    **tokens,
                    output_attentions=True,
                    return_dict=True,
                )
        finally:
            handle.remove()

        last_logits = out.logits[0, -1].detach().float().cpu()
        l2_norm = float(hidden_store["h"].norm(p=2).item())

        # L3 attention: out.attentions is a tuple len=num_layers,
        # each [batch, heads, seq, seq].  We want last-token row.
        attn_entropy = float("nan")
        attn_l3 = (
            out.attentions[self.attn_layer]
            if (out.attentions is not None and len(out.attentions) > self.attn_layer)
            else None
        )
        if attn_l3 is not None:
            last_tok_attn = attn_l3[0, :, -1, :].float().cpu()  # [heads, seq]
            eps = 1e-10
            ent_per_head = -(last_tok_attn * (last_tok_attn + eps).log()).sum(dim=-1)
            attn_entropy = float(ent_per_head.mean().item())

        return last_logits, l2_norm, attn_entropy

    # ------------------------------------------------------------------
    # Mahalanobis distance
    # ------------------------------------------------------------------

    def _fit_mahalanobis(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit μ and Σ⁻¹ on calibration features.

        Falls back to diagonal covariance if the matrix is singular or sklearn
        is unavailable.
        """
        mu = features.mean(axis=0)
        try:
            from sklearn.covariance import EmpiricalCovariance  # noqa: PLC0415

            ec = EmpiricalCovariance(assume_centered=False).fit(features)
            prec = ec.precision_
        except Exception:
            # Diagonal fallback
            var = features.var(axis=0) + 1e-8
            prec = np.diag(1.0 / var)
        return mu, prec

    def _mahalanobis(self, x: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> float:
        diff = x - mu
        return float(math.sqrt(max(diff @ prec @ diff, 0.0)))

    # ------------------------------------------------------------------
    # Degradation analysis
    # ------------------------------------------------------------------

    def _rolling_accuracy(
        self, scores: List[float], labels: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """Sort by score, compute rolling accuracy over window_size samples."""
        order = np.argsort(scores)
        sorted_labels = np.array([int(labels[i]) for i in order], dtype=float)
        ws = min(self.window_size, len(sorted_labels))
        roll_acc = []
        roll_score = []
        for start in range(len(sorted_labels) - ws + 1):
            chunk = sorted_labels[start : start + ws]
            roll_acc.append(float(chunk.mean()))
            # centre score for this window
            idx_chunk = order[start : start + ws]
            roll_score.append(float(np.mean([scores[i] for i in idx_chunk])))
        return roll_score, roll_acc

    def _spearman(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """Spearman ρ with p-value (scipy if available, else normal approx)."""
        if len(x) < 3:
            return float("nan"), float("nan")
        try:
            from scipy.stats import spearmanr  # noqa: PLC0415

            res = spearmanr(x, y)
            return float(res.statistic), float(res.pvalue)
        except Exception:
            # Normal approximation
            n = len(x)
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            rho = float(np.corrcoef(rx, ry)[0, 1])
            t_stat = rho * math.sqrt((n - 2) / max(1 - rho**2, 1e-10))
            # two-tailed p via normal approx of t
            z = abs(t_stat) / math.sqrt(1 + t_stat**2 / (n - 2))
            p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
            return rho, p

    def _bin_accuracy(self, scores: List[float], labels: List[bool]) -> List[Dict]:
        """Split into num_bins quantile bins, report mean accuracy per bin."""
        if not scores:
            return []
        arr = np.array(scores)
        lbl = np.array([int(b) for b in labels])
        bins = []
        percentiles = np.linspace(0, 100, self.num_bins + 1)
        edges = np.percentile(arr, percentiles)
        for i in range(self.num_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == self.num_bins - 1:
                mask = (arr >= lo) & (arr <= hi)
            else:
                mask = (arr >= lo) & (arr < hi)
            subset = lbl[mask]
            bins.append(
                {
                    "bin": i + 1,
                    "score_range": [round(float(lo), 4), round(float(hi), 4)],
                    "n": int(mask.sum()),
                    "accuracy": round(float(subset.mean()), 4) if len(subset) else None,
                }
            )
        return bins

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(
        self,
        dataset_name: str,
        norm_layer: int,
        n: int,
        n_cal: int,
        accuracy: float,
        spearman_rho: float,
        spearman_p: float,
        auroc: Optional[float],
        mean_score_correct: float,
        mean_score_incorrect: float,
        bins: List[Dict],
    ) -> None:
        print("\n" + "=" * 66)
        print(
            f"COMPOSITE SHIFT DETECTOR — {dataset_name}  (norm=L{norm_layer}, attn=L{self.attn_layer})"
        )
        print("=" * 66)
        print(f"  Samples          : {n}  (calibration: {n_cal})")
        print(f"  Accuracy         : {accuracy:.4f}")
        print()
        print(
            f"  Mahalanobis AUROC                : {auroc:.4f}"
            if auroc is not None
            else "  Mahalanobis AUROC                : n/a"
        )
        print(f"  Mean score (correct)             : {mean_score_correct:.4f}")
        print(f"  Mean score (incorrect)           : {mean_score_incorrect:.4f}")
        print()
        sig = "*" if (not math.isnan(spearman_p) and spearman_p < 0.05) else ""
        print(f"  Spearman ρ (score vs acc window) : {spearman_rho:.4f}{sig}  p={spearman_p:.4f}")
        print()
        print(f"  {'Bin':>4}  {'Score range':>22}  {'N':>5}  {'Accuracy':>8}")
        print("  " + "-" * 44)
        for b in bins:
            acc_str = f"{b['accuracy']:.4f}" if b["accuracy"] is not None else "  n/a "
            lo, hi = b["score_range"]
            print(f"  {b['bin']:>4}  [{lo:>8.4f}, {hi:>8.4f}]  {b['n']:>5}  {acc_str:>8}")
        print("=" * 66)

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
        """Run composite shift detector experiment."""

        norm_layer = self._resolve_norm_layer(backend)
        tokenizer = backend._tokenizer
        device = backend.device

        samples = (
            dataset.sample(self.num_samples, seed=self.seed) if self.num_samples else list(dataset)
        )

        # Switch to eager attention so output_attentions=True is honoured
        model = backend._model
        current_attn = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if current_attn != "eager" and hasattr(model, "set_attn_implementation"):
            print(f"Switching attention implementation: {current_attn} → eager")
            model.set_attn_implementation("eager")

        print(f"Model             : {backend.model_name}")
        print(f"Dataset           : {dataset.name}")
        print(f"Norm layer        : L{norm_layer}")
        print(f"Attention layer   : L{self.attn_layer}")
        print(f"Samples           : {len(samples)}")
        print(f"Calibration frac  : {self.calibration_fraction}")

        letter_ids = self._answer_letter_token_ids(tokenizer)

        per_sample: List[Dict] = []
        all_norms: List[float] = []
        all_entropies: List[float] = []
        all_labels: List[bool] = []

        for sample in tqdm(samples, desc="Composite signal"):
            answer_tok_id = self._answer_token_id(tokenizer, sample.label)

            prompt_str = (
                prompt_strategy.build_prompt(
                    {
                        "text": sample.text,
                        "question": sample.text,
                        "metadata": sample.metadata or {},
                    }
                )
                + self.answer_cue
            )
            tokens = self._tokenize(tokenizer, prompt_str, device)

            try:
                logits, l2_norm, attn_entropy = self._forward(backend, tokens, norm_layer)
            except Exception as exc:
                tqdm.write(f"  [skip] sample {sample.idx}: {exc}")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue

            if answer_tok_id is not None and letter_ids:
                best_letter_tok = max(letter_ids, key=lambda t: logits[t].item())
                is_correct = best_letter_tok == answer_tok_id
            else:
                is_correct = False

            all_norms.append(l2_norm)
            all_entropies.append(attn_entropy)
            all_labels.append(is_correct)

            per_sample.append(
                {
                    "sample_idx": sample.idx,
                    "is_correct": is_correct,
                    "l2_norm": round(l2_norm, 4),
                    "attn_entropy_l3": round(attn_entropy, 6)
                    if not math.isnan(attn_entropy)
                    else None,
                    "mahalanobis": None,  # filled in after calibration
                }
            )

            del logits
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ── Build feature matrix ────────────────────────────────────────
        n = len(all_labels)
        if n == 0:
            print("\n[composite_shift_detector] No samples collected — all were skipped. Aborting.")
            return ExperimentResult(
                experiment_name=self.name,
                model_name=backend.model_name,
                prompt_strategy=(
                    prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom"
                ),
                metrics={"error": "all_samples_skipped", "num_samples": 0},
                raw_outputs={},
                metadata={},
            )
        accuracy = sum(all_labels) / n if n else 0.0

        # Filter out NaN entropies; replace with mean for Mahalanobis
        ent_arr = np.array(all_entropies)
        valid_ent_mask = ~np.isnan(ent_arr)
        if valid_ent_mask.sum() > 0:
            ent_arr[~valid_ent_mask] = float(ent_arr[valid_ent_mask].mean())
        else:
            ent_arr = np.zeros(n)

        norm_arr = np.array(all_norms)
        features = np.column_stack([norm_arr, ent_arr])  # [n, 2]

        # ── Calibration: fit on first calibration_fraction samples ──────
        n_cal = max(2, int(n * self.calibration_fraction))
        cal_features = features[:n_cal]
        mu, prec = self._fit_mahalanobis(cal_features)

        # ── Mahalanobis scores for all samples ──────────────────────────
        scores = [self._mahalanobis(features[i], mu, prec) for i in range(n)]
        for i, rec in enumerate(per_sample):
            rec["mahalanobis"] = round(scores[i], 6)

        # ── AUROC: Mahalanobis vs is_correct (higher score → incorrect) ─
        lbl_arr = np.array([int(b) for b in all_labels])
        auroc = None
        if lbl_arr.sum() > 0 and lbl_arr.sum() < n and n >= 2:
            try:
                from sklearn.metrics import roc_auc_score  # noqa: PLC0415

                # Higher Mahalanobis → OOD → incorrect: negate for AUROC
                auroc = float(roc_auc_score(lbl_arr, [-s for s in scores]))
            except Exception:
                pass

        # ── Rolling accuracy + Spearman ─────────────────────────────────
        roll_scores, roll_acc = self._rolling_accuracy(scores, all_labels)
        spearman_rho, spearman_p = self._spearman(roll_scores, roll_acc)

        # ── Bin analysis ─────────────────────────────────────────────────
        bins = self._bin_accuracy(scores, all_labels)

        # ── Mean score by correctness ────────────────────────────────────
        correct_scores = [s for s, c in zip(scores, all_labels) if c]
        incorrect_scores = [s for s, c in zip(scores, all_labels) if not c]
        mean_score_corr = sum(correct_scores) / len(correct_scores) if correct_scores else 0.0
        mean_score_incorr = (
            sum(incorrect_scores) / len(incorrect_scores) if incorrect_scores else 0.0
        )

        self._print_summary(
            dataset_name=dataset.name,
            norm_layer=norm_layer,
            n=n,
            n_cal=n_cal,
            accuracy=accuracy,
            spearman_rho=spearman_rho,
            spearman_p=spearman_p,
            auroc=auroc,
            mean_score_correct=mean_score_corr,
            mean_score_incorrect=mean_score_incorr,
            bins=bins,
        )

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=(
                prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom"
            ),
            metrics={
                "dataset": dataset.name,
                "norm_layer": norm_layer,
                "attn_layer": self.attn_layer,
                "num_samples": n,
                "calibration_n": n_cal,
                "accuracy": round(accuracy, 4),
                "auroc_mahalanobis": round(auroc, 4) if auroc is not None else None,
                "spearman_rho_score_vs_acc": round(spearman_rho, 4)
                if not math.isnan(spearman_rho)
                else None,
                "spearman_p": round(spearman_p, 4) if not math.isnan(spearman_p) else None,
                "mean_score_correct": round(mean_score_corr, 4),
                "mean_score_incorrect": round(mean_score_incorr, 4),
                "accuracy_by_bin": bins,
            },
            raw_outputs={
                "per_sample": per_sample,
                "rolling": {"scores": roll_scores, "accuracy": roll_acc},
            },
            metadata={
                "norm_layer": norm_layer,
                "attn_layer": self.attn_layer,
                "calibration_fraction": self.calibration_fraction,
                "window_size": self.window_size,
                "num_bins": self.num_bins,
                "seed": self.seed,
            },
        )
