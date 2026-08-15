"""Residual Norm OOD Detection Experiment.

Tests whether the L2 norm of the residual stream at a target layer (default:
last transformer block) correlates with per-sample answer accuracy, and
whether a threshold on that norm can serve as a single-forward-pass OOD flag.

Two metrics are computed from the same single forward pass:

  1. Residual norm  ||h_L[-1]||₂
       The L2 norm of the last-token residual vector at layer L.
       Hypothesis: higher norm → model has formed a strong, confident
       representation → more likely correct.

  2. Logit entropy  -Σ p(letter) · log p(letter)   [MCQ mode]
       Entropy over the answer-letter (A/B/C/D/E) logit distribution.
       This is a single-pass approximation of Semantic Entropy for MCQ tasks.
       Hypothesis: lower entropy → model is more peaked on one answer → more
       likely correct.  AUROC is computed with (-entropy) so that higher
       values still indicate correctness.

Both AUROCs are reported so the norm can be directly compared to the logit
entropy baseline and benchmarked against published multi-pass Semantic Entropy
numbers (Kuhn et al. 2023, AUROC ~0.65–0.75 on medical QA).

Threshold analysis
------------------
A threshold τ* is found on the collected norm scores that maximises balanced
accuracy.  This τ can be applied at inference time as a zero-extra-cost OOD
flag.  Cross-dataset generalisation is evaluated by running the experiment on
multiple datasets and comparing AUROCs — a threshold trained on MedQA that
transfers to PubHealthBench would indicate the norm captures model-internal
state rather than dataset-specific surface features.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger

# MCQ answer letters and their common tokenisation variants.
_MCQ_LETTERS = list("ABCDEFGHIJ")


@Registry.register_experiment("residual_norm_ood")
class ResidualNormOODExperiment(BaseExperiment):
    """
    Single-pass residual norm OOD detection with logit entropy comparison.

    Hooks the residual stream at ``target_layer`` (default: last transformer
    block), extracts the last-token hidden state, computes its L2 norm, and
    collects the logit-entropy over answer letters — all from one forward pass.

    Reports AUROC for both metrics against per-sample correctness, plus a
    threshold search for the norm that maximises balanced accuracy.
    """

    def __init__(
        self,
        name: str = "residual_norm_ood",
        description: str = "L2 residual norm OOD flag vs logit entropy baseline",
        target_layer: Optional[int] = None,  # null = last transformer block
        num_samples: Optional[int] = None,  # null = full dataset
        seed: int = 42,
        max_input_tokens: int = 1024,
        answer_cue: str = "\n\nAnswer:",
        mcq_letters: Optional[List[str]] = None,
        threshold_percentile_step: int = 5,  # granularity of τ search
        batch_size: int = 8,
        **kwargs,
    ):
        self._name = name
        self.description = description
        self._target_layer_config = target_layer
        self.num_samples = num_samples
        self.seed = seed
        self.max_input_tokens = max_input_tokens
        self.answer_cue = answer_cue
        self.mcq_letters = mcq_letters or _MCQ_LETTERS
        self.threshold_percentile_step = threshold_percentile_step
        self.batch_size = batch_size
        self._answer_tok_cache: Dict[str, Optional[int]] = {}

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_layer(self, backend: InferenceBackend) -> int:
        if self._target_layer_config is not None:
            return int(self._target_layer_config)
        # Default: last transformer block (index = num_layers - 1).
        return backend.hook_manager.num_layers - 1

    def _tokenize(self, tokenizer, text: str, device: str) -> Dict[str, torch.Tensor]:
        return tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(device)

    def _tokenize_batch(self, tokenizer, texts: List[str], device: str) -> Dict[str, torch.Tensor]:
        """Left-pad a batch with position_ids remap (logit_lens precedent)."""
        orig_side = tokenizer.padding_side
        orig_pad = tokenizer.pad_token_id
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        try:
            tokens = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
                padding=True,
            ).to(device)
        finally:
            tokenizer.padding_side = orig_side
            tokenizer.pad_token_id = orig_pad

        attention_mask = tokens["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        tokens["position_ids"] = position_ids
        return tokens

    def _answer_letter_token_ids(self, tokenizer) -> List[int]:
        """Collect all plausible token ids for MCQ answer letters."""
        ids = set()
        for letter in self.mcq_letters:
            for prefix in (" ", "", "\n"):
                encoded = tokenizer.encode(prefix + letter, add_special_tokens=False)
                if encoded:
                    ids.add(encoded[-1])
        return sorted(ids)

    def _answer_token_id(self, tokenizer, label) -> Optional[int]:
        """Return the first token id of the label string (memoized per label)."""
        if label is None:
            return None
        label_str = str(label).strip()
        if not label_str:
            return None
        if label_str in self._answer_tok_cache:
            return self._answer_tok_cache[label_str]
        result = None
        for prefix in (" ", ""):
            ids = tokenizer.encode(prefix + label_str, add_special_tokens=False)
            if ids:
                result = ids[0]
                break
        self._answer_tok_cache[label_str] = result
        return result

    def _forward(
        self,
        backend: InferenceBackend,
        tokens: Dict[str, torch.Tensor],
        target_layer: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass.

        Returns:
            last_logits : float32 CPU tensor [vocab_size]
            last_hidden  : float32 CPU tensor [d_model]  at target_layer
        """
        last_hidden_store: Dict[str, torch.Tensor] = {}

        def hook(module, inp, output):
            tensor = output[0] if isinstance(output, tuple) else output
            with torch.inference_mode():
                last_hidden_store["h"] = tensor[0, -1].detach().float().cpu()

        mod = backend.hook_manager.get_residual_module(target_layer)
        handle = mod.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = backend._model(**tokens)
        finally:
            handle.remove()

        last_logits = out.logits[0, -1].detach().float().cpu()
        return last_logits, last_hidden_store["h"]

    def _forward_batch(
        self,
        backend: InferenceBackend,
        tokens: Dict[str, torch.Tensor],
        target_layer: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched forward over a left-padded batch (per-row last token).

        Returns:
            last_logits  : [B, vocab] float32 CPU
            last_hidden  : [B, d_model] float32 CPU
        """
        last_hidden_store: Dict[str, torch.Tensor] = {}

        def hook(module, inp, output):
            tensor = output[0] if isinstance(output, tuple) else output
            with torch.inference_mode():
                last_hidden_store["h"] = tensor[:, -1, :].detach().float().cpu()  # [B, d]

        mod = backend.hook_manager.get_residual_module(target_layer)
        handle = mod.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = backend._model(**tokens)
        finally:
            handle.remove()

        last_logits = out.logits[:, -1, :].detach().float().cpu()  # [B, vocab]
        return last_logits, last_hidden_store["h"]

    # ------------------------------------------------------------------
    # Per-sample metric computation
    # ------------------------------------------------------------------

    def _compute_norm(self, hidden: torch.Tensor) -> float:
        return float(hidden.norm(p=2).item())

    def _compute_logit_entropy(self, logits: torch.Tensor, letter_ids: List[int]) -> float:
        """Entropy over answer-letter logits (MCQ single-pass SE proxy).

        Softmax is applied only over the letter token ids so the distribution
        sums to 1 over the MCQ choice set.
        """
        if not letter_ids:
            return float("nan")
        letter_logits = logits[letter_ids]
        probs = torch.softmax(letter_logits, dim=0)
        entropy = -float((probs * (probs + 1e-10).log()).sum().item())
        return entropy

    # ------------------------------------------------------------------
    # Threshold search
    # ------------------------------------------------------------------

    def _find_threshold(self, norms: List[float], labels: List[bool]) -> Tuple[float, float]:
        """Find τ* that maximises balanced accuracy on (norms > τ) → correct.

        Returns:
            (tau_star, balanced_acc_at_tau)
        """
        arr = np.array(norms)
        lbl = np.array(labels, dtype=int)
        percentiles = np.arange(
            self.threshold_percentile_step,
            100,
            self.threshold_percentile_step,
        )
        best_tau, best_ba = float(arr.mean()), 0.0
        unique_labels = set(lbl.tolist())
        if len(unique_labels) < 2:
            return best_tau, best_ba
        for pct in percentiles:
            tau = float(np.percentile(arr, pct))
            preds = (arr >= tau).astype(int)
            if preds.sum() == 0 or preds.sum() == len(preds):
                continue
            ba = balanced_accuracy_score(lbl, preds)
            if ba > best_ba:
                best_ba, best_tau = ba, tau
        return best_tau, best_ba

    # ------------------------------------------------------------------
    # Summary printing
    # ------------------------------------------------------------------

    def _print_summary(
        self,
        dataset_name: str,
        target_layer: int,
        n: int,
        accuracy: float,
        auroc_norm: float,
        auroc_entropy: float,
        tau: float,
        balanced_acc: float,
        mean_norm_correct: float,
        mean_norm_incorrect: float,
        mean_entropy_correct: float,
        mean_entropy_incorrect: float,
    ) -> None:
        print("\n" + "=" * 66)
        print(f"RESIDUAL NORM OOD — {dataset_name}  (L{target_layer})")
        print("=" * 66)
        print(f"  Samples      : {n}")
        print(f"  Accuracy     : {accuracy:.4f}")
        print()
        print(f"  {'Metric':<22}  {'AUROC':>7}  {'mean(corr)':>11}  {'mean(incorr)':>12}")
        print("  " + "-" * 56)
        auc_n = f"{auroc_norm:.4f}" if auroc_norm is not None else "  n/a "
        auc_e = f"{auroc_entropy:.4f}" if auroc_entropy is not None else "  n/a "
        print(
            f"  {'L2 norm':.<22}  {auc_n:>7}  "
            f"{mean_norm_correct:>11.2f}  {mean_norm_incorrect:>12.2f}"
        )
        print(
            f"  {'Logit entropy (neg)':.<22}  {auc_e:>7}  "
            f"{mean_entropy_correct:>11.4f}  {mean_entropy_incorrect:>12.4f}"
        )
        print()
        print(f"  Norm threshold τ* : {tau:.2f}")
        print(f"  Balanced acc @ τ* : {balanced_acc:.4f}")
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
        """Run residual norm OOD experiment."""

        target_layer = self._resolve_layer(backend)
        tokenizer = backend._tokenizer
        device = backend.device
        letter_ids = self._answer_letter_token_ids(tokenizer)

        samples = (
            dataset.sample(self.num_samples, seed=self.seed) if self.num_samples else list(dataset)
        )

        print(f"Model         : {backend.model_name}")
        print(f"Dataset       : {dataset.name}")
        print(f"Target layer  : L{target_layer}")
        print(f"Samples       : {len(samples)}")
        print(f"MCQ letter ids: {len(letter_ids)} token ids")

        per_sample: List[Dict] = []
        norms: List[float] = []
        entropies: List[float] = []
        labels: List[bool] = []

        batch_size = max(1, self.batch_size or 1)

        def build_prompt_text(sample):
            return (
                prompt_strategy.build_prompt(
                    {
                        "text": sample.text,
                        "question": sample.text,
                        "metadata": sample.metadata or {},
                    }
                )
                + self.answer_cue
            )

        def process_one(sample, logits, hidden) -> None:
            norm = self._compute_norm(hidden)
            entropy = self._compute_logit_entropy(logits, letter_ids)
            answer_tok_id = self._answer_token_id(tokenizer, sample.label)
            if answer_tok_id is not None and letter_ids:
                best_letter_tok = max(letter_ids, key=lambda t: logits[t].item())
                is_correct = best_letter_tok == answer_tok_id
            else:
                is_correct = False
            norms.append(norm)
            entropies.append(entropy)
            labels.append(is_correct)
            per_sample.append(
                {
                    "sample_idx": sample.idx,
                    "is_correct": is_correct,
                    "l2_norm": round(norm, 4),
                    "logit_entropy": round(entropy, 6) if not math.isnan(entropy) else None,
                }
            )

        chunks = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]
        for chunk in tqdm(chunks, desc="Residual norm"):
            prompts = [build_prompt_text(s) for s in chunk]
            B = len(prompts)

            try:
                if B == 1:
                    tokens = self._tokenize(tokenizer, prompts[0], device)
                    logits_b, hidden_b = self._forward(backend, tokens, target_layer)
                    logits_b = logits_b.unsqueeze(0)
                    hidden_b = hidden_b.unsqueeze(0)
                else:
                    tokens = self._tokenize_batch(tokenizer, prompts, device)
                    logits_b, hidden_b = self._forward_batch(backend, tokens, target_layer)
            except Exception as exc:
                tqdm.write(f"  [skip] batch starting at {chunk[0].idx}: {exc}")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                # Fall back to per-sample so one bad sample cannot drop the batch
                for sample in chunk:
                    try:
                        tokens = self._tokenize(tokenizer, build_prompt_text(sample), device)
                        lg, hd = self._forward(backend, tokens, target_layer)
                        process_one(sample, lg, hd)
                    except Exception as sexc:
                        tqdm.write(f"  [skip] sample {sample.idx}: {sexc}")
                continue

            for idx, sample in enumerate(chunk):
                process_one(sample, logits_b[idx], hidden_b[idx])

        # ── Aggregate ──────────────────────────────────────────────────
        n = len(labels)
        accuracy = sum(labels) / n if n else 0.0

        lbl_arr = np.array(labels, dtype=int)

        # AUROC — handle edge case where all labels are the same.
        def _auroc(scores: List[float], lbl: np.ndarray) -> Optional[float]:
            if lbl.sum() == 0 or lbl.sum() == len(lbl) or len(scores) < 2:
                return None
            return float(roc_auc_score(lbl, scores))

        auroc_norm = _auroc(norms, lbl_arr)
        # Negate entropy: higher entropy → less likely correct.
        neg_ent = [-e for e in entropies if not math.isnan(e)]
        lbl_ent = lbl_arr[[not math.isnan(e) for e in entropies]]
        auroc_entropy = _auroc(neg_ent, lbl_ent)

        # Threshold search on norm.
        tau, balanced_acc = self._find_threshold(norms, labels)

        # Mean metrics by correctness.
        correct_norms = [n for n, c in zip(norms, labels) if c]
        incorrect_norms = [n for n, c in zip(norms, labels) if not c]
        correct_ent = [e for e, c in zip(entropies, labels) if c and not math.isnan(e)]
        incorrect_ent = [e for e, c in zip(entropies, labels) if not c and not math.isnan(e)]

        mean_norm_corr = sum(correct_norms) / len(correct_norms) if correct_norms else 0.0
        mean_norm_incorr = sum(incorrect_norms) / len(incorrect_norms) if incorrect_norms else 0.0
        mean_ent_corr = sum(correct_ent) / len(correct_ent) if correct_ent else 0.0
        mean_ent_incorr = sum(incorrect_ent) / len(incorrect_ent) if incorrect_ent else 0.0

        self._print_summary(
            dataset_name=dataset.name,
            target_layer=target_layer,
            n=n,
            accuracy=accuracy,
            auroc_norm=auroc_norm or 0.0,
            auroc_entropy=auroc_entropy or 0.0,
            tau=tau,
            balanced_acc=balanced_acc,
            mean_norm_correct=mean_norm_corr,
            mean_norm_incorrect=mean_norm_incorr,
            mean_entropy_correct=mean_ent_corr,
            mean_entropy_incorrect=mean_ent_incorr,
        )

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=(
                prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom"
            ),
            metrics={
                "dataset": dataset.name,
                "target_layer": target_layer,
                "num_samples": n,
                "accuracy": round(accuracy, 4),
                "auroc_l2_norm": round(auroc_norm, 4) if auroc_norm is not None else None,
                "auroc_logit_entropy": round(auroc_entropy, 4)
                if auroc_entropy is not None
                else None,
                "norm_threshold_tau": round(tau, 4),
                "balanced_acc_at_tau": round(balanced_acc, 4),
                "mean_norm_correct": round(mean_norm_corr, 4),
                "mean_norm_incorrect": round(mean_norm_incorr, 4),
                "mean_entropy_correct": round(mean_ent_corr, 4),
                "mean_entropy_incorrect": round(mean_ent_incorr, 4),
            },
            raw_outputs={"per_sample": per_sample},
            metadata={
                "target_layer": target_layer,
                "num_samples": n,
                "seed": self.seed,
                "answer_cue": self.answer_cue,
                "se_literature_baseline_auroc": "0.65-0.75 (Kuhn et al. 2023)",
            },
        )
