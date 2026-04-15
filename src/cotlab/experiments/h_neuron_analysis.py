"""H-Neuron Analysis Experiment.

Identifies FFN neurons that predict hallucination/incorrect answers in MedGemma
using the CETT (Contribution to rEsidual sTream norm of Token t) metric from
arXiv:2512.01797 (Gao et al., 2025).

Pipeline
--------
Phase 1 — Feature Extraction:
  For each MCQ sample, run a single forward pass and hook every FFN down-projection
  layer to capture z_t (intermediate SwiGLU activations). Compute per-neuron CETT
  at the final prompt token (answer cue position). Label each sample correct/incorrect
  via ground-truth comparison.

  With contrastive_labeling=True (3-vs-1 strategy):
    - CETT is captured at the GENERATED answer token (A/B/C/D), not the last prompt token.
    - Each QA sample contributes TWO training rows:
        answer-token row:  y=1 if hallucinatory,  y=0 if faithful
        other-token row:   y=0 always  (negative control)
    - H-Neurons have POSITIVE weight → active during hallucination generation.
      Causal direction is inverted vs default: suppression RAISES accuracy,
      amplification LOWERS it.

Phase 2 — H-Neuron Discovery:
  Train an L1-regularised logistic regression on the (n_rows × n_features) CETT
  matrix. Neurons with positive weights are H-Neurons.

  A random-neuron baseline is always computed: same N randomly selected neurons,
  same probe, same hyperparameters. This replicates Table 1 of the paper.

Phase 3 — Causal Validation:
  Re-run inference on a held-out split with H-Neuron activations scaled by α ∈ [0, 2].
  Default labeling:       α = 0 drops accuracy, α > 1 raises accuracy.
  Contrastive labeling:   α = 0 raises accuracy, α > 1 drops accuracy.

CETT Formula
------------
  CETT(j, t) = |z_{j,t}| · ‖W_down[:, j]‖₂ / ‖h_t‖₂

  where z_t = SwiGLU output (input to W_down), h_t = W_down · z_t (FFN output).
  Column norms ‖W_down[:, j]‖₂ are precomputed once per layer.

Reference: arXiv:2512.01797 — "H-Neurons: On the Existence, Impact, and Origin
of Hallucination-Associated Neurons in LLMs"
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger

_MCQ_LETTERS = list("ABCDEFGHIJ")


@Registry.register_experiment("h_neuron_analysis")
class HNeuronAnalysisExperiment(BaseExperiment):
    """
    H-Neuron Analysis: find FFN neurons predicting hallucination in MedGemma.

    Implements the CETT metric (arXiv:2512.01797) adapted for MCQ tasks where
    ground-truth labels are available directly (no consistency-filter needed).
    """

    def __init__(
        self,
        name: str = "h_neuron_analysis",
        description: str = "CETT-based H-Neuron discovery and causal validation",
        num_samples: int = 500,
        validation_split: float = 0.2,  # Fraction held out for causal validation
        l1_C: float = 0.01,  # Inverse L1 strength — lower = sparser
        alpha_values: List[float] = None,  # Causal scaling factors
        layer_stride: int = 1,  # Sample every Nth layer (1 = all)
        seed: int = 42,
        max_input_tokens: int = 1024,
        answer_cue: str = "\n\nAnswer:",
        mcq_letters: Optional[List[str]] = None,
        contrastive_labeling: bool = False,  # 3-vs-1: CETT at generated answer token, hallucination=1
        **kwargs,
    ):
        self._name = name
        self.description = description
        self.num_samples = num_samples
        self.validation_split = validation_split
        self.l1_C = l1_C
        self.alpha_values = alpha_values if alpha_values is not None else [0.0, 0.5, 1.5, 2.0]
        self.layer_stride = layer_stride
        self.seed = seed
        self.max_input_tokens = max_input_tokens
        self.answer_cue = answer_cue
        self.mcq_letters = mcq_letters or _MCQ_LETTERS
        self.contrastive_labeling = contrastive_labeling

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Layer and tokenizer helpers
    # ------------------------------------------------------------------

    def _resolve_layers(self, backend: InferenceBackend) -> List[int]:
        all_layers = backend.hook_manager.available_layers
        return all_layers[:: self.layer_stride]

    def _get_letter_ids(self, backend: InferenceBackend) -> Dict[str, int]:
        """Map MCQ letters → single best token id for logit extraction."""
        tokenizer = backend._tokenizer
        letter_ids = {}
        for letter in self.mcq_letters:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            if ids:
                letter_ids[letter] = ids[0]
        return letter_ids

    def _build_prompt(self, prompt_strategy, sample) -> str:
        text = sample.text
        prompt = prompt_strategy.build_prompt(
            {"text": text, "question": text, "metadata": sample.metadata or {}}
        )
        return prompt + self.answer_cue

    def _tokenize(self, backend: InferenceBackend, prompt: str) -> Dict[str, torch.Tensor]:
        tokenizer = backend._tokenizer
        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        return {k: v.to(backend.device) for k, v in tokens.items()}

    def _predict_letter(self, logits: torch.Tensor, letter_ids: Dict[str, int]) -> str:
        return max(letter_ids.items(), key=lambda kv: logits[kv[1]].item())[0]

    def _ground_truth_letter(self, sample) -> Optional[str]:
        """Extract ground-truth answer letter from sample label."""
        label = sample.label
        if label is None:
            return None
        if isinstance(label, str) and len(label) == 1 and label.upper() in self.mcq_letters:
            return label.upper()
        if isinstance(label, dict):
            for key in ("answer", "label", "correct_answer"):
                val = label.get(key)
                if isinstance(val, str) and val.upper() in self.mcq_letters:
                    return val.upper()
        return None

    # ------------------------------------------------------------------
    # CETT computation
    # ------------------------------------------------------------------

    def _precompute_col_norms(
        self, backend: InferenceBackend, layers: List[int]
    ) -> Dict[int, torch.Tensor]:
        """Precompute ‖W_down[:, j]‖₂ for each layer. Shape per layer: (intermediate_dim,)."""
        col_norms = {}
        for layer_idx in layers:
            down_proj = backend.hook_manager.get_mlp_down_proj_module(layer_idx)
            W = down_proj.weight.detach().float()  # (hidden_dim, intermediate_dim)
            col_norms[layer_idx] = torch.norm(W, dim=0).cpu()  # (intermediate_dim,)
        return col_norms

    def _forward_cett(
        self,
        backend: InferenceBackend,
        tokens: Dict[str, torch.Tensor],
        layers: List[int],
        col_norms: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single forward pass — extract CETT features at the final token for all layers.

        Returns:
            cett_vec: (n_layers * intermediate_dim,) float32 — concatenated CETT values
            logits:   (vocab_size,) float32 — output logits at final token
        """
        z_cache: Dict[int, torch.Tensor] = {}
        h_cache: Dict[int, torch.Tensor] = {}
        handles = []

        for layer_idx in layers:
            down_proj = backend.hook_manager.get_mlp_down_proj_module(layer_idx)

            def make_hook(idx: int):
                def hook(module, input, output):
                    z = input[0]  # (batch, seq_len, intermediate_dim)
                    h = output  # (batch, seq_len, hidden_dim)
                    # Capture last prompt token only
                    z_cache[idx] = z[0, -1, :].detach().float().cpu()
                    h_cache[idx] = h[0, -1, :].detach().float().cpu()
                    return output

                return hook

            handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

        try:
            with torch.no_grad():
                out = backend._model(**tokens)
        finally:
            for h in handles:
                h.remove()

        logits = out.logits[0, -1, :].detach().float().cpu()

        # Compute CETT per layer and concatenate
        cett_parts = []
        for layer_idx in layers:
            z_last = z_cache[layer_idx]  # (intermediate_dim,)
            h_last = h_cache[layer_idx]  # (hidden_dim,)
            h_norm = torch.norm(h_last).item() + 1e-8
            cett = (z_last * col_norms[layer_idx]) / h_norm  # signed: preserves direction for probe
            cett_parts.append(cett)

        return torch.cat(cett_parts, dim=0), logits

    def _forward_cett_at_answer_token(
        self,
        backend: InferenceBackend,
        tokens: Dict[str, torch.Tensor],
        predicted_letter: str,
        letter_ids: Dict[str, int],
        layers: List[int],
        col_norms: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Append the predicted answer letter token to the prompt and run a forward pass.
        Captures CETT at the generated answer token position (last token).

        Used by contrastive_labeling=True: hallucination neurons fire
        specifically when the model is generating its (wrong) answer token.

        Returns:
            cett_answer: (n_layers * intermediate_dim,) float32 — CETT at answer token
        """
        letter_token_id = letter_ids[predicted_letter]
        input_ids = tokens["input_ids"]  # (1, seq_len)
        letter_t = torch.tensor([[letter_token_id]], device=input_ids.device)
        extended_ids = torch.cat([input_ids, letter_t], dim=1)

        extended_tokens: Dict[str, torch.Tensor] = {"input_ids": extended_ids}
        if "attention_mask" in tokens:
            m = tokens["attention_mask"]
            extended_tokens["attention_mask"] = torch.cat(
                [m, torch.ones((1, 1), device=m.device, dtype=m.dtype)], dim=1
            )

        z_cache: Dict[int, torch.Tensor] = {}
        h_cache: Dict[int, torch.Tensor] = {}
        handles = []

        for layer_idx in layers:
            down_proj = backend.hook_manager.get_mlp_down_proj_module(layer_idx)

            def make_hook(idx: int):
                def hook(module, input, output):
                    z = input[0]
                    h = output
                    z_cache[idx] = z[0, -1, :].detach().float().cpu()  # answer token
                    h_cache[idx] = h[0, -1, :].detach().float().cpu()
                    return output

                return hook

            handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

        try:
            with torch.no_grad():
                backend._model(**extended_tokens)
        finally:
            for h in handles:
                h.remove()

        cett_parts = []
        for layer_idx in layers:
            h_norm = torch.norm(h_cache[layer_idx]).item() + 1e-8
            cett = (z_cache[layer_idx] * col_norms[layer_idx]) / h_norm
            cett_parts.append(cett)

        return torch.cat(cett_parts, dim=0)

    # ------------------------------------------------------------------
    # Causal validation
    # ------------------------------------------------------------------

    def _forward_with_suppression(
        self,
        backend: InferenceBackend,
        tokens: Dict[str, torch.Tensor],
        h_neurons: List[Tuple[int, int]],  # [(layer_idx, neuron_idx), ...]
        alpha: float,
        layers: List[int],
    ) -> torch.Tensor:
        """
        Forward pass scaling H-Neuron activations by alpha before W_down projection.

        Uses register_forward_pre_hook on down_proj so z_{j,t} is scaled before
        being multiplied through W_down, exactly as in the paper's intervention.
        """
        # Group H-neurons by layer for efficient masking
        neurons_by_layer: Dict[int, List[int]] = {}
        for layer_idx, neuron_idx in h_neurons:
            neurons_by_layer.setdefault(layer_idx, []).append(neuron_idx)

        handles = []
        for layer_idx in layers:
            if layer_idx not in neurons_by_layer:
                continue
            neuron_indices = torch.tensor(neurons_by_layer[layer_idx], dtype=torch.long)
            down_proj = backend.hook_manager.get_mlp_down_proj_module(layer_idx)

            def make_pre_hook(indices: torch.Tensor, a: float):
                def pre_hook(module, input):
                    z = input[0].clone()
                    z[..., indices.to(z.device)] *= a
                    return (z,) + input[1:]

                return pre_hook

            handles.append(
                down_proj.register_forward_pre_hook(make_pre_hook(neuron_indices, alpha))
            )

        try:
            with torch.no_grad():
                out = backend._model(**tokens)
        finally:
            for h in handles:
                h.remove()

        return out.logits[0, -1, :].detach().float().cpu()

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        logger: Optional["ExperimentLogger"] = None,
        **kwargs,
    ) -> ExperimentResult:
        layers = self._resolve_layers(backend)
        letter_ids = self._get_letter_ids(backend)
        samples = dataset.sample(self.num_samples, seed=self.seed)

        print(f"\n{'=' * 60}")
        print(f"H-NEURON ANALYSIS — {dataset.name}")
        print(f"{'=' * 60}")
        print(f"  Layers          : {len(layers)} (stride {self.layer_stride})")
        print(f"  Samples         : {len(samples)}")
        print(f"  L1 C            : {self.l1_C}")
        print(f"  Alpha vals      : {self.alpha_values}")

        # Precompute column norms once — shape per layer: (intermediate_dim,)
        print("\n[1/3] Precomputing W_down column norms...")
        col_norms = self._precompute_col_norms(backend, layers)
        intermediate_dim = next(iter(col_norms.values())).shape[0]
        n_features = len(layers) * intermediate_dim
        print(f"  Intermediate dim : {intermediate_dim}")
        print(f"  Total features   : {n_features:,}")

        # ----------------------------------------------------------------
        # Phase 1 — Extract CETT features + labels
        # ----------------------------------------------------------------
        print("\n[2/3] Extracting CETT features...")
        top_k = min(5000, n_features)

        # Welford online mean/variance accumulators (float64 for numerical stability)
        welford_n = 0
        welford_mean = np.zeros(n_features, dtype=np.float64)
        welford_M2 = np.zeros(n_features, dtype=np.float64)

        # cett_raw: one row per training example (2 rows per QA sample when contrastive_labeling)
        cett_raw: List[np.ndarray] = []
        labels: List[int] = []
        # Maps each cett_raw row back to its QA sample index in valid_samples
        row_to_sample_idx: List[int] = []

        valid_samples = []
        per_sample = []
        skipped = 0

        for sample in tqdm(samples, desc="CETT extraction"):
            gt = self._ground_truth_letter(sample)
            if gt is None:
                skipped += 1
                continue

            prompt = self._build_prompt(prompt_strategy, sample)
            tokens = self._tokenize(backend, prompt)

            try:
                cett_vec, logits = self._forward_cett(backend, tokens, layers, col_norms)
            except (RuntimeError, ValueError) as e:
                logging.warning(f"Error during forward_cett: {e}")
                skipped += 1
                continue

            pred = self._predict_letter(logits, letter_ids)
            is_correct = pred == gt

            sample_pos = len(valid_samples)
            valid_samples.append(sample)
            per_sample.append(
                {
                    "sample_idx": sample.idx,
                    "predicted": pred,
                    "ground_truth": gt,
                    "is_correct": is_correct,
                }
            )

            if self.contrastive_labeling:
                # Contrastive labeling: capture CETT at the generated answer token.
                # H-Neurons = neurons active during hallucination (wrong answer generation).
                try:
                    cett_answer_vec = self._forward_cett_at_answer_token(
                        backend, tokens, pred, letter_ids, layers, col_norms
                    )
                except (RuntimeError, ValueError, KeyError) as e:
                    logging.warning(f"Error during forward_cett_at_answer_token: {e}")
                    valid_samples.pop()
                    per_sample.pop()
                    skipped += 1
                    continue

                cett_answer = np.nan_to_num(
                    cett_answer_vec.numpy().astype(np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                cett_other = np.nan_to_num(
                    cett_vec.numpy().astype(np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                # Answer-token row: y=1 if hallucinatory, y=0 if faithful
                cett_raw.append(cett_answer)
                labels.append(0 if is_correct else 1)
                row_to_sample_idx.append(sample_pos)

                # Other-token row: always y=0 (negative control)
                cett_raw.append(cett_other)
                labels.append(0)
                row_to_sample_idx.append(sample_pos)

                for vec in (cett_answer, cett_other):
                    welford_n += 1
                    delta = vec.astype(np.float64) - welford_mean
                    welford_mean += delta / welford_n
                    welford_M2 += delta * (vec.astype(np.float64) - welford_mean)

            else:
                # Default: binary correct/incorrect label at last prompt token
                vec = np.nan_to_num(
                    cett_vec.numpy().astype(np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                cett_raw.append(vec)
                labels.append(int(is_correct))
                row_to_sample_idx.append(sample_pos)

                welford_n += 1
                delta = vec.astype(np.float64) - welford_mean
                welford_mean += delta / welford_n
                welford_M2 += delta * (vec.astype(np.float64) - welford_mean)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        n_valid = len(valid_samples)  # number of QA samples
        n_rows = len(labels)  # number of training rows (2× when contrastive_labeling)
        accuracy = sum(p["is_correct"] for p in per_sample) / n_valid if n_valid > 0 else 0.0
        print(f"  Valid QA samples : {n_valid}  (skipped {skipped})")
        print(f"  Training rows    : {n_rows}")
        print(f"  Accuracy         : {accuracy:.3f}")

        if n_valid < 20:
            print("  WARNING: too few valid samples for reliable probing.")

        # ----------------------------------------------------------------
        # Phase 2 — L1 Logistic Regression → H-Neurons
        # ----------------------------------------------------------------
        print("\n[3/3] Training L1 probe...")

        # Variance pre-selection using online-computed variance (no full matrix needed)
        feature_var = welford_M2 / max(welford_n - 1, 1)
        top_k_idx = np.argsort(feature_var)[-top_k:]
        print(f"  Pre-selected top-{top_k} features by CETT variance (from {n_features:,})")

        # Build compact matrix — only top-K columns
        X = np.stack([v[top_k_idx] for v in cett_raw], axis=0)  # (n_rows, top_k)
        del cett_raw  # free full-size vectors immediately
        y = np.array(labels)  # (n_rows,)

        # StandardScaler: zero-mean unit-variance
        col_mean = X.mean(axis=0)
        col_std = X.std(axis=0)
        col_std[col_std == 0] = 1.0
        X = (X - col_mean) / col_std

        # ---- Train/val split at QA-sample level to avoid leakage ----
        # Split sample indices, then expand to row indices.
        sample_correct = np.array([int(p["is_correct"]) for p in per_sample])
        sample_arr = np.arange(n_valid)
        can_stratify = sample_correct.sum() > 1 and (n_valid - sample_correct.sum()) > 1
        train_s, val_s = train_test_split(
            sample_arr,
            test_size=self.validation_split,
            random_state=self.seed,
            stratify=sample_correct if can_stratify else None,
        )
        train_set = set(train_s.tolist())
        val_set = set(val_s.tolist())
        train_row_idx = np.array([i for i, si in enumerate(row_to_sample_idx) if si in train_set])
        val_row_idx = np.array([i for i, si in enumerate(row_to_sample_idx) if si in val_set])

        X_train, X_val = X[train_row_idx], X[val_row_idx]
        y_train, y_val = y[train_row_idx], y[val_row_idx]

        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=self.l1_C,
            class_weight="balanced",
            max_iter=1000,
            random_state=self.seed,
        )
        clf.fit(X_train, y_train)

        val_pred = clf.predict(X_val)
        probe_accuracy = balanced_accuracy_score(y_val, val_pred)

        # H-Neurons: positive-weight neurons
        # Default labeling:  positive weight → active during CORRECT answers
        # Contrastive:       positive weight → active during HALLUCINATORY answers
        coef = clf.coef_[0]  # (top_k,)
        selected_flat = np.where(coef > 0)[0]

        # Map back to (layer_idx, neuron_idx_within_layer)
        h_neurons_decoded: List[Tuple[int, int]] = []
        for sel_idx in selected_flat:
            flat_idx = int(top_k_idx[sel_idx])
            layer_pos = flat_idx // intermediate_dim
            neuron_pos = flat_idx % intermediate_dim
            if layer_pos < len(layers):
                h_neurons_decoded.append((layers[layer_pos], int(neuron_pos)))

        # Layer distribution
        layer_counts: Dict[int, int] = {}
        for layer_idx, _ in h_neurons_decoded:
            layer_counts[layer_idx] = layer_counts.get(layer_idx, 0) + 1

        print(f"  Probe balanced acc : {probe_accuracy:.3f}")
        print(f"  H-Neurons found    : {len(h_neurons_decoded)}")
        print(f"  H-Neuron ratio     : {len(h_neurons_decoded) / n_features * 1000:.3f}‰")
        if layer_counts:
            top_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top layers (H-neurons): {top_layers}")

        # AUROC of probe score vs correctness label
        try:
            probe_scores_val = clf.predict_proba(X_val)[:, 1]
            probe_auroc = roc_auc_score(y_val, probe_scores_val)
        except ValueError as e:
            logging.warning(f"roc_auc_score failed (likely only one class present): {e}")
            probe_auroc = None

        # ---- Random neuron baseline ----
        # Same N neurons sampled randomly from the top-k variance pool.
        # Same L1 probe, same hyperparameters. Validates H-Neuron advantage.
        random_baseline_auroc = None
        random_baseline_bal_acc = None
        if len(h_neurons_decoded) > 0:
            rng = np.random.RandomState(self.seed + 1)
            n_rand = len(h_neurons_decoded)
            rand_col_idx = rng.choice(top_k, size=min(n_rand, top_k), replace=False)
            X_tr_rand = X_train[:, rand_col_idx]
            X_v_rand = X_val[:, rand_col_idx]

            clf_rand = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=self.l1_C,
                class_weight="balanced",
                max_iter=1000,
                random_state=self.seed,
            )
            clf_rand.fit(X_tr_rand, y_train)
            rand_pred = clf_rand.predict(X_v_rand)
            random_baseline_bal_acc = balanced_accuracy_score(y_val, rand_pred)
            try:
                rand_scores = clf_rand.predict_proba(X_v_rand)[:, 1]
                random_baseline_auroc = roc_auc_score(y_val, rand_scores)
            except ValueError as e:
                logging.warning(f"Random baseline roc_auc_score failed: {e}")

            print(f"  Random baseline acc  : {random_baseline_bal_acc:.3f}")
            print(f"  Random baseline AUROC: {random_baseline_auroc}")
            if probe_auroc is not None and random_baseline_auroc is not None:
                gap = probe_auroc - random_baseline_auroc
                print(f"  H-Neuron AUROC gap   : {gap:+.3f}  (paper claims >0.10)")

        # ----------------------------------------------------------------
        # Phase 3 — Causal Validation
        # ----------------------------------------------------------------
        causal_results: Dict[str, float] = {}
        if h_neurons_decoded and len(val_s) > 0:
            print(f"\n  Causal validation on {len(val_s)} held-out QA samples...")
            if self.contrastive_labeling:
                print(
                    "  (Contrastive: suppression should RAISE accuracy, amplification should LOWER it)"
                )
            else:
                print(
                    "  (Default labeling: suppression should LOWER accuracy, amplification should RAISE it)"
                )

            for alpha in self.alpha_values:
                correct_alpha = 0
                total_alpha = 0
                for si in val_s:
                    s = valid_samples[si]
                    gt = self._ground_truth_letter(s)
                    if gt is None:
                        continue
                    prompt = self._build_prompt(prompt_strategy, s)
                    tokens = self._tokenize(backend, prompt)
                    try:
                        logits = self._forward_with_suppression(
                            backend, tokens, h_neurons_decoded, alpha, layers
                        )
                        pred = self._predict_letter(logits, letter_ids)
                        correct_alpha += int(pred == gt)
                        total_alpha += 1
                    except (RuntimeError, ValueError) as e:
                        logging.warning(f"Error during causal validation forward pass: {e}")
                        continue
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                acc_alpha = correct_alpha / total_alpha if total_alpha > 0 else 0.0
                causal_results[f"accuracy_alpha_{alpha}"] = acc_alpha
                print(f"    α={alpha:.1f} → accuracy {acc_alpha:.3f}")

        # ----------------------------------------------------------------
        # Results
        # ----------------------------------------------------------------
        print(f"\n{'=' * 60}")
        print(f"H-NEURON SUMMARY — {dataset.name}")
        print(f"{'=' * 60}")
        print(f"  Accuracy (no intervention)  : {accuracy:.3f}")
        print(f"  Probe balanced accuracy     : {probe_accuracy:.3f}")
        print(f"  Probe AUROC                 : {probe_auroc}")
        print(f"  Random baseline AUROC       : {random_baseline_auroc}")
        print(f"  H-Neurons identified        : {len(h_neurons_decoded)}")

        metrics: Dict[str, Any] = {
            "dataset": dataset.name,
            "n_samples": n_valid,
            "n_training_rows": n_rows,
            "n_layers": len(layers),
            "intermediate_dim": intermediate_dim,
            "n_features": n_features,
            "accuracy": accuracy,
            "probe_balanced_accuracy": probe_accuracy,
            "probe_auroc": probe_auroc,
            "random_baseline_balanced_accuracy": random_baseline_bal_acc,
            "random_baseline_auroc": random_baseline_auroc,
            "n_h_neurons": len(h_neurons_decoded),
            "h_neuron_ratio_permille": len(h_neurons_decoded) / n_features * 1000,
            "layer_distribution": layer_counts,
            "top_h_neurons": [
                {"layer": li, "neuron": ni}
                for li, ni in h_neurons_decoded[:50]  # top 50 by layer order
            ],
            **causal_results,
        }

        return ExperimentResult(
            experiment_name=self.name,
            model_name=getattr(backend, "model_name", "unknown"),
            prompt_strategy=getattr(prompt_strategy, "name", "unknown"),
            metrics=metrics,
            raw_outputs={"per_sample": per_sample},
            metadata={
                "description": self.description,
                "layers": layers,
                "layer_stride": self.layer_stride,
                "l1_C": self.l1_C,
                "alpha_values": self.alpha_values,
                "num_samples_requested": self.num_samples,
            },
        )
