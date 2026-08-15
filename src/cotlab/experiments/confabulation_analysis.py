"""Confabulation Analysis Experiment.

Tests whether H-Neurons encode confabulation (confident + wrong) vs general
hallucination by comparing activation patterns across confidence × correctness
categories.

Categories
----------
1. High-Confidence CORRECT: Model confident and right
2. High-Confidence WRONG (confabulation): Model confident but wrong
3. Low-Confidence WRONG (uncertainty): Model uncertain and wrong

Hypothesis
----------
If H-Neurons specifically encode confabulation (overconfident errors), they
should show higher activation for high-conf wrong vs low-conf wrong.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset

_MCQ_LETTERS = list("ABCDEFGHIJ")


@Registry.register_experiment("confabulation_analysis")
class ConfabulationAnalysisExperiment(BaseExperiment):
    """Analyze H-Neuron activation across confidence × correctness categories."""

    def __init__(
        self,
        name: str = "confabulation_analysis",
        description: str = "Test if H-Neurons encode confabulation vs uncertainty",
        probe_path: Optional[str] = None,
        ood_dataset_path: Optional[str] = None,
        num_samples: int = 50,
        conf_high: float = 13.0,
        conf_low: float = 10.0,
        seed: int = 42,
        max_input_tokens: int = 1024,
        answer_cue: str = "\n\nAnswer:",
        mcq_letters: Optional[List[str]] = None,
        batch_size: int = 1,  # rows per forward; 1 = sequential (exact)
        **kwargs,
    ):
        self._name = name
        self.description = description
        self.probe_path = probe_path
        self.ood_dataset_path = ood_dataset_path
        self.num_samples = num_samples
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.seed = seed
        self.max_input_tokens = max_input_tokens
        self.answer_cue = answer_cue
        self.mcq_letters = mcq_letters or _MCQ_LETTERS
        self.batch_size = batch_size
        self._letter_ids_cache: Optional[Dict[str, int]] = None

    @property
    def name(self) -> str:
        return self._name

    def _load_probe(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Load probe weights and neuron indices."""
        if not self.probe_path:
            raise ValueError("probe_path required for confabulation analysis")

        with open(self.probe_path) as f:
            probe_data = json.load(f)

        # Handle both old format (weights/neurons) and new format (fit.*)
        if "weights" in probe_data:
            weights = np.array(probe_data["weights"])
            neurons = [(n["layer"], n["index"]) for n in probe_data["neurons"]]
        elif "fit" in probe_data:
            # Extract neurons from fit data
            h_neurons = probe_data["fit"]["h_neurons"]
            if h_neurons and isinstance(h_neurons[0], list):
                neurons = [(layer, idx) for layer, idx in h_neurons]
            else:
                neurons = [(n["layer"], n["index"]) for n in h_neurons]

            # Use uniform weights if not available (all neurons contribute equally)
            weights = np.ones(len(neurons))
        else:
            raise ValueError("Probe file missing weights/neurons data")

        return weights, neurons

    def _build_mcq_prompt(
        self, backend: InferenceBackend, question: str, options: Dict[str, str]
    ) -> str:
        """Build MCQ prompt in chat format."""
        text = f"{question}\n\n"
        for letter in sorted(options.keys()):
            text += f"{letter}. {options[letter]}\n"
        text += self.answer_cue

        messages = [{"role": "user", "content": text}]
        return backend._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _tokenize(self, backend: InferenceBackend, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and move to device."""
        tokens = backend._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        return {k: v.to(backend.device) for k, v in tokens.items()}

    def _tokenize_batch(
        self, backend: InferenceBackend, texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Left-pad a batch with position_ids remap (logit_lens precedent)."""
        tokenizer = backend._tokenizer
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
            )
        finally:
            tokenizer.padding_side = orig_side
            tokenizer.pad_token_id = orig_pad
        tokens = {k: v.to(backend.device) for k, v in tokens.items()}
        attention_mask = tokens["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        tokens["position_ids"] = position_ids
        return tokens

    def _get_prediction_and_confidence(
        self, backend: InferenceBackend, tokens: Dict[str, torch.Tensor]
    ) -> Tuple[str, float, float]:
        """Get model prediction, max logit, and entropy."""
        with torch.inference_mode():
            outputs = backend._model(**tokens)

        logits = outputs.logits[0, -1].float().cpu()

        # Get letter token IDs
        letter_ids = {}
        for letter in self.mcq_letters:
            ids = backend._tokenizer.encode(letter, add_special_tokens=False)
            if ids:
                letter_ids[letter] = ids[0]

        # Extract letter logits
        letter_logits = {letter: logits[tid].item() for letter, tid in letter_ids.items()}
        pred_letter = max(letter_logits.items(), key=lambda x: x[1])[0]
        max_logit = letter_logits[pred_letter]

        # Compute entropy
        logit_vals = torch.tensor(list(letter_logits.values()))
        probs = torch.softmax(logit_vals, dim=0)
        entropy = -float((probs * (probs + 1e-10).log()).sum().item())

        return pred_letter, max_logit, entropy

    def _extract_cett_features(
        self,
        backend: InferenceBackend,
        tokens: Dict[str, torch.Tensor],
        neurons: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Extract CETT features for probe neurons."""
        features = []
        cett_store = {}

        # Group neurons by layer
        layer_neurons = {}
        for layer, idx in neurons:
            layer_neurons.setdefault(layer, []).append(idx)

        # Hook each layer
        handles = []
        for layer, indices in layer_neurons.items():

            def make_hook(layer_id, neuron_indices):
                def hook(module, inp, output):
                    z = inp[0]  # Input to down_proj
                    h = output  # Output from down_proj

                    z_last = z[0, -1].detach().float()
                    h_last = h[0, -1].detach().float()
                    h_norm = h_last.norm(p=2).item()

                    # Get down_proj weights
                    w_down = module.weight.data
                    col_norms = w_down.norm(p=2, dim=0)

                    # Compute CETT for requested neurons
                    for idx in neuron_indices:
                        cett = (z_last[idx].abs() * col_norms[idx] / (h_norm + 1e-8)).item()
                        cett_store[(layer_id, idx)] = cett

                return hook

            down_proj = backend.hook_manager.get_mlp_down_proj_module(layer)
            handle = down_proj.register_forward_hook(make_hook(layer, indices))
            handles.append(handle)

        try:
            with torch.inference_mode():
                backend._model(**tokens)
        finally:
            for h in handles:
                h.remove()

        # Extract in order
        for layer, idx in neurons:
            features.append(cett_store.get((layer, idx), 0.0))

        return np.array(features)

    def _extract_prediction_and_cett(
        self,
        backend: InferenceBackend,
        tokens: Dict[str, torch.Tensor],
        neurons: List[Tuple[int, int]],
    ) -> Tuple[str, float, float, np.ndarray]:
        """Single forward pass computing prediction, confidence AND CETT features.

        The CETT hooks return ``None`` so the model outputs are untouched,
        making this exactly equivalent to running the two separate forwards in
        ``_get_prediction_and_confidence`` and ``_extract_cett_features``.
        """
        cett_store = {}

        layer_neurons = {}
        for layer, idx in neurons:
            layer_neurons.setdefault(layer, []).append(idx)

        handles = []
        for layer, indices in layer_neurons.items():

            def make_hook(layer_id, neuron_indices):
                def hook(module, inp, output):
                    z = inp[0]
                    h = output
                    z_last = z[0, -1].detach().float()
                    h_last = h[0, -1].detach().float()
                    h_norm = h_last.norm(p=2).item()
                    w_down = module.weight.data
                    col_norms = w_down.norm(p=2, dim=0)
                    for idx in neuron_indices:
                        cett = (z_last[idx].abs() * col_norms[idx] / (h_norm + 1e-8)).item()
                        cett_store[(layer_id, idx)] = cett

                return hook

            down_proj = backend.hook_manager.get_mlp_down_proj_module(layer)
            handle = down_proj.register_forward_hook(make_hook(layer, indices))
            handles.append(handle)

        try:
            with torch.inference_mode():
                outputs = backend._model(**tokens)
        finally:
            for h in handles:
                h.remove()

        logits = outputs.logits[0, -1].float().cpu()

        if self._letter_ids_cache is None:
            letter_ids = {}
            for letter in self.mcq_letters:
                ids = backend._tokenizer.encode(letter, add_special_tokens=False)
                if ids:
                    letter_ids[letter] = ids[0]
            self._letter_ids_cache = letter_ids
        letter_ids = self._letter_ids_cache

        letter_logits = {letter: logits[tid].item() for letter, tid in letter_ids.items()}
        pred_letter = max(letter_logits.items(), key=lambda x: x[1])[0]
        max_logit = letter_logits[pred_letter]

        logit_vals = torch.tensor(list(letter_logits.values()))
        probs = torch.softmax(logit_vals, dim=0)
        entropy = -float((probs * (probs + 1e-10).log()).sum().item())

        features = np.array([cett_store.get((layer, idx), 0.0) for layer, idx in neurons])

        return pred_letter, max_logit, entropy, features

    def _extract_prediction_and_cett_batch(
        self,
        backend: InferenceBackend,
        tokens: Dict[str, torch.Tensor],
        neurons: List[Tuple[int, int]],
        n_rows: int,
    ) -> Tuple[List[str], List[float], List[float], np.ndarray]:
        """Batched single forward computing prediction, confidence and CETT.

        Left-padding + position_ids remap keep each row's last-token logits and
        residuals identical to its single-sample run, so each row reproduces the
        sequential ``_extract_prediction_and_cett`` exactly (causal-mask row
        isolation, eval, no dropout).

        Returns per-row (pred, max_logit, entropy) lists and a [n_rows, d_sae]
        feature matrix.
        """
        cett_store = {}

        layer_neurons = {}
        for layer, idx in neurons:
            layer_neurons.setdefault(layer, []).append(idx)

        handles = []
        for layer, indices in layer_neurons.items():

            def make_hook(layer_id, neuron_indices):
                def hook(module, inp, output):
                    z = inp[0]
                    h = output
                    z_last = z[:, -1, :].detach().float()  # [B, d]
                    h_last = h[:, -1, :].detach().float()  # [B, d]
                    h_norm = h_last.norm(p=2, dim=-1)  # [B]
                    w_down = module.weight.data
                    col_norms = w_down.norm(p=2, dim=0)
                    for idx in neuron_indices:
                        vals = z_last[:, idx].abs() * col_norms[idx] / (h_norm + 1e-8)  # [B]
                        cett_store[(layer_id, idx)] = vals

                return hook

            down_proj = backend.hook_manager.get_mlp_down_proj_module(layer)
            handle = down_proj.register_forward_hook(make_hook(layer, indices))
            handles.append(handle)

        try:
            with torch.inference_mode():
                outputs = backend._model(**tokens)
        finally:
            for h in handles:
                h.remove()

        logits = outputs.logits[:, -1, :].float().cpu()  # [B, vocab]

        if self._letter_ids_cache is None:
            letter_ids = {}
            for letter in self.mcq_letters:
                ids = backend._tokenizer.encode(letter, add_special_tokens=False)
                if ids:
                    letter_ids[letter] = ids[0]
            self._letter_ids_cache = letter_ids
        letter_ids = self._letter_ids_cache
        letter_tids = sorted(letter_ids.values())
        # Rebuild the letter->id map in deterministic id order for vectorized use.
        letter_logits = logits[:, letter_tids]  # [B, n_letters]
        probs = torch.softmax(letter_logits, dim=-1)
        entropies = (-(probs * (probs + 1e-10).log())).sum(dim=-1).tolist()
        max_ids = torch.argmax(letter_logits, dim=-1).tolist()
        letter_by_tid = {tid: letter for letter, tid in letter_ids.items()}
        preds = [letter_by_tid[letter_tids[mi]] for mi in max_ids]
        max_logits = (
            logits[:, letter_tids]
            .gather(-1, torch.tensor(max_ids).unsqueeze(-1))
            .squeeze(-1)
            .tolist()
        )

        features = np.zeros((n_rows, len(neurons)))
        for col, (layer, idx) in enumerate(neurons):
            vals = cett_store.get((layer, idx))
            if vals is not None:
                features[:, col] = vals.cpu().numpy()

        return preds, max_logits, entropies, features

    def _compute_h_score(self, features: np.ndarray, weights: np.ndarray) -> float:
        """Compute H-Score = sigmoid(w · x)."""
        logit = np.dot(weights, features)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def _extract_category_data(
        self,
        backend: InferenceBackend,
        samples: List[Any],
        weights: np.ndarray,
        neurons: List[Tuple[int, int]],
        category_name: str,
    ) -> List[Dict[str, Any]]:
        """Extract data for one category.

        Batches the per-sample forwards (left-pad + position_ids remap), so each
        row reproduces the sequential prediction/confidence/CETT exactly.
        """
        results = []
        batch_size = max(1, self.batch_size or 1)

        def build_prompt(sample):
            # Build prompt - handle both dict and Sample object
            if isinstance(sample, dict):
                return self._build_mcq_prompt(
                    backend, sample["question"], sample["options"]
                ), sample["answer"]
            if hasattr(sample, "question") and hasattr(sample, "options"):
                return self._build_mcq_prompt(
                    backend, sample.question, sample.options
                ), sample.answer
            return sample.text + self.answer_cue, sample.label

        def process_row(pred, max_logit, entropy, features, gt_answer) -> None:
            h_score = self._compute_h_score(features, weights)
            results.append(
                {
                    "prediction": pred,
                    "ground_truth": gt_answer,
                    "correct": pred == gt_answer,
                    "max_logit": max_logit,
                    "entropy": entropy,
                    "h_score": h_score,
                }
            )

        chunks = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]
        for chunk in tqdm(chunks, desc=f"Processing {category_name}"):
            built = [build_prompt(s) for s in chunk]
            prompts = [p for p, _ in built]
            gts = [g for _, g in built]
            B = len(prompts)
            try:
                if B == 1:
                    tokens = self._tokenize(backend, prompts[0])
                    pred, max_logit, entropy, features = self._extract_prediction_and_cett(
                        backend, tokens, neurons
                    )
                    preds, max_logits, entropies, feats_b = (
                        [pred],
                        [max_logit],
                        [entropy],
                        features.reshape(1, -1),
                    )
                else:
                    tokens = self._tokenize_batch(backend, prompts)
                    preds, max_logits, entropies, feats_b = self._extract_prediction_and_cett_batch(
                        backend, tokens, neurons, B
                    )
            except (ValueError, KeyError, RuntimeError, IndexError, TypeError) as exc:
                tqdm.write(f"  [skip] batch starting at sample {chunk[0].idx}: {exc}")
                # Fall back to per-sample so one bad sample cannot drop the batch.
                for sample in chunk:
                    try:
                        tokens = self._tokenize(backend, build_prompt(sample)[0])
                        pred, max_logit, entropy, features = self._extract_prediction_and_cett(
                            backend, tokens, neurons
                        )
                        process_row(pred, max_logit, entropy, features, build_prompt(sample)[1])
                    except Exception as sexc:
                        tqdm.write(f"  [skip] sample {chunk[0].idx}: {sexc}")
                continue

            for row, sample in enumerate(chunk):
                process_row(preds[row], max_logits[row], entropies[row], feats_b[row], gts[row])

        return results

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        **kwargs,
    ) -> ExperimentResult:
        """Run confabulation analysis."""

        # Load probe
        weights, neurons = self._load_probe()

        print(f"Model         : {backend.model_name}")
        print(f"Dataset       : {dataset.name}")
        print(f"Probe neurons : {len(neurons)}")
        print(f"Conf high     : {self.conf_high}")
        print(f"Conf low      : {self.conf_low}")

        # Sample dataset
        samples = dataset.sample(self.num_samples * 3, seed=self.seed)

        # Category 1: High-Conf WRONG (OOD confabulation)
        if self.ood_dataset_path:
            with open(self.ood_dataset_path) as f:
                ood_samples = [json.loads(line) for line in f][: self.num_samples]
            high_conf_wrong = self._extract_category_data(
                backend, ood_samples, weights, neurons, "High-Conf WRONG (OOD)"
            )
        else:
            high_conf_wrong = []

        # Category 2 & 3: From main dataset, stratify by confidence
        all_results = self._extract_category_data(backend, samples, weights, neurons, "All samples")

        # Filter by confidence and correctness
        high_conf_correct = [
            r for r in all_results if r["correct"] and r["max_logit"] >= self.conf_high
        ][: self.num_samples]

        low_conf_wrong = [
            r for r in all_results if not r["correct"] and r["max_logit"] <= self.conf_low
        ][: self.num_samples]

        # Compute statistics
        categories = {
            "high_conf_correct": high_conf_correct,
            "high_conf_wrong": high_conf_wrong,
            "low_conf_wrong": low_conf_wrong,
        }

        metrics = {}
        for cat_name, cat_data in categories.items():
            if not cat_data:
                continue

            h_scores = [r["h_score"] for r in cat_data]
            metrics[f"{cat_name}_n"] = len(cat_data)
            metrics[f"{cat_name}_h_score_mean"] = float(np.mean(h_scores))
            metrics[f"{cat_name}_h_score_std"] = float(np.std(h_scores))

        # Statistical comparison
        if high_conf_wrong and low_conf_wrong:
            h_high = [r["h_score"] for r in high_conf_wrong]
            h_low = [r["h_score"] for r in low_conf_wrong]
            t_stat, p_val = stats.ttest_ind(h_high, h_low)
            metrics["ttest_statistic"] = float(t_stat)
            metrics["ttest_pvalue"] = float(p_val)
            metrics["effect_size_cohens_d"] = float(
                (np.mean(h_high) - np.mean(h_low))
                / np.sqrt((np.std(h_high) ** 2 + np.std(h_low) ** 2) / 2)
            )

        # Print summary
        print("\n" + "=" * 66)
        print("CONFABULATION ANALYSIS")
        print("=" * 66)
        for cat_name in ["high_conf_correct", "high_conf_wrong", "low_conf_wrong"]:
            if f"{cat_name}_n" in metrics:
                n = metrics[f"{cat_name}_n"]
                mean = metrics[f"{cat_name}_h_score_mean"]
                std = metrics[f"{cat_name}_h_score_std"]
                print(f"{cat_name:20s}: n={n:3d}  H-Score={mean:.3f}±{std:.3f}")

        if "ttest_pvalue" in metrics:
            print("\nHigh-Conf WRONG vs Low-Conf WRONG:")
            print(f"  t-statistic: {metrics['ttest_statistic']:.3f}")
            print(f"  p-value: {metrics['ttest_pvalue']:.4f}")
            print(f"  Cohen's d: {metrics['effect_size_cohens_d']:.3f}")
        print("=" * 66)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="mcq",
            metrics=metrics,
            raw_outputs=[categories],
            metadata={
                "probe_path": self.probe_path,
                "num_neurons": len(neurons),
                "conf_high": self.conf_high,
                "conf_low": self.conf_low,
            },
        )
