"""Logit Lens Experiment.

Visualize what the model "thinks" at each layer by projecting
intermediate activations through the unembedding matrix.

Supports both single-question mode (legacy) and dataset-loop mode
for aggregated analysis over many samples.
"""

from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

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
    Run over N dataset samples and aggregate:
      - per-layer top-1 correct rate  (how often is the right answer rank-1 at layer L?)
      - per-layer top-k correct rate
      - mean emergence layer           (first layer where correct answer enters top-k)
      - never-emerged rate             (samples where correct answer never appears in top-k)
      - final accuracy                 (model's actual last-token accuracy)
    """

    def __init__(
        self,
        name: str = "logit_lens",
        description: str = "Visualize layer-by-layer token predictions",
        target_layers: Optional[List[int]] = None,
        top_k: int = 10,
        num_samples: Optional[int] = None,
        layer_stride: int = 1,
        max_input_tokens: int = 1024,
        seed: int = 42,
        answer_cue: str = "\n\nAnswer:",
        batch_size: int = 1,
        # Legacy single-question field kept for backward compatibility
        question: str = "Patient presents with chest pain, sweating, and shortness of breath. What is the diagnosis?",
        **kwargs,
    ):
        self._name = name
        self.description = description
        self._target_layers_config = target_layers
        self.target_layers = target_layers  # resolved in run()
        self.top_k = top_k
        self.num_samples = num_samples
        self.layer_stride = layer_stride
        self.max_input_tokens = max_input_tokens
        self.seed = seed
        self.answer_cue = answer_cue  # appended to prompt so last token precedes answer letter
        self.batch_size = max(1, int(batch_size))
        self.question = question  # legacy fallback

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lm_head(self, model):
        if hasattr(model, "lm_head"):
            return model.lm_head
        return model.get_output_embeddings()

    def _resolve_layers(self, backend: InferenceBackend) -> List[int]:
        if self._target_layers_config is not None:
            return list(self._target_layers_config)
        all_layers = list(range(backend.hook_manager.num_layers))
        return all_layers[:: self.layer_stride]

    def _correct_token_ids(self, tokenizer, label) -> set:
        """Return all plausible token ids for a label.

        Handles three cases:
        - MCQ letter labels ("A"–"E"): tries with/without leading space/newline.
        - Boolean labels (True/False): maps to "Yes"/"No" which is what the
          model naturally outputs after an answer cue for binary questions.
        - Free-text labels ("Pneumonia", "LUAD" etc.): matches the first token
          of the label string (what the model would predict right after the cue).
        """
        if label is None:
            return set()

        # Boolean → Yes/No
        if isinstance(label, bool):
            label_str = "Yes" if label else "No"
        else:
            label_str = str(label).strip()

        candidates = set()

        # MCQ single-letter
        if len(label_str) == 1 and label_str.upper() in "ABCDEFG":
            upper = label_str.upper()
            for prefix in (" ", "", "\n", " \n"):
                ids = tokenizer.encode(prefix + upper, add_special_tokens=False)
                if ids:
                    candidates.add(ids[-1])
            return candidates

        # Free-text / Yes/No — match first token with/without leading space
        for prefix in (" ", ""):
            ids = tokenizer.encode(prefix + label_str, add_special_tokens=False)
            if ids:
                candidates.add(ids[0])
        return candidates

    def _run_batch(
        self,
        backend: InferenceBackend,
        prompt_strs: List[str],
        lm_head,
        tokenizer,
    ) -> List[tuple]:
        """
        Forward-pass a batch of prompts simultaneously.

        Left-pads sequences so that position [-1] is always the last real token
        for every sample in the batch, regardless of length differences.

        Returns:
            List of (layer_results, final_token_id) tuples, one per sample.
        """
        prompts_with_cue = [p + self.answer_cue for p in prompt_strs]
        B = len(prompts_with_cue)

        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokens = tokenizer(
            prompts_with_cue,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
            padding=True,
        ).to(backend.device)
        tokenizer.padding_side = orig_side

        # Compute position_ids that skip padding so positional embeddings
        # are identical to the non-padded single-sample case.
        # For left-padded input: e.g. [PAD, PAD, t0, t1, t2] → positions [0,0,0,1,2]
        attention_mask = tokens["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        tokens["position_ids"] = position_ids

        # batch_layer_results[layer_idx] = list of B per-sample dicts
        batch_layer_results: Dict[int, List[Dict]] = {}

        def make_hook(layer_idx: int):
            def hook(module, inp, output):
                tensor = output[0] if isinstance(output, tuple) else output
                # tensor: [B, seq_len, hidden]
                with torch.inference_mode():
                    last_hidden = tensor[:, -1, :]  # [B, hidden]
                    logits = lm_head(last_hidden)  # [B, vocab]
                    probs = torch.softmax(logits, dim=-1)
                    top_probs, top_ids = torch.topk(probs, self.top_k)  # [B, top_k]
                results_for_layer = []
                for b in range(B):
                    ids_list = top_ids[b].cpu().tolist()
                    results_for_layer.append(
                        {
                            "layer": layer_idx,
                            "top_ids": ids_list,
                            "top_probs": top_probs[b].cpu().tolist(),
                            "top_tokens": [tokenizer.decode([tid]) for tid in ids_list],
                        }
                    )
                batch_layer_results[layer_idx] = results_for_layer
                return output

            return hook

        handles = []
        for layer_idx in self.target_layers:
            if layer_idx < backend.hook_manager.num_layers:
                mod = backend.hook_manager.get_residual_module(layer_idx)
                handles.append(mod.register_forward_hook(make_hook(layer_idx)))

        try:
            with torch.inference_mode():
                final_logits = backend._model(**tokens).logits
        finally:
            for h in handles:
                h.remove()

        # final_logits: [B, seq_len, vocab] — last token is [-1]
        final_token_ids = torch.argmax(final_logits[:, -1, :], dim=-1).cpu().tolist()

        # Reorganise from layer→batch to batch→layer order
        per_sample = []
        for b in range(B):
            layer_results = [
                batch_layer_results[layer_idx][b]
                for layer_idx in sorted(batch_layer_results.keys())
            ]
            per_sample.append((layer_results, int(final_token_ids[b])))
        return per_sample

    def _run_single(
        self,
        backend: InferenceBackend,
        prompt_str: str,
        lm_head,
        tokenizer,
    ) -> tuple:
        """
        Forward-pass one prompt through the model with residual-stream hooks.

        Projection through lm_head is done INSIDE each hook so we only store
        tiny top-k results (ints + floats), never accumulating full hidden-state
        tensors on GPU simultaneously — prevents GPU page faults on long prompts.

        Returns:
            layer_results  – list of {layer, top_ids, top_probs, top_tokens}
            final_token_id – argmax of final logits at the last position
        """
        # Append answer cue so the final token position is right before the
        # model would predict the answer letter (e.g. A/B/C/D).
        prompt_with_cue = prompt_str + self.answer_cue
        tokens = tokenizer(
            prompt_with_cue,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(backend.device)

        # Store only lightweight top-k results, never the full hidden states
        layer_results_map: Dict[int, Dict] = {}

        def make_hook(layer_idx: int):
            def hook(module, inp, output):
                tensor = output[0] if isinstance(output, tuple) else output
                last_hidden = tensor[0, -1] if tensor.dim() == 3 else tensor[0]
                # Project and move to CPU immediately — release GPU memory
                with torch.inference_mode():
                    logits = lm_head(last_hidden.unsqueeze(0))
                    probs = torch.softmax(logits[0], dim=-1)
                    top_probs, top_ids = torch.topk(probs, self.top_k)
                ids_list = top_ids.cpu().tolist()
                layer_results_map[layer_idx] = {
                    "layer": layer_idx,
                    "top_ids": ids_list,
                    "top_probs": top_probs.cpu().tolist(),
                    "top_tokens": [tokenizer.decode([tid]) for tid in ids_list],
                }
                return output  # pass through unchanged

            return hook

        handles = []
        for layer_idx in self.target_layers:
            if layer_idx < backend.hook_manager.num_layers:
                mod = backend.hook_manager.get_residual_module(layer_idx)
                handles.append(mod.register_forward_hook(make_hook(layer_idx)))

        with torch.inference_mode():
            final_logits = backend._model(**tokens).logits

        for h in handles:
            h.remove()

        final_token_id = int(torch.argmax(final_logits[0, -1]).item())
        layer_results = [
            layer_results_map[layer_idx] for layer_idx in sorted(layer_results_map.keys())
        ]
        return layer_results, final_token_id

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        logger: Optional[ExperimentLogger] = None,
    ) -> ExperimentResult:
        """Run logit lens over dataset samples and aggregate per-layer metrics."""

        self.target_layers = self._resolve_layers(backend)
        lm_head = self._get_lm_head(backend._model)
        tokenizer = backend._tokenizer

        print(f"Model: {backend.model_name}")
        print(f"Layers ({len(self.target_layers)}): {self.target_layers}")
        print(f"Top-k: {self.top_k}")
        print(f"Batch size: {self.batch_size}")

        # --- Sample selection -------------------------------------------
        if self.num_samples is not None:
            samples = dataset.sample(self.num_samples, seed=self.seed)
        else:
            samples = list(dataset)
        n = len(samples)
        print(f"Samples: {n}\n")

        # --- Per-layer accumulators -------------------------------------
        layer_top1: Dict[int, List[bool]] = {layer_idx: [] for layer_idx in self.target_layers}
        layer_topk: Dict[int, List[bool]] = {layer_idx: [] for layer_idx in self.target_layers}
        emergence_layers: List[Optional[int]] = []
        never_emerged = 0
        final_correct = 0

        # Chunk samples into batches
        batches = [
            samples[i : i + self.batch_size] for i in range(0, len(samples), self.batch_size)
        ]

        for batch in tqdm(batches, desc="Logit lens"):
            # Build prompts for the whole batch
            prompt_strs = []
            for sample in batch:
                prompt_input = {
                    "text": sample.text,
                    "question": sample.text,
                    "report": sample.text,
                    "metadata": sample.metadata or {},
                }
                prompt_strs.append(prompt_strategy.build_prompt(prompt_input))

            try:
                if self.batch_size == 1:
                    # Use original single-sample path (no padding overhead)
                    layer_results, final_token_id = self._run_single(
                        backend, prompt_strs[0], lm_head, tokenizer
                    )
                    batch_outputs = [(layer_results, final_token_id)]
                else:
                    batch_outputs = self._run_batch(backend, prompt_strs, lm_head, tokenizer)
            except Exception as e:
                tqdm.write(f"  [skip] batch starting at {batch[0].idx}: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                n -= len(batch)
                continue

            for sample, (layer_results, final_token_id) in zip(batch, batch_outputs):
                correct_ids = self._correct_token_ids(tokenizer, sample.label)

                if correct_ids and final_token_id in correct_ids:
                    final_correct += 1

                # Track per-layer hit rates and emergence
                emerged = False
                for lr in layer_results:
                    lid = lr["layer"]
                    top_ids_list = lr["top_ids"]
                    in_top1 = bool(correct_ids) and (top_ids_list[0] in correct_ids)
                    in_topk = bool(correct_ids) and bool(correct_ids & set(top_ids_list))
                    layer_top1[lid].append(in_top1)
                    layer_topk[lid].append(in_topk)
                    if in_topk and not emerged:
                        emergence_layers.append(lid)
                        emerged = True

                if not emerged:
                    emergence_layers.append(None)
                    never_emerged += 1

            torch.cuda.empty_cache()

        # --- Aggregate --------------------------------------------------
        valid_emergence = [e for e in emergence_layers if e is not None]
        mean_emergence = (
            round(sum(valid_emergence) / len(valid_emergence), 2) if valid_emergence else None
        )
        per_layer_top1_rate = {
            layer_idx: round(sum(layer_top1[layer_idx]) / len(layer_top1[layer_idx]), 4)
            if layer_top1[layer_idx]
            else 0.0
            for layer_idx in self.target_layers
        }
        per_layer_topk_rate = {
            layer_idx: round(sum(layer_topk[layer_idx]) / len(layer_topk[layer_idx]), 4)
            if layer_topk[layer_idx]
            else 0.0
            for layer_idx in self.target_layers
        }

        # --- Print summary ----------------------------------------------
        print("\n" + "=" * 62)
        print("LOGIT LENS SUMMARY")
        print("=" * 62)
        print(f"Samples : {n}   |   Final accuracy : {final_correct / n:.1%}")
        if mean_emergence is not None:
            print(
                f"Mean emergence layer : {mean_emergence:.1f}  "
                f"(first layer correct answer enters top-{self.top_k})"
            )
        print(f"Never emerged (top-{self.top_k}) : {never_emerged / n:.1%}")
        print()
        print(f"{'Layer':>6}  {'Top-1 Rate':>10}  {'Top-K Rate':>10}")
        print("-" * 32)
        for layer_idx in self.target_layers:
            print(
                f"{layer_idx:>6}  {per_layer_top1_rate[layer_idx]:>10.1%}  {per_layer_topk_rate[layer_idx]:>10.1%}"
            )
        print("=" * 62)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=(
                prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom"
            ),
            metrics={
                "num_samples": n,
                "final_accuracy": round(final_correct / n, 4),
                "mean_emergence_layer": mean_emergence,
                "never_emerged_rate": round(never_emerged / n, 4),
                "per_layer_top1_rate": per_layer_top1_rate,
                "per_layer_topk_rate": per_layer_topk_rate,
            },
            raw_outputs={"emergence_layers": emergence_layers},
            metadata={
                "target_layers": self.target_layers,
                "top_k": self.top_k,
                "layer_stride": self.layer_stride,
                "num_samples": n,
                "seed": self.seed,
            },
        )
