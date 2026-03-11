"""Activation Patching experiment — causal intervention via residual-stream replacement.

Patching modes
--------------
``pairs``  (default — requires PatchingPairsDataset)
    clean   = sample.text
    corrupt = sample.metadata["corrupted_prompt"]
    Answers Q: which layers encode the specific diagnosis/fact?

``few_shot_contrast``  (works with ANY dataset)
    clean   = few-shot prompt of the sample  (prompt_strategy with few_shot=True)
    corrupt = zero-shot prompt of the sample (prompt_strategy with few_shot=False)
    Answers Q: which layers causally drive few-shot's benefit on OOD / non-OOD?

``introspect_contrast``  (works with ANY dataset)
    clean   = prompt + introspect instruction
    corrupt = prompt only
    Answers Q: which layers carry the "think deeply" reasoning signal?

``cot_contrast``  (works with ANY dataset)
    clean   = full CoT prompt (cot_trigger active, e.g. "Let's think through this step by step:")
    corrupt = zero-shot prompt (cot_trigger stripped — same structure, no reasoning nudge)
    Answers Q: which layers carry the chain-of-thought reasoning signal vs plain answering?
    Use as the default/baseline contrast alongside few_shot_contrast and introspect_contrast.

``token_group_contrast``  (works with ANY dataset)
    Hooks the attention weight matrix at a single target layer and zeros out
    one token group at a time (delimiter / choice / content).  Measures how
    much each group's removal shifts the answer logit.
    Answers Q: which token positions does <target_mask_layer> attend to causally?
    (Q1: run at layer 3 — the universal attention bottleneck)

Algorithm (logit-recovery metric, one sample, residual patching modes):
  1. Forward clean → cache per-layer residuals (CPU).
  2. Forward corrupt → baseline logit at last token.
  3. For each layer L (strided):
       Re-run corrupt with hook replacing layer L's output with cached clean.
       effect(L) = (logit_patched[clean_tok] - logit_corrupt[clean_tok])
                  / (logit_clean[clean_tok]  - logit_corrupt[clean_tok] + ε)
       1 = full recovery, 0 = no effect, negative = made things worse.

Algorithm (token_group_contrast):
  1. Single forward pass with no masking → logit_base.
  2. For each group G in {delimiter, choice, content}:
       Forward with attention weights at target_mask_layer zeroed for group G.
       importance(G) = |logit_base - logit_masked|
  3. dominant_group = argmax importance.

Memory safety: activations moved to CPU immediately inside each hook.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set

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
    - ``cot_contrast``       Any dataset — CoT prompt (clean) vs zero-shot (corrupt).
    """

    VALID_MODES = (
        "pairs",
        "few_shot_contrast",
        "introspect_contrast",
        "token_group_contrast",
        "cot_contrast",
    )

    # Tokens that are purely structural / formatting — not medical content.
    _DELIMITER_STRINGS: Set[str] = {
        "\n",
        ":",
        "#",
        "##",
        "###",
        "Options",
        "Options:",
        "Answer",
        "Answer:",
        "A.",
        "B.",
        "C.",
        "D.",
        "E.",
        "F.",
        "G.",
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)",
        "(F)",
        "(G)",
        "A)",
        "B)",
        "C)",
        "D)",
        "E)",
        "F)",
        "G)",
    }

    def __init__(
        self,
        name: str = "activation_patching",
        description: str = "Layer-wise causal activation patching (logit recovery)",
        patching_mode: str = "pairs",  # "pairs" | "few_shot_contrast" | "introspect_contrast"
        layer_stride: int = 2,
        num_samples: int = 50,
        max_input_tokens: int = 1024,
        seed: int = 42,
        answer_cue: str = "\n\nAnswer:",
        introspect_instruction: str = (
            "Think deeply about this problem. "
            "Carefully reason through the underlying mechanisms and consider "
            "all relevant factors before committing to your answer."
        ),
        # Legacy fields kept so old YAML configs don't break
        variants: Optional[List[Dict[str, Any]]] = None,
        patching: Optional[Dict[str, Any]] = None,
        # Token-group contrast params
        token_group_contrast_layer: int = 3,
        token_group_mode: str = "all",  # "all" | "delimiter" | "choice" | "content"
        **kwargs,
    ):
        if patching_mode not in self.VALID_MODES:
            raise ValueError(
                f"patching_mode must be one of {self.VALID_MODES}, got {patching_mode!r}"
            )
        self._name = name
        self.description = description
        self.patching_mode = patching_mode
        self.layer_stride = layer_stride
        self.num_samples = num_samples
        self.max_input_tokens = max_input_tokens
        self.seed = seed
        self.answer_cue = answer_cue
        self.introspect_instruction = introspect_instruction
        self.patching = patching or {}
        self.token_group_contrast_layer = int(token_group_contrast_layer)
        self.token_group_mode = token_group_mode

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
    # Token-group tagger and attention masking helpers
    # ------------------------------------------------------------------

    def _tag_tokens(
        self,
        input_ids: torch.Tensor,  # shape (seq_len,)
        tokenizer,
        metadata: dict,
    ) -> Dict[str, List[int]]:
        """Classify every token position into one of 3 groups.

        Groups
        ------
        delimiter : structural tokens (\\n, A., Options:, …)
        choice    : answer-option text (the words after A. / B. / …)
        content   : question stem + clinical entities

        For MedQA samples that carry ``metamap_phrases`` in metadata the
        content group is further split into ``entity`` and ``stem``.

        Returns
        -------
        dict mapping group name -> sorted list of 0-based token positions.
        """
        seq_len = input_ids.shape[0]
        labels = ["content"] * seq_len  # default everything to content

        # ── Pass 1: mark delimiter tokens ─────────────────────────────
        for i in range(seq_len):
            tok_raw = tokenizer.decode([input_ids[i].item()])
            tok_str = tok_raw.strip()
            # Match against stripped form OR raw form (catches \n, spaces, etc.)
            if tok_str in self._DELIMITER_STRINGS or tok_raw in self._DELIMITER_STRINGS:
                labels[i] = "delimiter"

        # ── Pass 2: mark answer-choice span ───────────────────────────
        # Options boundary detection: scan the full decoded text for the
        # first occurrence of a newline followed by an answer label pattern.
        # This handles tokenizers that split 'A)' into ['A', ')'] etc.
        ANSWER_LABEL_RE = re.compile(r"\n(?:Options\s*:?|(?:[A-G][.)\s]|\([A-G]\)))", re.IGNORECASE)
        options_start: Optional[int] = None

        # Build cumulative char offsets per token (same approach as entity split).
        cum_chars_pass2: list = []
        offset_p2 = 0
        for tid in input_ids.tolist():
            decoded = tokenizer.decode([tid])
            cum_chars_pass2.append(offset_p2)
            offset_p2 += len(decoded)

        full_text_p2 = tokenizer.decode(input_ids.tolist())
        match = ANSWER_LABEL_RE.search(full_text_p2)
        if match:
            boundary_char = match.start()  # char index of the '\n'
            # Find first token that starts at or after boundary_char.
            for i, tok_char_start in enumerate(cum_chars_pass2):
                if tok_char_start >= boundary_char:
                    options_start = i
                    break

        if options_start is not None:
            for i in range(options_start, seq_len):
                if labels[i] != "delimiter":
                    labels[i] = "choice"

        # ── Pass 3 (MedQA only): entity vs stem split ──────────────────
        metamap = metadata.get("metamap_phrases") if metadata else None
        if metamap:
            # metamap_phrases is a list of entity strings in their raw form.
            # We decode a window of tokens and look for substring matches.
            full_text = tokenizer.decode(input_ids.tolist())
            entity_spans: List[tuple] = []  # (char_start, char_end)
            for phrase in metamap:
                phrase_str = str(phrase).strip()
                if not phrase_str:
                    continue
                for m in re.finditer(re.escape(phrase_str), full_text, re.IGNORECASE):
                    entity_spans.append((m.start(), m.end()))

            # Map character spans back to token positions (approximate).
            if entity_spans:
                # Build cumulative char lengths per token.
                cum_chars = []
                offset = 0
                for tid in input_ids.tolist():
                    decoded = tokenizer.decode([tid])
                    cum_chars.append((offset, offset + len(decoded)))
                    offset += len(decoded)

                for i, (tok_start, tok_end) in enumerate(cum_chars):
                    if labels[i] != "content":
                        continue
                    for es, ee in entity_spans:
                        if tok_start < ee and tok_end > es:  # overlap
                            labels[i] = "entity"
                            break
                # Remaining "content" tokens become "stem".
                labels = ["stem" if label == "content" else label for label in labels]

        # ── Collect positions per group ────────────────────────────────
        groups: Dict[str, List[int]] = {}
        for i, lbl in enumerate(labels):
            groups.setdefault(lbl, []).append(i)

        # Always expose the 3 primary groups (even if empty).
        for g in ("delimiter", "choice", "content", "stem", "entity"):
            groups.setdefault(g, [])

        return groups

    def _forward_attention_masked(
        self,
        backend: InferenceBackend,
        tokens,
        mask_layer: int,
        zero_positions: List[int],
        answer_tok_id: int,
    ) -> float:
        """Forward pass suppressing ``zero_positions`` at ``mask_layer``'s attention.

        Strategy: register a pre-forward hook on the target layer's ``self_attn``
        module.  Inside the hook we add a large negative value (-1e4) to the
        attention_mask at the key-columns we want to suppress.  The additive causal
        mask is applied inside both ``eager`` and ``sdpa`` kernels before softmax, so
        the suppressed positions get ~zero weight after softmax, with no
        ``output_attentions`` flag required.

        Returns the logit (float32, CPU) for ``answer_tok_id`` at the last token.
        """
        if not zero_positions:
            with torch.no_grad():
                out = backend._model(**tokens)
            return float(out.logits[0, -1, answer_tok_id].detach().cpu().item())

        seq_len = tokens["input_ids"].shape[-1]
        device = tokens["input_ids"].device
        # Build a (1, 1, seq_len, seq_len) additive bias tensor.
        # -1e4 at every key-column in zero_positions, 0.0 elsewhere.
        bias = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32, device=device)
        valid_pos = [p for p in zero_positions if p < seq_len]
        if valid_pos:
            bias[:, :, :, valid_pos] = -1e4
        # Gemma 3 SDPA kernel requires bias dtype == query dtype (e.g. bfloat16).
        model_dtype = backend._model.dtype

        def _pre_hook(module, args, kwargs):
            # Gemma self_attn receives attention_mask as a keyword argument.
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                existing = kwargs["attention_mask"]
                # Add bias then cast to model dtype so SDPA dtype check passes.
                kwargs["attention_mask"] = (
                    existing + bias.to(dtype=existing.dtype, device=existing.device)
                ).to(dtype=model_dtype)
            else:
                kwargs["attention_mask"] = bias.to(dtype=model_dtype, device=device)
            return args, kwargs

        layer_mod = backend.hook_manager.get_layer_module(mask_layer)
        attn_mod = getattr(layer_mod, "self_attn", None)
        if attn_mod is None:
            tqdm.write(
                f"  [warn] token_group_contrast: no self_attn on layer {mask_layer}, skipping mask"
            )
            with torch.no_grad():
                out = backend._model(**tokens)
            return float(out.logits[0, -1, answer_tok_id].detach().cpu().item())

        handle = attn_mod.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                out = backend._model(**tokens)
        finally:
            handle.remove()

        return float(out.logits[0, -1, answer_tok_id].detach().float().cpu().item())

    # ------------------------------------------------------------------
    # Statistical correlation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_correlations(per_sample_results: List[Dict]) -> Dict[str, Any]:
        """Point-biserial correlations between each group's importance score and is_correct.

        Point-biserial r equals Pearson r when one variable is binary, so we
        compute standard Pearson r between the continuous importance score and
        the 0/1 correctness label.  A two-tailed p-value is derived from the
        t-distribution (df = n-2).  scipy is used for the CDF if available;
        otherwise a normal approximation is used as a fallback.

        Returns a dict keyed by group name, each with:
            r                       – point-biserial correlation coefficient
            p_value                 – two-tailed p-value
            n                       – number of samples used
            mean_importance_correct – mean importance when sample is correct
            mean_importance_incorrect – mean importance when sample is incorrect
        """
        valid = [s for s in per_sample_results if s.get("is_correct") is not None]
        if len(valid) < 3:
            return {}

        labels = [int(s["is_correct"]) for s in valid]

        # Collect all group names present across samples.
        groups: set = set()
        for s in valid:
            groups.update(s.get("group_importances", {}).keys())
        if any(s.get("entity_importance") is not None for s in valid):
            groups.add("entity")
        if any(s.get("stem_importance") is not None for s in valid):
            groups.add("stem")

        # Try to import scipy t-distribution CDF once.
        try:
            from scipy.stats import t as _t_dist  # noqa: PLC0415

            _t_cdf = _t_dist.cdf
        except ImportError:
            _t_cdf = None

        def _p_value(t_stat: float, df: int) -> float:
            if _t_cdf is not None:
                return float(2 * (1 - _t_cdf(abs(t_stat), df=df)))
            # Normal approximation fallback.
            return float(2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2)))))

        results: Dict[str, Any] = {}
        for group in sorted(groups):
            if group in ("entity", "stem"):
                scores = [s.get(f"{group}_importance") for s in valid]
            else:
                scores = [s.get("group_importances", {}).get(group) for s in valid]

            paired = [(y, x) for y, x in zip(labels, scores) if x is not None]
            if len(paired) < 3:
                continue

            ys = [p[0] for p in paired]
            xs = [p[1] for p in paired]
            n_g = len(paired)

            mean_x = sum(xs) / n_g
            mean_y = sum(ys) / n_g
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) + 1e-12)
            std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) + 1e-12)
            r = cov / (std_x * std_y)
            r = max(-1.0, min(1.0, r))  # clamp to [-1, 1]

            if abs(r) >= 1.0 - 1e-9:
                p_val = 0.0
            else:
                t_stat = r * math.sqrt((n_g - 2) / (1 - r**2 + 1e-12))
                p_val = _p_value(t_stat, n_g - 2)

            correct_scores = [x for x, y in zip(xs, ys) if y == 1]
            incorrect_scores = [x for x, y in zip(xs, ys) if y == 0]

            results[group] = {
                "r": round(r, 4),
                "p_value": round(p_val, 4),
                "n": n_g,
                "mean_importance_correct": (
                    round(sum(correct_scores) / len(correct_scores), 4) if correct_scores else None
                ),
                "mean_importance_incorrect": (
                    round(sum(incorrect_scores) / len(incorrect_scores), 4)
                    if incorrect_scores
                    else None
                ),
            }

        return results

    # ------------------------------------------------------------------
    # Token-group contrast: sample loop
    # ------------------------------------------------------------------

    def _run_token_group_contrast(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        logger: Optional["ExperimentLogger"] = None,
    ) -> "ExperimentResult":
        """Token-group attention masking loop.

        For each sample:
          1. Build a single prompt (standard, no clean/corrupt split).
          2. Tokenize and tag token positions into groups.
          3. Run baseline forward (no masking) → logit_base + is_correct.
          4. For each group, run _forward_attention_masked → logit_masked.
          5. importance(group) = |logit_base - logit_masked|.
          6. dominant_group = argmax importance.
        """
        tokenizer = backend._tokenizer
        mask_layer = self.token_group_contrast_layer

        print(f"Model        : {backend.model_name}")
        print("Patching mode: token_group_contrast")
        print(f"Mask layer   : L{mask_layer}")
        print(f"max_input_tokens: {self.max_input_tokens}")

        samples = dataset.sample(self.num_samples, seed=self.seed)
        n = len(samples)
        print(f"Samples: {n}  (each requires 4 forward passes)\n")

        # Primary groups to probe (entity/stem only appear for MedQA).
        PRIMARY_GROUPS = ("delimiter", "choice", "content")

        per_sample_results: List[Dict] = []
        # Accumulator: group → list of importance scores across samples.
        group_importances: Dict[str, List[float]] = {g: [] for g in PRIMARY_GROUPS}
        # Per-group: track whether dominant_group == group AND sample correct.
        accuracy_by_dominant: Dict[str, List[bool]] = {g: [] for g in PRIMARY_GROUPS}
        processed = 0

        for sample in tqdm(samples, desc="Token-group contrast"):
            answer_tok_id = self._answer_token_id(tokenizer, sample.label)
            if answer_tok_id is None:
                tqdm.write(f"  [skip] sample {sample.idx}: cannot resolve answer token")
                continue

            prompt_str = self._build_prompt(prompt_strategy, sample.text, sample.metadata or {})
            tokens = self._tokenize(tokenizer, prompt_str, backend.device)
            input_ids = tokens["input_ids"][0]  # (seq_len,)

            # Tag tokens into groups.
            try:
                groups = self._tag_tokens(input_ids, tokenizer, sample.metadata or {})
            except Exception as exc:
                tqdm.write(f"  [skip] sample {sample.idx} (tagging): {exc}")
                continue

            # Baseline forward (no masking) — also derive is_correct in one pass.
            try:
                with torch.no_grad():
                    out_base = backend._model(**tokens)
                last_logits = out_base.logits[0, -1].detach().float().cpu()
                logit_base = float(last_logits[answer_tok_id].item())
                predicted_tok = int(last_logits.argmax().item())
                is_correct = predicted_tok == answer_tok_id
                del out_base, last_logits
            except Exception as exc:
                tqdm.write(f"  [skip] sample {sample.idx} (baseline): {exc}")
                torch.cuda.empty_cache()
                continue

            # Masked forward passes per group.
            sample_importances: Dict[str, float] = {}
            for group in PRIMARY_GROUPS:
                zero_pos = groups.get(group, [])
                try:
                    logit_masked = self._forward_attention_masked(
                        backend, tokens, mask_layer, zero_pos, answer_tok_id
                    )
                    importance = abs(logit_base - logit_masked)
                except Exception as exc:
                    tqdm.write(f"  [skip] sample {sample.idx} group '{group}': {exc}")
                    importance = 0.0
                finally:
                    torch.cuda.empty_cache()

                sample_importances[group] = round(importance, 4)
                group_importances[group].append(importance)

            # Dominant group for this sample.
            dominant = max(sample_importances, key=lambda g: sample_importances[g])
            if is_correct is not None:
                accuracy_by_dominant[dominant].append(is_correct)

            # MedQA entity/stem breakdown (bonus — logged but not aggregated).
            entity_importance: Optional[float] = None
            stem_importance: Optional[float] = None
            if groups.get("entity"):
                try:
                    lm_e = self._forward_attention_masked(
                        backend, tokens, mask_layer, groups["entity"], answer_tok_id
                    )
                    entity_importance = round(abs(logit_base - lm_e), 4)
                except Exception:
                    pass
                finally:
                    torch.cuda.empty_cache()
            if groups.get("stem"):
                try:
                    lm_s = self._forward_attention_masked(
                        backend, tokens, mask_layer, groups["stem"], answer_tok_id
                    )
                    stem_importance = round(abs(logit_base - lm_s), 4)
                except Exception:
                    pass
                finally:
                    torch.cuda.empty_cache()

            per_sample_results.append(
                {
                    "sample_idx": sample.idx,
                    "is_correct": is_correct,
                    "logit_base": round(logit_base, 4),
                    "dominant_group": dominant,
                    "group_importances": sample_importances,
                    "token_counts": {g: len(groups.get(g, [])) for g in PRIMARY_GROUPS},
                    "entity_importance": entity_importance,
                    "stem_importance": stem_importance,
                }
            )
            processed += 1

        # ── Aggregate ─────────────────────────────────────────────────
        mean_importance: Dict[str, float] = {
            g: round(sum(v) / len(v), 4) if v else 0.0 for g, v in group_importances.items()
        }
        dominant_group_overall = max(mean_importance, key=lambda g: mean_importance[g])

        acc_by_dom: Dict[str, Optional[float]] = {}
        for g, hits in accuracy_by_dominant.items():
            acc_by_dom[g] = round(sum(hits) / len(hits), 4) if hits else None

        correlations = self._compute_correlations(per_sample_results)

        # ── Print summary ──────────────────────────────────────────────
        print("\n" + "=" * 70)
        print(f"TOKEN GROUP CONTRAST — L{mask_layer} attention masking")
        print("=" * 70)
        print(f"Processed samples   : {processed} / {n}")
        print(f"Dominant group (avg): {dominant_group_overall}")
        print()
        print(f"{'Group':<12}  {'Mean Importance':>16}  {'Acc when dominant':>18}")
        print("-" * 52)
        for g in PRIMARY_GROUPS:
            acc_str = f"{acc_by_dom[g]:.4f}" if acc_by_dom[g] is not None else "  n/a "
            print(f"{g:<12}  {mean_importance[g]:>16.4f}  {acc_str:>18}")

        if correlations:
            print()
            print("Point-biserial correlations (importance score → is_correct):")
            print(
                f"  {'Group':<12}  {'r':>7}  {'p':>8}  {'n':>5}  {'mean(corr)':>11}  {'mean(incorr)':>12}"
            )
            print("  " + "-" * 60)
            for g, c in correlations.items():
                sig = "*" if c["p_value"] < 0.05 else (" " if c["p_value"] < 0.10 else " ")
                mc = (
                    f"{c['mean_importance_correct']:.4f}"
                    if c["mean_importance_correct"] is not None
                    else "  n/a "
                )
                mi = (
                    f"{c['mean_importance_incorrect']:.4f}"
                    if c["mean_importance_incorrect"] is not None
                    else "  n/a "
                )
                print(
                    f"  {g:<12}  {c['r']:>+7.4f}  {c['p_value']:>8.4f}{sig}  {c['n']:>5}  {mc:>11}  {mi:>12}"
                )
            print("  (* p<0.05)")

        print("=" * 70)
        print()
        print("Interpretation:")
        print(
            "  Higher importance = removing this group from L",
            mask_layer,
            "attention shifts the answer more.",
        )
        print("  The dominant group is what the layer causally relies on most.")
        if correlations:
            print("  Positive r = higher importance of this group → more likely correct.")
            print("  Negative r = higher importance of this group → more likely incorrect.")

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=(
                prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom"
            ),
            metrics={
                "num_samples": processed,
                "mask_layer": mask_layer,
                "mean_importance_per_group": mean_importance,
                "dominant_group": dominant_group_overall,
                "accuracy_when_dominant": acc_by_dom,
                "point_biserial_correlations": correlations,
            },
            raw_outputs={"per_sample": per_sample_results},
            metadata={
                "mask_layer": mask_layer,
                "token_group_mode": self.token_group_mode,
                "num_samples": processed,
                "seed": self.seed,
                "answer_cue": self.answer_cue,
            },
        )

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

    def _build_prompt_introspect(
        self, prompt_strategy: Any, text: str, metadata: dict, introspect: bool
    ) -> str:
        """Build prompt with introspect instruction appended (clean) or omitted (corrupt).

        clean   (introspect=True)  → standard prompt + introspect_instruction prepended
        corrupt (introspect=False) → standard prompt only (no instruction)

        few_shot is kept at whatever the prompt strategy has configured so that
        the only variable between clean and corrupt is the introspect wording.
        """
        base = self._build_prompt(prompt_strategy, text, metadata)
        if introspect:
            # Prepend the instruction before the main prompt body so it sets
            # the reasoning intent from the first token.
            return self.introspect_instruction + "\n\n" + base
        return base

    def _build_prompt_cot(self, prompt_strategy: Any, text: str, metadata: dict, cot: bool) -> str:
        """Build prompt with CoT trigger active (clean) or stripped (corrupt).

        clean   (cot=True)  → full CoT prompt with cot_trigger intact
        corrupt (cot=False) → same prompt with cot_trigger set to "" (zero-shot)

        Only the cot_trigger attribute is toggled; few_shot and all other
        strategy settings are preserved so CoT is the sole variable.
        """
        orig = getattr(prompt_strategy, "cot_trigger", None)
        try:
            if hasattr(prompt_strategy, "cot_trigger"):
                prompt_strategy.cot_trigger = orig if cot else ""
            return self._build_prompt(prompt_strategy, text, metadata)
        finally:
            if orig is not None:
                prompt_strategy.cot_trigger = orig

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        logger: Optional[ExperimentLogger] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Run activation patching experiment.

        Dispatches to the token_group_contrast branch when
        ``patching_mode == 'token_group_contrast'``, otherwise runs the
        standard layer-sweep residual patching.
        """

        tokenizer = backend._tokenizer

        # ── Dispatch to token_group_contrast mode ─────────────────────
        if self.patching_mode == "token_group_contrast":
            return self._run_token_group_contrast(backend, dataset, prompt_strategy, logger)

        # ── Standard residual patching modes ──────────────────────────
        target_layers = self._resolve_layers(backend)

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
            elif self.patching_mode == "few_shot_contrast":
                # few-shot = clean  (more context → better answer representation)
                # zero-shot = corrupted
                clean_str = self._build_prompt_few_shot(
                    prompt_strategy, sample.text, sample.metadata or {}, few_shot=True
                )
                corr_str = self._build_prompt_few_shot(
                    prompt_strategy, sample.text, sample.metadata or {}, few_shot=False
                )
            elif self.patching_mode == "introspect_contrast":
                # introspect instruction prepended = clean
                # no instruction = corrupted
                clean_str = self._build_prompt_introspect(
                    prompt_strategy, sample.text, sample.metadata or {}, introspect=True
                )
                corr_str = self._build_prompt_introspect(
                    prompt_strategy, sample.text, sample.metadata or {}, introspect=False
                )
            else:  # cot_contrast
                # CoT trigger active = clean
                # CoT trigger stripped (zero-shot) = corrupted
                clean_str = self._build_prompt_cot(
                    prompt_strategy, sample.text, sample.metadata or {}, cot=True
                )
                corr_str = self._build_prompt_cot(
                    prompt_strategy, sample.text, sample.metadata or {}, cot=False
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
