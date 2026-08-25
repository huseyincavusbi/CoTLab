"""Safety Neurons Experiment.

Port of "Towards Understanding Safety Alignment: A Mechanistic Perspective
from Safety Neurons" (Chen et al., NeurIPS 2025, arXiv:2406.14144;
code: THU-KEG/SafetyNeuron) onto the CoTLab transformers backend.

Unlike weight-space criteria (e.g. ``confidence_regulation``), safety neurons
are found by *contrasting two checkpoints* of the same base model:

identify (paper Eq. 3):
    Inference-time activation contrasting. The second (aligned, e.g. DPO)
    checkpoint greedily generates completions; both checkpoints are then
    teacher-forced over identical token ids and the MLP-intermediate
    activations (down_proj input = TransformerLens ``hook_post``) are captured
    at the configured token positions. The change score of neuron ``i`` in
    layer ``l`` is the RMS activation difference over positions::

        S_i^(l) = sqrt(mean_pos (a_first - a_second)^2)

    Neurons ranked descending by S; top-k percent (or top-n) form the
    candidate safety-neuron set.

mediate (paper Eq. 4):
    Dynamic activation patching. Candidate neurons (re-derived via the
    identify contrast on the same guided sequences) have their down_proj
    inputs overwritten each greedy step with the aligned checkpoint's cached
    activations at that absolute position. The paper's cost-model causal
    effect C is applied post-hoc to persisted generations; the in-run proxy
    metric is token agreement of the patched continuation with the guided
    continuation versus an unpatched baseline.

IA3 checkpoint loading:
    Per the paper, alignment uses IA3 applied exclusively to MLP layers.
    Adapter ``adapter_model.safetensors`` vectors are applied as forward
    pre-hooks scaling down_proj inputs (peft IA3 semantics: ``Linear(x * v)``),
    so weights stay untouched, neuron identity is preserved, and one loaded
    base model serves both passes via hook toggling.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry

_DOWN_PROJ_SUFFIXES = ("down_proj", "w2", "fc2", "dense_4h_to_h")


@Registry.register_experiment("safety_neurons")
class SafetyNeuronsExperiment(BaseExperiment):
    """Identify and validate safety neurons via checkpoint activation contrasting."""

    def __init__(
        self,
        name: str = "safety_neurons",
        description: str = (
            "Safety-neuron identification via SFT-vs-DPO activation contrasting "
            "(Chen et al., NeurIPS 2025)"
        ),
        mode: str = "identify",
        first_peft_path: Optional[str] = None,
        second_peft_path: Optional[str] = None,
        token_type: str = "completion",
        num_samples: int = 200,
        batch_size: int = 8,
        max_new_tokens: int = 128,
        selection: str = "top_percent",
        top_percent: float = 0.05,
        top_n: int = 500,
        random_baseline_count: int = 500,
        mediate_num_samples: int = 50,
        seed: int = 42,
        **kwargs,
    ):
        valid_modes = ("identify", "mediate")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")
        if token_type not in ("completion", "prompt_last", "prompt"):
            raise ValueError(
                f"token_type must be completion|prompt_last|prompt, got '{token_type}'"
            )
        if selection not in ("top_n", "top_percent"):
            raise ValueError(f"selection must be 'top_n' or 'top_percent', got '{selection}'")
        if first_peft_path is None and second_peft_path is None:
            raise ValueError("at least one of first_peft_path/second_peft_path is required")

        self._name = name
        self.description = description
        self.mode = mode
        self.first_peft_path = first_peft_path
        self.second_peft_path = second_peft_path
        self.token_type = token_type
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.selection = selection
        self.top_percent = top_percent
        self.top_n = top_n
        self.random_baseline_count = random_baseline_count
        self.mediate_num_samples = mediate_num_samples
        self.seed = seed
        self._ia3_handles: List[Any] = []

    @property
    def name(self) -> str:
        return self._name

    def validate_backend(self, backend: InferenceBackend) -> None:
        if getattr(backend, "hook_manager", None) is None:
            raise ValueError("safety_neurons requires the transformers backend with hook support")

    # ------------------------------------------------------------------
    # IA3 adapter loading / toggling
    # ------------------------------------------------------------------

    @staticmethod
    def _load_ia3_vectors(peft_path: str) -> Dict[str, torch.Tensor]:
        """Load IA3 vectors from a peft dir, keeping FFN down-projection entries only.

        Keys look like ``base_model.model.model.layers.0.mlp.down_proj.ia3``.
        Attention k/v vectors are dropped (the paper applies IA3 to MLP layers
        exclusively); unknown suffixes are skipped with no error.
        """
        import os

        adapter_file = os.path.join(peft_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            raise FileNotFoundError(f"no adapter_model.safetensors under {peft_path}")
        from safetensors.torch import load_file

        raw = load_file(adapter_file)
        vectors: Dict[str, torch.Tensor] = {}
        for key, tensor in raw.items():
            stem = key.rsplit(".ia3", 1)[0]
            if isinstance(stem, str) and stem.endswith(_DOWN_PROJ_SUFFIXES):
                if stem in vectors:
                    raise ValueError(f"both .ia3 and .ia3_l present for {stem}")
                vectors[stem] = tensor.detach().float().cpu()
        return vectors

    @staticmethod
    def _match_module(model: torch.nn.Module, key: str):
        """Resolve an adapter key to a module via exact or unique-suffix match."""
        named = dict(model.named_modules())
        if key in named:
            return named[key]
        matches = [
            m for n, m in named.items() if n.endswith("." + key.split(".")[-1]) and n.endswith(key)
        ]
        if len(matches) == 1:
            return matches[0]
        candidates = [named[n] for n in named if n == key]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _apply_ia3(self, backend: InferenceBackend, peft_path: Optional[str]) -> int:
        """Attach IA3 input-scaling pre-hooks to matched down_proj modules.

        Returns the number of modules scaled. Safe to call when ``peft_path``
        is None (base model pass) — applies nothing.
        """
        if peft_path is None:
            return 0
        vectors = self._load_ia3_vectors(peft_path)
        applied = 0
        for key, vec in vectors.items():
            module = self._match_module(backend.model, key)
            if module is None:
                continue
            device = next(module.parameters(), torch.empty(0)).device
            dtype = next(module.parameters(), torch.empty(0)).dtype
            v = vec.to(device=device, dtype=dtype)

            def make_hook(scale):
                def hook(mod, inp):
                    x = inp[0] * scale.view(1, 1, -1).to(inp[0].device)
                    return (x,) + tuple(inp[1:])

                return hook

            self._ia3_handles.append(module.register_forward_pre_hook(make_hook(v)))
            applied += 1
        return applied

    def _clear_ia3(self) -> None:
        """Remove all IA3 hooks (restores exact base-model forwards)."""
        for handle in self._ia3_handles:
            handle.remove()
        self._ia3_handles = []

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------

    def _collect_prompts(self, dataset: Any) -> List[str]:
        samples = dataset.sample(self.num_samples, seed=self.seed)
        prompts = [s.text for s in samples]
        if not prompts:
            raise ValueError("dataset yielded no prompts")
        return prompts

    def _build_full_ids(self, backend: InferenceBackend, prompts: List[str]) -> List[torch.Tensor]:
        """Token id sequence per prompt that both passes will replay.

        completion   : prompt ids + greedy continuation generated by the SECOND
                       checkpoint (paper: completions from the aligned model).
        prompt_last  : prompt ids only.
        prompt       : prompt ids only.
        """
        tokenizer = backend.tokenizer
        rows = [
            tokenizer(p, return_tensors=None, add_special_tokens=True)["input_ids"] for p in prompts
        ]
        if self.token_type != "completion":
            return [torch.tensor(r, dtype=torch.long) for r in rows]

        needs_second = self.second_peft_path is not None
        if needs_second:
            self._apply_ia3(backend, self.second_peft_path)
        try:
            full_rows = []
            for prompt, row in zip(prompts, rows):
                out = backend.generate(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)
                full_rows.append(row + list(out.tokens))
            return [torch.tensor(r, dtype=torch.long) for r in full_rows]
        finally:
            if needs_second:
                self._clear_ia3()

    def _select_mask(self, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Boolean mask over positions whose activations enter the contrast."""
        T = full_ids.shape[-1]
        mask = torch.zeros(T, dtype=torch.bool)
        if self.token_type == "prompt":
            mask[:] = True
        elif self.token_type == "prompt_last":
            mask[min(prompt_len - 1, T - 1)] = True
        else:  # completion: generated span only
            mask[prompt_len:] = True
        return mask

    # ------------------------------------------------------------------
    # activation capture
    # ------------------------------------------------------------------

    def _capture_activations(
        self, backend: InferenceBackend, rows: List[torch.Tensor], masks: List[torch.Tensor]
    ) -> torch.Tensor:
        """Teacher-forced capture of every layer's down_proj input at masked positions.

        Returns ``(N, L, d_mlp)`` float32 CPU where rows follow the flattened
        (sequence, position) order of ``masks`` — so two calls over the same
        inputs produce position-aligned tensors ready for differencing.
        """
        device = backend.device
        num_layers = backend.hook_manager.num_layers
        per_layer_store: Dict[int, List[torch.Tensor]] = {ly: [] for ly in range(num_layers)}

        def make_hook(layer_idx):
            def hook(_mod, inp):
                per_layer_store[layer_idx].append(inp[0].detach().float().cpu())
                return None

            return hook

        handles = []
        try:
            for layer_idx in range(num_layers):
                mod = backend.hook_manager.get_mlp_down_proj_module(layer_idx)
                handles.append(mod.register_forward_pre_hook(make_hook(layer_idx)))
            for row in rows:
                if row.dim() == 2:  # tolerate pre-batched [1, T] rows
                    row = row[0]
                tokens = row.unsqueeze(0).to(device)
                with torch.inference_mode():
                    backend.model(tokens)
        finally:
            for h in handles:
                h.remove()

        stacked = [
            torch.stack([per_layer_store[ly][i] for ly in range(num_layers)], dim=-2)[0]
            for i in range(len(rows))
        ]  # each (T, L, d_mlp)
        selected = [s[mask] for s, mask in zip(stacked, masks)]
        return torch.cat(selected, dim=0)

    # ------------------------------------------------------------------
    # change scores + ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_change_scores(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """RMS activation difference over positions: ``(L, d_mlp)`` (paper Eq. 3)."""
        if first.shape != second.shape:
            raise ValueError(f"activation shape mismatch: {first.shape} vs {second.shape}")
        return (first - second).square().mean(dim=0).sqrt()

    def _select_neurons(self, scores: torch.Tensor) -> List[Tuple[int, int]]:
        """Top-k (layer, index) pairs by descending change score."""
        flat = scores.reshape(-1)
        if self.selection == "top_percent":
            n = max(1, int(self.top_percent * flat.numel()))
        else:
            n = min(self.top_n, flat.numel())
        idx = torch.topk(flat, n).indices.tolist()
        d_mlp = scores.shape[1]
        return [(int(i // d_mlp), int(i % d_mlp)) for i in idx]

    # ------------------------------------------------------------------
    # mediate: dynamic activation patching (paper Eq. 4)
    # ------------------------------------------------------------------

    def _capture_full(
        self, backend: InferenceBackend, rows: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Per-sequence full-position capture ``[T, L, d_mlp]`` (no masking)."""
        device = backend.device
        num_layers = backend.hook_manager.num_layers
        per_layer_store: Dict[int, List[torch.Tensor]] = {ly: [] for ly in range(num_layers)}

        def make_hook(layer_idx):
            def hook(_mod, inp):
                per_layer_store[layer_idx].append(inp[0].detach().float().cpu())
                return None

            return hook

        handles = []
        try:
            for layer_idx in range(num_layers):
                mod = backend.hook_manager.get_mlp_down_proj_module(layer_idx)
                handles.append(mod.register_forward_pre_hook(make_hook(layer_idx)))
            for row in rows:
                if row.dim() == 2:
                    row = row[0]
                with torch.inference_mode():
                    backend.model(row.unsqueeze(0).to(device))
        finally:
            for h in handles:
                h.remove()

        return [
            torch.stack([per_layer_store[ly][i] for ly in range(num_layers)], dim=-2)[0]
            for i in range(len(rows))
        ]

    @staticmethod
    def _model_logits(backend: InferenceBackend, tokens: torch.Tensor) -> torch.Tensor:
        """Logits from a model call (HF output object or bare tensor)."""
        out = backend.model(tokens)
        return out.logits if hasattr(out, "logits") else out

    @staticmethod
    def _teacher_forced_nll(backend: InferenceBackend, row: torch.Tensor, prompt_len: int) -> float:
        """Mean NLL of ``row[prompt_len:]`` under teacher forcing."""
        if row.dim() == 2:
            row = row[0]
        tokens = row.unsqueeze(0).to(backend.device)
        with torch.inference_mode():
            logits = SafetyNeuronsExperiment._model_logits(backend, tokens).float()
        targets = tokens[0, prompt_len:]
        pred = logits[0, prompt_len - 1 : -1]
        nll = torch.nn.functional.cross_entropy(pred, targets, reduction="mean")
        return float(nll)

    def _greedy_patch_loop(
        self,
        backend: InferenceBackend,
        row: torch.Tensor,
        prompt_len: int,
        guided_acts: torch.Tensor,
        candidates_by_layer: Dict[int, List[int]],
        eos_id: Optional[int],
    ) -> Tuple[List[int], bool]:
        """Greedy generation replacing candidate-neuron activations each step.

        At every generated position ``t`` the down_proj input of candidate
        neurons is overwritten with the guided checkpoint's cached value at
        that absolute position (paper Algorithm 1). Returns (tokens, patched).
        """
        device = backend.device
        if row.dim() == 2:
            row = row[0]
        current = row[:prompt_len].tolist()

        handles = []
        if candidates_by_layer:

            def make_patch_hook(layer_idx):
                idx = candidates_by_layer.get(layer_idx, [])
                cols = torch.tensor(idx, dtype=torch.long)

                def hook(_mod, inp):
                    x = inp[0].clone()
                    t = x.shape[1] - 1  # absolute position being processed
                    if t < guided_acts.shape[0]:
                        vals = guided_acts[t, layer_idx, cols].to(x.dtype)
                        x[0, -1, cols] = vals.to(x.device)
                    return (x,) + tuple(inp[1:])

                return hook

            handles = [
                backend.hook_manager.get_mlp_down_proj_module(ly).register_forward_pre_hook(
                    make_patch_hook(ly)
                )
                for ly in candidates_by_layer
            ]

        try:
            generated: List[int] = []
            for _step in range(self.max_new_tokens):
                tokens = torch.tensor([current], dtype=torch.long, device=device)
                with torch.inference_mode():
                    logits = self._model_logits(backend, tokens)[0, -1]
                nxt = int(torch.argmax(logits).item())
                if eos_id is not None and nxt == eos_id:
                    break
                current.append(nxt)
                generated.append(nxt)
            return generated, bool(candidates_by_layer)
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------------
    # modes
    # ------------------------------------------------------------------

    def _run_identify(self, backend: InferenceBackend, dataset: Any = None) -> ExperimentResult:
        prompts = self._collect_prompts(dataset) if dataset is not None else ["Hello world"]
        rows = self._build_full_ids(backend, prompts)

        masks = [
            self._select_mask(r, len(backend.tokenizer(p)["input_ids"]))
            for r, p in zip(rows, prompts)
        ]
        nonempty = [(r, m) for r, m in zip(rows, masks) if m.any()]
        if not nonempty:
            raise ValueError("select masks empty for all sequences")
        rows = [r for r, _ in nonempty]
        masks = [m for _, m in nonempty]

        n_applied_second = self._apply_ia3(backend, self.second_peft_path)
        try:
            second = self._capture_activations(backend, rows, masks)
        finally:
            self._clear_ia3()

        n_applied_first = self._apply_ia3(backend, self.first_peft_path)
        try:
            first = self._capture_activations(backend, rows, masks)
        finally:
            self._clear_ia3()

        scores = self._compute_change_scores(first, second)
        selected = self._select_neurons(scores)

        print("\n" + "=" * 66)
        print("SAFETY NEURONS -- IDENTIFY (activation contrasting)")
        print("=" * 66)
        print(f"Sequences      : {len(rows)} | token_type={self.token_type}")
        print(f"Positions      : {int(sum(m.sum() for m in masks))}")
        print(f"IA3 modules    : first={n_applied_first} second={n_applied_second}")
        print(f"Score range    : {float(scores.min()):.4f} .. {float(scores.max()):.4f}")
        print(f"Selected       : {len(selected)} neurons ({self.selection})")
        print("=" * 66)

        layer_counts: Dict[int, int] = {}
        for layer, _ in selected:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="n/a",
            metrics={
                "mode": "identify",
                "token_type": self.token_type,
                "n_sequences": len(rows),
                "n_positions": int(sum(m.sum() for m in masks)),
                "ia3_modules_first": n_applied_first,
                "ia3_modules_second": n_applied_second,
                "score_max": float(scores.max()),
                "score_mean": float(scores.mean()),
                "selected_count": len(selected),
                "layer_distribution": layer_counts,
            },
            metadata={
                "description": self.description,
                "selection": self.selection,
                "top_percent": self.top_percent,
                "top_n": self.top_n,
                "first_peft_path": self.first_peft_path,
                "second_peft_path": self.second_peft_path,
                "seed": self.seed,
                "selected_neurons": [{"layer": layer, "index": idx} for layer, idx in selected],
            },
        )

    # ------------------------------------------------------------------
    # mediate mode orchestration
    # ------------------------------------------------------------------

    def _run_mediate(self, backend: InferenceBackend, dataset: Any = None) -> ExperimentResult:
        """Dynamic activation patching over ``mediate_num_samples`` prompts.

        Candidates are re-derived by the identify contrast on the same guided
        sequences (keeps the pipeline self-contained). Behavioral proxy metric:
        greedy-patched continuation agreement with the guided checkpoint's
        continuation, against an unpatched baseline — the paper's cost-model C
        is applied post-hoc to the persisted generations.
        """
        prompts = (
            self._collect_prompts(dataset)[: self.mediate_num_samples]
            if dataset is not None
            else ["Hello world"]
        )
        eos_id = getattr(getattr(backend, "tokenizer", None), "eos_token_id", None)

        # guided continuations + their cached candidate activations
        rows = self._build_full_ids(backend, prompts)
        prompt_lens = [len(backend.tokenizer(p)["input_ids"]) for p in prompts]

        self._apply_ia3(backend, self.second_peft_path)
        try:
            guided_acts_all = self._capture_full(backend, rows)
            second_full_nll = [
                self._teacher_forced_nll(backend, r, pl) for r, pl in zip(rows, prompt_lens)
            ]
        finally:
            self._clear_ia3()

        # candidates via identify contrast restricted to completion positions
        masks = [self._select_mask(r, pl) for r, pl in zip(rows, prompt_lens)]
        self._apply_ia3(backend, self.first_peft_path)
        try:
            first_sel = self._capture_activations(backend, rows, masks)
            second_sel = self._capture_activations(backend, rows, masks)
            base_nll = [
                self._teacher_forced_nll(backend, r, pl) for r, pl in zip(rows, prompt_lens)
            ]
        finally:
            self._clear_ia3()
        scores = self._compute_change_scores(first_sel, second_sel)
        selected = self._select_neurons(scores)

        rng = torch.Generator().manual_seed(self.seed)
        d_mlp = scores.shape[1]
        num_layers = scores.shape[0]
        rand_flat = torch.randint(0, num_layers * d_mlp, (len(selected),), generator=rng)
        random_pairs = [(int(i // d_mlp), int(i % d_mlp)) for i in rand_flat.tolist()]

        def by_layer(pairs):
            out: Dict[int, List[int]] = {}
            for layer, idx in pairs:
                out.setdefault(layer, []).append(idx)
            return out

        cand_map, rand_map = by_layer(selected), by_layer(random_pairs)

        agreements_patched, agreements_base = [], []
        generations: List[Dict[str, Any]] = []
        for i, (row, pl) in enumerate(zip(rows, prompt_lens)):
            guided_tokens = row[pl:].tolist()
            patched_toks, _ = self._greedy_patch_loop(
                backend, row, pl, guided_acts_all[i], cand_map, eos_id
            )
            base_toks, _ = self._greedy_patch_loop(backend, row, pl, guided_acts_all[i], {}, eos_id)

            def agreement(gen):
                if not guided_tokens or not gen:
                    return 0.0
                n = min(len(guided_tokens), len(gen))
                same = sum(a == b for a, b in zip(guided_tokens[:n], gen[:n]))
                return same / max(1, n)

            agreements_patched.append(agreement(patched_toks))
            agreements_base.append(agreement(base_toks))
            generations.append(
                {
                    "prompt_idx": i,
                    "guided": backend.tokenizer.decode(guided_tokens),
                    "patched": backend.tokenizer.decode(patched_toks),
                    "baseline": backend.tokenizer.decode(base_toks),
                }
            )

        print("\n" + "=" * 66)
        print("SAFETY NEURONS -- MEDIATE (dynamic activation patching)")
        print("=" * 66)
        print(f"Prompts        : {len(rows)}")
        print(f"Candidates     : {len(selected)} | random control {len(random_pairs)}")
        print(
            f"Agree w/guided : patched {sum(agreements_patched):.2f} vs "
            f"base {sum(agreements_base):.2f} (token-rate sums)"
        )
        print(f"NLL guided     : second {sum(second_full_nll):.2f} vs first {sum(base_nll):.2f}")
        print("=" * 66)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="n/a",
            metrics={
                "mode": "mediate",
                "n_prompts": len(rows),
                "candidate_count": len(selected),
                "random_control_count": len(random_pairs),
                "mean_agreement_patched": sum(agreements_patched) / max(1, len(rows)),
                "mean_agreement_baseline": sum(agreements_base) / max(1, len(rows)),
                "guided_nll_second_mean": sum(second_full_nll) / max(1, len(rows)),
                "guided_nll_first_mean": sum(base_nll) / max(1, len(rows)),
            },
            raw_outputs=generations,
            metadata={
                "description": self.description,
                "candidates_by_layer": {str(k): v for k, v in cand_map.items()},
                "random_control_by_layer": {str(k): v for k, v in rand_map.items()},
                "seed": self.seed,
                "note": (
                    "cost-model C (BeaverTails RM) applied post-hoc; "
                    "agreement with guided continuation is the in-run proxy"
                ),
            },
        )

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------

    def run(
        self,
        backend: InferenceBackend,
        dataset: Any = None,
        prompt_strategy: Any = None,
        **kwargs,
    ) -> ExperimentResult:
        """Run the safety-neurons experiment in the configured mode."""
        self.validate_backend(backend)
        torch.manual_seed(self.seed)

        if self.mode == "identify":
            return self._run_identify(backend, dataset)
        if self.mode == "mediate":
            return self._run_mediate(backend, dataset)
        raise NotImplementedError(f"mode '{self.mode}' is not implemented")
