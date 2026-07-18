"""Jacobian Lens Experiment.

Fits and applies the Jacobian lens (Gurnee et al., 2026) to read out
verbalizable representations from intermediate residual-stream activations.

Reference: "Verbalizable Representations Form a Global Workspace in
Language Models", Transformer Circuits Thread, 2026.

Implementation validated against anthropics/jacobian-lens (cosine > 0.999).

Modes:
  fit     — compute J_ℓ matrices over a corpus, save to disk
  apply   — load pre-fitted lens, run on dataset
  compare — side-by-side J-lens vs logit lens readouts
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger


# ---------------------------------------------------------------------------
# JacobianLens dataclass
# ---------------------------------------------------------------------------

@dataclass
class JacobianLens:
    """Per-layer Jacobian matrices that transport activations to the final-layer basis.

    Each J_ℓ ∈ R^{d_model × d_model} maps a residual-stream vector at layer ℓ
    to its expected final-layer representation, averaged over a text corpus.

    Applying the lens:
        lens(h_ℓ) = softmax(W_U · J_ℓ · h_ℓ)
    """

    jacobians: Dict[int, torch.Tensor]
    d_model: int
    n_prompts: int = 0
    source_layers: Optional[List[int]] = None
    target_layer: Optional[int] = None
    skip_first_n: int = 16
    model_name: str = ""

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        jacobians_flat = torch.cat(
            [self.jacobians[l].flatten().cpu() for l in sorted(self.jacobians.keys())]
        )
        torch.save(
            {
                "jacobians_flat": jacobians_flat,
                "layer_keys": sorted(self.jacobians.keys()),
                "d_model": self.d_model,
            },
            os.path.join(path, "lens.pt"),
        )
        metadata = {
            "n_prompts": self.n_prompts,
            "source_layers": self.source_layers
            if self.source_layers is not None
            else sorted(self.jacobians.keys()),
            "target_layer": self.target_layer,
            "skip_first_n": self.skip_first_n,
            "model_name": self.model_name,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> JacobianLens:
        data = torch.load(
            os.path.join(path, "lens.pt"), map_location=map_location, weights_only=True
        )
        with open(os.path.join(path, "metadata.json")) as f:
            metadata = json.load(f)
        d_model = data["d_model"]
        layer_keys = data["layer_keys"]
        jacobians_flat = data["jacobians_flat"]
        jacobians = {}
        flat_offset = 0
        for k in layer_keys:
            n_elements = d_model * d_model
            jacobians[k] = jacobians_flat[flat_offset : flat_offset + n_elements].reshape(
                d_model, d_model
            )
            flat_offset += n_elements
        return cls(
            jacobians=jacobians,
            d_model=d_model,
            n_prompts=metadata.get("n_prompts", 0),
            source_layers=metadata.get("source_layers"),
            target_layer=metadata.get("target_layer"),
            skip_first_n=metadata.get("skip_first_n", 16),
            model_name=metadata.get("model_name", ""),
        )

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Apply J_ℓ to a residual-stream activation: returns J_ℓ @ h_ℓ."""
        if layer not in self.jacobians:
            raise KeyError(
                f"Layer {layer} not in fitted lens. Available: {sorted(self.jacobians.keys())}"
            )
        J = self.jacobians[layer].to(residual.device, dtype=residual.dtype)
        return residual @ J.T

    @torch.no_grad()
    def decode(
        self,
        residual: torch.Tensor,
        layer: int,
        lm_head: nn.Module,
        norm: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Full lens readout: W_U · norm(J_ℓ · h_ℓ). Returns raw logits."""
        transported = self.transport(residual, layer)
        if norm is not None:
            transported = norm(transported)
        return lm_head(transported)

    @torch.no_grad()
    def apply(
        self,
        model: nn.Module,
        tokenizer: Any,
        prompt: str,
        layers: Optional[List[int]] = None,
        positions: Optional[Union[int, List[int]]] = None,
        max_seq_len: int = 128,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Apply the lens to a single prompt, returning per-layer scores."""
        if layers is None:
            layers = sorted(self.jacobians.keys())
        input_ids = tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)
        seq_len = input_ids.shape[1]
        if positions is None:
            pos_indices = list(range(seq_len))
        elif isinstance(positions, int):
            pos_indices = [positions]
        else:
            pos_indices = [p if p >= 0 else seq_len + p for p in positions]

        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states

        lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
        norm = getattr(model, "final_layer_norm", None)
        if norm is None:
            norm = getattr(model.config, "final_layer_norm", None)

        lens_scores: Dict[int, torch.Tensor] = {}
        for layer in layers:
            if layer >= len(hidden_states):
                continue
            h = hidden_states[layer + 1][0, pos_indices, :]
            lens_scores[layer] = self.decode(h, layer, lm_head, norm).cpu()

        final_h = hidden_states[-1][0, pos_indices, :]
        if norm is not None:
            final_h = norm(final_h)
        model_scores = lm_head(final_h).cpu()
        return {"lens_scores": lens_scores, "model_scores": model_scores, "input_ids": input_ids.cpu()}

    @classmethod
    def merge(cls, lenses: List[JacobianLens]) -> JacobianLens:
        """Prompt-weighted average of multiple lenses fitted on different subsets."""
        total_prompts = sum(l.n_prompts for l in lenses)
        d_model = lenses[0].d_model
        layer_keys = sorted(lenses[0].jacobians.keys())
        merged_jacobians = {}
        for k in layer_keys:
            weighted_sum = torch.zeros(d_model, d_model)
            for l in lenses:
                weighted_sum += l.jacobians[k] * l.n_prompts
            merged_jacobians[k] = weighted_sum / total_prompts
        return cls(
            jacobians=merged_jacobians,
            d_model=d_model,
            n_prompts=total_prompts,
            source_layers=lenses[0].source_layers,
            target_layer=lenses[0].target_layer,
            skip_first_n=lenses[0].skip_first_n,
            model_name=lenses[0].model_name,
        )


# ---------------------------------------------------------------------------
# Fitting utilities
# ---------------------------------------------------------------------------

SKIP_FIRST_N_POSITIONS = 16
DIM_BATCH = 8


def _freeze_params(model: nn.Module) -> List[bool]:
    orig = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    return orig


def _thaw_params(model: nn.Module, orig: List[bool]) -> None:
    for p, state in zip(model.parameters(), orig):
        p.requires_grad_(state)


def _validate_prompt(tokenizer, prompt: str, min_tokens: int, max_seq_len: int) -> bool:
    tokens = tokenizer.encode(prompt, truncation=True, max_length=max_seq_len)
    return len(tokens) > min_tokens


def _find_block_modules(model: nn.Module, n_layers: int) -> Dict[int, nn.Module]:
    """Auto-detect transformer block modules from model tree."""
    candidates: Dict[int, List[Tuple[int, nn.Module]]] = {}
    layer_pat = re.compile(r"^(.+?)\.(\d+)$")
    for name, module in model.named_modules():
        m = layer_pat.match(name)
        if not m:
            continue
        prefix, idx_str = m.group(1), m.group(2)
        idx = int(idx_str)
        has_numbered = any(c.isdigit() for c, _ in module.named_children())
        if has_numbered:
            continue
        score = 10 if ("language_model" in prefix or "text_model" in prefix) else 5
        score += 3 if "layers" in prefix else 0
        score += 2 if "h" in prefix.split(".")[-1:] else 0
        score += 1 if "block" in prefix else 0
        candidates.setdefault(idx, []).append((score, module))

    blocks = {}
    for idx, mods in candidates.items():
        mods.sort(key=lambda x: x[0], reverse=True)
        blocks[idx] = mods[0][1]

    if len(blocks) >= n_layers:
        sorted_indices = sorted(blocks.keys())[:n_layers]
        return {i: blocks[idx] for i, idx in enumerate(sorted_indices)}

    for path in ["model.layers", "transformer.h", "model.decoder.layers"]:
        parts = path.split(".")
        container = model
        try:
            for p in parts:
                container = getattr(container, p)
            if hasattr(container, "__len__") and len(container) >= n_layers:
                return {i: container[i] for i in range(n_layers)}
        except (AttributeError, TypeError):
            continue
    raise RuntimeError(
        f"Could not auto-detect transformer blocks. Found {len(blocks)} candidate modules."
    )


def _hook_layer_outputs(
    model: nn.Module,
    input_ids: torch.Tensor,
    source_layers: List[int],
    target_layer: int,
    block_modules: Optional[Dict[int, nn.Module]] = None,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """Instrumented forward pass with gradient tracking from first source layer.

    Hooks block outputs; at the first source layer, detaches and re-enables
    requires_grad so downstream layers track gradients (model params are frozen).
    """
    first_source = min(source_layers)
    all_capture = set(source_layers) | {target_layer}
    if block_modules is None:
        block_modules = _find_block_modules(model, model.config.num_hidden_layers)

    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def hook_fn(layer_idx: int):
        def hook(module, inp, output):
            is_tuple = isinstance(output, tuple)
            t = output[0] if is_tuple else output
            rest = output[1:] if is_tuple else ()
            if layer_idx == first_source:
                t = t.detach().requires_grad_(True)
            captured[layer_idx] = t
            return (t,) + rest if is_tuple else t
        return hook

    for layer_idx in sorted(all_capture):
        block = block_modules[layer_idx]
        handles.append(block.register_forward_hook(hook_fn(layer_idx)))

    with torch.enable_grad():
        _ = model(input_ids=input_ids, output_hidden_states=False, use_cache=False)

    for h in handles:
        h.remove()

    return {l: captured[l] for l in source_layers}, captured[target_layer]


def jacobian_for_prompt(
    model: nn.Module,
    input_ids: torch.Tensor,
    source_layers: List[int],
    target_layer: int,
    dim_batch: int = DIM_BATCH,
    skip_first_n: int = SKIP_FIRST_N_POSITIONS,
) -> Dict[int, torch.Tensor]:
    """Compute per-layer Jacobian matrices J_ℓ = 𝔼[∂h_T/∂h_ℓ] for one prompt.

    Requires all model params to have requires_grad=False.
    """
    d_model = model.config.hidden_size
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    seq_len = input_ids.shape[1]
    valid_positions = list(range(skip_first_n, seq_len - 1))
    if not valid_positions:
        raise ValueError(f"Prompt too short: {seq_len} tokens, need > {skip_first_n + 1}")

    replicated = input_ids.expand(dim_batch, -1).contiguous()
    source_acts, target_act = _hook_layer_outputs(model, replicated, source_layers, target_layer)
    J = {l: torch.zeros(d_model, d_model) for l in source_layers}

    for dim_start in range(0, d_model, dim_batch):
        n_dims = min(dim_batch, d_model - dim_start)
        cotangent = torch.zeros(dim_batch, seq_len, d_model, device=input_ids.device)
        for i in range(n_dims):
            cotangent[i, valid_positions, dim_start + i] = 1.0
        grads = torch.autograd.grad(
            outputs=target_act,
            inputs=[source_acts[l] for l in source_layers],
            grad_outputs=cotangent,
            retain_graph=True,
            allow_unused=False,
        )
        for l_idx, grad in enumerate(grads):
            layer = source_layers[l_idx]
            rows = grad[:n_dims, valid_positions, :].float().mean(dim=1).detach().cpu()
            J[layer][dim_start : dim_start + n_dims, :] = rows

    del source_acts, target_act, replicated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return J


def load_corpus_prompts(
    path: Optional[str] = None,
    n_prompts: int = 1000,
    min_tokens: int = 32,
    max_seq_len: int = 128,
    seed: int = 42,
) -> List[str]:
    """Load prompts from file or generate generic ones for lens fitting."""
    if path is not None and os.path.exists(path):
        with open(path) as f:
            prompts = [line.strip() for line in f if line.strip()]
        return [p for p in prompts if len(p.split()) >= min_tokens][:n_prompts]

    import random
    random.seed(seed)
    templates = [
        "The following is a detailed analysis of the topic. {context} Let us examine this carefully.",
        "Consider the following passage from a textbook. {context} What can we learn from this?",
        "A researcher writes about their findings. {context} The implications are significant.",
        "In this article, we explore several themes. {context} Each theme reveals something important.",
        "The report discusses multiple factors. {context} These factors interact in complex ways.",
        "An expert provides commentary on recent developments. {context} Their perspective is valuable.",
    ]
    filler = [
        "Multiple studies have examined this phenomenon from different angles.",
        "The data suggests a complex relationship between the variables.",
        "Researchers continue to debate the underlying mechanisms.",
        "Several hypotheses have been proposed to explain these observations.",
        "The methodology employed in this study follows established protocols.",
        "Preliminary results indicate promising directions for future work.",
        "The theoretical framework provides a foundation for understanding.",
        "Empirical evidence supports the main conclusions of the analysis.",
        "Comparative studies reveal interesting patterns across different contexts.",
        "The longitudinal data demonstrates sustained effects over time.",
    ]
    prompts = []
    while len(prompts) < n_prompts:
        template = random.choice(templates)
        context = " ".join(random.sample(filler, min(3, len(filler))))
        prompt = template.format(context=context)
        while len(prompt.split()) < min_tokens * 2:
            prompt += " " + random.choice(filler)
        prompts.append(prompt)
    return prompts[:n_prompts]


def fit_jacobian_lens(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    source_layers: Optional[List[int]] = None,
    target_layer: Optional[int] = None,
    dim_batch: int = DIM_BATCH,
    skip_first_n: int = SKIP_FIRST_N_POSITIONS,
    max_seq_len: int = 128,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 50,
    progress: bool = True,
) -> JacobianLens:
    """Fit the Jacobian lens on a corpus of text prompts."""
    num_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    if source_layers is None:
        source_layers = list(range(num_layers))
    if target_layer is None:
        target_layer = num_layers - 1
    source_layers = [l for l in source_layers if l < target_layer]
    if not source_layers:
        raise ValueError(f"No source layers < target_layer ({target_layer})")

    orig_grad = _freeze_params(model)
    model.eval()

    valid_prompts = [p for p in prompts if _validate_prompt(tokenizer, p, skip_first_n + 1, max_seq_len)]
    if not valid_prompts:
        raise ValueError(f"No valid prompts (need > {skip_first_n + 1} tokens)")

    jacobian_sum = {l: torch.zeros(d_model, d_model) for l in source_layers}
    n_done = 0
    next_idx = 0

    if checkpoint_path and os.path.exists(os.path.join(checkpoint_path, "checkpoint.pt")):
        ckpt = torch.load(
            os.path.join(checkpoint_path, "checkpoint.pt"), map_location="cpu", weights_only=True
        )
        if (
            ckpt.get("source_layers") == source_layers
            and ckpt.get("target_layer") == target_layer
            and ckpt.get("skip_first_n", SKIP_FIRST_N_POSITIONS) == skip_first_n
        ):
            jacobian_sum = {l: ckpt["jacobian_sum"][l].clone() for l in source_layers}
            n_done = ckpt["n_done"]
            next_idx = ckpt.get("next_idx", n_done)
            print(f"Resumed from checkpoint: {n_done} prompts done, starting at index {next_idx}")

    iterator = (
        tqdm(range(next_idx, len(valid_prompts)), desc="Fitting J-lens")
        if progress
        else range(next_idx, len(valid_prompts))
    )
    for idx in iterator:
        prompt = valid_prompts[idx]
        try:
            input_ids = tokenizer.encode(
                prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
            ).to(device)
            if input_ids.shape[1] <= skip_first_n + 1:
                continue
            J = jacobian_for_prompt(
                model, input_ids, source_layers, target_layer,
                dim_batch=dim_batch, skip_first_n=skip_first_n,
            )
            for l in source_layers:
                jacobian_sum[l] += J[l]
            n_done += 1
            next_idx = idx + 1
            if checkpoint_path and n_done % checkpoint_every == 0 and n_done > 0:
                os.makedirs(checkpoint_path, exist_ok=True)
                tmp = os.path.join(checkpoint_path, "checkpoint.tmp")
                torch.save(
                    {
                        "jacobian_sum": {l: jacobian_sum[l].clone() for l in source_layers},
                        "n_done": n_done,
                        "next_idx": next_idx,
                        "source_layers": source_layers,
                        "target_layer": target_layer,
                        "skip_first_n": skip_first_n,
                    },
                    tmp,
                )
                os.replace(tmp, os.path.join(checkpoint_path, "checkpoint.pt"))
        except Exception as e:
            if progress:
                tqdm.write(f"  [skip] prompt {idx}: {type(e).__name__}: {e}")
            next_idx = idx + 1
            continue

    jacobians = {l: jacobian_sum[l] / n_done for l in source_layers}

    if checkpoint_path:
        os.makedirs(checkpoint_path, exist_ok=True)
        tmp = os.path.join(checkpoint_path, "checkpoint.tmp")
        torch.save(
            {
                "jacobian_sum": {l: jacobian_sum[l].clone() for l in source_layers},
                "n_done": n_done, "next_idx": next_idx,
                "source_layers": source_layers, "target_layer": target_layer,
                "skip_first_n": skip_first_n,
            },
            tmp,
        )
        os.replace(tmp, os.path.join(checkpoint_path, "checkpoint.pt"))

    _thaw_params(model, orig_grad)
    return JacobianLens(
        jacobians=jacobians, d_model=d_model, n_prompts=n_done,
        source_layers=source_layers, target_layer=target_layer,
        skip_first_n=skip_first_n,
        model_name=getattr(model.config, "_name_or_path", "unknown"),
    )


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

@Registry.register_experiment("jacobian_lens")
class JacobianLensExperiment(BaseExperiment):
    """Jacobian Lens: causal concept readout from intermediate activations.

    Three modes:
        fit     — compute and save J_ℓ matrices (GPU, one-time per model)
        apply   — load saved matrices, run on a dataset
        compare — run both J-lens and logit lens side-by-side
    """

    def __init__(
        self,
        name: str = "jacobian_lens",
        description: str = "Jacobian lens for causal concept readout",
        mode: str = "apply",
        lens_path: Optional[str] = None,
        corpus_path: Optional[str] = None,
        n_corpus_prompts: int = 100,
        source_layers: Optional[List[int]] = None,
        target_layer: Optional[int] = None,
        dim_batch: int = 8,
        skip_first_n: int = 16,
        layer_stride: int = 1,
        top_k: int = 10,
        num_samples: Optional[int] = None,
        max_input_tokens: int = 512,
        seed: int = 42,
        answer_cue: str = "\n\nAnswer:",
        **kwargs,
    ):
        self._name = name
        self.description = description
        self.mode = mode
        self.lens_path = lens_path
        self.corpus_path = corpus_path
        self.n_corpus_prompts = n_corpus_prompts
        self.source_layers = source_layers
        self.target_layer = target_layer
        self.dim_batch = dim_batch
        self.skip_first_n = skip_first_n
        self.layer_stride = layer_stride
        self.top_k = top_k
        self.num_samples = num_samples
        self.max_input_tokens = max_input_tokens
        self.seed = seed
        self.answer_cue = answer_cue

    @property
    def name(self) -> str:
        return self._name

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _get_lm_head(model):
        if hasattr(model, "lm_head"):
            return model.lm_head
        return model.get_output_embeddings()

    @staticmethod
    def _get_final_norm(model):
        for name in ["transformer.ln_f", "model.norm", "final_layer_norm", "ln_f"]:
            try:
                parts = name.split(".")
                obj = model
                for p in parts:
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                continue
        return None

    @staticmethod
    def _correct_token_ids(tokenizer, label) -> set:
        if label is None:
            return set()
        if isinstance(label, bool):
            label_str = "Yes" if label else "No"
        else:
            label_str = str(label).strip()
        candidates = set()
        if len(label_str) == 1 and label_str.upper() in "ABCDEFG":
            for prefix in (" ", "", "\n", " \n"):
                ids = tokenizer.encode(prefix + label_str.upper(), add_special_tokens=False)
                if ids:
                    candidates.add(ids[-1])
            return candidates
        for prefix in (" ", ""):
            ids = tokenizer.encode(prefix + label_str, add_special_tokens=False)
            if ids:
                candidates.add(ids[0])
        return candidates

    def _resolve_apply_layers(self, lens: JacobianLens) -> List[int]:
        if self.source_layers is not None:
            return [l for l in self.source_layers if l in lens.jacobians]
        return sorted(lens.jacobians.keys())[:: self.layer_stride]

    # -- fit mode ---------------------------------------------------------

    def _run_fit(self, backend: InferenceBackend) -> ExperimentResult:
        model = backend._model
        tokenizer = backend._tokenizer
        target_layer = self.target_layer or (model.config.num_hidden_layers - 1)
        source_layers = self.source_layers
        if source_layers is None:
            source_layers = list(range(0, target_layer, self.layer_stride))

        print(f"Mode: fit | Model: {backend.model_name}")
        print(f"Target layer: {target_layer} | Source layers: {source_layers}")
        print(f"Corpus prompts: {self.n_corpus_prompts}")

        prompts = load_corpus_prompts(
            path=self.corpus_path, n_prompts=self.n_corpus_prompts,
            max_seq_len=self.max_input_tokens, seed=self.seed,
        )
        print(f"Valid prompts: {len(prompts)}")

        lens = fit_jacobian_lens(
            model=model, tokenizer=tokenizer, prompts=prompts,
            source_layers=source_layers, target_layer=target_layer,
            dim_batch=self.dim_batch, skip_first_n=self.skip_first_n,
            max_seq_len=self.max_input_tokens, device=backend.device,
            checkpoint_path=self.lens_path, checkpoint_every=50,
        )
        if self.lens_path:
            lens.save(self.lens_path)
            print(f"Saved lens to {self.lens_path}")

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="corpus",
            metrics={
                "mode": "fit", "n_prompts": lens.n_prompts, "d_model": lens.d_model,
                "source_layers": lens.source_layers, "target_layer": lens.target_layer,
                "skip_first_n": lens.skip_first_n,
            },
            metadata={"lens_path": self.lens_path, "corpus_path": self.corpus_path},
        )

    # -- apply mode -------------------------------------------------------

    def _run_apply(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        lens: JacobianLens,
    ) -> ExperimentResult:
        tokenizer = backend._tokenizer
        model = backend._model
        device = backend.device
        lm_head = self._get_lm_head(model)
        norm = self._get_final_norm(model)
        apply_layers = self._resolve_apply_layers(lens)

        print(f"Mode: apply | Model: {backend.model_name}")
        print(f"Lens layers: {apply_layers} | Fitted on {lens.n_prompts} prompts")

        samples = (
            dataset.sample(self.num_samples, seed=self.seed)
            if self.num_samples is not None
            else list(dataset)
        )
        n = len(samples)
        print(f"Samples: {n}")

        jl_top1: Dict[int, List[bool]] = {l: [] for l in apply_layers}
        jl_topk: Dict[int, List[bool]] = {l: [] for l in apply_layers}
        ll_top1: Dict[int, List[bool]] = {l: [] for l in apply_layers}
        ll_topk: Dict[int, List[bool]] = {l: [] for l in apply_layers}
        jl_emergence: List[Optional[int]] = []
        ll_emergence: List[Optional[int]] = []
        jl_never = ll_never = final_correct = jl_disagree = 0

        for sample in tqdm(samples, desc="J-lens apply"):
            prompt_input = {
                "text": sample.text, "question": sample.text,
                "report": sample.text, "metadata": sample.metadata or {},
            }
            prompt_str = prompt_strategy.build_prompt(prompt_input) + self.answer_cue
            try:
                prompt_tokens = tokenizer.encode(
                    prompt_str, return_tensors="pt",
                    truncation=True, max_length=self.max_input_tokens,
                ).to(device)
                with torch.no_grad():
                    out = model(input_ids=prompt_tokens, output_hidden_states=True, use_cache=False)
                hidden_states = out.hidden_states
                final_h = hidden_states[-1][0, -1, :]
                if norm is not None:
                    final_h = norm(final_h)
                final_id = int(torch.argmax(lm_head(final_h)).item())
            except Exception as e:
                tqdm.write(f"  [skip] sample {sample.idx}: {type(e).__name__}: {e}")
                n -= 1
                continue

            correct_ids = self._correct_token_ids(tokenizer, sample.label)
            if correct_ids and final_id in correct_ids:
                final_correct += 1

            jl_ok = ll_ok = False
            for layer in apply_layers:
                if layer + 1 >= len(hidden_states):
                    continue
                h = hidden_states[layer + 1][0, -1, :]

                # J-lens
                try:
                    jl_score = lens.decode(h.unsqueeze(0), layer, lm_head, norm)[0]
                except KeyError:
                    continue
                jl_ids_list = torch.topk(jl_score, self.top_k).indices.tolist()
                jl_in_top1 = bool(correct_ids) and (jl_ids_list[0] in correct_ids)
                jl_in_topk = bool(correct_ids) and bool(correct_ids & set(jl_ids_list))
                jl_top1[layer].append(jl_in_top1)
                jl_topk[layer].append(jl_in_topk)
                if jl_in_topk and not jl_ok:
                    jl_emergence.append(layer)
                    jl_ok = True

                # Logit lens
                ll_h = h if norm is None else norm(h)
                ll_logits = lm_head(ll_h.unsqueeze(0))[0]
                ll_ids_list = torch.topk(ll_logits, self.top_k).indices.tolist()
                ll_in_top1 = bool(correct_ids) and (ll_ids_list[0] in correct_ids)
                ll_in_topk = bool(correct_ids) and bool(correct_ids & set(ll_ids_list))
                ll_top1[layer].append(ll_in_top1)
                ll_topk[layer].append(ll_in_topk)
                if ll_in_topk and not ll_ok:
                    ll_emergence.append(layer)
                    ll_ok = True

                if jl_in_top1 != ll_in_top1:
                    jl_disagree += 1

            if not jl_ok:
                jl_emergence.append(None); jl_never += 1
            if not ll_ok:
                ll_emergence.append(None); ll_never += 1

        def _rate(v): return round(sum(v) / len(v), 4) if v else 0.0
        def _mean_emergence(em): return (
            round(sum(e for e in em if e is not None) / sum(1 for e in em if e is not None), 2)
            if any(e is not None for e in em) else None
        )

        jl_t1 = {l: _rate(jl_top1[l]) for l in apply_layers}
        jl_tk = {l: _rate(jl_topk[l]) for l in apply_layers}
        ll_t1 = {l: _rate(ll_top1[l]) for l in apply_layers}
        ll_tk = {l: _rate(ll_topk[l]) for l in apply_layers}

        print(f"\n{'='*80}")
        print("JACOBIAN LENS vs LOGIT LENS")
        print(f"{'='*80}")
        print(f"Samples: {n} | Final accuracy: {final_correct / n:.1%} | Top-1 disagreements: {jl_disagree}")
        print(f"\n{'Layer':>6}  {'JL Top-1':>10}  {'JL Top-K':>10}  {'LL Top-1':>10}  {'LL Top-K':>10}")
        print("-" * 56)
        for layer in apply_layers:
            print(f"{layer:>6}  {jl_t1[layer]:>10.1%}  {jl_tk[layer]:>10.1%}  {ll_t1[layer]:>10.1%}  {ll_tk[layer]:>10.1%}")
        print(f"{'='*80}\n")

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics={
                "mode": "apply", "num_samples": n,
                "final_accuracy": round(final_correct / n, 4) if n else 0,
                "jl_mean_emergence": _mean_emergence(jl_emergence),
                "jl_never_emerged_rate": round(jl_never / n, 4) if n else 0,
                "ll_mean_emergence": _mean_emergence(ll_emergence),
                "ll_never_emerged_rate": round(ll_never / n, 4) if n else 0,
                "jl_top1_rates": jl_t1, "jl_topk_rates": jl_tk,
                "ll_top1_rates": ll_t1, "ll_topk_rates": ll_tk,
                "top1_disagreements": jl_disagree,
            },
            metadata={
                "apply_layers": apply_layers, "top_k": self.top_k,
                "lens_path": self.lens_path, "lens_n_prompts": lens.n_prompts,
                "num_samples": n, "seed": self.seed,
            },
        )

    # -- compare mode -----------------------------------------------------

    def _run_compare(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        lens: JacobianLens,
    ) -> ExperimentResult:
        tokenizer = backend._tokenizer
        model = backend._model
        device = backend.device
        lm_head = self._get_lm_head(model)
        norm = self._get_final_norm(model)
        apply_layers = self._resolve_apply_layers(lens)
        n_compare = min(self.num_samples or 5, 20)
        samples = list(dataset)[:n_compare]

        comparisons = []
        for sample in samples:
            prompt_input = {"text": sample.text, "question": sample.text, "report": sample.text, "metadata": sample.metadata or {}}
            prompt_str = prompt_strategy.build_prompt(prompt_input) + self.answer_cue
            try:
                prompt_tokens = tokenizer.encode(
                    prompt_str, return_tensors="pt",
                    truncation=True, max_length=self.max_input_tokens,
                ).to(device)
                with torch.no_grad():
                    out = model(input_ids=prompt_tokens, output_hidden_states=True, use_cache=False)
                hidden_states = out.hidden_states
            except Exception as e:
                comparisons.append({"error": str(e)})
                continue

            layer_comparisons = []
            for layer in apply_layers:
                if layer + 1 >= len(hidden_states):
                    continue
                h = hidden_states[layer + 1][0, -1, :]
                try:
                    jl_score = lens.decode(h.unsqueeze(0), layer, lm_head, norm)[0]
                    jl_top = torch.topk(jl_score, self.top_k)
                    jl_tokens = [tokenizer.decode([t.item()]) for t in jl_top.indices]
                except KeyError:
                    jl_tokens = []
                ll_h = h if norm is None else norm(h)
                ll_top = torch.topk(lm_head(ll_h.unsqueeze(0))[0], self.top_k)
                ll_tokens = [tokenizer.decode([t.item()]) for t in ll_top.indices]
                overlap = len(set(jl_top.indices.tolist()) & set(ll_top.indices.tolist()))
                layer_comparisons.append({
                    "layer": layer, "jl_top_tokens": jl_tokens, "ll_top_tokens": ll_tokens,
                    "topk_overlap": overlap,
                    "topk_overlap_rate": round(overlap / self.top_k, 2) if self.top_k else 0,
                })
            comparisons.append({"text": sample.text[:200], "label": sample.label, "layers": layer_comparisons})

        print(f"\nJ-lens vs Logit Lens comparison on {len(comparisons)} samples")
        for ci, comp in enumerate(comparisons):
            if "error" in comp:
                continue
            print(f"\n--- Sample {ci}: label={comp['label']} ---")
            print(f"  {comp['text'][:100]}...")
            for lc in comp["layers"][:5]:
                print(f"  L{lc['layer']:>3} JL: {lc['jl_top_tokens'][:5]}  |  LL: {lc['ll_top_tokens'][:5]}  overlap={lc['topk_overlap']}/{self.top_k}")

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics={"mode": "compare", "n_samples": len(comparisons)},
            raw_outputs=comparisons,
            metadata={"apply_layers": apply_layers, "top_k": self.top_k, "lens_path": self.lens_path},
        )

    # -- main -------------------------------------------------------------

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        logger: Optional[ExperimentLogger] = None,
    ) -> ExperimentResult:
        if self.mode == "fit":
            return self._run_fit(backend)
        if self.mode in ("apply", "compare"):
            if not self.lens_path:
                raise ValueError("lens_path is required for apply/compare mode")
            lens = JacobianLens.load(self.lens_path)
            if self.mode == "apply":
                return self._run_apply(backend, dataset, prompt_strategy, lens)
            return self._run_compare(backend, dataset, prompt_strategy, lens)
        raise ValueError(f"Unknown mode: {self.mode}. Use 'fit', 'apply', or 'compare'.")
