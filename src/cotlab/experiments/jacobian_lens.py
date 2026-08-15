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
    lens_type: str = "jlens"

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        jacobians_flat = torch.cat(
            [self.jacobians[k].flatten().cpu() for k in sorted(self.jacobians.keys())]
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
            "lens_type": self.lens_type,
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
            lens_type=metadata.get("lens_type", "jlens"),
        )

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Apply J_ℓ to a residual-stream activation: returns J_ℓ @ h_ℓ."""
        if layer not in self.jacobians:
            raise KeyError(
                f"Layer {layer} not in fitted lens. Available: {sorted(self.jacobians.keys())}"
            )
        J = self.jacobians[layer].to(residual.device, dtype=residual.dtype)
        return residual @ J.T

    @torch.inference_mode()
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

    @torch.inference_mode()
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

        with torch.inference_mode():
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
        return {
            "lens_scores": lens_scores,
            "model_scores": model_scores,
            "input_ids": input_ids.cpu(),
        }

    @classmethod
    def merge(cls, lenses: List[JacobianLens]) -> JacobianLens:
        """Prompt-weighted average of multiple lenses fitted on different subsets."""
        total_prompts = sum(lens.n_prompts for lens in lenses)
        d_model = lenses[0].d_model
        layer_keys = sorted(lenses[0].jacobians.keys())
        merged_jacobians = {}
        for k in layer_keys:
            weighted_sum = torch.zeros(d_model, d_model)
            for lens in lenses:
                weighted_sum += lens.jacobians[k] * lens.n_prompts
            merged_jacobians[k] = weighted_sum / total_prompts
        return cls(
            jacobians=merged_jacobians,
            d_model=d_model,
            n_prompts=total_prompts,
            source_layers=lenses[0].source_layers,
            target_layer=lenses[0].target_layer,
            skip_first_n=lenses[0].skip_first_n,
            model_name=lenses[0].model_name,
            lens_type=lenses[0].lens_type,
        )


# ---------------------------------------------------------------------------
# Fitting utilities
# ---------------------------------------------------------------------------

SKIP_FIRST_N_POSITIONS = 16
# Output-coordinate dims per backward pass in the fit loop. The cotangent is
# [dim_batch, seq_len, d_model] and the prompt is replicated dim_batch times, so
# VRAM scales ~linearly with dim_batch; higher = fewer autograd.grad calls
# (d_model/dim_batch), lower = less memory. Verified bit-identical across values.
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

    return {layer: captured[layer] for layer in source_layers}, captured[target_layer]


def jacobian_for_prompt(
    model: nn.Module,
    input_ids: torch.Tensor,
    source_layers: List[int],
    target_layer: int,
    dim_batch: int = DIM_BATCH,
    skip_first_n: int = SKIP_FIRST_N_POSITIONS,
    lrp: bool = False,
) -> Dict[int, torch.Tensor]:
    """Compute per-layer Jacobian matrices J_ℓ = 𝔼[∂h_T/∂h_ℓ] for one prompt.

    If lrp=True, the LRP rules are installed on the model so the backward pass
    computes relevance coefficients instead of raw gradients (R-lens / RelP).

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
    if lrp:
        from .lrp import lrp_context

        with lrp_context(model):
            source_acts, target_act = _hook_layer_outputs(
                model, replicated, source_layers, target_layer
            )
    else:
        source_acts, target_act = _hook_layer_outputs(
            model, replicated, source_layers, target_layer
        )
    J = {layer: torch.zeros(d_model, d_model) for layer in source_layers}

    # With device_map='auto' (multi-GPU), the target activation may live on a
    # different device than the inputs; the cotangent must match its device.
    device = target_act.device
    for dim_start in range(0, d_model, dim_batch):
        n_dims = min(dim_batch, d_model - dim_start)
        cotangent = torch.zeros(dim_batch, seq_len, d_model, device=device)
        for i in range(n_dims):
            cotangent[i, valid_positions, dim_start + i] = 1.0
        grads = torch.autograd.grad(
            outputs=target_act,
            inputs=[source_acts[layer] for layer in source_layers],
            grad_outputs=cotangent,
            retain_graph=True,
            allow_unused=False,
        )
        for l_idx, grad in enumerate(grads):
            layer = source_layers[l_idx]
            rows = grad[:n_dims, valid_positions, :].float().mean(dim=1).detach().cpu()
            J[layer][dim_start : dim_start + n_dims, :] = rows

    del source_acts, target_act, replicated
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
    lrp: bool = False,
) -> JacobianLens:
    """Fit the Jacobian lens on a corpus of text prompts.

    If lrp=True, the LRP rules (RelP) are installed during fitting so the
    resulting matrices are R-lens relevance coefficients rather than raw
    Jacobians. Forward values are identical; only the backward graph differs.
    """
    num_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    if source_layers is None:
        source_layers = list(range(num_layers))
    if target_layer is None:
        target_layer = num_layers - 1
    source_layers = [layer for layer in source_layers if layer < target_layer]
    if not source_layers:
        raise ValueError(f"No source layers < target_layer ({target_layer})")

    orig_grad = _freeze_params(model)
    model.eval()

    valid_prompts = [
        p for p in prompts if _validate_prompt(tokenizer, p, skip_first_n + 1, max_seq_len)
    ]
    if not valid_prompts:
        raise ValueError(f"No valid prompts (need > {skip_first_n + 1} tokens)")

    jacobian_sum = {layer: torch.zeros(d_model, d_model) for layer in source_layers}
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
            jacobian_sum = {layer: ckpt["jacobian_sum"][layer].clone() for layer in source_layers}
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
                model,
                input_ids,
                source_layers,
                target_layer,
                dim_batch=dim_batch,
                skip_first_n=skip_first_n,
                lrp=lrp,
            )
            for layer in source_layers:
                jacobian_sum[layer] += J[layer]
            n_done += 1
            next_idx = idx + 1
            if checkpoint_path and n_done % checkpoint_every == 0 and n_done > 0:
                os.makedirs(checkpoint_path, exist_ok=True)
                tmp = os.path.join(checkpoint_path, "checkpoint.tmp")
                torch.save(
                    {
                        "jacobian_sum": {
                            layer: jacobian_sum[layer].clone() for layer in source_layers
                        },
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

    jacobians = {layer: jacobian_sum[layer] / n_done for layer in source_layers}

    if checkpoint_path:
        os.makedirs(checkpoint_path, exist_ok=True)
        tmp = os.path.join(checkpoint_path, "checkpoint.tmp")
        torch.save(
            {
                "jacobian_sum": {layer: jacobian_sum[layer].clone() for layer in source_layers},
                "n_done": n_done,
                "next_idx": next_idx,
                "source_layers": source_layers,
                "target_layer": target_layer,
                "skip_first_n": skip_first_n,
            },
            tmp,
        )
        os.replace(tmp, os.path.join(checkpoint_path, "checkpoint.pt"))

    _thaw_params(model, orig_grad)
    return JacobianLens(
        jacobians=jacobians,
        d_model=d_model,
        n_prompts=n_done,
        source_layers=source_layers,
        target_layer=target_layer,
        skip_first_n=skip_first_n,
        model_name=getattr(model.config, "_name_or_path", "unknown"),
        lens_type="rlens" if lrp else "jlens",
    )


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------


@Registry.register_experiment("jacobian_lens")
class JacobianLensExperiment(BaseExperiment):
    """Jacobian Lens: causal concept readout and J-space interventions.

    Modes:
        fit       — compute and save J_ℓ matrices (GPU, one-time per model)
        apply     — load saved matrices, run on a dataset
        compare   — run both J-lens and logit lens side-by-side
        steer     — inject a concept J-lens vector, measure output shift
        swap      — lens-coordinate swap between two concepts
        ablate    — suppress J-space component, measure accuracy loss
        decompose — split activation into J-space vs non-J-space components
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
        lrp: bool = False,
        # intervention mode parameters
        steer_token: str = "",
        steer_alpha: float = 1.0,
        steer_positions: Optional[List[int]] = None,
        swap_source: str = "",
        swap_target: str = "",
        ablate_top_n: int = 5,
        intervention_layers: Optional[List[int]] = None,
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
        self.lrp = lrp
        self.steer_token = steer_token
        self.steer_alpha = steer_alpha
        self.steer_positions = steer_positions
        self.swap_source = swap_source
        self.swap_target = swap_target
        self.ablate_top_n = ablate_top_n
        self.intervention_layers = intervention_layers

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

    def _tokenize_batch(self, tokenizer, texts: List[str], device: str) -> Dict[str, torch.Tensor]:
        """Left-pad a batch with position_ids remap (logit_lens precedent).

        Each row's positional embeddings then match its single-sample run, so a
        per-row intervention in one batched forward reproduces the sequential
        per-sample forwards exactly.
        """
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

    def _resolve_apply_layers(self, lens: JacobianLens) -> List[int]:
        if self.source_layers is not None:
            return [layer for layer in self.source_layers if layer in lens.jacobians]
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
        print(f"Lens type: {'R-lens (LRP)' if self.lrp else 'J-lens'}")

        prompts = load_corpus_prompts(
            path=self.corpus_path,
            n_prompts=self.n_corpus_prompts,
            max_seq_len=self.max_input_tokens,
            seed=self.seed,
        )
        print(f"Valid prompts: {len(prompts)}")

        lens = fit_jacobian_lens(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=self.dim_batch,
            skip_first_n=self.skip_first_n,
            max_seq_len=self.max_input_tokens,
            device=backend.device,
            checkpoint_path=self.lens_path,
            checkpoint_every=50,
            lrp=self.lrp,
        )
        if self.lens_path:
            lens.save(self.lens_path)
            print(f"Saved lens to {self.lens_path}")

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="corpus",
            metrics={
                "mode": "fit",
                "lens_type": lens.lens_type,
                "n_prompts": lens.n_prompts,
                "d_model": lens.d_model,
                "source_layers": lens.source_layers,
                "target_layer": lens.target_layer,
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

        jl_top1: Dict[int, List[bool]] = {layer: [] for layer in apply_layers}
        jl_topk: Dict[int, List[bool]] = {layer: [] for layer in apply_layers}
        ll_top1: Dict[int, List[bool]] = {layer: [] for layer in apply_layers}
        ll_topk: Dict[int, List[bool]] = {layer: [] for layer in apply_layers}
        jl_emergence: List[Optional[int]] = []
        ll_emergence: List[Optional[int]] = []
        jl_never = ll_never = final_correct = jl_disagree = 0

        # Preload J matrices to the device once (avoids per-(sample, layer)
        # .to() device transfers in lens.transport).
        j_preloaded = {}
        for layer in apply_layers:
            if layer in lens.jacobians:
                j_preloaded[layer] = lens.jacobians[layer].to(device, dtype=torch.float32)
        layers_with_J = [layer for layer in apply_layers if layer in lens.jacobians]
        if layers_with_J:
            J_stack = torch.stack([j_preloaded[layer] for layer in layers_with_J])  # [L, d, d]

        for sample in tqdm(samples, desc="J-lens apply"):
            prompt_input = {
                "text": sample.text,
                "question": sample.text,
                "report": sample.text,
                "metadata": sample.metadata or {},
            }
            prompt_str = prompt_strategy.build_prompt(prompt_input) + self.answer_cue
            try:
                prompt_tokens = tokenizer.encode(
                    prompt_str,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_tokens,
                ).to(device)
                with torch.inference_mode():
                    out = model(input_ids=prompt_tokens, output_hidden_states=True, use_cache=False)
                hidden_states = out.hidden_states
                with torch.inference_mode():
                    final_h = hidden_states[-1][0, -1, :].detach().clone()
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

            # Batched per-layer decode: stack last-token hidden states across
            # the lens layers [L, d], one norm + lm_head call each. RMSNorm /
            # LayerNorm normalize per-token over the hidden dim, and lm_head is
            # a per-row matmul, so the batched rows are identical to the
            # per-layer decode.
            jl_ok = ll_ok = False
            valid_layers = [layer for layer in layers_with_J if layer + 1 < len(hidden_states)]
            if valid_layers:
                h_stack = torch.stack(
                    [hidden_states[layer + 1][0, -1, :] for layer in valid_layers]
                ).to(device)  # [L, d]

                # J-lens: transport J_ℓ @ h_ℓ for all layers, then norm + lm_head.
                # Wrapped in inference_mode like the per-layer lens.decode was.
                with torch.inference_mode():
                    transported = torch.bmm(
                        h_stack.unsqueeze(1), J_stack[: len(valid_layers)].transpose(-1, -2)
                    ).squeeze(1)  # [L, d]
                    if norm is not None:
                        transported = norm(transported)
                    jl_logits = lm_head(transported)  # [L, vocab]

                    # Logit lens: norm(h) then lm_head.
                    ll_h = h_stack if norm is None else norm(h_stack)
                    ll_logits = lm_head(ll_h)  # [L, vocab]

                jl_ids_lists = torch.topk(jl_logits, self.top_k, dim=-1).indices.tolist()
                ll_ids_lists = torch.topk(ll_logits, self.top_k, dim=-1).indices.tolist()

                for idx, layer in enumerate(valid_layers):
                    jl_ids_list = jl_ids_lists[idx]
                    ll_ids_list = ll_ids_lists[idx]
                    jl_in_top1 = bool(correct_ids) and (jl_ids_list[0] in correct_ids)
                    jl_in_topk = bool(correct_ids) and bool(correct_ids & set(jl_ids_list))
                    ll_in_top1 = bool(correct_ids) and (ll_ids_list[0] in correct_ids)
                    ll_in_topk = bool(correct_ids) and bool(correct_ids & set(ll_ids_list))
                    jl_top1[layer].append(jl_in_top1)
                    jl_topk[layer].append(jl_in_topk)
                    ll_top1[layer].append(ll_in_top1)
                    ll_topk[layer].append(ll_in_topk)
                    if jl_in_topk and not jl_ok:
                        jl_emergence.append(layer)
                        jl_ok = True
                    if ll_in_topk and not ll_ok:
                        ll_emergence.append(layer)
                        ll_ok = True
                    if jl_in_top1 != ll_in_top1:
                        jl_disagree += 1

            if not jl_ok:
                jl_emergence.append(None)
                jl_never += 1
            if not ll_ok:
                ll_emergence.append(None)
                ll_never += 1

        def _rate(v):
            return round(sum(v) / len(v), 4) if v else 0.0

        def _mean_emergence(em):
            return (
                round(sum(e for e in em if e is not None) / sum(1 for e in em if e is not None), 2)
                if any(e is not None for e in em)
                else None
            )

        jl_t1 = {layer: _rate(jl_top1[layer]) for layer in apply_layers}
        jl_tk = {layer: _rate(jl_topk[layer]) for layer in apply_layers}
        ll_t1 = {layer: _rate(ll_top1[layer]) for layer in apply_layers}
        ll_tk = {layer: _rate(ll_topk[layer]) for layer in apply_layers}

        print(f"\n{'=' * 80}")
        print("JACOBIAN LENS vs LOGIT LENS")
        print(f"{'=' * 80}")
        print(
            f"Samples: {n} | Final accuracy: {final_correct / n:.1%} | Top-1 disagreements: {jl_disagree}"
        )
        print(
            f"\n{'Layer':>6}  {'JL Top-1':>10}  {'JL Top-K':>10}  {'LL Top-1':>10}  {'LL Top-K':>10}"
        )
        print("-" * 56)
        for layer in apply_layers:
            print(
                f"{layer:>6}  {jl_t1[layer]:>10.1%}  {jl_tk[layer]:>10.1%}  {ll_t1[layer]:>10.1%}  {ll_tk[layer]:>10.1%}"
            )
        print(f"{'=' * 80}\n")

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics={
                "mode": "apply",
                "num_samples": n,
                "final_accuracy": round(final_correct / n, 4) if n else 0,
                "jl_mean_emergence": _mean_emergence(jl_emergence),
                "jl_never_emerged_rate": round(jl_never / n, 4) if n else 0,
                "ll_mean_emergence": _mean_emergence(ll_emergence),
                "ll_never_emerged_rate": round(ll_never / n, 4) if n else 0,
                "jl_top1_rates": jl_t1,
                "jl_topk_rates": jl_tk,
                "ll_top1_rates": ll_t1,
                "ll_topk_rates": ll_tk,
                "top1_disagreements": jl_disagree,
            },
            metadata={
                "apply_layers": apply_layers,
                "top_k": self.top_k,
                "lens_path": self.lens_path,
                "lens_n_prompts": lens.n_prompts,
                "num_samples": n,
                "seed": self.seed,
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
            prompt_input = {
                "text": sample.text,
                "question": sample.text,
                "report": sample.text,
                "metadata": sample.metadata or {},
            }
            prompt_str = prompt_strategy.build_prompt(prompt_input) + self.answer_cue
            try:
                prompt_tokens = tokenizer.encode(
                    prompt_str,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_tokens,
                ).to(device)
                with torch.inference_mode():
                    out = model(input_ids=prompt_tokens, output_hidden_states=True, use_cache=False)
                hidden_states = out.hidden_states
            except Exception as e:
                comparisons.append({"error": str(e)})
                continue

            layer_comparisons = []
            for layer in apply_layers:
                if layer + 1 >= len(hidden_states):
                    continue
                h = hidden_states[layer + 1][0, -1, :].detach().clone()
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
                layer_comparisons.append(
                    {
                        "layer": layer,
                        "jl_top_tokens": jl_tokens,
                        "ll_top_tokens": ll_tokens,
                        "topk_overlap": overlap,
                        "topk_overlap_rate": round(overlap / self.top_k, 2) if self.top_k else 0,
                    }
                )
            comparisons.append(
                {"text": sample.text[:200], "label": sample.label, "layers": layer_comparisons}
            )

        print(f"\nJ-lens vs Logit Lens comparison on {len(comparisons)} samples")
        for ci, comp in enumerate(comparisons):
            if "error" in comp:
                continue
            print(f"\n--- Sample {ci}: label={comp['label']} ---")
            print(f"  {comp['text'][:100]}...")
            for lc in comp["layers"][:5]:
                print(
                    f"  L{lc['layer']:>3} JL: {lc['jl_top_tokens'][:5]}  |  LL: {lc['ll_top_tokens'][:5]}  overlap={lc['topk_overlap']}/{self.top_k}"
                )

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics={"mode": "compare", "n_samples": len(comparisons)},
            raw_outputs=comparisons,
            metadata={
                "apply_layers": apply_layers,
                "top_k": self.top_k,
                "lens_path": self.lens_path,
            },
        )

    # ==================================================================
    # Steer mode: inject a concept along a J-lens vector
    # ==================================================================

    def _run_steer(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        lens: JacobianLens,
    ) -> ExperimentResult:
        """Inject steer_token into residual stream, measure output shift."""
        model = backend._model
        tokenizer = backend._tokenizer
        device = backend.device
        lm_head = self._get_lm_head(model)
        target_id = tokenizer.encode(self.steer_token, add_special_tokens=False)[0]
        layers = self.intervention_layers or sorted(lens.jacobians.keys())[-3:]

        samples = (
            dataset.sample(self.num_samples, seed=self.seed)
            if self.num_samples is not None
            else list(dataset)[:5]
        )

        results = []
        # Batched: one forward per layer over all samples as rows. Left-pad +
        # position_ids remap keep each row's positional embeddings identical to
        # its single-sample run, so each row reproduces the sequential steer
        # forward exactly (causal-mask row isolation, eval, no dropout).
        prompts_full = [
            prompt_strategy.build_prompt(
                {"text": s.text, "question": s.text, "metadata": s.metadata or {}}
            )
            + self.answer_cue
            for s in samples
        ]
        batch_tokens = self._tokenize_batch(tokenizer, prompts_full, device)

        for layer in layers:
            if layer not in lens.jacobians:
                continue
            J = lens.jacobians[layer].to(device, dtype=torch.float32)
            v_t = lm_head.weight[target_id].float() @ J  # [d_model]

            def make_steer_hook(vec):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        t, rest = output[0], output[1:]
                    else:
                        t, rest = output, ()
                    t[:, -1, :] = t[:, -1, :] + self.steer_alpha * vec
                    return (t,) + rest if rest else t

                return hook

            block = backend.hook_manager.get_layer_module(layer)
            handle = block.register_forward_hook(make_steer_hook(v_t))
            try:
                with torch.inference_mode():
                    out = model(
                        input_ids=batch_tokens["input_ids"],
                        attention_mask=batch_tokens["attention_mask"],
                        position_ids=batch_tokens["position_ids"],
                        output_hidden_states=False,
                        use_cache=False,
                    )
            finally:
                handle.remove()

            logits = out.logits[:, -1, :]  # [B, vocab]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, 5, dim=-1)
            sorted_probs, sorted_ids = probs.sort(descending=True, dim=-1)
            ranks = (sorted_ids == target_id).nonzero(as_tuple=True)[1]

            for row, sample in enumerate(samples):
                target_rank = int(ranks[row].item())
                results.append(
                    {
                        "sample": sample.idx,
                        "layer": layer,
                        "steer_token": self.steer_token,
                        "alpha": self.steer_alpha,
                        "target_rank": target_rank + 1,
                        "top_tokens": [tokenizer.decode([t.item()]) for t in top_ids[row]],
                    }
                )

        for r in results:
            print(
                f"  L{r['layer']:>3} sample={r['sample']}  rank({r['steer_token']})={r['target_rank']}  top={r['top_tokens'][:3]}"
            )

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics={"mode": "steer", "n_trials": len(results)},
            raw_outputs=results,
            metadata={
                "steer_token": self.steer_token,
                "steer_alpha": self.steer_alpha,
                "intervention_layers": layers,
            },
        )

    # ==================================================================
    # Swap mode: lens-coordinate concept swap
    # ==================================================================

    def _run_swap(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        lens: JacobianLens,
    ) -> ExperimentResult:
        """Swap v_src → v_tgt in J-space, measure output flip rate."""
        model = backend._model
        tokenizer = backend._tokenizer
        device = backend.device
        lm_head = self._get_lm_head(model)
        src_id = tokenizer.encode(self.swap_source, add_special_tokens=False)[0]
        tgt_id = tokenizer.encode(self.swap_target, add_special_tokens=False)[0]
        layers = self.intervention_layers or sorted(lens.jacobians.keys())[-3:]

        samples = (
            dataset.sample(self.num_samples, seed=self.seed)
            if self.num_samples is not None
            else list(dataset)[:10]
        )

        flips = 0
        # Batched: one forward per layer over all samples as rows (left-pad +
        # position_ids remap; causal-mask row isolation, eval, no dropout), so
        # each row reproduces the sequential swap forward exactly.
        prompts_full = [
            prompt_strategy.build_prompt(
                {"text": s.text, "question": s.text, "metadata": s.metadata or {}}
            )
            + self.answer_cue
            for s in samples
        ]
        batch_tokens = self._tokenize_batch(tokenizer, prompts_full, device)

        # Baseline: model output without swap (batched over all samples).
        with torch.inference_mode():
            base_out = model(
                input_ids=batch_tokens["input_ids"],
                attention_mask=batch_tokens["attention_mask"],
                position_ids=batch_tokens["position_ids"],
                output_hidden_states=False,
                use_cache=False,
            )
        base_probs = torch.softmax(base_out.logits[:, -1, :], dim=-1)  # [B, vocab]
        base_top1 = torch.argmax(base_probs, dim=-1).tolist()

        for layer in layers:
            if layer not in lens.jacobians:
                continue

            J = lens.jacobians[layer].to(device, dtype=torch.float32)
            v_src = (lm_head.weight[src_id].float() @ J).unsqueeze(1)  # [d_model, 1]
            v_tgt = (lm_head.weight[tgt_id].float() @ J).unsqueeze(1)  # [d_model, 1]
            V = torch.cat([v_src, v_tgt], dim=1)  # [d_model, 2]
            V_pinv = (
                torch.linalg.pinv(V.T @ V + 1e-6 * torch.eye(2, device=device)) @ V.T
            )  # [2, d_model]

            def make_swap_hook():
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        t, rest = output[0], output[1:]
                    else:
                        t, rest = output, ()
                    h = t[:, -1, :].unsqueeze(-1)  # [B, d_model, 1]
                    c = torch.bmm(V_pinv.unsqueeze(0).expand(h.shape[0], -1, -1), h)  # [B, 2, 1]
                    c_swapped = c.clone()
                    c_swapped[:, 0], c_swapped[:, 1] = c[:, 1].clone(), c[:, 0].clone()
                    h_new = h + torch.bmm(V.unsqueeze(0).expand(h.shape[0], -1, -1), c_swapped - c)
                    t[:, -1, :] = h_new.squeeze(-1)
                    return (t,) + rest if rest else t

                return hook

            block = backend.hook_manager.get_layer_module(layer)
            handle = block.register_forward_hook(make_swap_hook())
            try:
                with torch.inference_mode():
                    swap_out = model(
                        input_ids=batch_tokens["input_ids"],
                        attention_mask=batch_tokens["attention_mask"],
                        position_ids=batch_tokens["position_ids"],
                        output_hidden_states=False,
                        use_cache=False,
                    )
            finally:
                handle.remove()

            swap_probs = torch.softmax(swap_out.logits[:, -1, :], dim=-1)
            swap_top1 = torch.argmax(swap_probs, dim=-1).tolist()

            for row, sample in enumerate(samples):
                flipped = swap_top1[row] != base_top1[row]
                if flipped:
                    flips += 1
                print(
                    f"  L{layer:>3} sample={sample.idx}  base={tokenizer.decode([base_top1[row]])}"
                    f"  swap={tokenizer.decode([swap_top1[row]])}  flipped={flipped}"
                )

        n_trials = len(samples) * len([layer for layer in layers if layer in lens.jacobians])
        flip_rate = flips / n_trials if n_trials else 0

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics={
                "mode": "swap",
                "swap_source": self.swap_source,
                "swap_target": self.swap_target,
                "n_trials": n_trials,
                "flips": flips,
                "flip_rate": round(flip_rate, 4),
            },
            metadata={"intervention_layers": layers},
        )

    # ==================================================================
    # Ablate mode: suppress top-k J-space directions, measure accuracy
    # ==================================================================

    def _run_ablate(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        lens: JacobianLens,
    ) -> ExperimentResult:
        """Zero out top-k J-space components at each intervention layer."""
        model = backend._model
        tokenizer = backend._tokenizer
        device = backend.device
        lm_head = self._get_lm_head(model)
        norm = self._get_final_norm(model)
        layers = self.intervention_layers or sorted(lens.jacobians.keys())[:3]

        samples = (
            dataset.sample(self.num_samples, seed=self.seed)
            if self.num_samples is not None
            else list(dataset)[:20]
        )
        n = len(samples)

        baseline_correct = 0
        ablate_correct = 0

        # Hoist lm_head and the per-layer W_U·J product out of the sample loop.
        # W_U·J is a [vocab, d_model] matmul recomputed per (sample, layer)
        # before; now computed once per layer. Bit-identical matmul, just cached.
        vocab = lm_head.weight.float().to(device)
        wu_j_by_layer: dict = {}
        j_by_layer: dict = {}
        for layer in layers:
            if layer in lens.jacobians:
                J = lens.jacobians[layer].to(device, dtype=torch.float32)
                j_by_layer[layer] = J
                wu_j_by_layer[layer] = vocab @ J

        # Batched: one baseline + one ablated forward over all samples as rows
        # (left-pad + position_ids remap; causal-mask row isolation). The ablate
        # hook projects out top-k J-space directions per row.
        prompts_full = [
            prompt_strategy.build_prompt(
                {"text": s.text, "question": s.text, "metadata": s.metadata or {}}
            )
            + self.answer_cue
            for s in samples
        ]
        batch_tokens = self._tokenize_batch(tokenizer, prompts_full, device)

        # Baseline
        with torch.inference_mode():
            base_out = model(
                input_ids=batch_tokens["input_ids"],
                attention_mask=batch_tokens["attention_mask"],
                position_ids=batch_tokens["position_ids"],
                output_hidden_states=True,
                use_cache=False,
            )
        final_h = base_out.hidden_states[-1][:, -1, :]  # [B, d]
        if norm is not None:
            final_h = norm(final_h)
        base_ids = torch.argmax(lm_head(final_h), dim=-1).tolist()
        correct_sets = [self._correct_token_ids(tokenizer, s.label) for s in samples]
        baseline_correct = sum(1 for i, cid in enumerate(base_ids) if cid in correct_sets[i])

        # Ablate: at each intervention layer, project out top-k J-space directions
        def make_ablate_hook(layer_idx: int):
            wu_j = wu_j_by_layer[layer_idx]
            J = j_by_layer[layer_idx]

            def hook(module, inp, output):
                if isinstance(output, tuple):
                    t, rest = output[0], output[1:]
                else:
                    t, rest = output, ()
                h = t[:, -1, :].float()  # [B, d_model]
                # Project onto top-k J-lens directions and remove J-lens vectors
                # for the full vocabulary at this layer: W_U · J_ℓ → take top-k
                # by inner product with h (per row).
                all_scores = h @ wu_j.T  # [B, vocab]
                top_k_ids = torch.topk(all_scores, self.ablate_top_n, dim=-1).indices  # [B, k]
                for j in range(self.ablate_top_n):
                    v = vocab[top_k_ids[:, j]] @ J  # [B, d_model]
                    v_norm = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
                    h = h - torch.bmm(v_norm.unsqueeze(1), h.unsqueeze(-1)).squeeze(-1) * v_norm
                t[:, -1, :] = h.to(t.dtype)
                return (t,) + rest if rest else t

            return hook

        handles = []
        for layer in layers:
            if layer in lens.jacobians:
                block = backend.hook_manager.get_layer_module(layer)
                handles.append(block.register_forward_hook(make_ablate_hook(layer)))

        try:
            with torch.inference_mode():
                ablate_out = model(
                    input_ids=batch_tokens["input_ids"],
                    attention_mask=batch_tokens["attention_mask"],
                    position_ids=batch_tokens["position_ids"],
                    output_hidden_states=False,
                    use_cache=False,
                )
        finally:
            for h in handles:
                h.remove()

        ablate_ids = torch.argmax(ablate_out.logits[:, -1, :], dim=-1).tolist()
        ablate_correct = sum(1 for i, cid in enumerate(ablate_ids) if cid in correct_sets[i])

        base_acc = baseline_correct / n
        abl_acc = ablate_correct / n
        print(
            f"\nBaseline acc: {base_acc:.1%}  |  Ablate (top-{self.ablate_top_n}) acc: {abl_acc:.1%}  |  delta: {(abl_acc - base_acc):.1%}"
        )
        print(f"  Correct: {baseline_correct}/{n} → {ablate_correct}/{n}")
        print(f"  Ablated at layers: {layers}")

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics={
                "mode": "ablate",
                "baseline_accuracy": round(base_acc, 4),
                "ablate_accuracy": round(abl_acc, 4),
                "accuracy_delta": round(abl_acc - base_acc, 4),
                "ablate_top_n": self.ablate_top_n,
            },
            metadata={"intervention_layers": layers},
        )

    # ==================================================================
    # Decompose mode: J-space vs non-J-space components
    # ==================================================================

    def _run_decompose(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        lens: JacobianLens,
    ) -> ExperimentResult:
        """Decompose activations into J-space and non-J-space components."""
        model = backend._model
        tokenizer = backend._tokenizer
        device = backend.device
        lm_head = self._get_lm_head(model)
        layers = self.intervention_layers or sorted(lens.jacobians.keys())[-3:]

        samples = (
            dataset.sample(self.num_samples, seed=self.seed)
            if self.num_samples is not None
            else list(dataset)[:5]
        )

        decompositions = []

        # Hoist lm_head and per-layer W_U·J out of the sample loop (same as
        # ablate): the [vocab, d_model] product is recomputed per (sample,
        # layer) before; now computed once per layer.
        vocab = lm_head.weight.float().to(device)
        wu_j_by_layer: dict = {}
        j_by_layer: dict = {}
        for layer in layers:
            if layer in lens.jacobians:
                J = lens.jacobians[layer].to(device, dtype=torch.float32)
                j_by_layer[layer] = J
                wu_j_by_layer[layer] = vocab @ J

        # Batched: one forward over all samples as rows (left-pad + position_ids
        # remap; causal-mask row isolation), then per-layer computation on [B, d].
        prompts_full = [
            prompt_strategy.build_prompt(
                {"text": s.text, "question": s.text, "metadata": s.metadata or {}}
            )
            + self.answer_cue
            for s in samples
        ]
        batch_tokens = self._tokenize_batch(tokenizer, prompts_full, device)

        with torch.inference_mode():
            out = model(
                input_ids=batch_tokens["input_ids"],
                attention_mask=batch_tokens["attention_mask"],
                position_ids=batch_tokens["position_ids"],
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = out.hidden_states

        for layer in layers:
            if layer not in lens.jacobians or layer + 1 >= len(hidden_states):
                continue
            h = hidden_states[layer + 1][:, -1, :].float().detach()  # [B, d_model]
            wu_j = wu_j_by_layer[layer]
            J = j_by_layer[layer]

            # J-space: top-k J-lens directions via inner product (per row)
            all_scores = h @ wu_j.T  # [B, vocab]
            top_k_vals, top_k_ids = torch.topk(all_scores, self.top_k, dim=-1)
            j_tokens_by_row = [
                [tokenizer.decode([t.item()]) for t in top_k_ids[row]]
                for row in range(len(samples))
            ]

            # Non-J-space: project out top-k directions (per row, batched)
            h_nonj = h.clone()
            for j in range(self.top_k):
                v = vocab[top_k_ids[:, j]] @ J  # [B, d_model]
                v_norm = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
                h_nonj = (
                    h_nonj
                    - torch.bmm(v_norm.unsqueeze(1), h_nonj.unsqueeze(-1)).squeeze(-1) * v_norm
                )

            total_var = torch.var(h, dim=-1)
            j_component = h - h_nonj
            j_var = torch.where(
                total_var > 1e-8, torch.var(j_component, dim=-1), torch.zeros_like(total_var)
            )
            j_var_frac = torch.where(
                total_var > 1e-8, j_var / total_var, torch.zeros_like(total_var)
            )

            for row, sample in enumerate(samples):
                print(
                    f"  L{layer:>3} sample={sample.idx}  J-space({self.top_k}): {j_tokens_by_row[row][:5]}"
                    f"  J-var={float(j_var_frac[row].item()):.1%}"
                )

                decompositions.append(
                    {
                        "sample": sample.idx,
                        "layer": layer,
                        "j_space_tokens": j_tokens_by_row[row],
                        "j_variance_fraction": round(float(j_var_frac[row].item()), 4),
                        "total_variance": round(float(total_var[row].item()), 6),
                    }
                )

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics={"mode": "decompose", "n_results": len(decompositions)},
            raw_outputs=decompositions,
            metadata={"intervention_layers": layers, "top_k": self.top_k},
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
        if self.mode in ("apply", "compare", "steer", "swap", "ablate", "decompose"):
            if not self.lens_path:
                raise ValueError("lens_path is required for apply/compare/intervention modes")
            lens = JacobianLens.load(self.lens_path)
            if self.mode == "apply":
                return self._run_apply(backend, dataset, prompt_strategy, lens)
            if self.mode == "compare":
                return self._run_compare(backend, dataset, prompt_strategy, lens)
            if self.mode == "steer":
                return self._run_steer(backend, dataset, prompt_strategy, lens)
            if self.mode == "swap":
                return self._run_swap(backend, dataset, prompt_strategy, lens)
            if self.mode == "ablate":
                return self._run_ablate(backend, dataset, prompt_strategy, lens)
            return self._run_decompose(backend, dataset, prompt_strategy, lens)
        raise ValueError(
            f"Unknown mode: {self.mode}. "
            "Use 'fit', 'apply', 'compare', 'steer', 'swap', 'ablate', or 'decompose'."
        )
