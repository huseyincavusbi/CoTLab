"""Confidence Regulation Experiment.

Port of the entropy-neuron recipe from "Confidence Regulation Neurons in
Language Models" (Stolfo, Wu, Gurnee, Belinkov, Song, Sachan, Nanda --
NeurIPS 2024, arXiv:2406.16254) onto the CoTLab transformers backend.

Modes
-----
identify:
    Weight-space identification of final-layer entropy neurons. Three
    criteria are computed per neuron ``i`` of the final MLP layer:

    - ``norm_i``      : L2 norm of the neuron's output weights ``w_out^(i)``
      (columns of the MLP down-projection).
    - ``logit_var_i`` : variance over the vocabulary of the normalized logit
      projection ``W_U w_out / (col_norms(W_U) * ||w_out||)`` (Eq. 3 of the
      paper). Low values indicate a diffuse, softmax-invariant direct effect.
    - ``rho_i``       : fraction of ``w_out`` norm lying on the bottom-k
      right singular vectors of ``W_U`` -- the effective null space of the
      unembedding. High values indicate LayerNorm-mediated action.

    Entropy neurons combine high norm, low LogitVar and high rho. The
    default ranking follows the authors' released code
    (``get_potential_entropy_neurons_udark``): neurons are ranked by rho.
    On GPT-2 Small this reproduces 5/6 of the paper's named entropy
    neurons (584, 1611, 2044, 2123, 2870, 2910); the sixth (2378) is not
    returned by the authors' own released identification code either --
    it entered their named set through earlier manual analysis
    (Gurnee et al., arXiv:2401.12181).

mediate:
    Causal mediation via analytic mean-ablation on the cached final residual
    stream (paper Eqs. 4-6). Each candidate neuron's activation is set to its
    corpus mean and the residual-stream update is applied analytically -- no
    re-forward pass, valid only at the final layer where nothing intervenes
    before the final norm and unembedding. Total effect (TE) recomputes the
    final norm normally; direct effect with frozen scale (DE_LN) freezes the
    per-token norm denominator at its pre-ablation value. The LN-mediated
    fraction is ``1 - DE_LN / TE``; entropy neurons should be close to fully
    mediated (~1) versus ~0 for random neurons.

overlap:
    Jaccard overlap between identified neurons and H-Neuron probe
    sets (``probe_path``). Restricted to the final layer where the
    criterion is defined; reports enrichment over the hypergeometric
    random expectation.

neuron_family
-------------
``entropy`` (default) ranks by null-space fraction rho;
``frequency`` ranks by |cosine(write, v_freq)| with ``v_freq`` the
centered log-unigram direction (paper Sec. 4). In mediate mode the DE
pathway follows the family: entropy freezes the LayerNorm scale
(Eq. 6); frequency restores the v_freq logit component (Eq. 7).

Notes
-----
- The corpus for mean activations defaults to a small embedded public-domain
  text (configurable via ``corpus_text``); the original work used C4. This
  is an approximation that affects mean values but not the mechanism being
  validated (SEMANTIC-CHANGE class per working rules).
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry


@Registry.register_experiment("confidence_regulation")
class ConfidenceRegulationExperiment(BaseExperiment):
    """Identify and validate confidence-regulating (entropy) neurons."""

    _DEFAULT_CORPUS = (
        "The sun rose slowly over the quiet village, and the farmers began their daily work "
        "in the fields. In the market square, merchants arranged their goods while children "
        "played near the old stone fountain. The king had announced new laws that would change "
        "the way people lived, and everyone discussed the news with great interest. Scholars "
        "from the university came to study the ancient manuscripts preserved in the library, "
        "hoping to understand the history of the region. Travelers told stories of distant "
        "lands, describing mountains, rivers, and cities full of wonders. The seasons passed, "
        "and the people remembered both the hardships and the joys of the previous year. "
    )

    def __init__(
        self,
        name: str = "confidence_regulation",
        description: str = (
            "Entropy-neuron identification and mediation (Stolfo et al., NeurIPS 2024)"
        ),
        mode: str = "identify",
        selection: str = "top_n",
        top_n: int = 20,
        top_percent: float = 0.01,
        k_null: Optional[int] = None,
        logit_chunk_size: int = 256,
        fold_final_norm: bool = True,
        # --- mediate ---
        corpus_text: Optional[str] = None,
        n_tokens: int = 8192,
        seq_len: int = 256,
        mediate_sequences: int = 8,
        mediate_scope: str = "candidates",
        mediate_neuron_chunk: int = 16,
        random_baseline_count: int = 20,
        probe_path: Optional[str] = None,
        neuron_family: str = "entropy",
        unigram_path: Optional[str] = None,
        seed: int = 42,
        **kwargs,
    ):
        valid_modes = ("identify", "mediate", "overlap", "full")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")
        if selection not in ("top_n", "top_percent"):
            raise ValueError(f"selection must be 'top_n' or 'top_percent', got '{selection}'")
        if mediate_scope not in ("candidates", "all"):
            raise ValueError(f"mediate_scope must be 'candidates' or 'all', got '{mediate_scope}'")
        if neuron_family not in ("entropy", "frequency"):
            raise ValueError(
                f"neuron_family must be 'entropy' or 'frequency', got '{neuron_family}'"
            )

        self._name = name
        self.description = description
        self.mode = mode
        self.selection = selection
        self.top_n = top_n
        self.top_percent = top_percent
        self.k_null = k_null
        self.logit_chunk_size = logit_chunk_size
        self.fold_final_norm = fold_final_norm
        self.corpus_text = corpus_text
        self.n_tokens = n_tokens
        self.seq_len = seq_len
        self.mediate_sequences = mediate_sequences
        self.mediate_scope = mediate_scope
        self.mediate_neuron_chunk = mediate_neuron_chunk
        self.random_baseline_count = random_baseline_count
        self.probe_path = probe_path
        self.neuron_family = neuron_family
        self.unigram_path = unigram_path
        self.seed = seed

    @property
    def name(self) -> str:
        return self._name

    def validate_backend(self, backend: InferenceBackend) -> None:
        if getattr(backend, "hook_manager", None) is None:
            raise ValueError(
                "confidence_regulation requires the transformers backend with hook support"
            )

    # ------------------------------------------------------------------
    # weight access helpers
    # ------------------------------------------------------------------

    def _get_unembedding(self, backend: InferenceBackend, folded: bool = True) -> torch.Tensor:
        """Return ``W_U`` as a ``(vocab, d_model)`` fp32 CPU tensor.

        With ``folded=True`` (identify mode) the final-norm gain is folded in,
        matching the paper's weight preprocessing (TransformerLens
        ``fold_ln=True``): the null space that matters for entropy neurons is
        that of the effective unembedding.

        With ``folded=False`` (mediate mode) the raw matrix is returned --
        the mediation math applies the norm affine explicitly via
        ``_apply_norm``, so folding here would apply gamma twice.
        """
        w_u = backend.model.get_output_embeddings().weight.detach().float().cpu()
        if folded and self.fold_final_norm:
            gamma = self._get_final_norm_gain(backend)
            if gamma is not None:
                w_u = w_u * gamma.unsqueeze(0)
        return w_u

    @staticmethod
    def _get_final_norm_gain(backend: InferenceBackend):
        """Resolve the final normalization gain ``gamma`` (d_model,) or None."""
        model = backend.model
        containers = [
            model,
            getattr(model, "model", None),
            getattr(model, "transformer", None),
        ]
        for container in containers:
            if container is None:
                continue
            for attr in ("norm", "ln_f", "final_layernorm", "final_layer_norm"):
                mod = getattr(container, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    return mod.weight.detach().float().cpu()
        return None

    def _get_final_w_out(self, backend: InferenceBackend) -> torch.Tensor:
        """Return final-layer ``W_out`` as a ``(d_model, d_mlp)`` fp32 CPU tensor.

        Columns are the per-neuron output weights ``w_out^(i)``. Handles both
        parameterizations: ``nn.Linear`` stores ``(d_model, d_mlp)`` (columns
        are neurons) while GPT-2-style ``Conv1D`` stores ``(d_mlp, d_model)``
        (rows are neurons).
        """
        final_layer = backend.hook_manager.num_layers - 1
        w_down = backend.hook_manager.get_mlp_down_proj_module(final_layer).weight
        w_down = w_down.detach().float().cpu()
        if w_down.shape[0] != backend.model.get_input_embeddings().weight.shape[1]:
            w_down = w_down.T
        return w_down

    # ------------------------------------------------------------------
    # identify
    # ------------------------------------------------------------------

    def _compute_logit_vars(self, w_u: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
        """LogitVar per neuron (Eq. 3), chunked over neurons to bound memory.

        ``logit_var_i = Var_vocab( W_U w_out_i / (col_norms(W_U) * ||w_out_i||) )``
        """
        d_mlp = w_out.shape[1]
        wu_col_norms = w_u.norm(dim=1)  # (vocab,)
        w_norms = w_out.norm(dim=0)  # (d_mlp,)
        logit_vars = torch.empty(d_mlp, dtype=torch.float32)
        chunk = max(1, self.logit_chunk_size)
        for start in range(0, d_mlp, chunk):
            sl = slice(start, min(start + chunk, d_mlp))
            proj = w_u @ w_out[:, sl]  # (vocab, c)
            denom = wu_col_norms.unsqueeze(1) * w_norms[sl].unsqueeze(0)
            logit_vars[sl] = (proj / denom).var(dim=0)
        return logit_vars

    def _compute_rho(self, w_u: torch.Tensor, w_out: torch.Tensor):
        """Null-space fraction rho per neuron plus diagnostics.

        The bottom-k right singular vectors of ``W_U`` are obtained from the
        eigendecomposition of the small Gram matrix ``W_U^T W_U``
        (``d_model x d_model``) instead of a full SVD of the tall
        ``(vocab, d_model)`` matrix; the subspaces coincide up to sign.
        """
        d_model = w_u.shape[1]
        k = self.k_null if self.k_null is not None else max(1, round(0.01 * d_model))
        k = min(k, d_model)
        gram = w_u.T @ w_u
        eigvals, eigvecs = torch.linalg.eigh(gram)  # ascending eigenvalues
        v_bottom = eigvecs[:, :k]  # (d_model, k)
        rho = (v_bottom.T @ w_out).norm(dim=0) / w_out.norm(dim=0)
        diag = {
            "k_null": k,
            "bottom_eigval_min": float(eigvals[:k].min()),
            "bottom_eigval_max": float(eigvals[:k].max()),
            "median_eigval": float(eigvals.median()),
        }
        return rho, diag

    def _select_neurons(self, rho: torch.Tensor) -> List[int]:
        """Rank neurons by rho descending (authors' released-code criterion)."""
        if self.selection == "top_percent":
            n = max(1, int(self.top_percent * rho.numel()))
        else:
            n = min(self.top_n, rho.numel())
        return torch.topk(rho, n).indices.tolist()

    def _identify_arrays(self, backend: InferenceBackend) -> Dict[str, Any]:
        """Compute all identify-mode quantities once; shared by all modes."""
        w_u = self._get_unembedding(backend)
        w_out = self._get_final_w_out(backend)
        final_layer = backend.hook_manager.num_layers - 1

        norms = w_out.norm(dim=0)
        logit_vars = self._compute_logit_vars(w_u, w_out)
        rho, svd_diag = self._compute_rho(w_u, w_out)
        selected = self._select_neurons(rho)

        sel_norms = norms[selected]
        sel_lv = logit_vars[selected]
        sel_rho = rho[selected]

        summary = {
            "final_layer": final_layer,
            "d_model": w_u.shape[1],
            "d_mlp": w_out.shape[1],
            **svd_diag,
            "selected_count": len(selected),
            "selected_mean_norm": float(sel_norms.mean()),
            "selected_mean_logit_var": float(sel_lv.mean()),
            "selected_mean_rho": float(sel_rho.mean()),
            "all_mean_norm": float(norms.mean()),
            "all_mean_logit_var": float(logit_vars.mean()),
            "all_mean_rho": float(rho.mean()),
            "pearson_rho_norm": self._pearson(rho, norms),
            "pearson_rho_logit_var": self._pearson(rho, -logit_vars),
        }
        detail = [
            {
                "layer": final_layer,
                "index": int(i),
                "norm": float(norms[i]),
                "logit_var": float(logit_vars[i]),
                "rho": float(rho[i]),
            }
            for i in selected
        ]
        return {
            "w_out": w_out,
            "norms": norms,
            "logit_vars": logit_vars,
            "rho": rho,
            "score": rho,
            "score_name": "rho",
            "svd_diag": svd_diag,
            "final_layer": final_layer,
            "selected": selected,
            "summary": summary,
            "detail": detail,
        }

    # ------------------------------------------------------------------
    # token-frequency family (paper Sec. 4)
    # ------------------------------------------------------------------

    def _get_v_freq(self, backend: InferenceBackend) -> torch.Tensor:
        """Centered log-unigram direction ``v_freq`` over the vocabulary.

        From ``unigram_path`` (.npy of unigram counts/probs) when given --
        the authors ship OpenWebText counts -- otherwise derived from the
        experiment corpus text (config-gated approximation).
        """
        emb = backend.model.get_output_embeddings().weight
        vocab = emb.shape[0]
        if self.unigram_path:
            import numpy as np

            counts = torch.from_numpy(np.load(self.unigram_path).astype("float64"))
            if counts.numel() != vocab:
                raise ValueError(f"unigram file has {counts.numel()} entries, vocab is {vocab}")
        else:
            ids = torch.cat(
                [
                    torch.tensor(
                        backend.tokenizer(self._corpus_text_or_default())["input_ids"],
                        dtype=torch.long,
                    )
                    for _ in [0]
                ]
            )
            reps = -(-self.n_tokens // ids.numel())
            stream = ids.repeat(reps)[: self.n_tokens]
            counts = torch.bincount(stream, minlength=vocab).double()
        p = counts / counts.sum()
        log_p = torch.log(p.clamp_min(1e-12))
        return (log_p - log_p.mean()).float()

    def _corpus_text_or_default(self) -> str:
        return self.corpus_text if self.corpus_text else self._DEFAULT_CORPUS

    @staticmethod
    def _compute_freq_scores(
        w_u: torch.Tensor, w_out: torch.Tensor, v_freq: torch.Tensor
    ) -> torch.Tensor:
        """Signed cosine between each neuron's direct logit write and v_freq.

        The write of neuron i onto vocabulary space is ``W_U w_out^(i)``;
        centering it removes the softmax-invariant constant component. The
        sign encodes direction: positive boosts frequent tokens, negative
        suppresses them.
        """
        writes = w_u @ w_out  # (vocab, d_mlp)
        writes = writes - writes.mean(dim=0, keepdim=True)
        vf = v_freq - v_freq.mean()
        denom = writes.norm(dim=0) * vf.norm()
        return (writes.T @ vf) / denom.clamp_min(1e-12)

    def _identify_frequency_arrays(self, backend: InferenceBackend) -> Dict[str, Any]:
        """Frequency-family counterpart of :meth:`_identify_arrays`."""
        w_u = self._get_unembedding(backend)
        w_out = self._get_final_w_out(backend)
        final_layer = backend.hook_manager.num_layers - 1
        v_freq = self._get_v_freq(backend)

        norms = w_out.norm(dim=0)
        scores = self._compute_freq_scores(w_u, w_out, v_freq)
        ranked = scores.abs()
        selected = self._select_neurons(ranked)

        summary = {
            "final_layer": final_layer,
            "d_model": w_u.shape[1],
            "d_mlp": w_out.shape[1],
            "neuron_family": "frequency",
            "score_name": "abs_cosine(write, v_freq)",
            "selected_count": len(selected),
            "selected_mean_norm": float(norms[selected].mean()),
            "all_mean_norm": float(norms.mean()),
            "selected_mean_abs_score": float(ranked[selected].mean()),
            "all_mean_abs_score": float(ranked.mean()),
            "selected_positive_sign_count": int((scores[selected] > 0).sum()),
        }
        detail = [
            {
                "layer": final_layer,
                "index": int(i),
                "norm": float(norms[i]),
                "freq_cosine": float(scores[i]),
            }
            for i in selected
        ]
        return {
            "w_out": w_out,
            "norms": norms,
            "v_freq": v_freq,
            "score": ranked,
            "signed_score": scores,
            "score_name": "abs_cosine(write, v_freq)",
            "final_layer": final_layer,
            "selected": selected,
            "summary": summary,
            "detail": detail,
        }

    def _identify_dispatch(self, backend: InferenceBackend) -> Dict[str, Any]:
        if self.neuron_family == "frequency":
            return self._identify_frequency_arrays(backend)
        return self._identify_arrays(backend)

    @staticmethod
    def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a - a.mean()
        b = b - b.mean()
        denom = a.norm() * b.norm()
        return float((a @ b) / denom) if denom > 0 else 0.0

    def _run_identify(self, backend: InferenceBackend) -> ExperimentResult:
        ident = self._identify_dispatch(backend)
        s = ident["summary"]
        metrics = {
            "mode": "identify",
            "fold_final_norm": self.fold_final_norm,
            "neuron_family": self.neuron_family,
            **s,
        }

        print("\n" + "=" * 66)
        print(f"CONFIDENCE REGULATION -- IDENTIFY ({self.neuron_family})")
        print("=" * 66)
        print(f"Final layer      : {s['final_layer']} (d_model={s['d_model']}, d_mlp={s['d_mlp']})")
        if self.neuron_family == "entropy":
            print(
                f"Null space dim k : {s['k_null']} "
                f"(bottom eig {s['bottom_eigval_max']:.2e} vs median {s['median_eigval']:.2e})"
            )
            print(f"rho   selected   : {s['selected_mean_rho']:.4f} | all {s['all_mean_rho']:.4f}")
            print(
                f"logitVar selected: {s['selected_mean_logit_var']:.3e} | "
                f"all {s['all_mean_logit_var']:.3e}"
            )
            print(f"pearson(rho, norm)          : {s['pearson_rho_norm']:+.3f}")
            print(f"pearson(rho, -logitVar)     : {s['pearson_rho_logit_var']:+.3f}")
        else:
            print("Score            : |cosine(write, v_freq)|")
            print(
                f"|cos| selected   : {s['selected_mean_abs_score']:.4f} | "
                f"all {s['all_mean_abs_score']:.4f}"
            )
            print(
                f"sign split       : {s['selected_positive_sign_count']} positive "
                f"(boost frequent) / {s['selected_count']} total"
            )
        print(f"Selected         : {s['selected_count']} neurons ({self.selection})")
        print(f"norm  selected   : {s['selected_mean_norm']:.3f} | all {s['all_mean_norm']:.3f}")
        print("=" * 66)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="n/a",
            metrics=metrics,
            metadata={
                "description": self.description,
                "fold_final_norm": self.fold_final_norm,
                "neuron_family": self.neuron_family,
                "selection": self.selection,
                "top_n": self.top_n,
                "top_percent": self.top_percent,
                "seed": self.seed,
                "selected_neurons": ident["detail"],
            },
        )

    # ------------------------------------------------------------------
    # mediate
    # ------------------------------------------------------------------

    def _build_corpus_batches(self, backend: InferenceBackend) -> torch.Tensor:
        """Tokenize the corpus into an ``(n_sequences, seq_len)`` CPU tensor."""
        tokenizer = backend.tokenizer
        text = self.corpus_text if self.corpus_text else self._DEFAULT_CORPUS
        ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        ids = torch.tensor(ids, dtype=torch.long)
        reps = -(-self.n_tokens // ids.numel())
        stream = ids.repeat(reps)
        n_seq = max(1, stream.numel() // self.seq_len)
        return stream[: n_seq * self.seq_len].view(n_seq, self.seq_len)

    def _resolve_final_norm_module(self, backend: InferenceBackend):
        """Return the final normalization module (or None)."""
        model = backend.model
        containers = [model, getattr(model, "model", None), getattr(model, "transformer", None)]
        for container in containers:
            if container is None:
                continue
            for attr in ("norm", "ln_f", "final_layernorm", "final_layer_norm"):
                mod = getattr(container, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    return mod
        return None

    def _calibrate_norm(self, norm_mod, x: torch.Tensor) -> Dict[str, Any]:
        """Numerically calibrate how the final norm applies its affine gain.

        Freezing the norm denominator (DE_LN) requires reconstructing the norm
        output manually. Model families differ in gain semantics:
        ``y = normed * w + b`` (LayerNorm), ``y = normed * w`` (RMSNorm), or
        ``y = normed * (1 + w)`` (Gemma RMSNorm). We detect the variant by
        comparing against a real forward output rather than trusting class
        names.
        """
        eps = getattr(norm_mod, "variance_epsilon", None)
        if eps is None:
            eps = getattr(norm_mod, "eps", 1e-5)
        weight = norm_mod.weight.detach().float().cpu()
        bias = (
            norm_mod.bias.detach().float().cpu()
            if getattr(norm_mod, "bias", None) is not None
            else None
        )
        x0 = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            y0 = norm_mod(x0.to(norm_mod.weight.device)).float().cpu()
        is_rms = bias is None and ("rms" in type(norm_mod).__name__.lower())
        if is_rms:
            scale0 = (x0.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
            normed0 = x0 / scale0
        else:
            scale0 = (x0.var(dim=-1, unbiased=False, keepdim=True) + eps).sqrt()
            normed0 = (x0 - x0.mean(dim=-1, keepdim=True)) / scale0
        tol = 1e-3 * float(y0.abs().max()) + 1e-6
        candidates = {"affine": normed0 * weight + (bias if bias is not None else 0)}
        if bias is None:
            candidates["gemma"] = normed0 * (1 + weight)
        for name, recon in candidates.items():
            if float((recon - y0).abs().max()) < tol:
                return {
                    "eps": eps,
                    "weight": weight,
                    "bias": bias,
                    "gain_mode": name,
                    "is_rms": is_rms,
                }
        raise ValueError(f"could not reproduce final-norm outputs for {type(norm_mod).__name__}")

    def _apply_norm(self, x: torch.Tensor, cfg: Dict[str, Any], frozen_scale=None):
        """Apply the calibrated norm formula, optionally freezing the scale."""
        eps = cfg["eps"]
        if cfg["is_rms"]:
            mean = torch.zeros_like(x)
            scale = (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
        else:
            mean = x.mean(dim=-1, keepdim=True)
            scale = (x.var(dim=-1, unbiased=False, keepdim=True) + eps).sqrt()
        if frozen_scale is not None:
            scale = frozen_scale.view(*scale.shape).to(x.dtype).expand_as(scale)
        normed = (x - mean) / scale
        if cfg["gain_mode"] == "gemma":
            return normed * (1 + cfg["weight"])
        return normed * cfg["weight"] + (cfg["bias"] if cfg["bias"] is not None else 0)

    def _token_loss(self, logits: torch.Tensor, targets_flat: torch.Tensor) -> torch.Tensor:
        """Per-position CE loss without materializing log_softmax.

        ``logits`` is ``(n, L, vocab)``; every row shares the same target
        sequence, so the gather index is expanded to match the leading dim
        (a size-1 index would silently gather row 0's logits for all rows).
        """
        tgt = targets_flat.view(1, -1, 1).expand(logits.shape[0], -1, 1)
        gathered = logits.gather(-1, tgt).squeeze(-1)
        lse = logits.logsumexp(dim=-1)
        return -(gathered - lse)

    def _capture_sequences(self, backend: InferenceBackend):
        """Run forward passes caching final residual + final-layer activations.

        Returns (sequences, act_mean, norm_cfg): one dict per sequence with
        tokens/resid/acts, the corpus-mean activation per neuron, and the
        calibrated final-norm formula.
        """
        device = backend.device
        ident = self._identify_dispatch(backend)
        batches = self._build_corpus_batches(backend)
        norm_mod = self._resolve_final_norm_module(backend)
        if norm_mod is None:
            raise ValueError("could not resolve the final normalization module")

        down_mod = backend.hook_manager.get_mlp_down_proj_module(ident["final_layer"])
        captured: Dict[str, torch.Tensor] = {}

        def grab_acts(_mod, inp):
            # down_proj INPUT = post-activation hidden units (d_mlp)
            captured["acts"] = inp[0].detach().float().cpu()

        def grab_resid(_mod, inp):
            captured["resid"] = inp[0].detach().float().cpu()

        handle_down = down_mod.register_forward_pre_hook(grab_acts)
        handle_norm = norm_mod.register_forward_pre_hook(grab_resid)
        sequences: List[Dict[str, torch.Tensor]] = []
        try:
            means_sum = torch.zeros(ident["w_out"].shape[1])
            total_pos = 0
            for b in range(min(self.mediate_sequences, batches.shape[0])):
                tokens = batches[b : b + 1].to(device)
                with torch.no_grad():
                    backend.model(tokens)
                resid, acts = captured["resid"], captured["acts"]
                acts = acts.reshape(resid.shape[0], resid.shape[1], -1)
                sequences.append({"tokens": tokens.cpu(), "resid": resid[0], "acts": acts[0]})
                means_sum += acts.sum(dim=(0, 1))
                total_pos += acts.shape[0] * acts.shape[1]
        finally:
            handle_down.remove()
            handle_norm.remove()
        act_mean = means_sum / total_pos

        sample = sequences[0]["resid"][:1]
        norm_cfg = self._calibrate_norm(norm_mod, sample)
        return ident, sequences, act_mean, norm_cfg

    def _run_mediate(self, backend: InferenceBackend) -> ExperimentResult:
        """Analytic mean-ablation mediation (paper Eqs. 4-6).

        Default scope is candidate neurons only (identified set + random
        same-layer baseline) so the run stays tractable on CPU;
        ``mediate_scope="all"`` sweeps every final-layer neuron like the
        authors do on GPU.
        """
        ident, sequences, act_mean, norm_cfg = self._capture_sequences(backend)
        score = ident["score"]
        selected = ident["selected"]

        rng = torch.Generator().manual_seed(self.seed)
        rand_idx = torch.randperm(score.numel(), generator=rng)[
            : self.random_baseline_count
        ].tolist()
        if self.mediate_scope == "all":
            indices = list(range(score.numel()))
        else:
            indices = sorted(set(selected) | set(rand_idx))

        v_freq = ident.get("v_freq") if self.neuron_family == "frequency" else None
        stats = self._ablate_neurons(backend, sequences, act_mean, norm_cfg, indices, v_freq=v_freq)
        mediated = stats["mediated"]

        sel_rows = [indices.index(i) for i in selected]
        rand_rows = [indices.index(i) for i in rand_idx]

        spearman = self._spearman(score[torch.tensor(indices)], mediated)
        de_label = "de_freq" if v_freq is not None else "de_ln"
        metrics: Dict[str, Any] = {
            **{f"identify_{k}": v for k, v in ident["summary"].items()},
            "mode": "mediate",
            "neuron_family": self.neuron_family,
            "mediate_scope": self.mediate_scope,
            "n_ablated_neurons": len(indices),
            "n_sequences": len(sequences),
            "seq_len": self.seq_len,
            "total_positions": stats["positions"],
            "selected_mean_TE": float(stats["te"][sel_rows].mean()),
            f"selected_mean_{de_label}": float(stats["de"][sel_rows].mean()),
            "selected_mean_mediated": float(mediated[sel_rows].mean()),
            "random_baseline_mean_mediated": float(mediated[rand_rows].mean()),
            "random_baseline_max_mediated": float(mediated[rand_rows].max()),
            "random_baseline_neurons": {
                str(i): float(mediated[indices.index(i)]) for i in rand_idx
            },
            "spearman_score_mediated": spearman,
        }

        order = torch.argsort(mediated, descending=True)
        top_rows = order[: min(5, len(order))].tolist()

        print("\n" + "=" * 66)
        print(f"CONFIDENCE REGULATION -- MEDIATE ({self.neuron_family})")
        print("=" * 66)
        print(f"Scope            : {self.mediate_scope} ({len(indices)} neurons)")
        print(
            f"Sequences        : {len(sequences)} x {self.seq_len} tokens "
            f"({stats['positions']} positions)"
        )
        de_name = "freq-mediated" if v_freq is not None else "ln-mediated"
        print(f"Selected ({len(selected)}): {de_name} = {metrics['selected_mean_mediated']:.3f}")
        print(
            f"Random baseline  : mean {metrics['random_baseline_mean_mediated']:.3f} "
            f"max {metrics['random_baseline_max_mediated']:.3f} "
            f"(R={self.random_baseline_count})"
        )
        print(f"spearman(score, mediated)    : {spearman:+.3f}")
        print(f"Top-5 ablated neurons by {de_name} fraction:")
        for row in top_rows:
            tag = "*" if indices[row] in set(selected) else " "
            print(
                f"  {tag}{indices[row]:5d}  {de_name}={float(mediated[row]):+.3f}  "
                f"TE={float(stats['te'][row]):.4f}  DE={float(stats['de'][row]):.4f}"
            )
        print("=" * 66)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="n/a",
            metrics=metrics,
            metadata={
                "description": self.description,
                "corpus_default": self.corpus_text is None,
                "fold_final_norm": self.fold_final_norm,
                "neuron_family": self.neuron_family,
                "ablated_indices": indices,
                "per_neuron": [
                    {
                        "index": int(indices[row]),
                        "te": float(stats["te"][row]),
                        "de": float(stats["de"][row]),
                        "mediated": float(mediated[row]),
                        "score": float(score[indices[row]]),
                        "is_selected": indices[row] in set(selected),
                    }
                    for row in range(len(indices))
                ],
            },
        )

    def _ablate_neurons(
        self,
        backend: InferenceBackend,
        sequences: List[Dict[str, torch.Tensor]],
        act_mean: torch.Tensor,
        norm_cfg: Dict[str, Any],
        indices: List[int],
        v_freq: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Chunked analytic ablation with per-neuron separable effects.

        For a chunk of c neurons, builds c variants of the ablated final
        residual ``(c, T, d_model)``, applies the calibrated norm formula
        (fresh scale for TE; per-token frozen scale for DE_LN), projects to
        vocabulary space once per variant, and accumulates the absolute
        loss *change* vs the un-ablated baseline per neuron (paper Eq. 5/6
        are differences against the intact forward).

        For the token-frequency family (``v_freq`` given), DE instead
        restores each ablated logit vector's component along ``v_freq`` to
        its pre-ablation value (paper Eq. 7) -- the LayerNorm scale stays
        live in both TE and DE.
        """
        device = backend.device
        w_out_full = self._get_final_w_out(backend).to(device)
        with torch.no_grad():
            w_u = self._get_unembedding(backend, folded=False).to(device)
        emb = backend.model.get_output_embeddings()
        b_u = getattr(emb, "bias", None)
        b_u = b_u.detach().float().to(device) if b_u is not None else None
        vf = v_freq.to(device) if v_freq is not None else None
        vf2 = vf.pow(2).sum() if vf is not None else None

        te_sum = torch.zeros(len(indices))
        de_sum = torch.zeros(len(indices))
        positions = 0
        # Cap the neuron chunk so the (c*T, vocab) logits tensor stays ~<=1 GB.
        T0 = sequences[0]["tokens"].shape[1]
        vocab = self._get_unembedding(backend).shape[0]
        eff_chunk = max(1, min(self.mediate_neuron_chunk, int(2**28 / max(1, T0 * vocab))))
        chunk = eff_chunk
        eps = norm_cfg["eps"]

        with torch.no_grad():
            for seq in sequences:
                tokens = seq["tokens"]
                tgt_flat = tokens[:, 1:].reshape(-1).to(device)
                positions += tgt_flat.numel()
                x = seq["resid"].unsqueeze(0).to(device)  # (1, T, d)
                n = seq["acts"].unsqueeze(0).to(device)  # (1, T, m)
                delta_all = (act_mean.to(device).view(1, 1, -1) - n)[0]  # (T, m)
                if norm_cfg["is_rms"]:
                    scale0 = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
                else:
                    var0 = x.var(-1, unbiased=False, keepdim=True)
                    scale0 = (var0 + eps).sqrt()
                # un-ablated baseline loss for this sequence (Eq. 5 reference)
                base_logits = self._apply_norm(x, norm_cfg) @ w_u.T
                if b_u is not None:
                    base_logits = base_logits + b_u
                base_loss = self._token_loss(
                    base_logits[:, :-1], tokens[:, 1:].reshape(-1).to(device)
                ).to(device)
                comp_base = None
                if vf is not None:
                    # per-position component of baseline logits along v_freq
                    bl = base_logits[:, :-1].reshape(-1, base_logits.shape[-1])
                    comp_base = (bl @ vf) / vf2
                T = x.shape[1]
                for start in range(0, len(indices), chunk):
                    idx = indices[start : start + chunk]
                    deltas = delta_all[:, idx].T  # (c, T)
                    w_c = w_out_full[:, idx].T  # (c, d)
                    x_abl = x + torch.einsum(
                        "ct,cd->ctd", deltas.to(device), w_c
                    )  # (c, T, d); x's batch dim broadcasts over the chunk
                    xf = x_abl.reshape(-1, x_abl.shape[-1])  # (c*T, d)
                    gain = norm_cfg["weight"].to(device)
                    bias = norm_cfg["bias"].to(device) if norm_cfg["bias"] is not None else None

                    def forward_logits(xf_in, scale_frozen=None):
                        if norm_cfg["is_rms"]:
                            mean = torch.zeros_like(xf_in)
                            scale = (xf_in.pow(2).mean(-1, keepdim=True) + eps).sqrt()
                        else:
                            mean = xf_in.mean(-1, keepdim=True)
                            scale = (xf_in.var(-1, unbiased=False, keepdim=True) + eps).sqrt()
                        if scale_frozen is not None:
                            sf = (
                                scale_frozen.reshape(1, -1, 1)
                                .expand(x_abl.shape[0], T, 1)
                                .reshape(-1, 1)
                            )
                            scale = sf
                        normed = (xf_in - mean) / scale
                        if norm_cfg["gain_mode"] == "gemma":
                            normed = normed * (1 + gain)
                        elif bias is not None:
                            normed = normed * gain + bias
                        else:
                            normed = normed * gain
                        logits = normed @ w_u.T
                        if b_u is not None:
                            logits = logits + b_u
                        return logits.view(x_abl.shape[0], T, -1)

                    te_logits = forward_logits(xf)  # fresh scale (normal forward)
                    te_losses = self._token_loss(te_logits[:, :-1], tgt_flat)
                    if vf is not None:
                        # DE_freq: restore v_freq component to its baseline
                        # value (Eq. 7); LayerNorm stays live in TE and DE.
                        abl = te_logits[:, :-1].reshape(x_abl.shape[0], -1, te_logits.shape[-1])
                        comp_abl = torch.einsum("nlv,v->nl", abl, vf) / vf2
                        restored = abl + (comp_base.unsqueeze(0) - comp_abl).unsqueeze(-1) * vf
                        de_losses = self._token_loss(restored, tgt_flat)
                        del abl, restored
                    else:
                        de_logits = forward_logits(xf, scale_frozen=scale0)
                        de_losses = self._token_loss(de_logits[:, :-1], tgt_flat)
                        del de_logits
                    te_sum[start : start + len(idx)] += (
                        (te_losses - base_loss).abs().sum(dim=1).cpu()
                    )
                    de_sum[start : start + len(idx)] += (
                        (de_losses - base_loss).abs().sum(dim=1).cpu()
                    )
                    del x_abl, xf, te_logits, te_losses, de_losses

        te_mean = te_sum / positions
        de_mean = de_sum / positions
        safe_te = te_mean.clamp_min(1e-12)
        mediated = torch.where(te_mean > 1e-12, 1 - de_mean / safe_te, torch.zeros_like(te_mean))
        return {"te": te_mean, "de": de_mean, "mediated": mediated, "positions": positions}

    # ------------------------------------------------------------------
    # overlap
    # ------------------------------------------------------------------

    def _load_probe_neurons(self) -> List[Tuple[int, int]]:
        """Load H-Neuron ``(layer, index)`` pairs from a probe JSON.

        Handles both the legacy format (``neurons: [{layer, index}, ...]``)
        and the current fit format (``fit.h_neurons`` as ``[[l, i], ...]``).
        """
        import json

        if not self.probe_path:
            raise ValueError("probe_path required for overlap mode")
        with open(self.probe_path) as f:
            probe_data = json.load(f)
        if "neurons" in probe_data:
            return [(int(n["layer"]), int(n["index"])) for n in probe_data["neurons"]]
        if "fit" in probe_data and "h_neurons" in probe_data["fit"]:
            h_neurons = probe_data["fit"]["h_neurons"]
            if h_neurons and isinstance(h_neurons[0], (list, tuple)):
                return [(int(layer), int(i)) for layer, i in h_neurons]
            return [(int(n["layer"]), int(n["index"])) for n in h_neurons]
        raise ValueError("probe file missing neurons data")

    def _run_overlap(self, backend: InferenceBackend) -> ExperimentResult:
        """Jaccard overlap between identified entropy neurons and H-Neurons.

        The comparison is restricted to the final layer, where the entropy-
        neuron criterion is defined. Reports the hypergeometric expectation
        under random placement so the observed overlap can be read as an
        enrichment ratio.
        """
        ident = self._identify_dispatch(backend)
        final_layer = ident["final_layer"]
        d_mlp = ident["summary"]["d_mlp"]
        selected_set = set(ident["selected"])
        h_pairs = self._load_probe_neurons()

        h_all_count = len(h_pairs)
        h_final = sorted({i for (layer, i) in set(h_pairs) if layer == final_layer})
        overlap = sorted(set(h_final) & selected_set)

        n_sel = len(selected_set)
        n_h_final = len(h_final)
        expected = n_sel * n_h_final / d_mlp if d_mlp else 0.0
        union_size = n_sel + n_h_final - len(overlap)
        jaccard = len(overlap) / union_size if union_size else 0.0
        enrichment = len(overlap) / expected if expected > 0 else 0.0

        metrics: Dict[str, Any] = {
            **{f"identify_{k}": v for k, v in ident["summary"].items()},
            "mode": "overlap",
            "probe_path": self.probe_path,
            "h_neurons_total": h_all_count,
            "h_neurons_in_final_layer": n_h_final,
            "entropy_neuron_count": n_sel,
            "overlap_count": len(overlap),
            "jaccard_final_layer": jaccard,
            "expected_random_overlap": expected,
            "enrichment_observed_over_random": enrichment,
        }

        print("\n" + "=" * 66)
        print("CONFIDENCE REGULATION -- OVERLAP")
        print("=" * 66)
        print(f"Probe                 : {self.probe_path}")
        print(f"H-Neurons total       : {h_all_count} ({n_h_final} in final layer {final_layer})")
        print(f"Entropy neurons       : {n_sel}")
        print(f"Overlap               : {len(overlap)} {sorted(overlap)}")
        print(f"Jaccard (final layer) : {jaccard:.4f}")
        print(f"Expected at random    : {expected:.3f}  ->  enrichment x{enrichment:.2f}")
        print("=" * 66)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="n/a",
            metrics=metrics,
            metadata={
                "description": self.description,
                "h_neurons_final_layer": h_final,
                "entropy_selected": sorted(selected_set),
                "overlap": overlap,
            },
        )

    @staticmethod
    def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
        ra = a.argsort().argsort().float()
        rb = b.argsort().argsort().float()
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = ra.norm() * rb.norm()
        return float((ra @ rb) / denom) if denom > 0 else 0.0

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
        """Run the confidence-regulation experiment in the configured mode."""
        self.validate_backend(backend)
        torch.manual_seed(self.seed)

        if self.mode == "identify":
            return self._run_identify(backend)
        if self.mode == "mediate":
            return self._run_mediate(backend)
        if self.mode == "overlap":
            return self._run_overlap(backend)
        if self.mode == "full":
            self._run_identify(backend)
            self._run_mediate(backend)
            return self._run_overlap(backend)
        raise NotImplementedError(
            f"mode '{self.mode}' is not implemented yet; use identify|mediate|overlap|full"
        )
