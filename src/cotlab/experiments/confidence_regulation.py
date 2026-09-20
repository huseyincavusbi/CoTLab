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


def _hypergeom_sf(k: int, population: int, successes: int, draws: int) -> float:
    """Upper-tail hypergeometric P(X >= k) without a scipy dependency.

    ``population`` = neurons per layer, ``successes`` = entropy neurons,
    ``draws`` = H-Neurons in the layer. Returns 1.0 when undefined (no draws
    or no successes) and clamps ``k`` to the feasible range.
    """
    from math import comb

    n = min(draws, population)
    lo = max(0, n - (population - successes))
    hi = min(successes, n)
    if k <= lo:
        return 1.0
    if k > hi or n == 0 or successes == 0:
        return 0.0
    denom = comb(population, n)
    if denom == 0:
        return float("nan")
    tail = sum(comb(successes, i) * comb(population - successes, n - i) for i in range(k, hi + 1))
    return float(tail / denom)


def _percentile_rank(values: torch.Tensor) -> torch.Tensor:
    """Within-vector percentile rank in [0, 100) via argsort (ties by order)."""
    n = values.numel()
    if n <= 1:
        return torch.zeros(n, dtype=torch.float32)
    order = torch.argsort(values)
    ranks = torch.empty(n, dtype=torch.float32)
    ranks[order] = torch.arange(n, dtype=torch.float32)
    return ranks / n * 100.0


def _norm_containers(model) -> List[Any]:
    """Candidate modules that may hold the final normalization layer.

    Multimodal wrappers nest the language model (e.g. Gemma 3:
    ``model.model.language_model.model.norm``); without resolving these the
    final norm is silently dropped and rho is computed on the raw unembedding.
    """
    containers = [
        model,
        getattr(model, "model", None),
        getattr(model, "transformer", None),
        getattr(model, "gpt_neox", None),
    ]
    for root in list(containers):
        if root is None:
            continue
        lm = getattr(root, "language_model", None)
        if lm is not None:
            containers.append(lm)
            containers.append(getattr(lm, "model", None))
    return [c for c in containers if c is not None]


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
        # norm_logitvar criterion (paper Fig. 2a): keep neurons whose output
        # weight norm is in the top ``norm_percentile_min`` and whose logit
        # variance is in the bottom ``logit_var_percentile_max``.
        norm_percentile_min: float = 99.0,
        logit_var_percentile_max: float = 1.0,
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
        # Forward-engine intervention scale on target activations: 0.0 is the
        # paper's mean-ablation, 1.0 is a no-op, >1 amplifies.
        mediate_alpha: float = 0.0,
        probe_path: Optional[str] = None,
        overlap_layers: str = "final",
        neuron_family: str = "entropy",
        unigram_path: Optional[str] = None,
        mediate_engine: str = "auto",
        layer: Optional[int] = None,
        induction_half_len: int = 100,
        induction_sequences: int = 10,
        seed: int = 42,
        **kwargs,
    ):
        valid_modes = ("identify", "mediate", "overlap", "full", "induction")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")
        if selection not in ("top_n", "top_percent", "norm_logitvar"):
            raise ValueError(
                f"selection must be 'top_n', 'top_percent' or 'norm_logitvar', got '{selection}'"
            )
        if mediate_scope not in ("candidates", "all"):
            raise ValueError(f"mediate_scope must be 'candidates' or 'all', got '{mediate_scope}'")
        if neuron_family not in ("entropy", "frequency"):
            raise ValueError(
                f"neuron_family must be 'entropy' or 'frequency', got '{neuron_family}'"
            )
        if mediate_engine not in ("auto", "analytic", "forward"):
            raise ValueError(
                f"mediate_engine must be auto|analytic|forward, got '{mediate_engine}'"
            )
        if layer is not None and layer < 0:
            raise ValueError(f"layer must be a non-negative int or None, got {layer}")
        if mediate_alpha < 0:
            raise ValueError(f"mediate_alpha must be >= 0, got {mediate_alpha}")
        if overlap_layers not in ("final", "probe", "all"):
            raise ValueError(
                f"overlap_layers must be 'final', 'probe' or 'all', got '{overlap_layers}'"
            )

        self._name = name
        self.description = description
        self.mode = mode
        self.selection = selection
        self.top_n = top_n
        self.top_percent = top_percent
        self.norm_percentile_min = norm_percentile_min
        self.logit_var_percentile_max = logit_var_percentile_max
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
        self.mediate_alpha = mediate_alpha
        self.probe_path = probe_path
        self.overlap_layers = overlap_layers
        self.neuron_family = neuron_family
        self.unigram_path = unigram_path
        self.mediate_engine = mediate_engine
        self.layer = layer
        self.induction_half_len = induction_half_len
        self.induction_sequences = induction_sequences
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
        for container in _norm_containers(backend.model):
            for attr in ("norm", "ln_f", "final_layernorm", "final_layer_norm"):
                mod = getattr(container, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    return mod.weight.detach().float().cpu()
        return None

    def _resolve_layer(self, backend: InferenceBackend) -> int:
        """Effective analysis layer: configured ``layer`` or the final one."""
        num = backend.hook_manager.num_layers
        layer = self.layer if self.layer is not None else num - 1
        if layer >= num:
            raise ValueError(f"layer {layer} out of range for {num}-layer model")
        return layer

    def _is_final_layer(self, backend: InferenceBackend) -> bool:
        return self._resolve_layer(backend) == backend.hook_manager.num_layers - 1

    def _get_w_out(self, backend: InferenceBackend, layer: int) -> torch.Tensor:
        """Return ``W_out`` of ``layer`` as a ``(d_model, d_mlp)`` fp32 CPU tensor.

        Columns are the per-neuron output weights ``w_out^(i)``. Handles both
        parameterizations: ``nn.Linear`` stores ``(d_model, d_mlp)`` (columns
        are neurons) while GPT-2-style ``Conv1D`` stores ``(d_mlp, d_model)``
        (rows are neurons).
        """
        w_down = backend.hook_manager.get_mlp_down_proj_module(layer).weight
        w_down = w_down.detach().float().cpu()
        if w_down.shape[0] != backend.model.get_input_embeddings().weight.shape[1]:
            w_down = w_down.T
        return w_down

    def _get_final_w_out(self, backend: InferenceBackend) -> torch.Tensor:
        """Backward-compatible wrapper: ``W_out`` at the configured layer."""
        return self._get_w_out(backend, self._resolve_layer(backend))

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

    def _select_neurons(
        self,
        rho: torch.Tensor,
        norms: Optional[torch.Tensor] = None,
        logit_vars: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """Select neurons by the configured criterion.

        ``top_n``/``top_percent`` rank by the score passed as ``rho`` (the
        authors' released-code criterion). ``norm_logitvar`` instead matches the
        paper's Fig. 2a heuristic -- high output-weight norm AND low logit
        variance -- and requires ``norms``/``logit_vars``.
        """
        if self.selection == "norm_logitvar":
            if norms is None or logit_vars is None:
                raise ValueError("selection='norm_logitvar' requires norms and logit_vars")
            norm_pct = _percentile_rank(norms)
            lv_pct = _percentile_rank(logit_vars)
            mask = (norm_pct >= self.norm_percentile_min) & (
                lv_pct <= self.logit_var_percentile_max
            )
            return torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if self.selection == "top_percent":
            n = max(1, int(self.top_percent * rho.numel()))
        else:
            n = min(self.top_n, rho.numel())
        return torch.topk(rho, n).indices.tolist()

    @staticmethod
    def _norm_matched_indices(
        norms: torch.Tensor,
        targets: List[int],
        window: float = 0.05,
        exclude: Tuple[int, ...] = (),
        seed: int = 0,
    ) -> List[int]:
        """Random neurons with norms within ``+/- window`` of each target.

        Used as the honest control for weight-norm confounds when intervening on
        a selected set (H-Neurons / entropy neurons).
        """
        import random

        rng = random.Random(seed)
        used = set(targets) | set(exclude)
        out: List[int] = []
        for j in targets:
            lo, hi = norms[j] * (1 - window), norms[j] * (1 + window)
            pool = ((norms >= lo) & (norms <= hi)).nonzero(as_tuple=True)[0].tolist()
            pool = [p for p in pool if p not in used and p not in out]
            if not pool:
                order = torch.argsort((norms - norms[j]).abs()).tolist()
                pool = [p for p in order if p not in used and p not in out][:32]
            if pool:
                pick = rng.choice(pool)
                out.append(pick)
                used.add(pick)
        return sorted(out)

    def _identify_arrays(self, backend: InferenceBackend) -> Dict[str, Any]:
        """Compute all identify-mode quantities once; shared by all modes."""
        w_u = self._get_unembedding(backend)
        layer = self._resolve_layer(backend)
        w_out = self._get_w_out(backend, layer)

        norms = w_out.norm(dim=0)
        logit_vars = self._compute_logit_vars(w_u, w_out)
        rho, svd_diag = self._compute_rho(w_u, w_out)
        selected = self._select_neurons(rho, norms, logit_vars)

        sel_norms = norms[selected]
        sel_lv = logit_vars[selected]
        sel_rho = rho[selected]

        summary = {
            "layer": layer,
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
                "layer": layer,
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
            "layer": layer,
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
            if counts.numel() > vocab:
                raise ValueError(f"unigram file has {counts.numel()} entries, vocab is {vocab}")
            if counts.numel() < vocab:
                # e.g. Pythia pads its vocab (50277 -> 50304); pad with zero
                # counts -- such tokens are never produced by the model.
                counts = torch.cat([counts, torch.zeros(vocab - counts.numel())])
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
        final_layer = self._resolve_layer(backend)
        v_freq = self._get_v_freq(backend)

        norms = w_out.norm(dim=0)
        scores = self._compute_freq_scores(w_u, w_out, v_freq)
        ranked = scores.abs()
        selected = self._select_neurons(ranked)

        summary = {
            "layer": final_layer,
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
            "layer": final_layer,
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
        print(f"Analysis layer   : {s['layer']} (d_model={s['d_model']}, d_mlp={s['d_mlp']})")
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
        for container in _norm_containers(backend.model):
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

        down_mod = backend.hook_manager.get_mlp_down_proj_module(ident["layer"])
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
        norms = ident.get("norms")

        rng = torch.Generator().manual_seed(self.seed)
        rand_idx = torch.randperm(score.numel(), generator=rng)[
            : self.random_baseline_count
        ].tolist()

        # Optional probe arm: intervene on our H-Neurons at the analysis layer
        # plus a norm-matched control group (the honest baseline for norm).
        h_layer: List[int] = []
        if self.probe_path:
            h_layer = sorted({i for lyr, i in self._load_probe_neurons() if lyr == ident["layer"]})
        norm_matched: List[int] = []
        if norms is not None and h_layer:
            norm_matched = self._norm_matched_indices(
                norms, h_layer, exclude=tuple(selected), seed=self.seed
            )

        if self.mediate_scope == "all":
            indices = list(range(score.numel()))
        else:
            indices = sorted(set(selected) | set(rand_idx) | set(h_layer) | set(norm_matched))

        v_freq = ident.get("v_freq") if self.neuron_family == "frequency" else None
        engine = self._resolve_engine(backend)
        if engine == "analytic" and not self._is_final_layer(backend):
            raise ValueError(
                "analytic mediation requires the final layer (its single-pathway "
                f"decomposition is invalid mid-network); configured layer is "
                f"{self._resolve_layer(backend)}. Use mediate_engine='forward' "
                "for non-final layers."
            )
        if engine == "forward":
            stats = self._ablate_neurons_forward(
                backend, sequences, act_mean, indices, alpha=self.mediate_alpha
            )
        else:
            stats = self._ablate_neurons(
                backend, sequences, act_mean, norm_cfg, indices, v_freq=v_freq
            )
        mediated = stats["mediated"]

        sel_rows = [indices.index(i) for i in selected]
        rand_rows = [indices.index(i) for i in rand_idx]
        h_rows = [indices.index(i) for i in h_layer]
        nm_rows = [indices.index(i) for i in norm_matched]

        if mediated is not None:
            spearman = self._spearman(score[torch.tensor(indices)], mediated)
        else:
            spearman = self._spearman(score[torch.tensor(indices)], stats["te"])
        de_label = "de_freq" if v_freq is not None else "de_ln"
        metrics: Dict[str, Any] = {
            **{f"identify_{k}": v for k, v in ident["summary"].items()},
            "mode": "mediate",
            "neuron_family": self.neuron_family,
            "mediate_engine": engine,
            "mediate_scope": self.mediate_scope,
            "mediate_alpha": self.mediate_alpha,
            "n_ablated_neurons": len(indices),
            "n_sequences": len(sequences),
            "seq_len": self.seq_len,
            "total_positions": stats["positions"],
            "selected_mean_TE": float(stats["te"][sel_rows].mean()),
        }
        if mediated is not None:
            metrics.update(
                {
                    f"selected_mean_{de_label}": float(stats["de"][sel_rows].mean()),
                    "selected_mean_mediated": float(mediated[sel_rows].mean()),
                    "random_baseline_mean_mediated": float(mediated[rand_rows].mean()),
                    "random_baseline_max_mediated": float(mediated[rand_rows].max()),
                    "random_baseline_neurons": {
                        str(i): float(mediated[indices.index(i)]) for i in rand_idx
                    },
                    "spearman_score_mediated": spearman,
                }
            )
        else:
            metrics.update(
                {
                    "random_baseline_mean_TE": float(stats["te"][rand_rows].mean()),
                    "spearman_score_TE": spearman,
                    "selected_mean_d_entropy": float(stats["d_entropy"][sel_rows].mean()),
                    "selected_mean_abs_d_entropy": float(stats["abs_d_entropy"][sel_rows].mean()),
                    "selected_mean_flip_rate": float(stats["flip_rate"][sel_rows].mean()),
                    "selected_mean_d_max_prob": float(stats["d_max_prob"][sel_rows].mean()),
                    "random_baseline_mean_d_entropy": float(stats["d_entropy"][rand_rows].mean()),
                    "random_baseline_mean_flip_rate": float(stats["flip_rate"][rand_rows].mean()),
                    "random_baseline_mean_d_max_prob": float(stats["d_max_prob"][rand_rows].mean()),
                }
            )

        group_extra: Dict[str, Any] = {}
        if h_rows:
            group_extra["n_h_neurons_intervened"] = len(h_rows)
            group_extra["h_neuron_mean_TE"] = float(stats["te"][h_rows].mean())
        if nm_rows:
            group_extra["n_norm_matched_intervened"] = len(nm_rows)
            group_extra["norm_matched_mean_TE"] = float(stats["te"][nm_rows].mean())
        if "d_entropy" in stats:
            if h_rows:
                group_extra["h_neuron_mean_d_entropy"] = float(stats["d_entropy"][h_rows].mean())
                group_extra["h_neuron_mean_flip_rate"] = float(stats["flip_rate"][h_rows].mean())
                group_extra["h_neuron_mean_d_max_prob"] = float(stats["d_max_prob"][h_rows].mean())
            if nm_rows:
                group_extra["norm_matched_mean_d_entropy"] = float(
                    stats["d_entropy"][nm_rows].mean()
                )
                group_extra["norm_matched_mean_flip_rate"] = float(
                    stats["flip_rate"][nm_rows].mean()
                )
                group_extra["norm_matched_mean_d_max_prob"] = float(
                    stats["d_max_prob"][nm_rows].mean()
                )
        metrics.update(group_extra)

        if mediated is not None:
            order = torch.argsort(mediated, descending=True)
        else:
            order = torch.argsort(stats["te"], descending=True)
        top_rows = order[: min(5, len(order))].tolist()

        print("\n" + "=" * 66)
        print(f"CONFIDENCE REGULATION -- MEDIATE ({self.neuron_family}, engine={engine})")
        print("=" * 66)
        print(f"Scope            : {self.mediate_scope} ({len(indices)} neurons)")
        print(
            f"Sequences        : {len(sequences)} x {self.seq_len} tokens "
            f"({stats['positions']} positions)"
        )
        if mediated is not None:
            de_name = "freq-mediated" if v_freq is not None else "ln-mediated"
            print(
                f"Selected ({len(selected)}): {de_name} = {metrics['selected_mean_mediated']:.3f}"
            )
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
        else:
            # forward engine: causal effect only (post-FFN-norm architectures)
            sel_te = float(stats["te"][sel_rows].mean())
            rand_te = float(stats["te"][rand_rows].mean())
            reason = (
                "post-FFN-norm architecture"
                if self._has_post_ffn_norm(backend)
                else f"non-final layer {self._resolve_layer(backend)}"
            )
            print(f"Engine           : forward ({reason}, alpha={self.mediate_alpha})")
            print(f"Selected ({len(selected)}): mean |dLoss| = {sel_te:.4f}")
            print(f"Random baseline  : mean {rand_te:.4f} (R={self.random_baseline_count})")
            print(f"spearman(score, TE)          : {spearman:+.3f}")
            print(
                f"Confidence sign. : selected dH={metrics['selected_mean_d_entropy']:+.4f} "
                f"flip={metrics['selected_mean_flip_rate']:.3f} | "
                f"random dH={metrics['random_baseline_mean_d_entropy']:+.4f} "
                f"flip={metrics['random_baseline_mean_flip_rate']:.3f}"
            )
            print("Top-5 ablated neurons by causal effect (|dLoss|):")
            for row in top_rows:
                tag = "*" if indices[row] in set(selected) else " "
                print(f"  {tag}{indices[row]:5d}  |dLoss|={float(stats['te'][row]):.4f}")
        print("=" * 66)

        per_neuron_de = (
            [float(stats["de"][row]) for row in range(len(indices))]
            if mediated is not None
            else None
        )
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
                "layer": ident["layer"],
                "is_final_layer": self._is_final_layer(backend),
                "mediate_engine": engine,
                "ablated_indices": indices,
                "per_neuron": [
                    {
                        "index": int(indices[row]),
                        "te": float(stats["te"][row]),
                        "de": per_neuron_de[row] if per_neuron_de else None,
                        "mediated": (float(mediated[row]) if mediated is not None else None),
                        "score": float(ident.get("signed_score", score)[indices[row]]),
                        "is_selected": indices[row] in set(selected),
                        "is_h_neuron": indices[row] in set(h_layer),
                        "is_norm_matched": indices[row] in set(norm_matched),
                        **(
                            {
                                "d_entropy": float(stats["d_entropy"][row]),
                                "abs_d_entropy": float(stats["abs_d_entropy"][row]),
                                "flip_rate": float(stats["flip_rate"][row]),
                                "d_max_prob": float(stats["d_max_prob"][row]),
                            }
                            if mediated is None
                            else {}
                        ),
                    }
                    for row in range(len(indices))
                ],
            },
        )

    @staticmethod
    def _has_post_ffn_norm(backend: InferenceBackend) -> bool:
        """True for architectures whose MLP output is renormalized before the
        residual add (Gemma 2/3, MedGemma). There the analytic shortcut
        ``x' = x + dn * w_out`` on the cached final residual is invalid --
        the write is entangled with all other units through that norm."""
        layer = backend.hook_manager.get_layer_module(0)
        return hasattr(layer, "post_feedforward_layernorm")

    def _resolve_engine(self, backend: InferenceBackend) -> str:
        if self.mediate_engine != "auto":
            return self.mediate_engine
        return "forward" if self._has_post_ffn_norm(backend) else "analytic"

    @staticmethod
    def _distribution_stats(
        logits: torch.Tensor, logp: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Per-position output-distribution statistics for the causal signature.

        Returns entropy, max probability, top1-top2 margin and argmax. These let
        an intervention be read as confidence-regulation (entropy moves, argmax
        does not) versus a direct change to the prediction. Pass an already
        computed ``logp`` (log-softmax of ``logits``) to avoid a second full-vocab
        log-softmax pass per forward.
        """
        if logp is None:
            logp = torch.log_softmax(logits.float(), dim=-1)
        top2 = torch.topk(logp, 2, dim=-1).values
        max_prob = top2[..., 0].exp()
        return {
            "entropy": -(logp.exp() * logp).sum(-1),
            "max_prob": max_prob,
            "margin": max_prob - top2[..., 1].exp(),
            "argmax": logp.argmax(-1),
        }

    def _ablate_neurons_forward(
        self,
        backend: InferenceBackend,
        sequences: List[Dict[str, torch.Tensor]],
        act_mean: torch.Tensor,
        indices: List[int],
        alpha: float = 0.0,
    ) -> Dict[str, Any]:
        """Causal intervention via real forwards, batched over neurons.

        ``alpha == 0`` replaces each target activation with its corpus mean (the
        paper's ablation); ``alpha > 1`` amplifies it by that factor. Required
        for architectures with a post-FFN norm (Gemma 2/3 family) where no
        analytic shortcut exists. A single forward passes ``c`` copies of each
        sequence, one copy per intervened neuron (batch row), so the run scales
        with ``ceil(n_neurons / c) * n_sequences`` forwards of ``(c, T)`` instead
        of one ``(1, T)`` forward per neuron -- much larger matmuls, far better
        utilization. On CPU fp32 this is exactly equivalent to per-neuron
        sequential runs (validated path).

        Reports the total causal effect (mean |dLoss| per position) plus the
        confidence signature: signed change in output entropy, argmax-flip rate
        and change in max probability. The LN-mediated fraction is not defined
        on these architectures.
        """
        device = backend.device
        mod = backend.hook_manager.get_mlp_down_proj_module(self._resolve_layer(backend))
        n = len(indices)
        acc = {
            k: torch.zeros(n)
            for k in ("te", "d_entropy", "abs_d_entropy", "flip_rate", "d_max_prob")
        }
        counter = {"positions": 0}
        chunk = max(1, self.mediate_neuron_chunk)

        def intervene(m, inp, neuron_cols):
            new = inp[0].clone()
            for j, col in enumerate(neuron_cols):
                if alpha == 0.0:
                    new[j, :, col] = act_mean[col].to(inp[0].device)
                else:
                    new[j, :, col] = new[j, :, col] * alpha
            return (new,) + tuple(inp[1:])

        with torch.no_grad():
            base_loss, base_ent, base_maxp, base_arg = [], [], [], []
            for seq in sequences:
                tokens = seq["tokens"].to(device)
                out = backend.model(tokens)
                logits = out.logits.float()[:, :-1]
                lp = torch.log_softmax(logits, dim=-1)
                base_loss.append(
                    -lp.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1).reshape(-1).cpu()
                )
                st = self._distribution_stats(logits, logp=lp)
                base_ent.append(st["entropy"].reshape(-1).cpu())
                base_maxp.append(st["max_prob"].reshape(-1).cpu())
                base_arg.append(st["argmax"].reshape(-1).cpu())
                counter["positions"] += base_loss[-1].numel()
            base_loss = torch.cat(base_loss)
            base_ent = torch.cat(base_ent)
            base_maxp = torch.cat(base_maxp)
            base_arg = torch.cat(base_arg)

            for start in range(0, n, chunk):
                rows = list(range(start, min(start + chunk, n)))
                neuron_cols = [indices[r] for r in rows]
                c = len(rows)

                def hook(m, inp, _cols=neuron_cols):
                    return intervene(m, inp, _cols)

                h = mod.register_forward_pre_hook(hook)
                try:
                    loss_rows, ent_rows, maxp_rows, arg_rows = [], [], [], []
                    for seq in sequences:
                        tokens = seq["tokens"].to(device)
                        inp = tokens.repeat(c, 1)
                        out = backend.model(inp)
                        logits = out.logits.float()[:, :-1]
                        lp = torch.log_softmax(logits, dim=-1)
                        loss_rows.append(-lp.gather(-1, inp[:, 1:].unsqueeze(-1)).squeeze(-1))
                        st = self._distribution_stats(logits, logp=lp)
                        ent_rows.append(st["entropy"])
                        maxp_rows.append(st["max_prob"])
                        arg_rows.append(st["argmax"])
                    loss = torch.cat(loss_rows, dim=1).cpu()
                    ent = torch.cat(ent_rows, dim=1).cpu()
                    maxp = torch.cat(maxp_rows, dim=1).cpu()
                    arg = torch.cat(arg_rows, dim=1).cpu()
                finally:
                    h.remove()

                d_ent = ent - base_ent
                acc["te"][start : start + c] += (loss - base_loss).abs().sum(dim=1)
                acc["d_entropy"][start : start + c] += d_ent.sum(dim=1)
                acc["abs_d_entropy"][start : start + c] += d_ent.abs().sum(dim=1)
                acc["flip_rate"][start : start + c] += (arg != base_arg).float().sum(dim=1)
                acc["d_max_prob"][start : start + c] += (maxp - base_maxp).sum(dim=1)

        pos = max(1, counter["positions"])
        out = {k: acc[k] / pos for k in acc}
        out["positions"] = counter["positions"]
        out["de"] = torch.zeros(n)
        out["mediated"] = None
        return out

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
        # Gemma-2-style final-logit softcapping: applied by the model after
        # the lm_head matmul; must be reproduced or analytic losses diverge.
        model_cfg = getattr(backend.model, "config", None)
        softcap = getattr(model_cfg, "final_logit_softcapping", None)
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
                if softcap is not None:
                    base_logits = torch.tanh(base_logits / softcap) * softcap
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
                        if softcap is not None:
                            logits = torch.tanh(logits / softcap) * softcap
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

        ``overlap_layers='final'`` compares only the configured analysis layer
        (paper-faithful, back-compatible). ``'probe'``/``'all'`` compare every
        layer that hosts an H-Neuron (or every layer), emitting per-layer and
        pooled overlap with hypergeometric p-values. The entropy criterion is
        defined w.r.t. the final ``W_U``, so mid-layer descriptors are candidate
        features rather than the operating mechanism.
        """
        if self.overlap_layers == "final":
            return self._run_overlap_single_layer(backend)
        return self._run_overlap_multi_layer(backend)

    def _run_overlap_single_layer(self, backend: InferenceBackend) -> ExperimentResult:
        ident = self._identify_dispatch(backend)
        analysis_layer = ident["layer"]
        d_mlp = ident["summary"]["d_mlp"]
        selected_set = set(ident["selected"])
        h_pairs = self._load_probe_neurons()

        h_all_count = len(h_pairs)
        h_final = sorted({i for (l_, i) in set(h_pairs) if l_ == analysis_layer})
        overlap = sorted(set(h_final) & selected_set)

        n_sel = len(selected_set)
        n_h_final = len(h_final)
        expected = n_sel * n_h_final / d_mlp if d_mlp else 0.0
        union_size = n_sel + n_h_final - len(overlap)
        jaccard = len(overlap) / union_size if union_size else 0.0
        enrichment = len(overlap) / expected if expected > 0 else 0.0
        hypergeom_p = _hypergeom_sf(len(overlap), d_mlp, n_sel, n_h_final)

        metrics: Dict[str, Any] = {
            **{f"identify_{k}": v for k, v in ident["summary"].items()},
            "mode": "overlap",
            "overlap_layers": "final",
            "probe_path": self.probe_path,
            "h_neurons_total": h_all_count,
            "h_neurons_in_analysis_layer": n_h_final,
            "entropy_neuron_count": n_sel,
            "overlap_count": len(overlap),
            "jaccard_analysis_layer": jaccard,
            "expected_random_overlap": expected,
            "enrichment_observed_over_random": enrichment,
            "hypergeom_p": hypergeom_p,
        }

        print("\n" + "=" * 66)
        print("CONFIDENCE REGULATION -- OVERLAP")
        print("=" * 66)
        print(f"Probe                 : {self.probe_path}")
        print(
            f"H-Neurons total       : {h_all_count} "
            f"({n_h_final} in analysis layer {analysis_layer})"
        )
        print(f"Entropy neurons       : {n_sel}")
        print(f"Overlap               : {len(overlap)} {sorted(overlap)}")
        print(f"Jaccard (final layer) : {jaccard:.4f}")
        print(f"Expected at random    : {expected:.3f}  ->  enrichment x{enrichment:.2f}")
        print(f"Hypergeometric p      : {hypergeom_p:.3e}")
        print("=" * 66)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="n/a",
            metrics=metrics,
            metadata={
                "description": self.description,
                "h_neurons_analysis_layer": h_final,
                "entropy_selected": sorted(selected_set),
                "overlap": overlap,
            },
        )

    def _overlap_one_layer(self, backend, w_u, layer, h_indices) -> Dict[str, Any]:
        w_out = self._get_w_out(backend, layer)
        d_mlp = w_out.shape[1]
        norms = w_out.norm(dim=0)
        rho, _ = self._compute_rho(w_u, w_out)
        logit_vars = (
            self._compute_logit_vars(w_u, w_out) if self.selection == "norm_logitvar" else None
        )
        selected = set(self._select_neurons(rho, norms, logit_vars))
        h_set = set(h_indices)
        overlap = sorted(h_set & selected)
        n_sel, n_h = len(selected), len(h_set)
        expected = n_sel * n_h / d_mlp if d_mlp else 0.0
        union = n_sel + n_h - len(overlap)
        return {
            "layer": int(layer),
            "n_neurons_in_layer": int(d_mlp),
            "h_neurons": n_h,
            "entropy_neurons": n_sel,
            "overlap_count": len(overlap),
            "overlap_neurons": overlap,
            "expected_random_overlap": expected,
            "enrichment_observed_over_random": (len(overlap) / expected) if expected else 0.0,
            "jaccard": (len(overlap) / union) if union else 0.0,
            "hypergeom_p": _hypergeom_sf(len(overlap), d_mlp, n_sel, n_h),
        }

    def _run_overlap_multi_layer(self, backend: InferenceBackend) -> ExperimentResult:
        if self.neuron_family != "entropy":
            raise ValueError("overlap_layers != 'final' supports neuron_family='entropy' only")
        w_u = self._get_unembedding(backend)
        h_pairs = self._load_probe_neurons()
        if self.overlap_layers == "probe":
            layers = sorted({lyr for lyr, _ in h_pairs})
        else:
            layers = list(range(backend.hook_manager.num_layers))

        by_layer: Dict[int, List[int]] = {lyr: [] for lyr in layers}
        for lyr, i in h_pairs:
            if lyr in by_layer:
                by_layer[lyr].append(i)

        per_layer = [
            self._overlap_one_layer(backend, w_u, layer, sorted(set(by_layer[layer])))
            for layer in layers
        ]
        pop = sum(r["n_neurons_in_layer"] for r in per_layer)
        tot_sel = sum(r["entropy_neurons"] for r in per_layer)
        tot_h = sum(r["h_neurons"] for r in per_layer)
        tot_ov = sum(r["overlap_count"] for r in per_layer)
        pooled_expected = tot_sel * tot_h / pop if pop else 0.0
        enrichment = (tot_ov / pooled_expected) if pooled_expected else 0.0
        metrics: Dict[str, Any] = {
            "mode": "overlap",
            "overlap_layers": self.overlap_layers,
            "probe_path": self.probe_path,
            "h_neurons_total": len(h_pairs),
            "layers_analyzed": layers,
            "pooled_entropy_neurons": tot_sel,
            "pooled_h_neurons": tot_h,
            "pooled_overlap_count": tot_ov,
            "pooled_expected_random_overlap": pooled_expected,
            "pooled_enrichment_observed_over_random": enrichment,
            "pooled_hypergeom_p": _hypergeom_sf(tot_ov, pop, tot_sel, tot_h),
            "per_layer": per_layer,
        }

        print("\n" + "=" * 66)
        print(f"CONFIDENCE REGULATION -- OVERLAP ({self.overlap_layers} layers)")
        print("=" * 66)
        print(f"Probe             : {self.probe_path}")
        print(f"Layers analysed   : {layers}")
        print(f"Pooled overlap    : {tot_ov} / {tot_h} H-Neurons in {tot_sel} entropy neurons")
        print(f"Expected at random: {pooled_expected:.4f}  ->  enrichment x{enrichment:.2f}")
        print(f"Pooled hypergeom p: {metrics['pooled_hypergeom_p']:.3e}")
        print("-" * 66)
        for r in per_layer:
            print(
                f"  L{r['layer']:>2} H={r['h_neurons']:>2} ent={r['entropy_neurons']:>4} "
                f"ov={r['overlap_count']:>2} jac={r['jaccard']:.3f} "
                f"enr={r['enrichment_observed_over_random']:5.1f} p={r['hypergeom_p']:.2e}"
            )
        print("=" * 66)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="n/a",
            metrics=metrics,
            metadata={"description": self.description, "per_layer": per_layer},
        )

    # ------------------------------------------------------------------
    # induction function mode (paper Sec. 6)
    # ------------------------------------------------------------------

    def _build_induction_batches(self, backend: InferenceBackend) -> torch.Tensor:
        """Duplicated-sequence inputs ``(n_sequences, 2*half_len)``.

        Successive non-overlapping ``half_len``-token blocks from the corpus
        text, each concatenated with itself (AB...A -> B setup).
        """
        tokenizer = backend.tokenizer
        ids = torch.tensor(
            tokenizer(self._corpus_text_or_default(), add_special_tokens=False)["input_ids"],
            dtype=torch.long,
        )
        half = self.induction_half_len
        n_blocks = len(ids) // half
        if n_blocks < 1:
            raise ValueError("corpus too short for induction_half_len")
        if n_blocks < self.induction_sequences:
            reps = -(-self.induction_sequences // n_blocks)
            ids = ids.repeat(reps)
            n_blocks = len(ids) // half
        blocks = ids[: n_blocks * half].view(n_blocks, half)
        dup = torch.cat([blocks, blocks], dim=1)  # (n_blocks, 2*half)
        return dup[: self.induction_sequences]

    @staticmethod
    def _entropy_per_row(logits: torch.Tensor) -> torch.Tensor:
        log_p = torch.log_softmax(logits.float(), dim=-1)
        return -(log_p.exp() * log_p).sum(-1)

    def _run_induction(self, backend: InferenceBackend) -> ExperimentResult:
        """Function case study: hedging on repeated sequences (paper Sec. 6).

        For each candidate neuron of the final MLP layer, activations are
        clipped to their **first-occurrence mean** at second-occurrence
        positions via a real forward hook. If the neuron hedges (raises
        entropy against copying confidence), clipping *reduces*
        second-occurrence entropy; the reported Δ is negative in that case.
        """
        device = backend.device
        ident = self._identify_dispatch(backend)
        selected = ident["selected"]
        batches = self._build_induction_batches(backend)
        mod = backend.hook_manager.get_mlp_down_proj_module(ident["layer"])
        captured: Dict[str, Any] = {}

        def grab_acts(_mod, inp):
            captured["acts"] = inp[0].detach().float().cpu()

        def forward(seq_batch, clip_neuron=None, clip_value=None, positions=None):
            h = None
            if clip_neuron is not None:

                def hook(m, inp):
                    new = inp[0].clone()
                    new[:, positions:, clip_neuron] = float(clip_value)
                    return (new,) + tuple(inp[1:])

                h = mod.register_forward_pre_hook(hook)
            with torch.no_grad():
                out = backend.model(seq_batch.to(device))
            if h is not None:
                h.remove()
            logits = out.logits  # (B, L, vocab)
            H = self._entropy_per_row(logits[:, :-1])  # row r predicts token r+1
            return H, captured["acts"]

        half = self.induction_half_len
        rng = torch.Generator().manual_seed(self.seed)
        rand_idx = torch.randperm(ident["score"].numel(), generator=rng)[
            : self.random_baseline_count
        ].tolist()

        h_acts = mod.register_forward_pre_hook(grab_acts)

        # baseline pass: entropies + first-occurrence means per sequence
        base_second, first_means = [], []
        ent_first_all, ent_second_all = [], []
        with torch.no_grad():
            for b in range(batches.shape[0]):
                H, acts = forward(batches[b : b + 1])
                # rows [0..half-2] predict first-half tokens 1..half-1
                ent_first_all.append(H[0, : half - 1])
                # rows [half-1..2half-3] predict second-half tokens up to last
                ent_second_all.append(H[0, half - 1 : 2 * half - 2])
                first_means.append(acts[0, :half].mean(dim=0))
                base_second.append(H[0, half - 1 : 2 * half - 2])
        base_second_mean = torch.stack(base_second).mean()
        ent_first_mean = torch.stack(ent_first_all).mean()
        base_per_seq = torch.stack(base_second)  # (n_seq, half-1)

        def clip_pass(neuron: int, means: torch.Tensor) -> float:
            ds = []
            for b in range(batches.shape[0]):
                H, _ = forward(
                    batches[b : b + 1],
                    clip_neuron=neuron,
                    clip_value=float(means[b][neuron]),
                    positions=half,
                )
                ds.append(H[0, half - 1 : 2 * half - 2])
            # per-position delta vs un-clipped baseline (paper Fig. 5b metric)
            return torch.stack(ds) - base_per_seq

        base_pos = base_per_seq.mean(dim=0).clamp_min(1e-6)
        indices = sorted(set(selected) | set(rand_idx))
        d_entropy, peak_red, frac_drop = {}, {}, {}
        try:
            for i in indices:
                dp = clip_pass(i, first_means)
                d_entropy[i] = float(dp.mean())
                rel = dp.mean(dim=0) / base_pos
                peak_red[i] = float(-rel.min())
                frac_drop[i] = float((rel < -0.2).float().mean())
        finally:
            h_acts.remove()

        sel_d = [d_entropy[i] for i in selected]
        rand_d = [d_entropy[i] for i in rand_idx]
        strongest = min(sel_d) if sel_d else float("nan")

        metrics: Dict[str, Any] = {
            **{f"identify_{k}": v for k, v in ident["summary"].items()},
            "mode": "induction",
            "n_sequences": batches.shape[0],
            "seq_len": 2 * half,
            "baseline_entropy_first_occurrence": float(ent_first_mean),
            "baseline_entropy_second_occurrence": float(base_second_mean),
            "selected_mean_d_entropy": float(sum(sel_d) / len(sel_d)) if sel_d else 0.0,
            "selected_max_reduction_d_entropy": strongest,
            "random_baseline_mean_d_entropy": float(sum(rand_d) / len(rand_d)) if rand_d else 0.0,
            "per_neuron_d_entropy": {str(k): v for k, v in sorted(d_entropy.items())},
            "selected_peak_position_reduction": max(peak_red[i] for i in selected)
            if selected
            else 0.0,
            "random_peak_position_reduction": max(peak_red[i] for i in rand_idx)
            if rand_idx
            else 0.0,
            "selected_n_any_pos_gt20pct": sum(frac_drop[i] > 0 for i in selected),
        }

        print("\n" + "=" * 66)
        print("CONFIDENCE REGULATION -- INDUCTION (function case study)")
        print("=" * 66)
        print(
            f"Sequences        : {batches.shape[0]} x {2 * half} tokens (first {half} duplicated)"
        )
        print(f"Entropy first occ: {float(ent_first_mean):.3f}")
        print(
            f"Entropy second   : {float(base_second_mean):.3f} (confidence rise on repeat expected)"
        )
        print(
            f"Selected ({len(selected)}): mean dEntropy = "
            f"{metrics['selected_mean_d_entropy']:+.4f} "
            f"(strongest {strongest:+.4f})"
        )
        print(f"Random baseline  : mean {metrics['random_baseline_mean_d_entropy']:+.4f}")
        print(
            f"Peak pos. reduction: selected {metrics['selected_peak_position_reduction']:.1%} "
            f"| random {metrics['random_peak_position_reduction']:.1%}"
        )
        print(
            f"Selected w/ any >20% pos. drop: "
            f"{metrics['selected_n_any_pos_gt20pct']}/{len(selected)}"
        )
        print("Most-negative dEntropy (hedgers):")
        top = sorted(d_entropy.items(), key=lambda kv: kv[1])[:5]
        for i, dv in top:
            tag = "*" if i in set(selected) else " "
            rel = dv / float(base_second_mean) if base_second_mean else 0.0
            print(f"  {tag}{i:5d}  dEntropy={dv:+.4f}  ({rel:+.1%} of second-occ entropy)")
        print("=" * 66)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="teacher_forced",
            metrics=metrics,
            metadata={
                "description": self.description,
                "neuron_family": self.neuron_family,
                "layer": ident["layer"],
                "induction_half_len": half,
                "clip_source": "first_occurrence_mean",
                "selected_neurons": ident["detail"],
                "d_entropy": {str(k): v for k, v in d_entropy.items()},
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
        if self.mode == "induction":
            return self._run_induction(backend)
        if self.mode == "full":
            self._run_identify(backend)
            self._run_mediate(backend)
            return self._run_overlap(backend)
        raise NotImplementedError(
            f"mode '{self.mode}' is not implemented yet; use identify|mediate|overlap|full"
        )
