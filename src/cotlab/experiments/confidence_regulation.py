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

mediate:
    Causal mediation via analytic mean-ablation on the cached final residual
    stream: total effect (TE) vs direct effect with the final-norm scale
    frozen (DE_LN); LN-mediated fraction = 1 - DE_LN / TE. Not implemented
    yet (follow-up commit).

overlap:
    Jaccard overlap between identified entropy neurons and H-Neuron probe
    sets. Not implemented yet (follow-up commit).
"""

from typing import Any, Dict, List, Optional

import torch

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry


@Registry.register_experiment("confidence_regulation")
class ConfidenceRegulationExperiment(BaseExperiment):
    """Identify and validate confidence-regulating (entropy) neurons."""

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
        seed: int = 42,
        **kwargs,
    ):
        valid_modes = ("identify", "mediate", "overlap", "full")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")
        if selection not in ("top_n", "top_percent"):
            raise ValueError(f"selection must be 'top_n' or 'top_percent', got '{selection}'")

        self._name = name
        self.description = description
        self.mode = mode
        self.selection = selection
        self.top_n = top_n
        self.top_percent = top_percent
        self.k_null = k_null
        self.logit_chunk_size = logit_chunk_size
        self.fold_final_norm = fold_final_norm
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
    # identify
    # ------------------------------------------------------------------

    def _get_unembedding(self, backend: InferenceBackend) -> torch.Tensor:
        """Return the *effective* ``W_U`` as a ``(vocab, d_model)`` fp32 CPU tensor.

        Following the paper (Sec. 2) the final-norm gain is folded into the
        unembedding: logits = LN(x) @ W_U^T = x_normed @ (W_U ⊙ γ)^T. The
        null space that matters for entropy neurons is that of the folded
        matrix (TransformerLens ``fold_ln=True`` equivalent).
        """
        w_u = backend.model.get_output_embeddings().weight.detach().float().cpu()
        if self.fold_final_norm:
            gamma = self._get_final_norm_gain(backend)
            if gamma is not None:
                w_u = w_u * gamma.unsqueeze(0)
        return w_u

    @staticmethod
    def _get_final_norm_gain(backend: InferenceBackend):
        """Resolve the final normalization gain ``γ`` (d_model,) or None."""
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
        """Return final-layer ``W_out`` as a ``(d_model, d_mlp)`` float32 CPU tensor.

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

    def _compute_logit_vars(self, w_u: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
        """LogitVar per neuron (Eq. 3), chunked over neurons to bound memory.

        ``logit_var_i = Var_vocab( W_U w_out_i / (col_norms(W_U) * ||w_out_i||) )``
        """
        vocab, d_model = w_u.shape
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
        del vocab, d_model
        return logit_vars

    def _compute_rho(self, w_u: torch.Tensor, w_out: torch.Tensor) -> tuple:
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

    def _run_identify(self, backend: InferenceBackend) -> ExperimentResult:
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

        def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
            a = a - a.mean()
            b = b - b.mean()
            denom = a.norm() * b.norm()
            return float((a @ b) / denom) if denom > 0 else 0.0

        metrics: Dict[str, Any] = {
            "mode": "identify",
            "fold_final_norm": self.fold_final_norm,
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
            "pearson_rho_norm": _pearson(rho, norms),
            "pearson_rho_logit_var": _pearson(rho, -logit_vars),
        }

        print("\n" + "=" * 66)
        print("CONFIDENCE REGULATION -- IDENTIFY")
        print("=" * 66)
        print(
            f"Final layer      : {final_layer} (d_model={metrics['d_model']}, "
            f"d_mlp={metrics['d_mlp']})"
        )
        print(
            f"Null space dim k : {svd_diag['k_null']} "
            f"(bottom eig {svd_diag['bottom_eigval_max']:.2e} vs median "
            f"{svd_diag['median_eigval']:.2e})"
        )
        print(f"Selected         : {len(selected)} neurons ({self.selection})")
        print(
            f"rho   selected   : {float(sel_rho.mean()):.4f} ± {float(sel_rho.std()):.4f} "
            f"| all {metrics['all_mean_rho']:.4f}"
        )
        print(
            f"norm  selected   : {float(sel_norms.mean()):.3f} ± {float(sel_norms.std()):.3f} "
            f"| all {metrics['all_mean_norm']:.3f}"
        )
        print(
            f"logitVar selected: {float(sel_lv.mean()):.3e} ± {float(sel_lv.std()):.3e} "
            f"| all {metrics['all_mean_logit_var']:.3e}"
        )
        print(f"pearson(rho, norm)          : {metrics['pearson_rho_norm']:+.3f}")
        print(f"pearson(rho, -logitVar)     : {metrics['pearson_rho_logit_var']:+.3f}")
        print("=" * 66)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="n/a",
            metrics=metrics,
            metadata={
                "description": self.description,
                "fold_final_norm": self.fold_final_norm,
                "selection": self.selection,
                "top_n": self.top_n,
                "top_percent": self.top_percent,
                "seed": self.seed,
                "selected_neurons": [
                    {
                        "layer": final_layer,
                        "index": int(i),
                        "norm": float(norms[i]),
                        "logit_var": float(logit_vars[i]),
                        "rho": float(rho[i]),
                    }
                    for i in selected
                ],
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
        """Run the confidence-regulation experiment in the configured mode."""
        self.validate_backend(backend)
        torch.manual_seed(self.seed)

        if self.mode in ("identify", "full"):
            result = self._run_identify(backend)
            if self.mode == "identify":
                return result
        raise NotImplementedError(f"mode '{self.mode}' is not implemented yet; use mode='identify'")
