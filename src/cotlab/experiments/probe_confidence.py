"""Probe-Confidence Correlation Experiment.

Tests whether a trained probe's score tracks the model's own output
uncertainty: correlate per-sample probe scores (e.g. H-Score for
hallucination probes) with token-level output entropy of the model on the
same responses.

Per sample a single teacher-forced forward over ``prompt + response``
yields both quantities:

- **H-Score** — probe applied to the CETT-style feature vector (per-layer
  mean of MLP down-projection inputs over response-token positions).
- **Entropy** — mean/max/last-token entropy read from the logit rows that
  predict the response tokens (row ``t`` predicts token ``t+1``; no extra
  forwards).

Statistics: Spearman correlation overall and per label, AUROC(probe) vs
AUROC(entropy-alone), partial Spearman controlling response length, and
bootstrap confidence intervals.

Verdict convention (pre-registered): |rho| > 0.5 → probe largely tracks
confidence; |rho| < 0.2 → independent signal.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry


@Registry.register_experiment("probe_confidence")
class ProbeConfidenceExperiment(BaseExperiment):
    """Correlate probe scores with model output entropy per sample."""

    def __init__(
        self,
        name: str = "probe_confidence",
        description: str = ("Correlate probe scores with model output entropy per sample"),
        probe_path: Optional[str] = None,
        samples: Optional[List[Dict[str, Any]]] = None,
        bootstrap_iters: int = 1000,
        batch_size: int = 1,
        feature_last_layer_only: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        if bootstrap_iters < 0:
            raise ValueError("bootstrap_iters must be non-negative")
        self._name = name
        self.description = description
        self.probe_path = probe_path
        self.samples = samples
        self.bootstrap_iters = bootstrap_iters
        self.batch_size = max(1, batch_size)
        self.feature_last_layer_only = feature_last_layer_only
        self.seed = seed

    @property
    def name(self) -> str:
        return self._name

    def validate_backend(self, backend: InferenceBackend) -> None:
        if getattr(backend, "hook_manager", None) is None:
            raise ValueError("probe_confidence requires the transformers backend with hook support")

    # ------------------------------------------------------------------
    # probe loading (dense weights or top-k safetensors convention)
    # ------------------------------------------------------------------

    def _load_probe_weights(self) -> Dict[str, Any]:
        """Load coef/intercept/scaler from probe JSON or sibling safetensors."""
        if not self.probe_path:
            raise ValueError("probe_path required")
        path = Path(self.probe_path)
        with open(path) as f:
            data = json.load(f)

        fit = data.get("fit", data)
        if isinstance(fit.get("weights"), list) and fit.get("intercept") is not None:
            return {
                "coef": torch.tensor(fit["weights"], dtype=torch.float32),
                "intercept": float(fit["intercept"]),
                "mean": torch.tensor(fit.get("mean", [0.0]), dtype=torch.float32),
                "std": torch.tensor(fit.get("std", [1.0]), dtype=torch.float32),
                "top_idx": None,
            }

        sibling = path.parent / (path.stem + ".safetensors")
        if sibling.exists():
            from safetensors.torch import load_file

            st = load_file(str(sibling))
            coef = st["clf_coef"].float()
            top_idx = st["top_k_idx"].long() if "top_k_idx" in st else None
            mean = st["col_mean"].float()
            std = st["col_std"].float()
            intercept = float(st["clf_intercept"].item())
            return {
                "coef": coef,
                "intercept": intercept,
                "mean": mean,
                "std": std,
                "top_idx": top_idx,
            }
        raise ValueError(f"no usable probe weights found for {path}")

    def _score_features(self, x: torch.Tensor, probe: Dict[str, Any]) -> float:
        z = (x - probe["mean"]) / probe["std"].clamp_min(1e-8)
        coef = probe["coef"]
        top_idx = probe["top_idx"]
        if top_idx is not None:
            z = z[top_idx]
            coef = coef[top_idx]
        return float(torch.sigmoid(z @ coef + probe["intercept"]))

    # ------------------------------------------------------------------
    # per-sample forward: features + entropies in one pass
    # ------------------------------------------------------------------

    def _sample_forward(
        self, backend: InferenceBackend, prompt_ids: List[int], resp_ids: List[int]
    ) -> Dict[str, Any]:
        device = backend.device
        input_ids = torch.tensor(prompt_ids + resp_ids, dtype=torch.long).unsqueeze(0).to(device)
        captured: List[torch.Tensor] = []

        def grab(_mod, inp):
            captured.append(inp[0].detach().float().cpu())

        handles = []
        num_layers = backend.hook_manager.num_layers
        layers = [num_layers - 1] if self.feature_last_layer_only else range(num_layers)
        try:
            for l_ in layers:
                handles.append(
                    backend.hook_manager.get_mlp_down_proj_module(l_).register_forward_pre_hook(
                        grab
                    )
                )
            with torch.no_grad():
                out = backend.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        logits = out.logits.float()[0]  # (L, vocab)
        lp_rows = logits[prompt_len - 1 : -1] if (prompt_len := len(prompt_ids)) else None
        log_p = torch.log_softmax(lp_rows, dim=-1)
        ent_t = -(log_p.exp() * log_p).sum(-1)  # (len(resp),)

        # features: per-layer mean over response-token hidden positions
        span = slice(len(prompt_ids), len(prompt_ids) + len(resp_ids))
        feats = []
        for acts in captured:  # each (1, L, d_mlp)
            feats.append(acts[0, span].mean(dim=0))
        x = torch.cat(feats) if not self.feature_last_layer_only else feats[0]

        return {
            "x": x,
            "entropy_mean": float(ent_t.mean()),
            "entropy_max": float(ent_t.max()),
            "entropy_last": float(ent_t[-1]),
            "n_response_tokens": len(resp_ids),
        }

    # ------------------------------------------------------------------
    # statistics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rankdata(a: torch.Tensor) -> torch.Tensor:
        order = a.argsort()
        ranks = torch.empty_like(a, dtype=torch.float32)
        ranks[order] = torch.arange(len(a), dtype=torch.float32)
        # average ties
        unique, inv, counts = torch.unique(a, return_inverse=True, return_counts=True)
        sums = torch.zeros(len(unique)).index_add_(0, inv, ranks)
        avg = sums / counts
        return avg[inv]

    @classmethod
    def _spearman(cls, a: torch.Tensor, b: torch.Tensor) -> float:
        ra, rb = cls._rankdata(a), cls._rankdata(b)
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = ra.norm() * rb.norm()
        return float((ra @ rb) / denom) if denom > 0 else 0.0

    @staticmethod
    def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
        """Mann–Whitney AUROC; labels boolean (positive class = True)."""
        pos = scores[labels]
        neg = scores[~labels]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        gt = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
        eq = (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum()
        return float((gt + 0.5 * eq) / (len(pos) * len(neg)))

    def _bootstrap_ci(
        self, scores: torch.Tensor, entropies: torch.Tensor, rng: torch.Generator
    ) -> tuple:
        if self.bootstrap_iters == 0 or len(scores) < 2:
            return (float("nan"), float("nan"))
        vals = []
        n = len(scores)
        for _ in range(self.bootstrap_iters):
            idx = torch.randint(0, n, (n,), generator=rng)
            vals.append(self._spearman(scores[idx], entropies[idx]))
        lo, hi = torch.quantile(torch.tensor(vals), torch.tensor([0.025, 0.975]))
        return (float(lo), float(hi))

    @classmethod
    def _partial_spearman_length(
        cls, scores: torch.Tensor, entropies: torch.Tensor, lengths: torch.Tensor
    ) -> float:
        rs, re, rl = cls._rankdata(scores), cls._rankdata(entropies), cls._rankdata(lengths)

        def resid(y, x):
            x1 = torch.cat([x.unsqueeze(1), torch.ones(len(x), 1)], dim=1)
            beta = torch.linalg.lstsq(x1, y.unsqueeze(1)).solution
            return (y.unsqueeze(1) - x1 @ beta).squeeze(1)

        return cls._spearman(resid(rs, rl), resid(re, rl))

    # ------------------------------------------------------------------
    # main run
    # ------------------------------------------------------------------

    def run(
        self,
        backend: InferenceBackend,
        dataset: Any = None,
        prompt_strategy: Any = None,
        **kwargs,
    ) -> ExperimentResult:
        self.validate_backend(backend)
        torch.manual_seed(self.seed)

        samples = self.samples if self.samples is not None else list(dataset or [])
        if not samples:
            raise ValueError("no samples provided (pass `samples` or a dataset)")
        probe = self._load_probe_weights()

        rows: List[Dict[str, Any]] = []
        for s in samples:
            prompt_ids = backend.tokenizer(s["prompt"], add_special_tokens=False)["input_ids"]
            resp_ids = backend.tokenizer(s["response"], add_special_tokens=False)["input_ids"]
            fwd = self._sample_forward(backend, prompt_ids, resp_ids)
            rows.append(
                {
                    "label": bool(s["label"]),
                    "h_score": self._score_features(fwd["x"], probe),
                    "entropy_mean": fwd["entropy_mean"],
                    "entropy_max": fwd["entropy_max"],
                    "entropy_last": fwd["entropy_last"],
                    "length": fwd["n_response_tokens"],
                }
            )

        scores = torch.tensor([r["h_score"] for r in rows])
        ents = torch.tensor([r["entropy_mean"] for r in rows])
        labels = torch.tensor([r["label"] for r in rows])
        lengths = torch.tensor([r["length"] for r in rows], dtype=torch.float32)
        rng = torch.Generator().manual_seed(self.seed)
        ci_lo, ci_hi = self._bootstrap_ci(scores, ents, rng)

        mask_f = ~labels
        metrics: Dict[str, Any] = {
            "n_samples": len(rows),
            "spearman_overall": self._spearman(scores, ents),
            "spearman_faithful": self._spearman(scores[mask_f], ents[mask_f])
            if mask_f.any()
            else float("nan"),
            "spearman_hallucinated": self._spearman(scores[labels], ents[labels])
            if labels.any()
            else float("nan"),
            "spearman_ci95": [ci_lo, ci_hi],
            "auroc_probe": self._auroc(scores, labels),
            "auroc_entropy_mean": self._auroc(ents, labels),
            "partial_spearman_controlled_length": self._partial_spearman_length(
                scores, ents, lengths
            ),
        }
        verdict = (
            "confounded"
            if abs(metrics["spearman_overall"]) > 0.5
            else "independent"
            if abs(metrics["spearman_overall"]) < 0.2
            else "mixed"
        )
        metrics["verdict"] = verdict

        print("\n" + "=" * 66)
        print("PROBE CONFIDENCE CORRELATION")
        print("=" * 66)
        print(
            f"Samples              : {len(rows)} "
            f"(faithful={int(mask_f.sum())}, hallucinated={int(labels.sum())})"
        )
        print(
            f"Spearman overall     : {metrics['spearman_overall']:+.3f} "
            f"CI95 [{ci_lo:+.3f}, {ci_hi:+.3f}]"
        )
        print(f"  faithful only      : {metrics['spearman_faithful']:+.3f}")
        print(f"  hallucinated only  : {metrics['spearman_hallucinated']:+.3f}")
        print(f"AUROC probe          : {metrics['auroc_probe']:.3f}")
        print(f"AUROC entropy alone  : {metrics['auroc_entropy_mean']:.3f}")
        print(f"Partial rho (length) : {metrics['partial_spearman_controlled_length']:+.3f}")
        print(f"Verdict              : {verdict}")
        print("=" * 66)

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy="teacher_forced",
            metrics=metrics,
            metadata={
                "description": self.description,
                "probe_path": self.probe_path,
                "feature_last_layer_only": self.feature_last_layer_only,
                "per_sample": rows,
            },
        )
