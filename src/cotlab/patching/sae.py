"""GemmaScope-2 JumpReLU Sparse Autoencoder loader.

Loads SAE weights directly from HuggingFace (no SAELens / TransformerLens
dependency) and exposes a minimal encode() forward pass compatible with the
CoTLab TransformersBackend hook infrastructure.

Architecture (JumpReLU SAE, Lieberum et al. 2024)
--------------------------------------------------
    h       = x - b_dec                   # centre around decoder bias
    pre_act = h @ W_enc + b_enc           # linear projection  [... , d_sae]
    features = pre_act * (pre_act > θ)    # JumpReLU gate

Weight layout in params.npz (GemmaScope-2 convention)
------------------------------------------------------
    W_enc     float32  [d_model, d_sae]
    b_enc     float32  [d_sae]
    W_dec     float32  [d_sae, d_model]
    b_dec     float32  [d_model]
    threshold float32  [d_sae]
"""

import re
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class GemmaScopeLayer(nn.Module):
    """JumpReLU SAE for a single residual-stream layer."""

    # Maps l0 label → (lo_inclusive, hi_exclusive) average_l0 range.
    _L0_RANGES = {
        "small": (0, 30),
        "medium": (30, 90),
        "big": (90, float("inf")),
    }

    def __init__(
        self,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        threshold: torch.Tensor,
        layer: int,
        repo_id: str,
        source_path: str,
    ):
        super().__init__()
        self.register_buffer("W_enc", W_enc)  # [d_model, d_sae]
        self.register_buffer("b_enc", b_enc)  # [d_sae]
        self.register_buffer("W_dec", W_dec)  # [d_sae, d_model]
        self.register_buffer("b_dec", b_dec)  # [d_model]
        self.register_buffer("threshold", threshold)  # [d_sae]
        self.layer = layer
        self.repo_id = repo_id
        self.source_path = source_path

    @property
    def d_model(self) -> int:
        return self.W_enc.shape[0]

    @property
    def d_sae(self) -> int:
        return self.W_enc.shape[1]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        layer: int,
        site: str = "resid_post_all",
        width: str = "16k",
        l0_label: str = "small",
        token: Optional[str] = None,
    ) -> "GemmaScopeLayer":
        """Download and load a GemmaScope-2 SAE from HuggingFace.

        Args:
            repo_id:   HF repo, e.g. ``"google/gemma-scope-2-270m-it"``
            layer:     Residual stream layer index (0-based).
            site:      SAE training site.  ``"resid_post_all"`` covers every
                       layer; other options are ``"resid_post"`` (4 depths only),
                       ``"attn_out_all"``, ``"mlp_out_all"``.
            width:     Feature dictionary width: ``"16k"`` or ``"262k"``.
            l0_label:  Target sparsity bucket: ``"small"`` (<30), ``"medium"``
                       (30-90), or ``"big"`` (>90).  Picks the lowest-l0 file
                       inside the range; falls back to any match if the range
                       has no entries.
            token:     HuggingFace API token (optional; reads HF_TOKEN env var
                       automatically via huggingface_hub).
        """
        from huggingface_hub import hf_hub_download, list_repo_files  # noqa: PLC0415

        print(f"  [SAE] Listing files in {repo_id} …")
        all_files = list(list_repo_files(repo_id, token=token))

        # Filter to files that match site / layer / width and end in params.npz.
        layer_tag = f"layer_{layer}"
        width_tag = f"width_{width}"
        candidates = [
            f
            for f in all_files
            if site in f and layer_tag in f and width_tag in f and f.endswith("params.npz")
        ]

        if not candidates:
            available_layers = sorted(
                {
                    re.search(r"layer_(\d+)", f).group(1)
                    for f in all_files
                    if "params.npz" in f and re.search(r"layer_(\d+)", f)
                },
                key=int,
            )
            raise FileNotFoundError(
                f"No SAE found for site={site!r}, layer={layer}, width={width!r} "
                f"in {repo_id}.\n"
                f"Available layers (any site/width): {available_layers}"
            )

        # Pick by l0_label.
        lo, hi = cls._L0_RANGES.get(l0_label, (0, float("inf")))

        def _l0(path: str) -> int:
            m = re.search(r"average_l0_(\d+)", path)
            return int(m.group(1)) if m else 9999

        in_range = [f for f in candidates if lo <= _l0(f) < hi]
        chosen = sorted(in_range or candidates, key=_l0)[0]

        print(f"  [SAE] Downloading {chosen} …")
        local_path = hf_hub_download(repo_id=repo_id, filename=chosen, token=token)

        data = np.load(local_path)
        W_enc = torch.from_numpy(np.array(data["W_enc"])).float()  # [d_model, d_sae]
        b_enc = torch.from_numpy(np.array(data["b_enc"])).float()  # [d_sae]
        W_dec = torch.from_numpy(np.array(data["W_dec"])).float()  # [d_sae, d_model]
        b_dec = torch.from_numpy(np.array(data["b_dec"])).float()  # [d_model]
        threshold = torch.from_numpy(np.array(data["threshold"])).float()  # [d_sae]

        print(
            f"  [SAE] Loaded  layer={layer}  d_model={W_enc.shape[0]}  "
            f"d_sae={W_enc.shape[1]}  l0_path={chosen.split('/')[-2]}"
        )
        return cls(
            W_enc=W_enc,
            b_enc=b_enc,
            W_dec=W_dec,
            b_dec=b_dec,
            threshold=threshold,
            layer=layer,
            repo_id=repo_id,
            source_path=chosen,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """JumpReLU encode.

        Args:
            x: Residual stream activations, shape ``[..., d_model]``.
               Must be on the same device as the SAE buffers.

        Returns:
            Sparse feature activations, shape ``[..., d_sae]``.
        """
        orig_dtype = x.dtype
        x = x.float()
        h = x - self.b_dec
        pre = h @ self.W_enc + self.b_enc
        features = pre * (pre > self.threshold).float()
        return features.to(orig_dtype)

    def __repr__(self) -> str:
        return (
            f"GemmaScopeLayer(layer={self.layer}, "
            f"d_model={self.d_model}, d_sae={self.d_sae}, "
            f"repo={self.repo_id})"
        )
