"""GemmaScope-2 JumpReLU Sparse Autoencoder loader.

Loads SAE weights directly from HuggingFace (no SAELens / TransformerLens
dependency) and exposes a minimal encode() forward pass compatible with the
CoTLab TransformersBackend hook infrastructure.

Architecture (JumpReLU SAE, Lieberum et al. 2024)
--------------------------------------------------
    h        = x - b_dec                  # centre around decoder bias
    pre_act  = h @ w_enc + b_enc          # linear projection  [..., d_sae]
    features = pre_act * (pre_act > θ)    # JumpReLU gate

Weight layout in params.safetensors (GemmaScope-2 convention)
--------------------------------------------------------------
    w_enc     float32  [d_model, d_sae]
    b_enc     float32  [d_sae]
    w_dec     float32  [d_sae, d_model]
    b_dec     float32  [d_model]
    threshold float32  [d_sae]

HF path pattern
---------------
    {site}/layer_{N}_width_{width}_l0_{l0_label}/params.safetensors
    e.g. resid_post_all/layer_9_width_16k_l0_small/params.safetensors
"""

import re
from typing import Optional

import torch
import torch.nn as nn


class GemmaScopeLayer(nn.Module):
    """JumpReLU SAE for a single residual-stream layer."""

    def __init__(
        self,
        w_enc: torch.Tensor,
        b_enc: torch.Tensor,
        w_dec: torch.Tensor,
        b_dec: torch.Tensor,
        threshold: torch.Tensor,
        layer: int,
        repo_id: str,
        source_path: str,
    ):
        super().__init__()
        self.register_buffer("w_enc", w_enc)  # [d_model, d_sae]
        self.register_buffer("b_enc", b_enc)  # [d_sae]
        self.register_buffer("w_dec", w_dec)  # [d_sae, d_model]
        self.register_buffer("b_dec", b_dec)  # [d_model]
        self.register_buffer("threshold", threshold)  # [d_sae]
        self.layer = layer
        self.repo_id = repo_id
        self.source_path = source_path

    @property
    def d_model(self) -> int:
        return self.w_enc.shape[0]

    @property
    def d_sae(self) -> int:
        return self.w_enc.shape[1]

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
            l0_label:  Sparsity label used in the directory name: ``"small"``,
                       ``"medium"``, or ``"big"``.  If not found, falls back to
                       any available file for that layer/width combination.
            token:     HuggingFace API token (optional; reads HF_TOKEN env var
                       automatically via huggingface_hub).
        """
        from huggingface_hub import hf_hub_download, list_repo_files  # noqa: PLC0415
        from safetensors import safe_open  # noqa: PLC0415

        # Try the canonical direct path first (avoids full repo listing).
        direct = f"{site}/layer_{layer}_width_{width}_l0_{l0_label}/params.safetensors"
        print(f"  [SAE] Fetching layer={layer} ({direct}) …")

        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=direct, token=token)
            chosen = direct
        except Exception:
            print("  [SAE] Direct path not found — scanning repo …")
            all_files = list(list_repo_files(repo_id, token=token))
            layer_tag = f"layer_{layer}_"
            width_tag = f"width_{width}_"
            candidates = [
                f
                for f in all_files
                if site in f
                and layer_tag in f
                and width_tag in f
                and f.endswith("params.safetensors")
            ]
            if not candidates:
                available = sorted(
                    {
                        re.search(r"layer_(\d+)_", f).group(1)
                        for f in all_files
                        if "params.safetensors" in f and re.search(r"layer_(\d+)_", f)
                    },
                    key=int,
                )
                raise FileNotFoundError(
                    f"No SAE found for site={site!r}, layer={layer}, width={width!r} "
                    f"in {repo_id}.\nAvailable layers (any site/width): {available}"
                )
            preferred = [f for f in candidates if f"l0_{l0_label}" in f]
            chosen = preferred[0] if preferred else candidates[0]
            local_path = hf_hub_download(repo_id=repo_id, filename=chosen, token=token)

        with safe_open(local_path, framework="pt") as f:
            w_enc = f.get_tensor("w_enc").float()  # [d_model, d_sae]
            b_enc = f.get_tensor("b_enc").float()  # [d_sae]
            w_dec = f.get_tensor("w_dec").float()  # [d_sae, d_model]
            b_dec = f.get_tensor("b_dec").float()  # [d_model]
            threshold = f.get_tensor("threshold").float()  # [d_sae]

        print(f"  [SAE] Loaded  layer={layer}  d_model={w_enc.shape[0]}  d_sae={w_enc.shape[1]}")
        return cls(
            w_enc=w_enc,
            b_enc=b_enc,
            w_dec=w_dec,
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
        pre = h @ self.w_enc + self.b_enc
        features = pre * (pre > self.threshold).float()
        return features.to(orig_dtype)

    def __repr__(self) -> str:
        return (
            f"GemmaScopeLayer(layer={self.layer}, "
            f"d_model={self.d_model}, d_sae={self.d_sae}, "
            f"repo={self.repo_id})"
        )
