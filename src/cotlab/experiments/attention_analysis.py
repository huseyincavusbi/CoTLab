"""Attention Pattern Analysis Experiment.

Extracts attention weights at critical layers (55-60) and computes
attention entropy to understand which tokens each prompt strategy focuses on.

Enhanced to support multiple dataset samples for statistical robustness.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger


@Registry.register_experiment("attention_analysis")
class AttentionAnalysisExperiment(BaseExperiment):
    """
    Analyze attention patterns at critical layers.

    Computes:
    1. Last-token attention entropy per head (legacy metric)
    2. All-tokens mean attention entropy per head (primary metric)
    3. Optional last-k-tokens mean attention entropy per head
    4. Optional generated-answer-token span entropy
    5. Top-attended tokens for focused heads
    6. Aggregated statistics across multiple samples
    """

    def __init__(
        self,
        name: str = "attention_analysis",
        description: str = "Analyze attention patterns at critical layers",
        target_layers: Optional[List[int]] = None,
        all_layers: bool = False,
        force_eager_reload: bool = True,
        num_samples: Optional[int] = None,
        last_k_tokens: int = 16,
        max_input_tokens: Optional[int] = 1024,
        analyze_generated_tokens: bool = False,
        generated_max_new_tokens: int = 16,
        generated_do_sample: bool = False,
        generated_temperature: float = 0.7,
        generated_top_p: float = 0.9,
        question: str = "Patient presents with chest pain, sweating, and shortness of breath. What is the diagnosis?",
        batch_size: int = 1,
        **kwargs,
    ):
        self._name = name
        self.description = description
        # Default to layers 55-60 (critical reasoning layers found earlier)
        self._target_layers_config = target_layers or [55, 56, 57, 58, 59, 60]
        self.all_layers = all_layers
        self.force_eager_reload = force_eager_reload
        self.target_layers = self._target_layers_config
        self.num_samples = num_samples
        self.last_k_tokens = max(1, int(last_k_tokens))
        self.max_input_tokens = (
            max(1, int(max_input_tokens)) if max_input_tokens is not None else None
        )
        self.analyze_generated_tokens = bool(analyze_generated_tokens)
        self.generated_max_new_tokens = max(1, int(generated_max_new_tokens))
        self.generated_do_sample = bool(generated_do_sample)
        self.generated_temperature = float(generated_temperature)
        self.generated_top_p = float(generated_top_p)
        self.question = question  # Fallback if no dataset
        self.batch_size = max(1, int(batch_size))
        self._generated_analysis_disabled = False

    @property
    def name(self) -> str:
        return self._name

    def _compute_entropy(self, attn_dist: torch.Tensor) -> float:
        """Compute entropy of attention distribution.

        Note: Use bfloat16 (not float16) for the model to avoid NaN attention weights.
        """
        eps = 1e-10
        # Compute entropy in float32 for numerical stability.
        probs = attn_dist.float()
        return -torch.sum(probs * torch.log(probs + eps)).item()

    def _compute_mean_entropy_over_queries(self, attn_qk: torch.Tensor) -> float:
        """Compute mean entropy over query positions for one head.

        Args:
            attn_qk: Attention tensor of shape (num_queries, seq_len).
        """
        eps = 1e-10
        probs = attn_qk.float()
        entropies = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        return float(entropies.mean().item())

    def _analyze_generated_token_span(
        self,
        model,
        tokenizer,
        inputs: Dict[str, torch.Tensor],
        num_heads: int,
    ) -> tuple[Dict[int, Dict[str, Any]], int]:
        """Analyze attention entropy over generated answer-token steps.

        Returns:
            per_layer_stats, num_generated_steps
        """
        pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = getattr(model.config, "eos_token_id", None)

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.generated_max_new_tokens,
            "output_attentions": True,
            "return_dict_in_generate": True,
            "pad_token_id": pad_token_id,
            "do_sample": self.generated_do_sample,
        }
        if self.generated_do_sample:
            generate_kwargs["temperature"] = self.generated_temperature
            generate_kwargs["top_p"] = self.generated_top_p

        with torch.no_grad():
            gen_outputs = model.generate(**inputs, **generate_kwargs)

        gen_attentions = getattr(gen_outputs, "attentions", None)
        if not gen_attentions:
            return {}, 0

        num_generated_steps = len(gen_attentions)
        per_layer_stats: Dict[int, Dict[str, Any]] = {}

        for layer_idx in self.target_layers:
            per_head_step_entropies: List[List[float]] = [[] for _ in range(num_heads)]

            for step_attn in gen_attentions:
                layer_attn = None
                if isinstance(step_attn, (tuple, list)):
                    if layer_idx < len(step_attn):
                        layer_attn = step_attn[layer_idx]
                elif torch.is_tensor(step_attn):
                    layer_attn = step_attn

                if layer_attn is None:
                    continue

                # Typical shape: (batch, heads, q_len, k_len)
                # Some impls may provide (batch, heads, k_len)
                if layer_attn.dim() == 4:
                    attn_qk = layer_attn[0]  # (heads, q_len, k_len)
                elif layer_attn.dim() == 3:
                    attn_qk = layer_attn[0].unsqueeze(1)  # (heads, 1, k_len)
                else:
                    continue

                n_heads_eff = min(num_heads, attn_qk.shape[0])
                for h in range(n_heads_eff):
                    per_head_step_entropies[h].append(
                        self._compute_mean_entropy_over_queries(attn_qk[h])
                    )

            head_entropies_generated_tokens: List[float] = []
            for vals in per_head_step_entropies:
                if vals:
                    head_entropies_generated_tokens.append(float(np.nanmean(vals)))
                else:
                    head_entropies_generated_tokens.append(float("nan"))

            if np.all(np.isnan(head_entropies_generated_tokens)):
                continue

            per_layer_stats[layer_idx] = {
                "avg_entropy_generated_tokens": float(np.nanmean(head_entropies_generated_tokens)),
                "std_entropy_generated_tokens": float(np.nanstd(head_entropies_generated_tokens)),
                "head_entropies_generated_tokens": head_entropies_generated_tokens,
                "generated_tokens_analyzed": num_generated_steps,
            }

        return per_layer_stats, num_generated_steps

    def _analyze_batch(
        self,
        model,
        tokenizer,
        prompts: List[str],
        device: str,
        num_heads: int,
    ) -> List[Optional[Dict[str, Any]]]:
        """Analyze attention for a batch of samples in a single forward pass.

        Tokenizes all prompts together with padding, runs one batched forward pass
        with output_attentions=True, then slices out per-sample entropy stats
        accounting for left/right padding.  Attention tensors for each layer are
        freed immediately after processing to keep peak VRAM low.
        """
        tokenizer_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
        }
        if self.max_input_tokens is not None:
            tokenizer_kwargs.update({"truncation": True, "max_length": self.max_input_tokens})

        tokens = tokenizer(prompts, **tokenizer_kwargs).to(device)
        input_ids = tokens["input_ids"]          # (B, padded_seq)
        attention_mask = tokens["attention_mask"]  # (B, padded_seq)
        batch_size_actual = input_ids.shape[0]
        total_len = input_ids.shape[1]

        # Number of real tokens per sample (padding tokens have mask=0)
        seq_lengths = attention_mask.sum(dim=1).tolist()
        pad_left = getattr(tokenizer, "padding_side", "right") == "left"

        with torch.no_grad():
            outputs = model(**tokens, output_attentions=True, return_dict=True)

        attentions = outputs.attentions  # tuple[(B, heads, padded_seq, padded_seq)] * num_layers
        del outputs  # release non-attention outputs immediately

        if attentions is None or len(attentions) == 0:
            return [None] * batch_size_actual

        # Build per-sample result containers
        results: List[Dict] = [{} for _ in range(batch_size_actual)]

        for layer_idx in self.target_layers:
            if layer_idx >= len(attentions):
                continue

            layer_attn = attentions[layer_idx]  # (B, heads, padded_seq, padded_seq)

            for sample_i in range(batch_size_actual):
                seq_len = int(seq_lengths[sample_i])
                if seq_len == 0:
                    continue

                # Determine real-token slice (exclude padding positions)
                if pad_left:
                    start, end = total_len - seq_len, total_len
                else:
                    start, end = 0, seq_len

                # sample_attn: (heads, seq_len, seq_len) — padding stripped
                sample_attn = layer_attn[sample_i, :, start:end, start:end]

                last_token_attn = sample_attn[:, -1, :]   # (heads, seq_len)
                last_k = min(self.last_k_tokens, seq_len)
                last_k_tokens_attn = sample_attn[:, seq_len - last_k:, :]  # (heads, k, seq_len)

                head_entropies_last_token: List[float] = []
                head_entropies_all_tokens: List[float] = []
                head_entropies_last_k_tokens: List[float] = []
                for h in range(num_heads):
                    head_entropies_last_token.append(self._compute_entropy(last_token_attn[h]))
                    head_entropies_all_tokens.append(
                        self._compute_mean_entropy_over_queries(sample_attn[h])
                    )
                    head_entropies_last_k_tokens.append(
                        self._compute_mean_entropy_over_queries(last_k_tokens_attn[h])
                    )

                avg_entropy_last_token = np.mean(head_entropies_last_token)
                avg_entropy_all_tokens = np.mean(head_entropies_all_tokens)
                avg_entropy_last_k_tokens = np.mean(head_entropies_last_k_tokens)
                min_head = int(np.argmin(head_entropies_last_token))

                # Top-attended tokens for the most focused head
                focused_attn = last_token_attn[min_head]
                top_positions = torch.topk(focused_attn, k=min(5, seq_len))
                top_tokens = []
                for pos, weight in zip(
                    top_positions.indices.tolist(), top_positions.values.tolist()
                ):
                    actual_pos = start + pos
                    token_str = tokenizer.decode([input_ids[sample_i, actual_pos]])
                    top_tokens.append({"token": token_str, "weight": weight})

                results[sample_i][layer_idx] = {
                    # Legacy fields preserved
                    "avg_entropy": avg_entropy_last_token,
                    "head_entropies": head_entropies_last_token,
                    "min_entropy": min(head_entropies_last_token),
                    "max_entropy": max(head_entropies_last_token),
                    # Explicit metrics
                    "avg_entropy_last_token": avg_entropy_last_token,
                    "avg_entropy_all_tokens": avg_entropy_all_tokens,
                    "avg_entropy_last_k_tokens": avg_entropy_last_k_tokens,
                    "head_entropies_last_token": head_entropies_last_token,
                    "head_entropies_all_tokens": head_entropies_all_tokens,
                    "head_entropies_last_k_tokens": head_entropies_last_k_tokens,
                    "last_k_tokens_used": last_k,
                    # Generated-token fields filled in below
                    "avg_entropy_generated_tokens": None,
                    "std_entropy_generated_tokens": None,
                    "head_entropies_generated_tokens": None,
                    "generated_tokens_analyzed": 0,
                    "focused_head": min_head,
                    "top_tokens": top_tokens,
                }

            # Free this layer's tensor immediately to keep VRAM headroom
            del layer_attn

        del attentions
        torch.cuda.empty_cache()

        # Generated-token analysis: run per-sample (auto-regressive, inherently sequential)
        if self.analyze_generated_tokens and not self._generated_analysis_disabled:
            for sample_i in range(batch_size_actual):
                if not results[sample_i]:
                    continue
                seq_len = int(seq_lengths[sample_i])
                if pad_left:
                    s = total_len - seq_len
                    single_ids = input_ids[sample_i : sample_i + 1, s:]
                    single_mask = attention_mask[sample_i : sample_i + 1, s:]
                else:
                    single_ids = input_ids[sample_i : sample_i + 1, :seq_len]
                    single_mask = attention_mask[sample_i : sample_i + 1, :seq_len]
                single_inputs = {"input_ids": single_ids, "attention_mask": single_mask}
                try:
                    gen_stats, gen_steps = self._analyze_generated_token_span(
                        model=model,
                        tokenizer=tokenizer,
                        inputs=single_inputs,
                        num_heads=num_heads,
                    )
                    for layer_idx in self.target_layers:
                        if layer_idx in results[sample_i] and layer_idx in gen_stats:
                            results[sample_i][layer_idx].update(
                                {
                                    "avg_entropy_generated_tokens": gen_stats[layer_idx].get(
                                        "avg_entropy_generated_tokens"
                                    ),
                                    "std_entropy_generated_tokens": gen_stats[layer_idx].get(
                                        "std_entropy_generated_tokens"
                                    ),
                                    "head_entropies_generated_tokens": gen_stats[layer_idx].get(
                                        "head_entropies_generated_tokens"
                                    ),
                                    "generated_tokens_analyzed": gen_stats[layer_idx].get(
                                        "generated_tokens_analyzed", gen_steps
                                    ),
                                }
                            )
                except Exception as e:
                    print(
                        f"Warning: generated-token analysis failed for sample {sample_i} "
                        f"and will be disabled for this run: {type(e).__name__}: {e}"
                    )
                    self._generated_analysis_disabled = True
                    break

        return [r if r else None for r in results]

    def _analyze_single_sample(
        self,
        model,
        tokenizer,
        prompt: str,
        device: str,
        num_heads: int,
    ) -> Dict[str, Any]:
        """Analyze attention for a single sample (delegates to _analyze_batch)."""
        results = self._analyze_batch(model, tokenizer, [prompt], device, num_heads)
        return results[0]

    def _analyze_single_sample_legacy(
        self,
        model,
        tokenizer,
        prompt: str,
        device: str,
        num_heads: int,
    ) -> Dict[str, Any]:
        """Original single-sample implementation kept for reference."""
        tokenizer_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
        if self.max_input_tokens is not None:
            tokenizer_kwargs.update(
                {
                    "truncation": True,
                    "max_length": self.max_input_tokens,
                }
            )
        tokens = tokenizer(prompt, **tokenizer_kwargs).to(device)
        input_ids = tokens["input_ids"]

        with torch.no_grad():
            outputs = model(**tokens, output_attentions=True, return_dict=True)

        attentions = outputs.attentions

        if attentions is None or len(attentions) == 0:
            return None

        sample_results = {}

        generated_layer_stats: Dict[int, Dict[str, Any]] = {}
        generated_steps = 0
        if self.analyze_generated_tokens and not self._generated_analysis_disabled:
            try:
                generated_layer_stats, generated_steps = self._analyze_generated_token_span(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=tokens,
                    num_heads=num_heads,
                )
            except Exception as e:
                print(
                    "Warning: generated-token attention analysis failed once and will be disabled "
                    f"for this run: {type(e).__name__}: {e}"
                )
                self._generated_analysis_disabled = True

        for layer_idx in self.target_layers:
            if layer_idx >= len(attentions):
                continue

            attn = attentions[layer_idx]  # (batch, heads, seq, seq)
            seq_len = attn.shape[-1]
            last_token_attn = attn[0, :, -1, :]  # (heads, seq)
            all_tokens_attn = attn[0, :, :, :]  # (heads, seq, seq)

            last_k = min(self.last_k_tokens, seq_len)
            last_k_tokens_attn = attn[0, :, seq_len - last_k :, :]  # (heads, k, seq)

            head_entropies_last_token = []
            head_entropies_all_tokens = []
            head_entropies_last_k_tokens = []
            for h in range(num_heads):
                head_entropies_last_token.append(self._compute_entropy(last_token_attn[h]))
                head_entropies_all_tokens.append(
                    self._compute_mean_entropy_over_queries(all_tokens_attn[h])
                )
                head_entropies_last_k_tokens.append(
                    self._compute_mean_entropy_over_queries(last_k_tokens_attn[h])
                )

            avg_entropy_last_token = np.mean(head_entropies_last_token)
            avg_entropy_all_tokens = np.mean(head_entropies_all_tokens)
            avg_entropy_last_k_tokens = np.mean(head_entropies_last_k_tokens)
            min_head = int(np.argmin(head_entropies_last_token))

            # Get top-attended tokens for the most focused head
            focused_head_attn = last_token_attn[min_head]
            top_positions = torch.topk(focused_head_attn, k=min(5, input_ids.shape[1]))

            top_tokens = []
            for pos, weight in zip(top_positions.indices.tolist(), top_positions.values.tolist()):
                token_str = tokenizer.decode([input_ids[0, pos]])
                top_tokens.append({"token": token_str, "weight": weight})

            sample_results[layer_idx] = {
                # Legacy fields preserved (last-token)
                "avg_entropy": avg_entropy_last_token,
                "head_entropies": head_entropies_last_token,
                "min_entropy": min(head_entropies_last_token),
                "max_entropy": max(head_entropies_last_token),
                # Explicit metrics
                "avg_entropy_last_token": avg_entropy_last_token,
                "avg_entropy_all_tokens": avg_entropy_all_tokens,
                "avg_entropy_last_k_tokens": avg_entropy_last_k_tokens,
                "head_entropies_last_token": head_entropies_last_token,
                "head_entropies_all_tokens": head_entropies_all_tokens,
                "head_entropies_last_k_tokens": head_entropies_last_k_tokens,
                "last_k_tokens_used": last_k,
                "avg_entropy_generated_tokens": generated_layer_stats.get(layer_idx, {}).get(
                    "avg_entropy_generated_tokens"
                ),
                "std_entropy_generated_tokens": generated_layer_stats.get(layer_idx, {}).get(
                    "std_entropy_generated_tokens"
                ),
                "head_entropies_generated_tokens": generated_layer_stats.get(layer_idx, {}).get(
                    "head_entropies_generated_tokens"
                ),
                "generated_tokens_analyzed": generated_layer_stats.get(layer_idx, {}).get(
                    "generated_tokens_analyzed", generated_steps
                ),
                "focused_head": min_head,
                "top_tokens": top_tokens,
            }

        return sample_results

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        num_samples: Optional[int] = None,
        logger: Optional[ExperimentLogger] = None,
    ) -> ExperimentResult:
        """Run attention analysis experiment on multiple samples."""

        tokenizer = backend._tokenizer
        model = backend._model

        # Get model config
        config = model.config
        if hasattr(config, "text_config"):
            config = config.text_config
        num_heads = config.num_attention_heads

        if self.all_layers:
            num_layers = getattr(config, "num_hidden_layers", None) or getattr(
                config, "num_layers", None
            )
            if num_layers is None:
                num_layers = backend.num_layers()
            self.target_layers = list(range(int(num_layers)))

        print(f"Model: {backend.model_name}")
        print(f"Attention heads: {num_heads}")
        print(f"All layers enabled: {self.all_layers}")
        print(f"Resolved layers: {self.target_layers}")
        print(
            f"Max input tokens: {self.max_input_tokens if self.max_input_tokens is not None else 'None'}"
        )
        print(f"Batch size: {self.batch_size}")
        print(f"Analyze generated tokens: {self.analyze_generated_tokens}")
        if self.analyze_generated_tokens:
            print(f"Generated max_new_tokens: {self.generated_max_new_tokens}")

        # Set eager attention to enable output_attentions by reloading if necessary
        # We need to check if the model is already using eager attention
        current_attn = getattr(model, "config", None) and getattr(
            model.config, "_attn_implementation", None
        )

        if current_attn != "eager":
            if hasattr(model, "set_attn_implementation") and not self.force_eager_reload:
                print(f"Current attention implementation: {current_attn}")
                print("Switching attention implementation to 'eager' in-place...")
                model.set_attn_implementation("eager")
                current_attn = getattr(model.config, "_attn_implementation", None)
            elif self.force_eager_reload:
                print(f"Current attention implementation: {current_attn}")
                print(
                    "Reloading model with attn_implementation='eager' to support output_attentions=True..."
                )
                # We need to preserve the model name before unloading
                model_name = backend.model_name
                backend.unload()
                # Reload with eager attention
                backend.load_model(model_name, attn_implementation="eager")
                model = backend._model
                tokenizer = backend._tokenizer

        # Get samples from dataset
        n_samples = num_samples if num_samples is not None else self.num_samples
        samples = list(dataset) if n_samples is None else (dataset.sample(n_samples) if n_samples < len(dataset) else list(dataset))
        print(f"\nAnalyzing attention on {len(samples)} samples (batch_size={self.batch_size})...")

        # Aggregate statistics across samples
        layer_entropy_stats_last_token: Dict[int, List[float]] = defaultdict(list)
        layer_entropy_stats_all_tokens: Dict[int, List[float]] = defaultdict(list)
        layer_entropy_stats_last_k_tokens: Dict[int, List[float]] = defaultdict(list)
        layer_entropy_stats_generated_tokens: Dict[int, List[float]] = defaultdict(list)
        layer_head_entropy_stats_last_token: Dict[int, List[List[float]]] = defaultdict(list)
        layer_head_entropy_stats_all_tokens: Dict[int, List[List[float]]] = defaultdict(list)
        layer_head_entropy_stats_last_k_tokens: Dict[int, List[List[float]]] = defaultdict(list)
        layer_head_entropy_stats_generated_tokens: Dict[int, List[List[float]]] = defaultdict(list)
        all_top_tokens: Dict[int, List[str]] = defaultdict(list)

        sample_results = []

        # Build batches
        batches = [
            samples[i : i + self.batch_size] for i in range(0, len(samples), self.batch_size)
        ]

        for batch_samples in tqdm(batches, desc="Processing batches"):
            prompts = [
                prompt_strategy.build_prompt(
                    {"question": s.text, "text": s.text, "metadata": s.metadata or {}}
                )
                for s in batch_samples
            ]

            batch_results = self._analyze_batch(
                model, tokenizer, prompts, backend.device, num_heads
            )

            for sample, result in zip(batch_samples, batch_results):
                if result is None:
                    print(f"\nWarning: Attention not available for sample {sample.idx}")
                    continue

                sample_results.append(
                    {
                        "sample_idx": sample.idx,
                        "layer_results": result,
                    }
                )

                # Aggregate stats for this sample
                for layer_idx, layer_data in result.items():
                    layer_entropy_stats_last_token[layer_idx].append(
                        layer_data["avg_entropy_last_token"]
                    )
                    layer_entropy_stats_all_tokens[layer_idx].append(
                        layer_data["avg_entropy_all_tokens"]
                    )
                    layer_entropy_stats_last_k_tokens[layer_idx].append(
                        layer_data["avg_entropy_last_k_tokens"]
                    )
                    layer_head_entropy_stats_last_token[layer_idx].append(
                        layer_data["head_entropies_last_token"]
                    )
                    layer_head_entropy_stats_all_tokens[layer_idx].append(
                        layer_data["head_entropies_all_tokens"]
                    )
                    layer_head_entropy_stats_last_k_tokens[layer_idx].append(
                        layer_data["head_entropies_last_k_tokens"]
                    )
                    gen_entropy = layer_data.get("avg_entropy_generated_tokens")
                    gen_head_entropies = layer_data.get("head_entropies_generated_tokens")
                    if gen_entropy is not None:
                        layer_entropy_stats_generated_tokens[layer_idx].append(gen_entropy)
                    if gen_head_entropies:
                        layer_head_entropy_stats_generated_tokens[layer_idx].append(gen_head_entropies)
                    for tok in layer_data["top_tokens"][:3]:  # Top 3 tokens
                        all_top_tokens[layer_idx].append(tok["token"])

        if not sample_results:
            return ExperimentResult(
                experiment_name=self.name,
                model_name=backend.model_name,
                prompt_strategy=prompt_strategy.name
                if hasattr(prompt_strategy, "name")
                else "custom",
                metrics={"error": "attention_not_supported", "num_layers_analyzed": 0},
                raw_outputs=[],
                metadata={"target_layers": self.target_layers},
            )

        # Compute aggregated statistics
        print("\n" + "=" * 70)
        print("ATTENTION ANALYSIS: Aggregated Statistics Across Samples")
        print("=" * 70)
        header = (
            f"{'Layer':<8} | {'LastTok μ':<10} | {'AllTok μ':<10} | "
            f"{f'Last{self.last_k_tokens} μ':<10} | {'AllTok σ':<10}"
        )
        if self.analyze_generated_tokens:
            header += f" | {'GenTok μ':<10}"
        header += " | Top Tokens"
        print(header)
        print("-" * 106)

        aggregated_results = []

        for layer_idx in sorted(layer_entropy_stats_last_token.keys()):
            entropies_last_token = layer_entropy_stats_last_token[layer_idx]
            entropies_all_tokens = layer_entropy_stats_all_tokens[layer_idx]
            entropies_last_k_tokens = layer_entropy_stats_last_k_tokens[layer_idx]

            mean_entropy_last_token = float(np.nanmean(entropies_last_token))
            std_entropy_last_token = float(np.nanstd(entropies_last_token))
            mean_entropy_all_tokens = float(np.nanmean(entropies_all_tokens))
            std_entropy_all_tokens = float(np.nanstd(entropies_all_tokens))
            mean_entropy_last_k_tokens = float(np.nanmean(entropies_last_k_tokens))
            std_entropy_last_k_tokens = float(np.nanstd(entropies_last_k_tokens))
            if layer_entropy_stats_generated_tokens[layer_idx]:
                mean_entropy_generated_tokens = float(
                    np.nanmean(layer_entropy_stats_generated_tokens[layer_idx])
                )
                std_entropy_generated_tokens = float(
                    np.nanstd(layer_entropy_stats_generated_tokens[layer_idx])
                )
            else:
                mean_entropy_generated_tokens = float("nan")
                std_entropy_generated_tokens = float("nan")

            # Count most common top tokens
            tokens = all_top_tokens[layer_idx]
            from collections import Counter

            token_counts = Counter(tokens)
            top_3_tokens = token_counts.most_common(5)
            top_tokens_str = ", ".join([f"'{t}'" for t, _ in top_3_tokens[:3]])

            # Aggregate head-level entropies for each metric
            head_entropies_last_token = np.array(layer_head_entropy_stats_last_token[layer_idx])
            head_entropies_all_tokens = np.array(layer_head_entropy_stats_all_tokens[layer_idx])
            head_entropies_last_k_tokens = np.array(
                layer_head_entropy_stats_last_k_tokens[layer_idx]
            )
            mean_per_head_last_token = np.nanmean(head_entropies_last_token, axis=0).tolist()
            std_per_head_last_token = np.nanstd(head_entropies_last_token, axis=0).tolist()
            mean_per_head_all_tokens = np.nanmean(head_entropies_all_tokens, axis=0).tolist()
            std_per_head_all_tokens = np.nanstd(head_entropies_all_tokens, axis=0).tolist()
            mean_per_head_last_k_tokens = np.nanmean(head_entropies_last_k_tokens, axis=0).tolist()
            std_per_head_last_k_tokens = np.nanstd(head_entropies_last_k_tokens, axis=0).tolist()
            if layer_head_entropy_stats_generated_tokens[layer_idx]:
                head_entropies_generated_tokens = np.array(
                    layer_head_entropy_stats_generated_tokens[layer_idx]
                )
                mean_per_head_generated_tokens = np.nanmean(
                    head_entropies_generated_tokens, axis=0
                ).tolist()
                std_per_head_generated_tokens = np.nanstd(
                    head_entropies_generated_tokens, axis=0
                ).tolist()
            else:
                mean_per_head_generated_tokens = []
                std_per_head_generated_tokens = []

            aggregated_results.append(
                {
                    "layer": layer_idx,
                    # Legacy keys (last-token metric)
                    "mean_entropy": mean_entropy_last_token,
                    "std_entropy": std_entropy_last_token,
                    "mean_per_head": mean_per_head_last_token,
                    "std_per_head": std_per_head_last_token,
                    # Explicit metrics
                    "mean_entropy_last_token": mean_entropy_last_token,
                    "std_entropy_last_token": std_entropy_last_token,
                    "mean_entropy_all_tokens": mean_entropy_all_tokens,
                    "std_entropy_all_tokens": std_entropy_all_tokens,
                    "mean_entropy_last_k_tokens": mean_entropy_last_k_tokens,
                    "std_entropy_last_k_tokens": std_entropy_last_k_tokens,
                    "mean_entropy_generated_tokens": (
                        None
                        if np.isnan(mean_entropy_generated_tokens)
                        else mean_entropy_generated_tokens
                    ),
                    "std_entropy_generated_tokens": (
                        None
                        if np.isnan(std_entropy_generated_tokens)
                        else std_entropy_generated_tokens
                    ),
                    "last_k_tokens": self.last_k_tokens,
                    "mean_per_head_last_token": mean_per_head_last_token,
                    "std_per_head_last_token": std_per_head_last_token,
                    "mean_per_head_all_tokens": mean_per_head_all_tokens,
                    "std_per_head_all_tokens": std_per_head_all_tokens,
                    "mean_per_head_last_k_tokens": mean_per_head_last_k_tokens,
                    "std_per_head_last_k_tokens": std_per_head_last_k_tokens,
                    "mean_per_head_generated_tokens": mean_per_head_generated_tokens,
                    "std_per_head_generated_tokens": std_per_head_generated_tokens,
                    "top_tokens": [{"token": t, "count": c} for t, c in top_3_tokens],
                }
            )

            row = (
                f"L{layer_idx:<7} | {mean_entropy_last_token:<10.4f} | "
                f"{mean_entropy_all_tokens:<10.4f} | {mean_entropy_last_k_tokens:<10.4f} | "
                f"{std_entropy_all_tokens:<10.4f}"
            )
            if self.analyze_generated_tokens:
                if np.isnan(mean_entropy_generated_tokens):
                    row += f" | {'NA':<10}"
                else:
                    row += f" | {mean_entropy_generated_tokens:<10.4f}"
            row += f" | {top_tokens_str}"
            print(row)

        print("-" * 106)

        # Overall metrics
        all_mean_entropies_last_token = [r["mean_entropy_last_token"] for r in aggregated_results]
        all_mean_entropies_all_tokens = [r["mean_entropy_all_tokens"] for r in aggregated_results]
        all_mean_entropies_last_k_tokens = [
            r["mean_entropy_last_k_tokens"] for r in aggregated_results
        ]
        all_mean_entropies_generated_tokens = [
            r["mean_entropy_generated_tokens"]
            for r in aggregated_results
            if r["mean_entropy_generated_tokens"] is not None
        ]
        overall_mean_last_token = (
            float(np.nanmean(all_mean_entropies_last_token))
            if all_mean_entropies_last_token
            else 0.0
        )
        overall_mean_all_tokens = (
            float(np.nanmean(all_mean_entropies_all_tokens))
            if all_mean_entropies_all_tokens
            else 0.0
        )
        overall_mean_last_k_tokens = (
            float(np.nanmean(all_mean_entropies_last_k_tokens))
            if all_mean_entropies_last_k_tokens
            else 0.0
        )
        overall_mean_generated_tokens = (
            float(np.nanmean(all_mean_entropies_generated_tokens))
            if all_mean_entropies_generated_tokens
            else None
        )

        # Most focused layer using all-tokens metric (primary)
        valid_layers_all_tokens = [
            r for r in aggregated_results if not np.isnan(r["mean_entropy_all_tokens"])
        ]
        most_focused_layer = (
            min(valid_layers_all_tokens, key=lambda x: x["mean_entropy_all_tokens"])["layer"]
            if valid_layers_all_tokens
            else None
        )
        most_focused_entropy = (
            min(r["mean_entropy_all_tokens"] for r in valid_layers_all_tokens)
            if valid_layers_all_tokens
            else 0.0
        )

        # Legacy most-focused values for last-token metric
        valid_layers_last_token = [
            r for r in aggregated_results if not np.isnan(r["mean_entropy_last_token"])
        ]
        most_focused_layer_last_token = (
            min(valid_layers_last_token, key=lambda x: x["mean_entropy_last_token"])["layer"]
            if valid_layers_last_token
            else None
        )
        most_focused_entropy_last_token = (
            min(r["mean_entropy_last_token"] for r in valid_layers_last_token)
            if valid_layers_last_token
            else 0.0
        )

        metrics = {
            "num_samples_analyzed": len(sample_results),
            "num_layers_analyzed": len(aggregated_results),
            "num_heads": num_heads,
            # Primary metrics (all-tokens)
            "overall_mean_entropy": float(overall_mean_all_tokens),
            "overall_mean_entropy_all_tokens": float(overall_mean_all_tokens),
            "overall_mean_entropy_last_token": float(overall_mean_last_token),
            "overall_mean_entropy_last_k_tokens": float(overall_mean_last_k_tokens),
            "last_k_tokens": self.last_k_tokens,
            "most_focused_layer": most_focused_layer,
            "most_focused_entropy": float(most_focused_entropy),
            "most_focused_layer_all_tokens": most_focused_layer,
            "most_focused_entropy_all_tokens": float(most_focused_entropy),
            "most_focused_layer_last_token": most_focused_layer_last_token,
            "most_focused_entropy_last_token": float(most_focused_entropy_last_token),
            "analyze_generated_tokens": self.analyze_generated_tokens,
        }
        if overall_mean_generated_tokens is not None:
            metrics["overall_mean_entropy_generated_tokens"] = float(overall_mean_generated_tokens)

        print(f"\nOverall mean entropy (all tokens): {overall_mean_all_tokens:.4f}")
        print(f"Overall mean entropy (last token): {overall_mean_last_token:.4f}")
        print(
            f"Overall mean entropy (last {self.last_k_tokens} tokens): {overall_mean_last_k_tokens:.4f}"
        )
        if overall_mean_generated_tokens is not None:
            print(f"Overall mean entropy (generated tokens): {overall_mean_generated_tokens:.4f}")
        print(
            f"Most focused layer (all tokens): L{most_focused_layer} (entropy: {most_focused_entropy:.4f})"
        )

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics=metrics,
            raw_outputs=aggregated_results,
            metadata={
                "target_layers": self.target_layers,
                "last_k_tokens": self.last_k_tokens,
                "analyze_generated_tokens": self.analyze_generated_tokens,
                "generated_max_new_tokens": self.generated_max_new_tokens,
                "batch_size": self.batch_size,
                "num_samples": len(samples),
                "sample_results": sample_results,  # Include per-sample data
            },
        )
