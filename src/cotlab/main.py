"""Main entry point for the CoT research framework."""

from __future__ import annotations

import inspect
import os
import random
import re
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .core import create_component
from .experiment import ExperimentDocumenter
from .logging import ExperimentLogger


class _TeeStream:
    """Mirror writes to multiple text streams."""

    def __init__(self, *streams: TextIO):
        self._streams = streams
        self._primary = streams[0]

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        isatty = getattr(self._primary, "isatty", None)
        return bool(isatty()) if callable(isatty) else False

    @property
    def encoding(self) -> str:
        return getattr(self._primary, "encoding", "utf-8")

    def __getattr__(self, name: str):
        # Delegate other stream attributes (e.g. fileno, buffer).
        return getattr(self._primary, name)


@contextmanager
def _tee_stdout_stderr(log_path: Path) -> Iterator[None]:
    """Tee stdout/stderr to the given log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")

    try:
        sys.stdout = _TeeStream(original_stdout, log_file)
        sys.stderr = _TeeStream(original_stderr, log_file)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


def _safe_model_name(model_value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_value).strip("_").lower()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_model_config_file(model_value: str, safe_name: str) -> None:
    """Create a minimal model config for a HF model id if it doesn't exist."""
    if not safe_name:
        return
    config_dir = _repo_root() / "conf" / "model"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{safe_name}.yaml"
    if config_path.exists():
        return

    content = "\n".join(
        [
            "# Auto-generated model config",
            f"name: {model_value}",
            "max_new_tokens: 512",
            "temperature: 0.7",
            "top_p: 0.9",
            f"safe_name: {safe_name}",
            "",
        ]
    )
    config_path.write_text(content)


def _rewrite_hf_model_override(argv: list[str]) -> list[str]:
    """Allow `model=<hf-id>` by rewriting to `model.name=<hf-id>` overrides."""
    rewritten: list[str] = []
    changed = False

    for arg in argv:
        if arg.startswith("model="):
            model_value = arg.split("=", 1)[1]
            # Treat any value with a "/" or absolute/relative path as HF/local model id.
            if "/" in model_value or model_value.startswith(".") or model_value.startswith("/"):
                changed = True
                safe_name = _safe_model_name(model_value)
                _ensure_model_config_file(model_value, safe_name)
                rewritten.append(f"model.name={model_value}")
                # Provide a safe_name override for output paths.
                if safe_name:
                    rewritten.append(f"model.safe_name={safe_name}")
                continue
        rewritten.append(arg)

    if not changed:
        return argv
    return rewritten


def _maybe_rewrite_argv() -> None:
    """Rewrite argv in-place to support `model=<hf-id>` overrides."""
    if os.environ.get("COTLAB_DISABLE_HF_MODEL_REWRITE") == "1":
        return
    sys.argv[:] = _rewrite_hf_model_override(sys.argv)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_backend_load_kwargs(cfg_backend: DictConfig) -> dict:
    backend_cfg = OmegaConf.to_container(cfg_backend, resolve=True)
    keys = [
        "load_in_4bit",
        "load_in_8bit",
        "bnb_4bit_quant_type",
        "bnb_4bit_compute_dtype",
        "bnb_4bit_use_double_quant",
    ]
    return {key: backend_cfg.get(key) for key in keys if backend_cfg.get(key) is not None}


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running experiments.

    Uses Hydra for configuration management. Override configs via CLI:
        python -m cotlab.main model.name=google/gemma-3-1b-it
        python -m cotlab.main -m prompt=chain_of_thought,direct_answer  # multirun
    """
    # Get output directory from Hydra
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    main_log_path = output_dir / "main.log"

    with _tee_stdout_stderr(main_log_path):
        try:
            # Print config if verbose
            if cfg.verbose:
                print("=" * 60)
                print("Configuration:")
                print("=" * 60)
                print(OmegaConf.to_yaml(cfg))
                print("=" * 60)

            # Set seed
            set_seed(cfg.seed)

            # Initialize logger
            logger = ExperimentLogger(str(output_dir))
            logger.log_config(cfg)

            # Initialize experiment documenter and create initial EXPERIMENT.md
            documenter = ExperimentDocumenter(cfg, output_dir)
            doc_path = documenter.save()
            print(f"Created experiment documentation: {doc_path}")

            if cfg.dry_run:
                print("Dry run - exiting without running experiment")
                return

            # Create components from config
            print(f"Loading backend: {cfg.backend._target_}")
            backend = create_component(cfg.backend)

            print(f"Loading model: {cfg.model.name}")
            backend.load_model(cfg.model.name, **_extract_backend_load_kwargs(cfg.backend))

            print(f"Creating prompt strategy: {cfg.prompt.name}")
            prompt_strategy = create_component(cfg.prompt)

            print(f"Loading dataset: {cfg.dataset.name}")
            dataset = create_component(cfg.dataset)

            print(f"Creating experiment: {cfg.experiment.name}")
            experiment = create_component(cfg.experiment)

            # Run experiment
            print("=" * 60)
            print(f"Running experiment: {cfg.experiment.name}")
            print("=" * 60)

            try:
                # Track experiment timing
                import time

                start_time = time.time()

                # Generation kwargs (passed through to backend.generate_batch via experiments that accept **kwargs).
                gen_kwargs: dict = {
                    "max_new_tokens": cfg.model.max_new_tokens,
                    "temperature": cfg.model.temperature,
                    "top_p": cfg.model.top_p,
                }
                # HF Transformers uses `do_sample`; vLLM's SamplingParams does not accept it.
                # For vLLM, greedy decoding is achieved by setting temperature=0.
                if str(cfg.backend._target_).endswith("TransformersBackend"):
                    gen_kwargs["do_sample"] = bool(
                        cfg.model.temperature and cfg.model.temperature > 0
                    )

                # num_samples is optional for some experiments
                num_samples = OmegaConf.select(cfg, "experiment.num_samples", default=None)
                if num_samples is not None and isinstance(num_samples, int) and num_samples > 0:
                    dataset_size = len(dataset)
                    if num_samples > dataset_size:
                        dataset_name = getattr(dataset, "name", type(dataset).__name__)
                        print(
                            "Warning: experiment.num_samples="
                            f"{num_samples} exceeds dataset size ({dataset_size}) "
                            f"for '{dataset_name}'. Using {dataset_size} samples."
                        )
                        num_samples = dataset_size

                # Only pass kwargs that the experiment.run signature can accept.
                # Some analysis experiments do not accept generation args.
                run_sig = inspect.signature(experiment.run)
                run_params = run_sig.parameters
                accepts_var_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in run_params.values()
                )

                run_kwargs = {
                    "backend": backend,
                    "dataset": dataset,
                    "prompt_strategy": prompt_strategy,
                    "logger": logger,
                }
                extra_kwargs = dict(gen_kwargs)
                if num_samples is not None:
                    extra_kwargs["num_samples"] = num_samples

                if accepts_var_kwargs:
                    run_kwargs.update(extra_kwargs)
                else:
                    run_kwargs.update({k: v for k, v in extra_kwargs.items() if k in run_params})

                result = experiment.run(**run_kwargs)

                # Calculate duration
                duration = time.time() - start_time

                # Save results
                results_path = logger.save_results(result)
                print(f"\nResults saved to: {results_path}")

                # Print metrics summary
                print("\nMetrics:")
                for name, value in result.metrics.items():
                    print(f"  {name}: {value}")

                # Update EXPERIMENT.md with results
                results_dict = {**result.metrics, "total_samples": len(result.raw_outputs)}
                updated_doc = documenter.update_with_results(results_dict, duration)
                documenter.save(updated_doc)
                print(f"\nExperiment documentation updated: {doc_path}")

            finally:
                # Cleanup
                backend.unload()
                print("\nExperiment complete.")

        except Exception:
            # Ensure traceback is captured in main.log before Hydra handles it.
            traceback.print_exc()
            raise


if __name__ == "__main__":
    _maybe_rewrite_argv()
    main()
