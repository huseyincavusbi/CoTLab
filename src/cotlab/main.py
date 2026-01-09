"""Main entry point for the CoT research framework."""

import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .core import create_component
from .experiment import ExperimentDocumenter
from .logging import ExperimentLogger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running experiments.

    Uses Hydra for configuration management. Override configs via CLI:
        python -m cotlab.main model.name=google/gemma-3-1b-it
        python -m cotlab.main -m prompt=chain_of_thought,direct_answer  # multirun
    """
    # Print config if verbose
    if cfg.verbose:
        print("=" * 60)
        print("Configuration:")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 60)

    # Set seed
    set_seed(cfg.seed)

    # Get output directory from Hydra
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

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
    backend.load_model(cfg.model.name)

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

        # num_samples is optional for some experiments
        num_samples = OmegaConf.select(cfg, "experiment.num_samples", default=None)
        result = experiment.run(
            backend=backend,
            dataset=dataset,
            prompt_strategy=prompt_strategy,
            logger=logger,
            **(dict(num_samples=num_samples) if num_samples is not None else {}),
        )

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


if __name__ == "__main__":
    main()
