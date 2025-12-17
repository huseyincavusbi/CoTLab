import argparse
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf

from cotlab.core import create_component
from cotlab.logging import ExperimentLogger


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run CoTLab experiment batch")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "transformers", "transformers_mps"],
        help="Backend to use (default: vllm)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medgemma_27b_text_it",
        help="Model config name (default: medgemma_27b_text_it)",
    )
    args = parser.parse_args()

    backend_name = args.backend
    model_name = args.model

    # Setup base output directory with model name
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    base_output_dir = Path(f"outputs/{timestamp}_{model_name}_{backend_name}")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {base_output_dir}")

    # Initialize Hydra
    with initialize(version_base=None, config_path="../../conf"):
        # 1. Load Backend & Model ONCE
        print("=" * 60)
        print(f"Initializing Backend ({backend_name}) and Model ({model_name})...")
        print("=" * 60)

        # Base config to load model
        base_cfg = compose(
            config_name="config",
            overrides=[
                f"backend={backend_name}",
                f"model={model_name}",
                "experiment=cot_faithfulness",  # Dummy
                "dataset=pediatrics",  # Dummy
                "prompt=chain_of_thought",  # Dummy
            ],
        )

        backend = create_component(base_cfg.backend)
        backend.load_model(base_cfg.model.name)
        print("Backend initialized successfully.")

        # Define Experiment Grid
        grid = [
            # 1. Main CoT Faithfulness Sweep
            {
                "experiment": "cot_faithfulness",
                "datasets": ["pediatrics", "synthetic", "patching_pairs", "radiology"],
                "prompts": [
                    "chain_of_thought",
                    "direct_answer",
                    "sycophantic",
                    "adversarial",
                    "uncertainty",
                    "expert_persona",
                    "few_shot",
                    "contrarian",
                    "arrogance",
                    "simple",
                    "socratic",
                    "no_instruction",
                    "radiology",
                ],
            },
            # 2. Specialized Radiology Experiment
            {"experiment": "radiology", "datasets": ["radiology"], "prompts": ["radiology"]},
        ]

        total_jobs = sum(len(g["datasets"]) * len(g["prompts"]) for g in grid)
        print(f"Starting execution of {total_jobs} jobs...")

        job_idx = 0
        for group in grid:
            exp_name = group["experiment"]

            for dataset_name in group["datasets"]:
                for prompt_name in group["prompts"]:
                    job_idx += 1
                    print(
                        f"\n[Job {job_idx}/{total_jobs}] exp={exp_name} dataset={dataset_name} prompt={prompt_name}"
                    )

                    # Create specific config override
                    overrides = [
                        f"backend={backend_name}",
                        f"model={model_name}",
                        f"experiment={exp_name}",
                        f"dataset={dataset_name}",
                        f"prompt={prompt_name}",
                    ]

                    cfg = compose(config_name="config", overrides=overrides)

                    # Instantiate Components for this run
                    try:
                        dataset = create_component(cfg.dataset)
                        prompt_strategy = create_component(cfg.prompt)
                        experiment = create_component(cfg.experiment)

                        # Setup individual run logger
                        run_name = f"{exp_name}_{dataset_name}_{prompt_name}"
                        run_dir = base_output_dir / run_name
                        run_dir.mkdir(exist_ok=True)

                        logger = ExperimentLogger(str(run_dir))
                        logger.log_config(cfg)

                        # Run Experiment
                        num_samples = OmegaConf.select(cfg, "experiment.num_samples", default=None)
                        result = experiment.run(
                            backend=backend,
                            dataset=dataset,
                            prompt_strategy=prompt_strategy,
                            logger=logger,
                            **(dict(num_samples=num_samples) if num_samples is not None else {}),
                        )

                        # Save results
                        results_path = logger.save_results(result)
                        print(f"Results saved: {results_path}")

                        # Basic metrics print
                        metrics_str = ", ".join(
                            [
                                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in result.metrics.items()
                            ]
                        )
                        print(f"Metrics: {metrics_str}")

                    except Exception as e:
                        print(f"ERROR in job {job_idx}: {e}")
                        import traceback

                        traceback.print_exc()

        print("\n" + "=" * 60)
        print("All jobs completed.")
        print(f"Results stored in: {base_output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
