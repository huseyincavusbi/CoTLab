"""JSON-based experiment logging."""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..core.base import ExperimentResult


class ExperimentLogger:
    """
    Log experiments to JSON files.

    Provides structured logging for:
    - Experiment configurations
    - Individual sample results
    - Aggregate metrics
    - Intermediate checkpoints

    Example:
        >>> logger = ExperimentLogger("outputs/2024-01-01")
        >>> logger.log_config(cfg)
        >>> logger.log_sample(0, {"input": "...", "output": "..."})
        >>> logger.save_results(experiment_result)
    """

    def __init__(self, output_dir: str):
        """
        Initialize logger with output directory.

        Args:
            output_dir: Directory to write logs to
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._samples: List[Dict[str, Any]] = []
        self._metadata: Dict[str, Any] = {}
        self._start_time = datetime.now()

    def log_config(self, config: Any) -> None:
        """
        Log experiment configuration.

        Args:
            config: Hydra DictConfig or dict
        """
        from omegaconf import OmegaConf

        if hasattr(config, "_content"):  # OmegaConf DictConfig
            config_dict = OmegaConf.to_container(config, resolve=True)
        elif is_dataclass(config):
            config_dict = asdict(config)
        else:
            config_dict = dict(config)

        self._metadata["config"] = config_dict
        self._metadata["start_time"] = self._start_time.isoformat()

        # Save config immediately
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    def log_sample(self, idx: int, sample_data: Dict[str, Any], checkpoint: bool = False) -> None:
        """
        Log a single sample result.

        Args:
            idx: Sample index
            sample_data: Sample input/output data
            checkpoint: Whether to save to disk immediately
        """
        sample_data["idx"] = idx
        sample_data["timestamp"] = datetime.now().isoformat()
        self._samples.append(sample_data)

        if checkpoint:
            self._save_checkpoint()

    def log_intermediate(self, step: str, data: Dict[str, Any]) -> None:
        """
        Log intermediate results (e.g., per-layer patching results).

        Args:
            step: Step identifier
            data: Data to log
        """
        intermediate_path = self.output_dir / f"intermediate_{step}.json"
        with open(intermediate_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def save_results(self, result: ExperimentResult, filename: str = "results.json") -> Path:
        """
        Save final experiment results.

        Args:
            result: ExperimentResult dataclass
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        # Combine metadata with result
        full_result = {
            "metadata": self._metadata,
            "experiment": result.experiment_name,
            "model": result.model_name,
            "prompt_strategy": result.prompt_strategy,
            "metrics": result.metrics,
            "raw_outputs": result.raw_outputs,  # Include all layer results
            "num_samples": len(self._samples),
            "samples": self._samples,
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self._start_time).total_seconds(),
        }

        with open(output_path, "w") as f:
            json.dump(full_result, f, indent=2, default=str)

        return output_path

    def save_summary(self, metrics: Dict[str, Any], filename: str = "summary.json") -> Path:
        """
        Save a metrics summary without full sample data.

        Args:
            metrics: Computed metrics
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(self._samples),
            "metrics": metrics,
            "config": self._metadata.get("config", {}),
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return output_path

    def _save_checkpoint(self) -> None:
        """Save current samples as checkpoint."""
        checkpoint_path = self.output_dir / "checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(
                {"samples": self._samples, "timestamp": datetime.now().isoformat()},
                f,
                indent=2,
                default=str,
            )
