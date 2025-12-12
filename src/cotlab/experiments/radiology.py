"""Radiology-specific experiment using JSON structured output."""

from typing import Any, Dict, List
from tqdm import tqdm

from ..core.base import BaseExperiment, ExperimentResult, BasePromptStrategy
from ..core.registry import Registry
from ..backends.base import InferenceBackend
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger


@Registry.register_experiment("radiology")
class RadiologyExperiment(BaseExperiment):
    """
    Radiology pathological fracture detection experiment.
    
    Uses structured JSON output for reliable answer extraction.
    Compares model prediction (pathological_fracture: true/false) 
    against ground truth labels.
    """
    
    def __init__(
        self,
        name: str = "radiology",
        description: str = "Pathological fracture detection from radiology reports",
        num_samples: int = 50,
        **kwargs
    ):
        self._name = name
        self.description = description
        self.num_samples = num_samples
    
    @property
    def name(self) -> str:
        return self._name
    
    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: BasePromptStrategy,
        num_samples: int = None,
        logger: ExperimentLogger = None,
        **kwargs
    ) -> ExperimentResult:
        """Run the radiology experiment."""
        n_samples = num_samples or self.num_samples
        samples = dataset.sample(n_samples) if n_samples < len(dataset) else list(dataset)
        
        results = []
        metrics = {
            "correct": 0,
            "incorrect": 0,
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "parse_errors": 0,
        }
        
        print(f"Running Radiology Experiment on {len(samples)} samples...")
        
        for sample in tqdm(samples, desc="Processing reports"):
            input_data = {"text": sample.text, "report": sample.text}
            
            # Build prompt and generate
            prompt = prompt_strategy.build_prompt(input_data)
            output = backend.generate(prompt, **kwargs)
            
            # Parse response (RadiologyPromptStrategy returns structured data)
            parsed = prompt_strategy.parse_response(output.text)
            
            # Extract prediction
            if parsed.get("parse_error"):
                metrics["parse_errors"] += 1
                predicted = None
            else:
                predicted = parsed.get("pathological_fracture", False)
            
            # Ground truth (sample.label is True/False for pathological fracture)
            ground_truth = sample.label
            
            # Calculate metrics
            if predicted is not None:
                if predicted == ground_truth:
                    metrics["correct"] += 1
                    if predicted:
                        metrics["true_positives"] += 1
                    else:
                        metrics["true_negatives"] += 1
                else:
                    metrics["incorrect"] += 1
                    if predicted:
                        metrics["false_positives"] += 1
                    else:
                        metrics["false_negatives"] += 1
            
            result = {
                "sample_idx": sample.idx,
                "input": sample.text[:500] + "..." if len(sample.text) > 500 else sample.text,
                "response": output.text,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": predicted == ground_truth if predicted is not None else None,
                "reasoning": parsed.get("reasoning", ""),
                "findings": parsed.get("findings", []),
                "fracture_mentioned": parsed.get("fracture_mentioned", None),
            }
            results.append(result)
            
            if logger:
                logger.log_sample(sample.idx, result)
        
        # Calculate final metrics
        n = len(samples)
        total_valid = metrics["correct"] + metrics["incorrect"]
        
        metrics["accuracy"] = metrics["correct"] / total_valid if total_valid > 0 else 0
        metrics["parse_error_rate"] = metrics["parse_errors"] / n if n > 0 else 0
        
        # Precision, Recall, F1
        tp, fp, fn = metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]
        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"]) \
                        if (metrics["precision"] + metrics["recall"]) > 0 else 0
        
        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name or "unknown",
            prompt_strategy=prompt_strategy.name,
            metrics=metrics,
            raw_outputs=results,
            metadata={
                "num_samples": n,
                "description": self.description
            }
        )
