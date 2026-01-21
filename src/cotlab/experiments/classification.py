"""Generic classification experiment for medical reports (binary and multi-class)."""

from collections import Counter
from typing import Optional

from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, BasePromptStrategy, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger


@Registry.register_experiment("classification")
class ClassificationExperiment(BaseExperiment):
    """
    Generic classification experiment for medical reports.

    Supports both binary classification (True/False labels) and
    multi-class classification (string labels like cancer types).
    Uses structured JSON output for reliable answer extraction.
    """

    def __init__(
        self,
        name: str = "classification",
        description: str = "Classification from medical reports",
        num_samples: int = -1,  # Default to -1 (all samples)
        **kwargs,
    ):
        self._name = name
        self.description = description
        self.num_samples = num_samples

    @property
    def name(self) -> str:
        return self._name

    def _compute_multiclass_metrics(self, y_true: list, y_pred: list, labels: list) -> dict:
        """Compute multi-class classification metrics."""
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        metrics = {}

        # Per-class metrics via classification_report
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        metrics["classification_report"] = report

        # Macro and weighted averages
        metrics["macro_precision"] = precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )
        metrics["macro_recall"] = recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )
        metrics["macro_f1"] = f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )
        metrics["weighted_f1"] = f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["class_labels"] = labels

        # Find top confused pairs
        confused_pairs = []
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                if i != j and cm[i][j] > 0:
                    confused_pairs.append((true_label, pred_label, int(cm[i][j])))
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        metrics["top_confused_pairs"] = confused_pairs[:10]  # Top 10

        # Class distribution
        true_dist = Counter(y_true)
        pred_dist = Counter(y_pred)
        metrics["true_class_distribution"] = dict(true_dist)
        metrics["pred_class_distribution"] = dict(pred_dist)

        return metrics

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: BasePromptStrategy,
        num_samples: Optional[int] = None,
        logger: Optional[ExperimentLogger] = None,
        **kwargs,
    ) -> ExperimentResult:
        """Run the classification experiment."""
        n_samples = num_samples if num_samples is not None else self.num_samples

        if n_samples > 0 and n_samples < len(dataset):
            samples = dataset.sample(n_samples)
        else:
            samples = list(dataset)

        results = []
        metrics = {
            "correct": 0,
            "incorrect": 0,
            "parse_errors": 0,
        }

        # For multi-class metrics
        y_true = []
        y_pred = []

        # Get prediction field from prompt strategy
        prediction_field = getattr(
            prompt_strategy, "get_prediction_field", lambda: "pathological_fracture"
        )()
        print(f"Running Classification Experiment on {len(samples)} samples...")
        print(f"  Prediction field: {prediction_field}")

        # Prepare inputs
        inputs = [{"text": s.text, "report": s.text, "metadata": s.metadata} for s in samples]

        # Batch Generate
        print("Generating responses...")
        prompts = [prompt_strategy.build_prompt(i) for i in inputs]
        outputs = backend.generate_batch(prompts, **kwargs)

        # Process results
        print("Analyzing results...")
        for i, sample in enumerate(tqdm(samples, desc="Analyzing reports")):
            output = outputs[i]
            prompt = prompts[i]

            # Parse response
            parsed = prompt_strategy.parse_response(output.text)

            # Extract prediction
            if parsed.get("parse_error"):
                metrics["parse_errors"] += 1
                predicted = None
            else:
                predicted = parsed.get(prediction_field, None)

            # Ground truth
            ground_truth = sample.label

            # Calculate metrics
            if predicted is not None:
                y_true.append(ground_truth)
                y_pred.append(predicted)

                if predicted == ground_truth:
                    metrics["correct"] += 1
                else:
                    metrics["incorrect"] += 1

            result = {
                "sample_idx": sample.idx,
                "input": sample.text[:500] + "..." if len(sample.text) > 500 else sample.text,
                "prompt": prompt,
                "response": output.text,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": predicted == ground_truth if predicted is not None else None,
                "reasoning": parsed.get("reasoning", ""),
            }
            results.append(result)

            if logger:
                logger.log_sample(sample.idx, result)

        # Calculate final metrics
        n = len(samples)
        total_valid = metrics["correct"] + metrics["incorrect"]

        metrics["accuracy"] = metrics["correct"] / total_valid if total_valid > 0 else 0
        metrics["parse_error_rate"] = metrics["parse_errors"] / n if n > 0 else 0

        # Determine if multi-class or binary
        is_multiclass = len(y_true) > 0 and isinstance(y_true[0], str)

        if is_multiclass and len(y_true) > 0:
            # Multi-class metrics
            all_labels = sorted(set(y_true + y_pred))
            multiclass_metrics = self._compute_multiclass_metrics(y_true, y_pred, all_labels)
            metrics.update(multiclass_metrics)
            metrics["num_classes"] = len(all_labels)
        else:
            # Binary metrics (legacy support)
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
            tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)

            metrics["true_positives"] = tp
            metrics["true_negatives"] = tn
            metrics["false_positives"] = fp
            metrics["false_negatives"] = fn
            metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics["f1"] = (
                2
                * (metrics["precision"] * metrics["recall"])
                / (metrics["precision"] + metrics["recall"])
                if (metrics["precision"] + metrics["recall"]) > 0
                else 0
            )

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name or "unknown",
            prompt_strategy=prompt_strategy.name,
            metrics=metrics,
            raw_outputs=results,
            metadata={"num_samples": n, "description": self.description},
        )
