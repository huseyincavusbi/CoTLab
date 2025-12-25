"""Probing Classifier Experiment.

Trains linear probes on hidden states at specific layers to test
whether different prompts encode answers differently at each layer.
"""

from typing import Any, Dict, List, Optional

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger


@Registry.register_experiment("probing_classifier")
class ProbingClassifierExperiment(BaseExperiment):
    """
    Train linear probes to test answer encoding at different layers.
    
    Extracts hidden states at specified layers and trains LogisticRegression
    to predict labels, measuring how well each layer encodes the answer.
    """

    def __init__(
        self,
        name: str = "probing_classifier",
        description: str = "Train linear probes on hidden states",
        target_layers: Optional[List[int]] = None,
        num_samples: int = 50,
        **kwargs,
    ):
        self._name = name
        self.description = description
        # Default to layers 30, 40, 50, 58 (spanning early to critical)
        self._target_layers_config = target_layers or [30, 40, 50, 58]
        self.target_layers = self._target_layers_config
        self.num_samples = num_samples

    @property
    def name(self) -> str:
        return self._name

    def run(
        self,
        backend: InferenceBackend,
        dataset: BaseDataset,
        prompt_strategy: Any,
        num_samples: Optional[int] = None,
        logger: Optional[ExperimentLogger] = None,
    ) -> ExperimentResult:
        """Run probing classifier experiment."""

        n_samples = num_samples or self.num_samples
        samples = dataset.sample(n_samples) if n_samples < len(dataset) else list(dataset)

        tokenizer = backend._tokenizer
        model = backend._model

        print(f"Model: {backend.model_name}")
        print(f"Target layers: {self.target_layers}")
        print(f"Samples: {len(samples)}")

        # Collect hidden states and labels
        layer_hidden_states = {layer: [] for layer in self.target_layers}
        labels = []

        print("\nExtracting hidden states...")
        
        for i, sample in enumerate(samples):
            if i % 10 == 0:
                print(f"  Processing sample {i+1}/{len(samples)}")
            
            # Build prompt
            question = sample.text
            prompt = prompt_strategy.build_prompt({"question": question})
            tokens = tokenizer(prompt, return_tensors="pt").to(backend.device)
            
            # Get hidden states
            with torch.no_grad():
                outputs = model(
                    **tokens,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)
            
            # Extract last token hidden state from each target layer
            for layer_idx in self.target_layers:
                if layer_idx < len(hidden_states):
                    # Get last token representation (cast to float32 for numpy)
                    h = hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
                    layer_hidden_states[layer_idx].append(h)
            
            # Get label from sample metadata
            label = sample.metadata.get("label", sample.metadata.get("ground_truth", 0))
            if isinstance(label, str):
                label = 1 if label.lower() in ["yes", "true", "positive", "1"] else 0
            labels.append(label)

        labels = np.array(labels)
        
        # Train probes for each layer
        print("\n" + "=" * 60)
        print("PROBING CLASSIFIER: Accuracy per Layer")
        print("=" * 60)
        print(f"{'Layer':<8} | {'Train Acc':<12} | {'Test Acc':<12} | {'N Train':<8} | {'N Test':<8}")
        print("-" * 60)

        results = []
        layer_accuracies = {}

        for layer_idx in self.target_layers:
            X = np.array(layer_hidden_states[layer_idx])
            y = labels
            
            if len(np.unique(y)) < 2:
                print(f"L{layer_idx:<7} | Skipped - only one class present")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train logistic regression probe
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            test_acc = accuracy_score(y_test, clf.predict(X_test))
            
            layer_accuracies[layer_idx] = {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
            
            print(f"L{layer_idx:<7} | {train_acc:<12.4f} | {test_acc:<12.4f} | {len(X_train):<8} | {len(X_test):<8}")
            
            results.append({
                "layer": layer_idx,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "n_train": len(X_train),
                "n_test": len(X_test),
            })

        print("-" * 60)

        # Find best layer
        if layer_accuracies:
            best_layer = max(layer_accuracies.keys(), 
                           key=lambda l: layer_accuracies[l]["test_accuracy"])
            best_acc = layer_accuracies[best_layer]["test_accuracy"]
            print(f"\nBest probing layer: L{best_layer} (test accuracy: {best_acc:.4f})")
        else:
            best_layer = None
            best_acc = 0

        metrics = {
            "num_samples": len(samples),
            "num_layers_probed": len(results),
            "best_layer": best_layer,
            "best_test_accuracy": best_acc,
            "label_distribution": {
                "positive": int(np.sum(labels)),
                "negative": int(len(labels) - np.sum(labels)),
            }
        }

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics=metrics,
            raw_outputs=results,
            metadata={
                "target_layers": self.target_layers,
            },
        )
