"""Probing Classifier Experiment.

Trains linear probes on hidden states at specific layers to test
whether different prompts encode answers differently at each layer.
"""

from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ..backends.base import InferenceBackend
from ..core.base import BaseExperiment, ExperimentResult
from ..core.registry import Registry
from ..datasets.loaders import BaseDataset
from ..logging import ExperimentLogger


class GPULinearProbe(nn.Module):
    """Simple linear classifier for GPU-accelerated probing."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


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
        label_field: str = "category",  # Use category instead of individual diagnosis
        use_gpu_probe: bool = False,  # Use PyTorch GPU probe for speedup
        **kwargs,
    ):
        self._name = name
        self.description = description
        # None means auto-detect all layers at runtime
        self._target_layers_config = target_layers
        self.target_layers = target_layers
        self.num_samples = num_samples
        self.label_field = label_field
        self.use_gpu_probe = use_gpu_probe

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

        # 1. Collect all samples and extract labels based on label_field
        print("Collecting samples...")
        all_samples = list(dataset)
        raw_labels = []
        
        for s in all_samples:
            # Try to get label from specified field
            if self.label_field == "category":
                lbl = s.metadata.get("category", "unknown")
            elif self.label_field == "label":
                lbl = s.label if s.label is not None else s.metadata.get("label", "unknown")
            else:
                lbl = s.metadata.get(self.label_field, "unknown")
            raw_labels.append(str(lbl))

        # 2. Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(raw_labels)
        label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
        print(f"Found {len(label_mapping)} unique labels: {list(label_mapping.keys())}")

        # Count samples per class
        unique, counts = np.unique(encoded_labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        print(f"Class distribution: {class_counts}")

        # 3. Filter to classes with at least 2 samples for stratification
        valid_classes = {cls for cls, count in class_counts.items() if count >= 2}
        
        if len(valid_classes) < 2:
            print("Warning: Not enough classes with 2+ samples. Using all samples without stratification.")
            selected_indices = list(range(min(n_samples, len(all_samples))))
        else:
            # Filter samples to valid classes
            valid_indices = [i for i, lbl in enumerate(encoded_labels) if lbl in valid_classes]
            valid_labels = encoded_labels[valid_indices]
            
            # Sample from valid indices
            if len(valid_indices) <= n_samples:
                selected_indices = valid_indices
            else:
                try:
                    _, selected_indices = train_test_split(
                        valid_indices,
                        test_size=n_samples,
                        stratify=valid_labels,
                        random_state=42,
                    )
                except ValueError:
                    # Fallback to random sampling
                    import random
                    random.seed(42)
                    selected_indices = random.sample(valid_indices, n_samples)

        samples = [all_samples[i] for i in selected_indices]
        sample_labels = encoded_labels[np.array(selected_indices)]

        print(f"Selected {len(samples)} samples for probing.")
        
        # Re-encode labels to be contiguous (important for classification)
        le2 = LabelEncoder()
        sample_labels = le2.fit_transform(sample_labels)
        print(f"Final class count: {len(np.unique(sample_labels))}")

        tokenizer = backend._tokenizer
        model = backend._model

        # Auto-detect all layers if not specified
        if self.target_layers is None:
            # Get number of layers from model config
            config = model.config
            if hasattr(config, 'text_config'):
                config = config.text_config
            num_layers = config.num_hidden_layers
            self.target_layers = list(range(num_layers))
            print(f"Auto-detected {num_layers} layers from model")

        print(f"Model: {backend.model_name}")
        print(f"Target layers: {self.target_layers}")

        # Collect hidden states
        layer_hidden_states = {layer: [] for layer in self.target_layers}

        print("\nExtracting hidden states...")

        for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
            # Build prompt
            question = sample.text
            prompt = prompt_strategy.build_prompt({"question": question, "text": question})
            tokens = tokenizer(prompt, return_tensors="pt").to(backend.device)

            # Get hidden states
            with torch.no_grad():
                outputs = model(**tokens, output_hidden_states=True, return_dict=True)

            hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)

            # Extract last token hidden state from each target layer
            for layer_idx in self.target_layers:
                if layer_idx < len(hidden_states):
                    # Get last token representation (cast to float32 for numpy)
                    h = hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
                    layer_hidden_states[layer_idx].append(h)

        # Labels are already prepared in sample_labels
        labels = sample_labels

        # Train probes for each layer
        print("\n" + "=" * 60)
        print("PROBING CLASSIFIER: Accuracy per Layer")
        print("=" * 60)
        print(
            f"{'Layer':<8} | {'Train Acc':<12} | {'Test Acc':<12} | {'N Train':<8} | {'N Test':<8}"
        )
        print("-" * 60)

        results = []
        layer_accuracies = {}

        for layer_idx in self.target_layers:
            if layer_idx not in layer_hidden_states or not layer_hidden_states[layer_idx]:
                print(f"L{layer_idx:<7} | Skipped - no hidden states available")
                continue
                
            X = np.array(layer_hidden_states[layer_idx])
            y = labels

            if len(np.unique(y)) < 2:
                print(f"L{layer_idx:<7} | Skipped - only one class present")
                continue

            # Split data - use smaller test size if needed
            test_size = min(0.2, max(2, len(X) // 5) / len(X))
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            except ValueError:
                # If stratification fails, try without
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

            # Train probe (GPU or CPU)
            if self.use_gpu_probe:
                train_acc, test_acc = self._train_gpu_probe(
                    X_train, X_test, y_train, y_test, backend.device
                )
            else:
                # Train sklearn logistic regression probe (CPU)
                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(X_train, y_train)
                train_acc = accuracy_score(y_train, clf.predict(X_train))
                test_acc = accuracy_score(y_test, clf.predict(X_test))

            layer_accuracies[layer_idx] = {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }

            print(
                f"L{layer_idx:<7} | {train_acc:<12.4f} | {test_acc:<12.4f} | {len(X_train):<8} | {len(X_test):<8}"
            )

            results.append(
                {
                    "layer": layer_idx,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                }
            )

        print("-" * 60)

        # Find best layer
        if layer_accuracies:
            best_layer = max(
                layer_accuracies.keys(), key=lambda layer: layer_accuracies[layer]["test_accuracy"]
            )
            best_acc = layer_accuracies[best_layer]["test_accuracy"]
            print(f"\nBest probing layer: L{best_layer} (test accuracy: {best_acc:.4f})")
        else:
            best_layer = None
            best_acc = 0

        # Count class distribution in final labels
        unique_final, counts_final = np.unique(labels, return_counts=True)
        
        metrics = {
            "num_samples": len(samples),
            "num_layers_probed": len(results),
            "num_classes": len(unique_final),
            "best_layer": best_layer,
            "best_test_accuracy": best_acc,
            "label_field": self.label_field,
        }

        return ExperimentResult(
            experiment_name=self.name,
            model_name=backend.model_name,
            prompt_strategy=prompt_strategy.name if hasattr(prompt_strategy, "name") else "custom",
            metrics=metrics,
            raw_outputs=results,
            metadata={
                "target_layers": self.target_layers,
                "class_distribution": dict(zip(unique_final.tolist(), counts_final.tolist())),
            },
        )

    def _train_gpu_probe(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        device: str,
    ) -> tuple[float, float]:
        """Train a linear probe using PyTorch on GPU matching sklearn's LogisticRegression."""
        
        # Convert to tensors and normalize (sklearn does this internally)
        X_train_t = torch.from_numpy(X_train).float().to(device)
        X_test_t = torch.from_numpy(X_test).float().to(device)
        y_train_t = torch.from_numpy(y_train).long().to(device)
        y_test_t = torch.from_numpy(y_test).long().to(device)
        
        # Create model
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        model = GPULinearProbe(input_dim, num_classes).to(device)
        
        # Match sklearn's LogisticRegression settings:
        # - C=1.0 (L2 regularization strength = 1/C)
        # - max_iter=1000
        # - tolerance=1e-4
        criterion = nn.CrossEntropyLoss()
        l2_lambda = 1.0  # sklearn's C=1.0 means regularization strength = 1.0
        
        # LBFGS optimizer with sklearn-like settings
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=20,  # iterations per step
            max_eval=None,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=100,
            line_search_fn='strong_wolfe'
        )
        
        # Training loop - call optimizer.step() multiple times until convergence
        max_steps = 50  # max outer steps (total iterations = 20 * 50 = 1000)
        prev_loss = float('inf')
        tolerance = 1e-4
        
        for step in range(max_steps):
            def closure():
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                
                # Add L2 regularization (match sklearn's C=1.0)
                l2_reg = 0.0
                for param in model.parameters():
                    l2_reg += torch.norm(param, 2) ** 2
                loss = loss + (l2_lambda / 2.0) * l2_reg
                
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            
            # Check convergence
            if abs(prev_loss - loss.item()) < tolerance:
                break
            prev_loss = loss.item()
        
        # Evaluate
        with torch.no_grad():
            # Train accuracy
            train_outputs = model(X_train_t)
            train_preds = torch.argmax(train_outputs, dim=1)
            train_acc = (train_preds == y_train_t).float().mean().item()
            
            # Test accuracy
            test_outputs = model(X_test_t)
            test_preds = torch.argmax(test_outputs, dim=1)
            test_acc = (test_preds == y_test_t).float().mean().item()
        
        return train_acc, test_acc
