"""Probing Classifier Experiment.

Trains linear probes on hidden states at the ANSWER position to test
whether different prompts encode diagnoses differently at each layer.

This probes AFTER the model generates an answer, not at input position.
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
        description: str = "Train linear probes on answer hidden states",
        target_layers: Optional[List[int]] = None,
        num_samples: int = 30,
        probe_target: str = "diagnosis",  # "diagnosis", "category", or "correctness"
        max_new_tokens: int = 128,
        use_gpu_probe: bool = False,
        batch_size: int = 128,
        random_seed: int = 42,
        **kwargs,
    ):
        self._name = name
        self.description = description
        self._target_layers_config = target_layers
        self.target_layers = target_layers
        self.num_samples = num_samples
        self.probe_target = probe_target
        self.max_new_tokens = max_new_tokens
        self.use_gpu_probe = use_gpu_probe
        self.batch_size = batch_size
        self.random_seed = random_seed

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

        # Collect samples
        print("Collecting samples...")
        all_samples = list(dataset)[:n_samples]
        print(f"Using {len(all_samples)} samples")

        tokenizer = backend.tokenizer
        model = backend.model

        # Set random seeds
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        # Auto-detect layers
        if self.target_layers is None:
            config = model.config
            if hasattr(config, "text_config"):
                config = config.text_config
            num_layers = config.num_hidden_layers
            self.target_layers = list(range(num_layers))
            print(f"Auto-detected {num_layers} layers")

        print(f"Model: {backend.model_name}")
        print(f"Target layers: {len(self.target_layers)} layers")
        print(f"Probe target: {self.probe_target}")

        # Storage
        layer_hidden_states = {layer: [] for layer in self.target_layers}
        labels = []
        correctness_labels = []
        model_answers = []

        print("\nGenerating answers and extracting hidden states at answer position...")

        for sample in tqdm(all_samples, desc="Processing"):
            question = sample.text
            ground_truth = sample.label
            category = sample.metadata.get("category", "unknown")

            # Build prompt
            prompt = prompt_strategy.build_prompt({"question": question, "text": question})
            inputs = tokenizer(prompt, return_tensors="pt").to(backend.device)
            prompt_length = inputs.input_ids.shape[1]

            # Generate answer with hidden states
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Extract generated tokens
            generated_ids = outputs.sequences[0, prompt_length:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
            model_answers.append(answer)

            # Check correctness (simple substring match)
            is_correct = ground_truth.lower() in answer.lower()
            correctness_labels.append(1 if is_correct else 0)

            # Get hidden states at the LAST generated token (answer position)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                last_step_hidden = outputs.hidden_states[-1]

                for layer_idx in self.target_layers:
                    if layer_idx < len(last_step_hidden):
                        h = last_step_hidden[layer_idx][0, -1, :].float().cpu().numpy()
                        layer_hidden_states[layer_idx].append(h)

            # Store label based on probe_target
            if self.probe_target == "diagnosis":
                labels.append(ground_truth)
            elif self.probe_target == "category":
                labels.append(category)
            elif self.probe_target == "correctness":
                labels.append(1 if is_correct else 0)

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Encode labels
        if self.probe_target in ["diagnosis", "category"]:
            le = LabelEncoder()
            encoded_labels = le.fit_transform(labels)
            label_names = list(le.classes_)
        else:
            encoded_labels = np.array(labels)
            label_names = ["incorrect", "correct"]

        print(f"\nLabels: {len(set(encoded_labels))} unique classes")
        print(f"Correctness: {sum(correctness_labels)}/{len(correctness_labels)} correct")

        # Use encoded_labels for probing
        labels = encoded_labels

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

            # Split data
            test_size = 0.25

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_seed, stratify=y
                )
            except ValueError:
                # If stratification fails, try without
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_seed
                )

            # Train probe (GPU or CPU)
            if self.use_gpu_probe:
                train_acc, test_acc = self._train_gpu_probe(
                    X_train, X_test, y_train, y_test, backend.device
                )
            else:
                # Train sklearn logistic regression probe (CPU)
                # Normalize features (same as GPU path) - sklearn doesn't auto-normalize
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                clf = LogisticRegression(max_iter=1000, random_state=self.random_seed)
                clf.fit(X_train_scaled, y_train)
                train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
                test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))

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

        # Count class distribution
        unique_final, counts_final = np.unique(labels, return_counts=True)

        metrics = {
            "num_samples": len(all_samples),
            "num_correct": sum(correctness_labels),
            "accuracy_rate": sum(correctness_labels) / len(correctness_labels)
            if correctness_labels
            else 0,
            "num_layers_probed": len(results),
            "num_classes": len(unique_final),
            "best_layer": best_layer,
            "best_test_accuracy": best_acc,
            "probe_target": self.probe_target,
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
                "label_names": label_names,
                "model_answers": model_answers[:5],  # Save first 5 for inspection
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

        # Set random seed for reproducibility
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        # 1. NORMALIZE features (sklearn does this internally with lbfgs solver)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        X_train_normalized = (X_train - mean) / std
        X_test_normalized = (X_test - mean) / std  # Use train mean/std

        # Convert to tensors
        X_train_t = torch.from_numpy(X_train_normalized).float().to(device)
        X_test_t = torch.from_numpy(X_test_normalized).float().to(device)
        y_train_t = torch.from_numpy(y_train).long().to(device)
        y_test_t = torch.from_numpy(y_test).long().to(device)

        # Create model
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        model = GPULinearProbe(input_dim, num_classes).to(device)

        # Match sklearn's LogisticRegression settings:
        # - C=1.0 (regularization strength = 1/C = 1.0)
        # - max_iter=1000
        # - tolerance=1e-4
        # - penalty='l2' (applied to weights only, NOT bias)
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
            line_search_fn="strong_wolfe",
        )

        # Training loop - call optimizer.step() multiple times until convergence
        max_steps = 50  # max outer steps (total iterations = 20 * 50 = 1000)
        prev_loss = float("inf")
        tolerance = 1e-4

        for step in range(max_steps):

            def closure():
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)

                # 2. L2 regularization on WEIGHTS ONLY (not bias) - matches sklearn
                # sklearn's LogisticRegression does NOT regularize the intercept
                l2_reg = torch.norm(model.linear.weight, 2) ** 2
                loss = loss + (l2_lambda / (2.0 * len(y_train))) * l2_reg

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
