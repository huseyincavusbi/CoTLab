# CoTLab

A research toolkit for investigating Chain of Thought reasoning, faithfulness, and mechanistic interpretability in Large Language Models.

## Features

- **Dual Backend**: vLLM for throughput, Transformers for interpretability
- **Activation Patching**: From-scratch PyTorch hooks for causal interventions
- **Modular Prompts**: Easily swap between CoT, direct, arrogance strategies
- **Hydra Config**: Compose experiments from YAML configs
- **JSON Structured Output**: Reliable answer extraction for evaluation

## Installation

```bash
uv venv cotlab --python 3.11
source cotlab/bin/activate
uv pip install -e ".[dev]"
```

## Quick Start

```bash
# Run experiment with any HuggingFace model
python -m cotlab.main model.name=google/gemma-3-1b-it

# Run CoT faithfulness experiment
python -m cotlab.main experiment=cot_faithfulness

# Run radiology experiment with structured JSON output
python -m cotlab.main experiment=radiology prompt=radiology dataset=radiology

# Sweep prompt strategies
python -m cotlab.main -m prompt=chain_of_thought,direct_answer,simple
```

## Configuration

Override any config via CLI:
```bash
python -m cotlab.main \
    model.name=meta-llama/Llama-3.1-8B-Instruct \
    backend.device=cuda \
    experiment.num_samples=100
```

## Project Structure

```
src/cotlab/
├── backends/      # vLLM & Transformers inference
├── patching/      # Activation patching (PyTorch hooks)
├── prompts/       # Prompt strategies
├── datasets/      # Dataset loaders
├── experiments/   # Research experiments
├── analysis/      # CoT parsing & metrics
└── logging/       # JSON structured logging
```

## Experiments

| Experiment | Description |
|------------|-------------|
| `cot_faithfulness` | Compare CoT vs Direct answers |
| `activation_patching` | Causal interventions via layer patching |
| `radiology` | Structured JSON classification task |

## License

MIT
