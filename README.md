# CoTLab

A research toolkit for investigating Chain of Thought (CoT) reasoning, faithfulness, and mechanistic interpretability in Large Language Models.

## Features

- Experiments for CoT faithfulness, patching, logit‑lens, steering, and probing
- Diverse prompt strategies (CoT, direct, adversarial, contrarian, few‑shot, etc.)
- Configurable models, datasets, and backends (vLLM + Transformers)
- Auto‑detect layers/heads at runtime
- Hydra config for easy composition and multiruns
----
- Project overview (DeepWiki): https://deepwiki.com/huseyincavusbi/CoTLab/1-overview
- Official docs: https://huseyincavusbi.github.io/CoTLab/
---
## Installation

```bash
git clone https://github.com/huseyincavusbi/CoTLab.git
cd CoTLab
uv venv cotlab --python 3.11
source cotlab/bin/activate
uv pip install -e ".[dev]"

# GPU Setup:
# NVIDIA: uv pip install vllm
# AMD ROCm: ./scripts/cotlab-rocm.sh (uses Docker)

# AMD ROCm (Transformers backend): install ROCm PyTorch wheels
# uv pip install --reinstall --index-url https://download.pytorch.org/whl/rocm6.4 torch torchvision torchaudio

# Apple Silicon: requires Python 3.12 and vllm-metal plugin
# See docs/getting-started/installation.md for Metal setup instructions
```

See [Installation Docs](docs/getting-started/installation.md) for detailed GPU setup.

## Backend Compatibility

CoTLab supports two inference backends with different strengths:

### 1. vLLM Backend (High Performance)
Best for large-scale generation experiments.
- **Supported Experiments**: `cot_faithfulness`, `radiology`
- **Supported Models**: All text-only models (e.g., `gemma_270m`, `medgemma_27b_text_it`)
- **Limitation**: Does NOT support activation patching or internal state access.
- **Note**: Gemma 3 multimodal models (e.g., `medgemma_4b_it`) are currently incompatible with vLLM 0.12.0 due to architecture detection issues. Use `transformers` backend for these.

### 2. Transformers Backend (Full Access)
Best for mechanistic interpretability and activation patching.
- **Supported Experiments**: ALL experiments.
- **Supported Models**: ALL models.
- **Limitation**: Slower.

To switch backends:
```bash
# Use vLLM (fast generation)
python -m cotlab.main backend=vllm ...

# Use Transformers (activation access)
python -m cotlab.main backend=transformers ...
```

## Quick Start

```bash
# Run logit lens on MedGemma
python -m cotlab.main experiment=logit_lens model=medgemma_4b

# Find sycophancy heads
python -m cotlab.main experiment=sycophancy_heads model=medgemma_4b

# Test CoT ablation on pediatrics dataset
python -m cotlab.main experiment=cot_ablation dataset=pediatrics

# Compare prompt strategies
python -m cotlab.main -m prompt=chain_of_thought,direct_answer,sycophantic
```

## Supported Models

CoTLab ships config files for some models, but in principle it supports any model
that the selected backend can load. Mechanistic experiments can still fail for
models with unusual architectures.

You can add a model config file for more control over hyperparameters, but you
can also run any experiment by passing a Hugging Face model name directly.

```bash
# Use a built-in model config
python -m cotlab.main model=medgemma_4b

# Or pass any HF model name directly
python -m cotlab.main model.name=google/gemma-3-270m
```

## Datasets

| Dataset | Samples | Domain |
|---------|---------|--------|
| `pediatrics` | 100 | General pediatrics scenarios |
| `synthetic` | 100 | General medical QA |
| `patching_pairs` | 100 | Clean/corrupted pairs for activation patching |
| `radiology` | 9 | Radiology reports for testing |

## Experiments

| Experiment | Technique | Purpose |
|------------|-----------|---------|
| `cot_ablation` | Token ablation | Zero CoT tokens, measure effect |
| `cot_heads` | Head patching | Find heads encoding CoT |
| `logit_lens` | Early decoding | See layer-by-layer predictions |
| `sycophancy_heads` | Head patching | Find sycophancy heads |
| `steering_vectors` | Activation steering | Control behavior at inference |
| `full_layer_cot` | Layer patching | Patch full layers for CoT |
| `activation_patching` | Residual patching | Causal interventions |

## Prompt Strategies

| Strategy | Description |
|----------|-------------|
| `chain_of_thought` | Step-by-step reasoning |
| `direct_answer` | Answer only, no explanation |
| `sycophantic` | Suggest wrong answer |
| `adversarial` | Challenge the model |
| `uncertainty` | Express confidence levels |
| `expert_persona` | Specialist personas |
| `few_shot` | Include examples |

## Configuration

All configs auto-detect layers/heads at runtime. Override via CLI:

```bash
python -m cotlab.main \
    model=medgemma_4b \
    dataset=pediatrics \
    prompt=chain_of_thought \
    experiment.top_k=10
```

## License

MIT
