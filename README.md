# CoTLab

A research toolkit for investigating Chain of Thought (CoT) reasoning, faithfulness, and mechanistic interpretability in Large Language Models.

## Features

- **12 Experiments**: CoT ablation, head patching, layer patching, logit lens, steering vectors
- **13 Prompt Strategies**: CoT, Direct, Sycophantic, Adversarial, Uncertainty, and more
- **20 Model Configs**: Gemma 3, MedGemma, DeepSeek-R1, Olmo-Think, Ministral, Nemotron
- **4 Medical Datasets**: 100 samples each (pediatrics, synthetic, patching pairs, radiology)
- **Auto-Detection**: Layers and heads detected at runtime, not hardcoded
- **Hydra Config**: Compose any experiment + model + dataset + prompt

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

| Family | Models |
|--------|--------|
| **Gemma 3** | 270m, 1b, 4b, 12b, 27b (pt + it) |
| **MedGemma** | 4b-pt, 4b-it, 27b-text-it |
| **Reasoning** | DeepSeek-R1-32B, Ministral-14B, Olmo-3/3.1-32B-Think, Nemotron-30B |

> **Note on Model Compatibility:**
> - **Nemotron-30B**: Uses Mamba (SSM) + MoE hybrid architecture. Only `logit_lens` is supported; activation patching fails due to MoE routing.
> - **Ministral-14B**: Multimodal model requiring `AutoModelWithImageTextToText`. Not supported with current text-only backend.

```bash
# Use any model config
python -m cotlab.main model=medgemma_4b
python -m cotlab.main model=deepseek_r1_32b
python -m cotlab.main model=olmo_31_32b_think
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
