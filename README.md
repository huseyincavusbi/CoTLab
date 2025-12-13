# CoTLab

A research toolkit for investigating Chain of Thought reasoning, faithfulness, and mechanistic interpretability in Large Language Models.

## Features

- **Dual Backend**: vLLM for throughput, Transformers for interpretability
- **Activation Patching**: Residual stream patching with auto-detected layer paths
- **12 Prompt Strategies**: CoT, Direct, Adversarial, Sycophantic, Uncertainty, and more
- **CoT Ablation**: Zero out reasoning tokens to test causal effects
- **Hydra Config**: Compose experiments from YAML configs

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

# Run activation patching on MedGemma
python -m cotlab.main experiment=activation_patching model=medgemma_4b

# Test different prompt strategies
python -m cotlab.main -m prompt=chain_of_thought,direct_answer,adversarial,sycophantic
```

## Prompt Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `chain_of_thought` | Step-by-step reasoning | Baseline CoT behavior |
| `direct_answer` | No explanation, just answer | Compare accuracy without reasoning |
| `no_instruction` | Raw question only | Test default model behavior |
| `arrogance` | Force overconfidence | Test hedging suppression |
| `adversarial` | Hostile/threatening prompts | Safety testing (low/medium/high/extreme) |
| `uncertainty` | Express confidence levels | Calibration testing |
| `socratic` | Ask clarifying questions first | Information gathering |
| `contrarian` | Argue against obvious answer | Test reasoning flexibility |
| `expert_persona` | Specialist personas | Compare cardiologist vs ER vs psychiatrist |
| `sycophantic` | Suggest wrong answer | Test sycophancy vulnerability |
| `few_shot` | Provide examples | In-context learning |
| `simple` | Minimal formatting | Basic prompting |

## Experiments

| Experiment | Description |
|------------|-------------|
| `cot_faithfulness` | Compare CoT vs Direct answers for consistency |
| `cot_ablation` | Zero reasoning tokens to test causal effects |
| `cot_heads` | Find attention heads encoding CoT reasoning |
| `multi_head_cot` | Patch multiple CoT heads simultaneously |
| `activation_patching` | Causal interventions via residual stream patching |
| `sycophancy_heads` | Find attention heads causing sycophancy |
| `multi_head_patching` | Patch multiple sycophancy heads |
| `full_layer_patching` | Patch attention + MLP for complete behavior reversal |
| `steering_vectors` | Inference-time behavior control via activation differences |
| `logit_lens` | Visualize layer-by-layer token predictions |
| `radiology` | Structured JSON classification task |

## Configuration

Override any config via CLI:
```bash
python -m cotlab.main \
    model.name=google/medgemma-4b-it \
    backend.device=mps \
    experiment.num_samples=100
```

## License

MIT
