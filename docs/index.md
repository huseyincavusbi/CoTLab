# CoTLab

Chain of Thought research toolkit for LLM experiments.

## What It Does

- Run mechanistic interpretability experiments on language models
- Test different prompt strategies on medical datasets
- Analyze attention heads, activations, and reasoning patterns

## Install

```bash
git clone https://github.com/huseyincavusbi/CoTLab.git
cd CoTLab
uv venv cotlab --python 3.11
source cotlab/bin/activate
uv pip install -e ".[dev]"
```

## Basic Usage

```bash
python -m cotlab.main experiment=logit_lens model=medgemma_4b
python -m cotlab.main experiment=cot_ablation dataset=pediatrics
python -m cotlab.main prompt=radiology dataset=radiology
```

## Project Structure

```
conf/           # Hydra configuration files
  experiment/   # 14 experiment configs
  model/        # 21 model configs
  prompt/       # 19 prompt configs
  dataset/      # 8 dataset configs
src/cotlab/
  experiments/  # Experiment implementations
  prompts/      # Prompt strategies
  backends/     # vLLM and Transformers backends
  core/         # Base classes
data/           # Datasets (100 samples each)
```

## Experiments

| Experiment | Purpose |
|------------|---------|
| `logit_lens` | Layer-by-layer predictions |
| `cot_ablation` | Remove CoT tokens, measure effect |
| `cot_heads` | Find heads encoding reasoning |
| `sycophancy_heads` | Find sycophancy-related heads |
| `activation_patching` | Causal interventions |
| `steering_vectors` | Control behavior at inference |

## Models

Supports Gemma 3, MedGemma, DeepSeek-R1, Olmo-Think, and more.

## License

MIT
