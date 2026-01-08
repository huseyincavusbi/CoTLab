# Quick Start

## Running Experiments

```bash
# Logit lens on MedGemma
python -m cotlab.main experiment=logit_lens model=medgemma_4b

# CoT ablation on pediatrics dataset
python -m cotlab.main experiment=cot_ablation dataset=pediatrics

# Radiology classification
python -m cotlab.main prompt=radiology dataset=radiology
```

## Overriding Config

```bash
# Change model
python -m cotlab.main experiment=logit_lens model=gemma_1b

# Change backend
python -m cotlab.main experiment=logit_lens backend=vllm

# Multiple runs
python -m cotlab.main -m prompt=radiology,cardiology
```

## Output

Results saved to `outputs/YYYY-MM-DD/HH-MM-SS/`:

- `results.json` - Predictions and metrics
- `EXPERIMENT.md` - Run documentation
- `config.yaml` - Config used
