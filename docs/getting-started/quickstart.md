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

## Batch Experiments

Use Hydra multirun (`-m`) or a simple shell loop for batches.

```bash
# Sweep prompt flags (Hydra multirun)
python -m cotlab.main -m \
  experiment=classification \
  dataset=medqa \
  prompt=mcq \
  prompt.answer_first=true,false \
  prompt.few_shot=true \
  experiment.num_samples=20

# Sweep datasets (Hydra multirun)
python -m cotlab.main -m \
  experiment=classification \
  prompt=mcq \
  dataset=medqa,medmcqa,medxpertqa,mmlu_medical,afrimedqa \
  prompt.few_shot=true \
  experiment.num_samples=20

# Shell loop (no Hydra multirun)
for ds in medqa medmcqa medxpertqa mmlu_medical afrimedqa; do
  python -m cotlab.main experiment=classification dataset=$ds prompt=mcq prompt.few_shot=true experiment.num_samples=20
done
```

Multirun outputs are stored under `multirun/YYYY-MM-DD/HH-MM-SS/`.

## Output

Results saved to `outputs/YYYY-MM-DD/HH-MM-SS/`:

- `results.json` - Predictions and metrics
- `EXPERIMENT.md` - Run documentation
- `config.yaml` - Config used
