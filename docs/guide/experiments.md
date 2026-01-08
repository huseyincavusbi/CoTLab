# Experiments

## Available Experiments

| Experiment | Technique | Purpose |
|------------|-----------|---------|
| `logit_lens` | Early decoding | Layer-by-layer predictions |
| `cot_ablation` | Token ablation | Zero CoT tokens, measure effect |
| `cot_heads` | Head patching | Find heads encoding CoT |
| `cot_faithfulness` | Comparison | Compare CoT vs direct answers |
| `sycophancy_heads` | Head patching | Find sycophancy heads |
| `activation_patching` | Residual patching | Causal interventions |
| `steering_vectors` | Activation steering | Control behavior |
| `full_layer_cot` | Layer patching | Patch full layers |
| `probing_classifier` | Probing | Train probes on hidden states |
| `radiology` | Classification | Medical report classification |

## Running

```bash
python -m cotlab.main experiment=logit_lens model=medgemma_4b
python -m cotlab.main experiment=cot_ablation dataset=pediatrics
```

## Output

Each run creates:

- `results.json` - Data and metrics
- `EXPERIMENT.md` - Auto-generated documentation
- `config.yaml` - Full config used
