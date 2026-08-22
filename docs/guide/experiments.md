# Experiments

## Available Experiments

| Experiment | Technique | Purpose |
|------------|-----------|---------|
| `logit_lens` | Early decoding | Layer-by-layer predictions |
| `jacobian_lens` | Jacobian lens | Causal concept readout from residuals; `lrp: true` fits R-lens (LRP/RelP) instead of J-lens |
| `cot_ablation` | Token ablation | Zero CoT tokens, measure effect |
| `cot_heads` | Head patching | Find heads encoding CoT |
| `cot_faithfulness` | Comparison | Compare CoT vs direct answers |
| `sycophancy_heads` | Head patching | Find sycophancy heads |
| `activation_patching` | Residual patching | Causal interventions |
| `steering_vectors` | Activation steering | Control behavior |
| `full_layer_cot` | Layer patching | Patch full layers |
| `probing_classifier` | Probing | Train probes on hidden states |
| `radiology` | Classification | Medical report classification |
| `confabulation_analysis` | Probing | H-Score of H-Neurons across confidence × correctness categories |
| `entropy_neuron_overlap` | Weight analysis | Norm-based overlap between high-norm neurons and H-Neurons |
| `confidence_regulation` | Entropy/frequency-neuron recipe | Two neuron families in one experiment: entropy (norm/LogitVar/null-space ρ, frozen-scale mediation) and token-frequency (v_freq cosine, component-restoration mediation); Jaccard vs probe sets (Stolfo et al., NeurIPS 2024) |
| `probe_confidence` | Probe diagnostics | Correlate probe scores with model output entropy per sample; Spearman/AUROC with length control — is the probe detecting behavior or just low confidence? |

## Creating a probe (for `confabulation_analysis`)

Probes are discovered with [hprobes](https://github.com/huseyincavusbi/hprobes) — it
finds H-Neurons whose activations separate confident-correct from confident-wrong
answers:

```bash
hprobes run --model google/gemma-3-1b-it --data medqa.jsonl --out results/my_probe
```

This writes two files that must stay side by side:

- `results/my_probe.json` — the readable result (H-Neurons, AUROC, config)
- `results/my_probe.safetensors` — the learned classifier coefficients,
  intercept, and per-feature mean/std

Point `probe_path` at the JSON:

```yaml
# conf/experiment/confabulation_analysis.yaml
probe_path: results/my_probe.json
```

The experiment reads the neurons from the JSON and the real learned weights +
standardization stats from the sibling `.safetensors`, so the reported H-Score
is scored exactly like hprobes (`sigmoid(w·(x−mean)/(std+1e-8) + intercept)`).
If no weights are found (no safetensors sibling), it warns and falls back to
uniform weights.

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
