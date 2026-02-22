# Models

## Model Support

### CoT & Benchmarking: ANY vLLM Model ✓

CoTLab works with **any model on HuggingFace** that vLLM supports.

```bash
# Use directly without config
python -m cotlab.main model.name=meta-llama/Llama-3.1-8B experiment=cot_faithfulness
python -m cotlab.main model.name=Qwen/Qwen2.5-7B experiment=radiology
python -m cotlab.main model.name=mistralai/Mistral-7B-v0.1 experiment=cot_ablation
```

### Mechanistic Experiments: Architecture Dependent

Head patching, activation patching, and logit lens require standard Transformer architecture.

## Pre-configured Models

CoTLab includes configs for commonly used models:

```bash
python -m cotlab.main model=medgemma_4b
python -m cotlab.main model=gemma_1b
python -m cotlab.main model=deepseek_r1_32b
```

See `conf/model/` for available configs.

## Adding New Models

### Option 1: Direct Override (Quick)

No config file needed:

```bash
python -m cotlab.main model.name=your/model-name
```

### Option 2: Create Config (Recommended)

Use base templates:

```bash
# Copy template
cp conf/model/_base/vllm_default.yaml conf/model/my_model.yaml

# Edit the file:
# - Change 'name' to your model
# - Adjust parameters as needed

# Use it:
python -m cotlab.main model=my_model
```

### Option 3: CLI Helper (Optional)

```bash
cotlab-template meta-llama/Llama-3.1-8B
# Creates conf/model/meta_llama_llama_3_1_8b.yaml
```

Use this when you want to pre-create a model config file before the first run.
This is optional: CoTLab can also auto-generate a model config at runtime when
you run with `model=org/repo-id`.

### vLLM Compatibility

Check vLLM supported models: https://docs.vllm.ai/en/latest/models/supported_models.html
