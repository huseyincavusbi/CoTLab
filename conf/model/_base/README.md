# Base Model Templates

Use these templates to quickly add support for new models.

## Templates

- `vllm_default.yaml` - For any vLLM-compatible model (CoT/benchmarking)
- `transformers_default.yaml` - For Transformers backend (mechanistic experiments)

## Quick Start

```bash
# Copy template
cp conf/model/_base/vllm_default.yaml conf/model/my_model.yaml

# Edit the model name and parameters
# Then use:
python -m cotlab.main model=my_model
```

## CLI Helper

```bash
# Auto-generate from template
cotlab-template meta-llama/Llama-3.1-8B
```
