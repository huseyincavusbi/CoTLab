# Configuration

Uses [Hydra](https://hydra.cc/) for configuration.

## Config Files

```
conf/
├── config.yaml        # Main config
├── experiment/        # 14 experiments
├── model/             # 21 models
├── prompt/            # 19 prompt strategies
├── dataset/           # 8 datasets
└── backend/           # vllm, transformers
```

## CLI Override

```bash
python -m cotlab.main \
    experiment=logit_lens \
    model=medgemma_4b \
    dataset=pediatrics \
    backend=transformers
```

## Prompt Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `few_shot` | bool | Include examples |
| `answer_first` | bool | Conclude first, then justify |
| `contrarian` | bool | Skeptical reasoning |
| `output_format` | str | json/toml/yaml/xml/plain |

## Using Custom Models

CoTLab supports ANY vLLM-compatible model:

```bash
# Use any model directly
python -m cotlab.main model.name=meta-llama/Llama-3.1-8B

# Override parameters
python -m cotlab.main \
  model.name=Qwen/Qwen2.5-7B \
  model.max_tokens=4096
```

### Create Custom Config (Optional)

```bash
# Copy base template
cp conf/model/_base/vllm_default.yaml conf/model/my_model.yaml

# Edit parameters
# Then use:
python -m cotlab.main model=my_model
```

See [Models Guide](models.md) for compatibility details.
