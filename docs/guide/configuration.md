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
