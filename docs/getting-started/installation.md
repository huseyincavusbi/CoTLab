# Installation

## Requirements

- Python 3.11+
- GPU recommended for model inference

## Install with uv (Recommended)

```bash
git clone https://github.com/huseyincavusbi/CoTLab.git
cd CoTLab
uv venv cotlab --python 3.11
source cotlab/bin/activate
uv pip install -e ".[dev]"
```

## Install with pip

```bash
git clone https://github.com/huseyincavusbi/CoTLab.git
cd CoTLab
python -m venv cotlab
source cotlab/bin/activate
pip install -e ".[dev]"
```

## Install with conda/mamba

```bash
git clone https://github.com/huseyincavusbi/CoTLab.git
cd CoTLab
conda create -n cotlab python=3.11
conda activate cotlab
pip install -e ".[dev]"
```

Or with mamba (faster):

```bash
mamba create -n cotlab python=3.11
mamba activate cotlab
pip install -e ".[dev]"
```

## Environment Setup

Create `.env` file with your HuggingFace token:

```bash
HF_TOKEN=your_token_here
```

## Verify

```bash
python -c "import cotlab; print('OK')"
```
