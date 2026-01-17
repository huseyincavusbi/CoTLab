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

## GPU Setup (vLLM Backend)

CoTLab supports high-performance inference via vLLM. Installation varies by GPU.

> **For AMD GPU users**: See the comprehensive [ROCm Setup Guide](../rocm-setup.md) for detailed Docker-based installation.

### NVIDIA GPU (CUDA)

```bash
# Standard installation - pulls CUDA-enabled vLLM from PyPI
uv pip install vllm
```

### AMD GPU (ROCm) - Docker (Recommended)

The official way to run vLLM on AMD GPUs is via Docker:

```bash
# Run experiments using the ROCm Docker wrapper
./scripts/cotlab-rocm.sh model=gemma_270m

# First run downloads base image (~10 GB) and compiles kernels (~30 sec)
# Subsequent runs start in seconds (cached)
```

**Base Image**: `rocm/vllm-dev:rocm7.1.1_navi_ubuntu24.04_py3.12_pytorch_2.8_vllm_0.10.2rc1`
- Native RDNA 4 support
- ROCm 7.1.1, PyTorch 2.8, vLLM 0.10.2

**Requirements:**
- Docker installed
- ROCm drivers on host (`rocminfo` should show your GPU)
- User in `docker`, `video`, and `render` groups

## Environment Setup

Create `.env` file with your HuggingFace token:

```bash
HF_TOKEN=your_token_here
```

## Verify

```bash
python -c "import cotlab; print('OK')"