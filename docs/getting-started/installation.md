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

### Apple Silicon (Metal)

vLLM runs on the Apple GPU through the **vllm-metal** plugin (MLX-accelerated).
Requirements: **Apple Silicon**, **macOS 15 (Sequoia) or later**, **Python 3.12**.

> **Note:** Upstream vLLM ships its macOS wheel tagged `+cpu` (upstream has no
> Metal kernels). That tag only describes the engine host — the `vllm-metal`
> plugin supplies the Metal/MLX compute path, so `backend=vllm` runs on the GPU,
> not the CPU.

Install with the official installer (recommended, no compiler — ships the
matching prebuilt vLLM + plugin wheels into a dedicated environment):

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
```

> Alternatively, Homebrew provides the `vllm` CLI:
> ```bash
> brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
> brew install vllm-project/vllm-metal/vllm-metal
> ```

Add CoTLab to that environment (its runtime deps are already present, so skip
dependency resolution to avoid clobbering the plugin's pinned `torch`):

```bash
cd /path/to/CoTLab
uv pip install --no-deps -e .
uv pip install omegaconf hydra-core
```

Verify both the plugin and CoTLab's backend:

```bash
python -c "from vllm import LLM; import vllm_metal; print('Metal ready!')"
python -c "from cotlab.backends.vllm_backend import VLLMBackend; print('CoTLab vLLM backend OK')"
```

Metal is auto-detected when you run with `backend=vllm` — no additional
configuration needed. You will see `MLX device set to: Device(gpu, 0)` and
`PyTorch device set to: mps` in the logs when it is running on the GPU.

## Environment Setup

Create `.env` file with your HuggingFace token:

```bash
HF_TOKEN=your_token_here
```

## Verify

```bash
python -c "import cotlab; print('OK')"
```
