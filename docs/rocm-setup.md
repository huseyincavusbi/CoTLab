# AMD ROCm Setup Guide

> **Note**: Tested on RDNA 4 in Ubuntu 24.04. Other setups may require different configuration.

## Quick Start

```bash
./scripts/cotlab-rocm.sh model=gemma_270m
```

## Prerequisites

### 1. ROCm Drivers

Verify ROCm is installed and sees your GPU:

```bash
/opt/rocm/bin/rocminfo | grep gfx
```

### 2. Docker Setup

```bash
# Add yourself to required groups
sudo usermod -aG docker,video,render $USER

# Activate
newgrp docker
```

### 3. HuggingFace Cache

```bash
mkdir -p ~/.cache/huggingface
```

## How It Works

CoTLab uses Docker with the official AMD vLLM image:

```
Host System (ROCm drivers)
    │
    └── Docker Container
        ├── rocm/vllm-dev:rocm7.1.1_navi_...
        ├── vLLM pre-compiled
        └── GPU access via /dev/kfd, /dev/dri
```

## Troubleshooting

### Permission Denied

```bash
sudo usermod -aG docker,video,render $USER
newgrp docker
```

### GPU Stuck at 100%

```bash
sudo rocm-smi --resetclocks
```

## Known Limitations

vLLM does not support activation extraction (required for `logit_lens` and similar mechanistic experiments).
