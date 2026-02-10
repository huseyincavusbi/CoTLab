#!/bin/bash
# Run CoTLab experiments on AMD GPU using Docker
#
# Usage:
#   ./scripts/cotlab-rocm.sh model=gemma_270m
#   ./scripts/cotlab-rocm.sh experiment=faithfulness model=medgemma_4b

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Ensure HOST_UID/HOST_GID are set for docker-compose user mapping.
# Note: bash sets UID as a readonly variable, so we must not assign to it.
if [ -z "${HOST_UID:-}" ]; then
  HOST_UID="$(id -u)"
  export HOST_UID
fi
if [ -z "${HOST_GID:-}" ]; then
  HOST_GID="$(id -g)"
  export HOST_GID
fi

# Always build (fast with Docker cache)
echo "Building CoTLab ROCm Docker image..."
docker compose -f docker-compose.rocm.yml build

# Run with all arguments passed through
docker compose -f docker-compose.rocm.yml run --rm cotlab "$@"
