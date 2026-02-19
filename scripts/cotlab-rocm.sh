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

# Ensure outputs directory exists on host before docker bind-mounts it.
# If this path is missing, docker may create it as root, causing permission issues.
OUTPUTS_DIR="$PROJECT_DIR/outputs"
mkdir -p "$OUTPUTS_DIR"

# Always build
echo "Building CoTLab ROCm Docker image..."
docker compose -f docker-compose.rocm.yml build

# Try to self-heal outputs permissions if they drifted
if [ ! -w "$OUTPUTS_DIR" ]; then
  echo "Warning: '$OUTPUTS_DIR' is not writable. Attempting automatic ownership repair..."
  docker compose -f docker-compose.rocm.yml run --rm \
    --user root \
    --entrypoint bash \
    cotlab \
    -lc "mkdir -p /app/outputs && chown -R ${HOST_UID}:${HOST_GID} /app/outputs" || true
fi

# Fail fast with a clear manual fix if auto-repair did not resolve permissions.
if [ ! -w "$OUTPUTS_DIR" ]; then
  echo "Error: '$OUTPUTS_DIR' is still not writable by $(id -un) (${HOST_UID}:${HOST_GID})."
  echo "Run once to fix ownership:"
  echo "  sudo chown -R ${HOST_UID}:${HOST_GID} \"$OUTPUTS_DIR\""
  echo "  chmod -R u+rwX \"$OUTPUTS_DIR\""
  exit 1
fi

# Run with all arguments passed through
docker compose -f docker-compose.rocm.yml run --rm cotlab "$@"
