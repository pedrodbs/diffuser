#!/bin/bash

# change to root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${DIR}/../.." || exit
clear

# parameters
OUTPUT="output/diffuser"
CONFIG="config.maze2d"
DATASET="maze2d-large-v1"

GPU=1 #0
export CUDA_VISIBLE_DEVICES="${GPU}"

python diffuser/scripts/train.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --logbase "${OUTPUT}" \
  --device "cuda:0"

echo "========================================"
echo "Saving pip packages..."
pip freeze >"${OUTPUT}/packages.txt"
