#!/bin/bash
set -e

MODEL_DIR="${MODEL_DIR:-/runpod-volume/ckpts}"

echo "===== Downloading HunyuanVideo-1.5 model checkpoints ====="
echo "Destination: $MODEL_DIR"

# Install CLI tools
pip install -U "huggingface_hub[cli]" modelscope --quiet

# Create model directory
mkdir -p "$MODEL_DIR/text_encoder"

# 1. Download VAE + scheduler + config (~200 MB)
echo "[1/5] Downloading VAE, scheduler, config..."
hf download tencent/HunyuanVideo-1.5 \
  --include "vae/*" "scheduler/*" "config.json" "LICENSE" "NOTICE" \
  --local-dir "$MODEL_DIR"

# 2. Download 480p text-to-video transformer only (~16.6 GB)
echo "[2/5] Downloading 480p T2V transformer..."
hf download tencent/HunyuanVideo-1.5 \
  --include "transformer/480p_t2v/*" \
  --local-dir "$MODEL_DIR"

# 3. Download 480p image-to-video transformer (~16.6 GB)
echo "[3/5] Downloading 480p I2V transformer..."
hf download tencent/HunyuanVideo-1.5 \
  --include "transformer/480p_i2v/*" \
  --local-dir "$MODEL_DIR"

# 4. Download MLLM text encoder (Qwen2.5-VL-7B-Instruct, ~14.5 GB)
echo "[4/5] Downloading text encoder (Qwen2.5-VL-7B-Instruct)..."
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "$MODEL_DIR/text_encoder/llm"

# 5. Download byT5-small + Glyph-SDXL-v2 (~500 MB)
echo "[5/5] Downloading byT5-small and Glyph-SDXL-v2..."
hf download google/byt5-small --local-dir "$MODEL_DIR/text_encoder/byt5-small"
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir "$MODEL_DIR/text_encoder/Glyph-SDXL-v2"

echo "===== All checkpoints downloaded to $MODEL_DIR (~48 GB) ====="
echo ""
echo "Directory sizes:"
du -sh "$MODEL_DIR"/*
