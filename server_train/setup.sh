#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Setup môi trường cho CodeLlama-13B Training
# Chạy 1 lần trước khi train: bash setup.sh
# ══════════════════════════════════════════════════════════════════════════════

set -e

echo "════════════════════════════════════════════════════════════════════"
echo "  Setup môi trường cho CodeLlama-13B v12 Training"
echo "════════════════════════════════════════════════════════════════════"

# Kiểm tra GPU
echo ""
echo ">>> Kiểm tra GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️  nvidia-smi không tìm thấy! Cần GPU NVIDIA."
    exit 1
fi

# Cài đặt packages
echo ""
echo ">>> Cài đặt Python packages..."
pip install -q -U \
    "bitsandbytes>=0.46.1" \
    "transformers>=4.48.0" \
    "accelerate>=1.0.1" \
    "peft>=0.13.2" \
    "tokenizers>=0.20.3" \
    scikit-learn \
    sentencepiece

# Kiểm tra bitsandbytes CUDA
echo ""
echo ">>> Kiểm tra bitsandbytes CUDA..."
python3 -c "
import bitsandbytes as bnb
import torch
print(f'bitsandbytes: {bnb.__version__}')
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Kiểm tra dataset
echo ""
echo ">>> Kiểm tra dataset..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_FILE="$SCRIPT_DIR/final-dataset-v12-train.json"
if [ ! -f "$DATA_FILE" ]; then
    DATA_FILE="$SCRIPT_DIR/final-dataset-train.json"
fi

if [ -f "$DATA_FILE" ]; then
    SIZE=$(du -h "$DATA_FILE" | cut -f1)
    COUNT=$(python3 -c "import json; print(len(json.load(open('$DATA_FILE'))))")
    echo "✅ Dataset tìm thấy: $DATA_FILE ($SIZE, $COUNT mẫu)"
else
    echo "❌ Chưa có dataset v12! Copy file vào đây:"
    echo "   cp /path/to/final-dataset-v12-train.json $SCRIPT_DIR/"
    exit 1
fi

# Pre-download model (optional, tránh timeout khi train)
echo ""
echo ">>> Pre-download CodeLlama-13B (nếu chưa có cache)..."
python3 -c "
from transformers import AutoTokenizer
print('Downloading tokenizer...')
tok = AutoTokenizer.from_pretrained('codellama/CodeLlama-13b-hf', use_fast=True)
print('✅ Tokenizer downloaded successfully')
print('💡 Model weights sẽ được download khi bắt đầu train (~7.3GB 4-bit)')
"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✅ Setup hoàn tất!"
echo ""
echo "  Chạy train:"
echo "    nohup python3 train_codellama13b.py > train.log 2>&1 &"
echo ""
echo "  Theo dõi:"
echo "    tail -f train.log"
echo "════════════════════════════════════════════════════════════════════"
