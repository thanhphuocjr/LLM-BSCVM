#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Setup môi trường cho Qwen2.5-Coder-14B v13 (train + API server)
# Chạy 1 lần trước khi train: bash setup.sh
# ══════════════════════════════════════════════════════════════════════════════

set -e

echo "════════════════════════════════════════════════════════════════════"
echo "  Setup môi trường cho Qwen2.5-Coder-14B v13"
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

if [ -f "$DATA_FILE" ]; then
    SIZE=$(du -h "$DATA_FILE" | cut -f1)
    COUNT=$(python3 -c "import json; print(len(json.load(open('$DATA_FILE'))))")
    echo "✅ Dataset tìm thấy: $DATA_FILE ($SIZE, $COUNT mẫu)"
else
    echo "❌ Chưa có dataset v12! Copy file vào đây:"
    echo "   Từ máy local: ./gpu_server.sh deploy"
    exit 1
fi

# Pre-download model (optional, tránh timeout khi train)
echo ""
echo ">>> Pre-download tokenizer Qwen2.5-Coder-14B-Instruct (nếu chưa có cache)..."
python3 -c "
from transformers import AutoTokenizer
print('Downloading tokenizer...')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-14B-Instruct', use_fast=True)
print('✅ Tokenizer downloaded successfully')
print('💡 Model weights sẽ được download khi bắt đầu train / khởi động API')
"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✅ Setup hoàn tất!"
echo ""
echo "  Chạy train v13 (background, log train_v13_<timestamp>.log):"
echo "    bash run_train.sh"
echo ""
echo "  Bật API inference v13:"
echo "    bash service.sh start"
echo "════════════════════════════════════════════════════════════════════"
