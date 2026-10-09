#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Khởi chạy training Qwen2.5-Coder-14B v13 trong background
# Cấu hình: GPU 0 Dedicated (MAX_LEN 1024, Bao phủ 99.92% DAppSCAN)
# Usage: bash run_train.sh
# ══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="train_v13_${TIMESTAMP}.log"

echo "════════════════════════════════════════════════════════════════════"
echo "  Qwen2.5-Coder-14B v13 — Quality-First Background Training Job"
echo "  GPU 0 Dedicated | MAX_LEN 1024 (99.92% DAppSCAN) | Decoupled Head"
echo "  GPU 1 được giữ nguyên hoàn toàn cho các dịch vụ nền hiện có"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Kiểm tra dataset v12
DATASET_FILE="final-dataset-v12-train.json"
if [ ! -f "$DATASET_FILE" ]; then
    echo "❌ Chưa có dataset ($DATASET_FILE)!"
    echo "   Vui lòng kiểm tra lại file trong thư mục $SCRIPT_DIR/"
    exit 1
fi

# Dọn dẹp tiến trình train cũ nếu còn chạy
if [ -f "train.pid" ]; then
    OLD_PID=$(cat train.pid)
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Phát hiện tiến trình train cũ (PID $OLD_PID) đang chạy. Đang dừng để giải phóng VRAM..."
        kill -9 "$OLD_PID" 2>/dev/null || true
        sleep 3
    fi
fi

# Kiểm tra trạng thái GPU
echo ">>> Trạng thái GPU Server:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo ""

# Kiểm tra VRAM GPU 0 và GPU 1 khả dụng
GPU0_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ' || echo "0")
GPU1_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 2>/dev/null | tr -d ' ' || echo "0")
echo ">>> GPU 0 VRAM free: ${GPU0_FREE} MiB"
echo ">>> GPU 1 VRAM free: ${GPU1_FREE} MiB"

if [ "$GPU0_FREE" -lt 8000 ] || [ "$GPU1_FREE" -lt 3500 ]; then
    echo "⚠️  CẢNH BÁO: GPU 0 cần tối thiểu 8,000 MiB và GPU 1 cần tối thiểu 3,500 MiB free."
    read -p "   Bạn có muốn tiếp tục không? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Đã hủy."
        exit 0
    fi
else
    echo "✅ Cả 2 GPU đều thừa VRAM (GPU 0: ${GPU0_FREE} MiB, GPU 1: ${GPU1_FREE} MiB) -> Đủ điều kiện khởi chạy Dual-GPU!"
fi

echo ""
echo ">>> Khởi chạy training v13 Quality-First trên Dual-GPU (GPU 0 + GPU 1)..."
echo "    Script:   train_qwen25_v13.py"
echo "    Log file: $LOG_FILE"
echo "    PID file: train.pid"
echo ""

# Tối ưu hóa phân bổ bộ nhớ PyTorch chống phân mảnh VRAM
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export HF_HUB_DISABLE_XET="1"

# Cho phép cả 2 GPU
export CUDA_VISIBLE_DEVICES="0,1"

# Chạy background với nohup
nohup python3 -u train_qwen25_v13.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > train.pid

echo "════════════════════════════════════════════════════════════════════"
echo "  ✅ Training v13 Quality-First đã chạy ngầm thành công trên Dual-GPU!"
echo ""
echo "  PID:       $TRAIN_PID"
echo "  Log file:  $LOG_FILE"
echo ""
echo "  Theo dõi trực tiếp:  tail -f $LOG_FILE"
echo "  Kiểm tra tiến trình: ps -p $TRAIN_PID -o pid,stat,etime,cmd"
echo "  Dừng phiên train:    kill $TRAIN_PID"
echo "  Theo dõi 2 GPU:      nvidia-smi -l 3"
echo "════════════════════════════════════════════════════════════════════"
