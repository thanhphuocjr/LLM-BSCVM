#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# run_api_server.sh
# Khởi chạy Unified AI Inference Server (Version 13) trên GPU 0 Server
# Host: 0.0.0.0 | Port: 8000
# GPU: CHỈ DÙNG GPU 0 (NVIDIA RTX 4070 Ti SUPER 16GB)
# ══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="api_server.pid"
LOG_FILE="api_server.log"
PORT="${PORT:-6868}"

echo "════════════════════════════════════════════════════════════════════"
echo "  LLM-BSCVM v13 Unified AI Inference Server"
echo "  Dedicated Device: GPU 0 (RTX 4070 Ti SUPER)"
echo "  Port: $PORT | Mode: CausalLM 4-bit (NF4) + LoRA v13 Detector"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# 1. Dọn dẹp tiến trình cũ CỦA CHÍNH DỊCH VỤ NÀY (nếu đang chạy từ api_server.pid)
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && ps -p "$OLD_PID" > /dev/null 2>&1; then
        CMD_CHECK=$(ps -p "$OLD_PID" -o args= 2>/dev/null || echo "")
        if echo "$CMD_CHECK" | grep -q "serve_v13_api.py"; then
            echo "ℹ️  Phát hiện tiến trình serve_v13_api cũ của dịch vụ (PID $OLD_PID). Đang dừng để khởi động lại..."
            kill -15 "$OLD_PID" 2>/dev/null || true
            sleep 2
            if ps -p "$OLD_PID" > /dev/null 2>&1; then
                kill -9 "$OLD_PID" 2>/dev/null || true
            fi
            echo "✅ Đã dừng phiên bản cũ của dịch vụ."
        fi
    fi
    rm -f "$PID_FILE"
fi

# 2. KIỂM TRA AN TOÀN CỔNG: Tuyệt đối KHÔNG tự ý kill cổng của dịch vụ khác!
PORT_IN_USE=$(python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    res = s.connect_ex(('127.0.0.1', int('$PORT')))
    sys.exit(0 if res == 0 else 1)
except Exception:
    sys.exit(1)
finally:
    s.close()
" && echo "yes" || echo "no")

if [ "$PORT_IN_USE" = "yes" ]; then
    echo "════════════════════════════════════════════════════════════════════"
    echo "❌ LỖI AN TOÀN: Cổng $PORT hiện đang được sử dụng bởi một tiến trình khác!"
    echo "════════════════════════════════════════════════════════════════════"
    echo ">>> Thông tin tiến trình đang chiếm giữ cổng $PORT:"
    
    # Hiển thị chi tiết tiến trình đang dùng cổng (không kill)
    if command -v lsof > /dev/null 2>&1; then
        lsof -i :$PORT 2>/dev/null || true
    elif command -v ss > /dev/null 2>&1; then
        ss -tulpn | grep ":$PORT " 2>/dev/null || true
    fi
    
    echo ""
    echo "🛡️  QUY TẮC BẢO VỆ: Hệ thống DỪNG LẠI và TUYỆT ĐỐI KHÔNG kill tiến trình trên."
    echo "👉 Vui lòng chọn một cổng khác chưa sử dụng để khởi chạy, ví dụ:"
    echo "     PORT=6869 bash service.sh start"
    echo "   hoặc kiểm tra lại tiến trình đang chạy ở trên."
    echo "════════════════════════════════════════════════════════════════════"
    exit 1
fi
echo "✅ Cổng $PORT đang hoàn toàn trống và sẵn sàng."

# 2. Kiểm tra tài nguyên GPU 0
echo ">>> Kiểm tra trạng thái GPU 0:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv,noheader -i 0
GPU0_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ' || echo "0")
echo ">>> GPU 0 VRAM free: ${GPU0_FREE} MiB"

if [ "$GPU0_FREE" -lt 8000 ]; then
    echo "⚠️  CẢNH BÁO: GPU 0 có ${GPU0_FREE} MiB free (khuyến nghị tối thiểu 8,500 MiB)."
    echo "    Vui lòng kiểm tra lại các tiến trình đang chiếm GPU 0 bằng: nvidia-smi"
else
    echo "✅ GPU 0 đủ VRAM để khởi chạy mô hình 14B 4-bit (~8.5 GB)!"
fi

# 3. Kiểm tra các thư viện cần thiết
echo ""
echo ">>> Kiểm tra dependencies (fastapi, uvicorn)..."
python3 -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "Đang cài đặt fastapi và uvicorn..."
    pip install fastapi uvicorn --quiet
}

# 4. Thiết lập biến môi trường GPU (CHỈ DÙNG GPU 0, GIỮ NGUYÊN GPU 1)
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

# 5. Khởi chạy server background
echo ""
echo ">>> Đang khởi chạy serve_v13_api.py trên cổng $PORT..."
nohup python3 -u serve_v13_api.py --host 0.0.0.0 --port $PORT > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✅ Dịch vụ AI v13 đã được khởi chạy trong background!"
echo ""
echo "  PID:       $SERVER_PID"
echo "  Log file:  $SCRIPT_DIR/$LOG_FILE"
echo "  Health:    http://127.0.0.1:$PORT/api/v1/health"
echo "  Docs:      http://127.0.0.1:$PORT/docs"
echo ""
echo "  Lệnh theo dõi log trực tiếp:  tail -f $LOG_FILE"
echo "  Lệnh dừng dịch vụ:           kill \$(cat $PID_FILE)"
echo "════════════════════════════════════════════════════════════════════"
