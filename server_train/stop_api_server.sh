#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# stop_api_server.sh
# Dừng ngay lập tức Unified AI Inference Server v13 trên Server
# và giải phóng hoàn toàn VRAM trên GPU 0 để nhường tài nguyên cho dịch vụ khác.
# ══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="api_server.pid"
PORT="${PORT:-6868}"

echo "════════════════════════════════════════════════════════════════════"
echo "  DỪNG DỊCH VỤ AI V13 & GIẢI PHÓNG VRAM GPU 0"
echo "════════════════════════════════════════════════════════════════════"

STOPPED=0

# 1. Dừng theo PID file (chỉ dừng nếu chính là serve_v13_api.py)
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
        CMD_CHECK=$(ps -p "$PID" -o args= 2>/dev/null || echo "")
        if echo "$CMD_CHECK" | grep -q "serve_v13_api.py"; then
            echo ">>> Đang dừng tiến trình API server (PID $PID)..."
            kill -15 "$PID" 2>/dev/null || true
            sleep 2
            if ps -p "$PID" > /dev/null 2>&1; then
                kill -9 "$PID" 2>/dev/null || true
                sleep 1
            fi
            STOPPED=1
            echo "✅ Đã dừng tiến trình API server (PID $PID)."
        else
            echo "⚠️  PID $PID không phải là tiến trình serve_v13_api.py. Bỏ qua để bảo vệ dịch vụ khác."
        fi
    fi
    rm -f "$PID_FILE"
fi

# 2. Quét dọn các tiến trình serve_v13_api.py còn sót (chỉ tìm đúng tên serve_v13_api.py)
LEFTOVER_PIDS=$(pgrep -f "serve_v13_api.py" 2>/dev/null || echo "")
if [ -n "$LEFTOVER_PIDS" ]; then
    for p in $LEFTOVER_PIDS; do
        if [ "$p" != "$$" ]; then
            echo ">>> Đang dừng tiến trình serve_v13_api.py (PID: $p)..."
            kill -15 "$p" 2>/dev/null || true
            sleep 1
            if ps -p "$p" > /dev/null 2>&1; then
                kill -9 "$p" 2>/dev/null || true
            fi
            STOPPED=1
        fi
    done
fi

if [ "$STOPPED" -eq 0 ]; then
    echo "ℹ️  Dịch vụ AI Server v13 hiện không chạy."
fi

# 4. Kiểm tra và hiển thị lại VRAM GPU 0
echo ""
echo ">>> Trạng thái GPU 0 sau khi dừng dịch vụ:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv,noheader -i 0 2>/dev/null || true
GPU0_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ' || echo "0")
echo ">>> VRAM GPU 0 khả dụng: ${GPU0_FREE} MiB"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✅ Dịch vụ đã tắt! GPU 0 đã được giải phóng hoàn toàn."
echo "  Tài nguyên đã sẵn sàng cho các dịch vụ khác của bạn."
echo "════════════════════════════════════════════════════════════════════"
