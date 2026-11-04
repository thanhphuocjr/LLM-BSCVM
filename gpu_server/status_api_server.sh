#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# status_api_server.sh
# Kiểm tra trạng thái hoạt động của Unified AI Inference Server v13 & GPU 0
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="api_server.pid"
PORT="${PORT:-6868}"

echo "════════════════════════════════════════════════════════════════════"
echo "  TRẠNG THÁI DỊCH VỤ AI V13 & TÀI NGUYÊN GPU 0"
echo "════════════════════════════════════════════════════════════════════"

RUNNING=0
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
        RUNNING=1
        echo "🟢 Trạng thái tiến trình: ĐANG CHẠY (PID: $PID)"
    fi
fi

if [ "$RUNNING" -eq 0 ]; then
    # Kiểm tra xem có tiến trình nào đang listen port 8000
    PORT_PID=$(lsof -ti :$PORT 2>/dev/null || true)
    if [ -n "$PORT_PID" ]; then
        echo "🟡 Cổng $PORT đang mở (PID: $PORT_PID), nhưng PID file không khớp."
        RUNNING=1
    else
        echo "🔴 Trạng thái dịch vụ: ĐÃ TẮT (OFFLINE)"
    fi
fi

# Kiểm tra Health endpoint
if [ "$RUNNING" -eq 1 ]; then
    echo ""
    echo ">>> Phản hồi từ Health API (http://localhost:$PORT/api/v1/health):"
    HEALTH_RESP=$(curl -s --max-time 2 "http://localhost:$PORT/api/v1/health" 2>/dev/null || echo "")
    if [ -n "$HEALTH_RESP" ]; then
        echo "$HEALTH_RESP" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_RESP"
    else
        echo "⚠️  Chưa phản hồi (mô hình có thể đang nạp weights trong vài giây...)"
    fi
fi

# Kiểm tra trạng thái GPU 0
echo ""
echo ">>> Thông số tài nguyên GPU 0:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader -i 0 2>/dev/null || true

echo "════════════════════════════════════════════════════════════════════"
