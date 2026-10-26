#!/usr/bin/env bash
# ==============================================================================
# Script khởi chạy Giao Diện Web App Kiểm Toán Smart Contract (LLM-BSCVM v12)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Kiểm tra môi trường ảo Python
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "❌ Lỗi: Không tìm thấy Python!"
    exit 1
fi

PORT="${PORT:-8080}"
# Mặc định chỉ nghe trên localhost: API này ghi file và khởi chạy tiến trình phân tích.
# Muốn mở cho mạng LAN thì đặt HOST=0.0.0.0 một cách có chủ đích.
HOST="${HOST:-127.0.0.1}"
RELOAD_FLAG=""
if [ "${RELOAD:-0}" = "1" ]; then
    RELOAD_FLAG="--reload"
fi
echo "======================================================================"
echo "🛡️  LLM-BSCVM v13 Unified AI Security Audit Studio Web App"
echo "🌐  Địa chỉ truy cập: http://$HOST:$PORT"
echo "======================================================================"

exec "$PYTHON" -m uvicorn webapp.app:app --host "$HOST" --port "$PORT" $RELOAD_FLAG
