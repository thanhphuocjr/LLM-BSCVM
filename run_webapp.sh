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
echo "======================================================================"
echo "🛡️  LLM-BSCVM v13 Unified AI Security Audit Studio Web App"
echo "🌐  Địa chỉ truy cập: http://localhost:$PORT"
echo "======================================================================"

exec "$PYTHON" -m uvicorn webapp.app:app --host 0.0.0.0 --port "$PORT" --reload
