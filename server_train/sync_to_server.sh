#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Script đồng bộ code và data từ local (Mac) lên GPU Server
# Server: ai@192.168.91.176 (Port 2225)
# ══════════════════════════════════════════════════════════════════════════════

set -e

SERVER_USER="ai"
SERVER_IP="192.168.91.176"
SERVER_PORT="2225"
REMOTE_DIR="~/server_train"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "════════════════════════════════════════════════════════════════════"
echo "  Đồng bộ thư mục server_train (v13 API & Scripts) lên Server"
echo "  Target: ${SERVER_USER}@${SERVER_IP}:${SERVER_PORT} -> ${REMOTE_DIR}"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Đồng bộ qua rsync với SSH port 2225
# LOẠI TRỪ các checkpoint và weight lớn để sync nhanh trong vài giây
rsync -avz --progress \
    -e "ssh -p ${SERVER_PORT}" \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude "*.log" \
    --exclude "*.pid" \
    --exclude "codellama-*" \
    --exclude "checkpoint-*" \
    --exclude "*.safetensors" \
    --exclude "*.bin" \
    "${LOCAL_DIR}/" "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/"

# Đồng bộ bổ sung threshold_config.json v13 nếu có
THRESHOLD_CFG="${LOCAL_DIR}/../models/qwen25-v13-14b-detector-final/threshold_config.json"
if [ -f "$THRESHOLD_CFG" ]; then
    echo ">>> Đang đồng bộ threshold_config.json v13..."
    rsync -avz -e "ssh -p ${SERVER_PORT}" "$THRESHOLD_CFG" "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/qwen25-v13-14b-detector-final/" || true
fi

echo ""
echo "✅ Đồng bộ hoàn tất!"
echo ""
echo "Các lệnh điều khiển nhanh từ máy Local (Mac):"
echo "  • Bật dịch vụ:      ./server_start.sh"
echo "  • Tắt dịch vụ:      ./server_stop.sh      (giải phóng VRAM GPU 0 lập tức)"
echo "  • Xem trạng thái:   ./server_status.sh"
echo ""
echo "Hoặc điều khiển trực tiếp trên Server:"
echo "  ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP}"
echo "  cd ~/server_train"
echo "  bash service.sh start | stop | status | logs"
echo "════════════════════════════════════════════════════════════════════"
