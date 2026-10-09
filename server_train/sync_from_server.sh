#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Script kéo kết quả huấn luyện (Model LoRA Adapter + Logs) từ GPU Server về máy local
# Server: ai@192.168.91.176 (Port 2225)
# ══════════════════════════════════════════════════════════════════════════════

set -e

SERVER_USER="ai"
SERVER_IP="192.168.91.176"
SERVER_PORT="2225"
REMOTE_DIR="~/server_train"

# Thư mục gốc project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOCAL_MODELS_DIR="${PROJECT_ROOT}/models"
LOCAL_SERVER_TRAIN_DIR="${SCRIPT_DIR}"

# Mặc định lấy bản v13 mới nhất (hoặc truyền tham số: bash sync_from_server.sh v12)
VERSION="${1:-v13}"
if [ "$VERSION" = "v13" ]; then
    MODEL_NAME="qwen25-v13-14b-detector-final"
else
    MODEL_NAME="codellama-${VERSION}-13b-detector-final"
fi

mkdir -p "${LOCAL_MODELS_DIR}"

echo "════════════════════════════════════════════════════════════════════"
echo "  KÉO KẾT QUẢ MODEL VÀ LOG TỪ SERVER VỀ MÁY LOCAL (${VERSION})"
echo "  Server: ${SERVER_USER}@${SERVER_IP}:${SERVER_PORT}"
echo "  Source: ${REMOTE_DIR}/${MODEL_NAME}"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# 1. Kéo log files về thư mục server_train/
echo "==> [1/2] Đang tải log huấn luyện (*.log)..."
rsync -avz --progress \
    -e "ssh -p ${SERVER_PORT}" \
    --include="*.log" \
    --exclude="*" \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/" "${LOCAL_SERVER_TRAIN_DIR}/"

echo ""
# 2. Kéo thư mục model LoRA adapter hoàn chỉnh
# LƯU Ý: Chỉ kéo thư mục -final (~300MB), TUYỆT ĐỐI KHÔNG kéo thư mục checkpoint trung gian (hàng chục GB).
echo "==> [2/2] Đang tải model hoàn chỉnh (${MODEL_NAME})..."
rsync -avz --progress \
    -e "ssh -p ${SERVER_PORT}" \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_DIR}/${MODEL_NAME}/" \
    "${LOCAL_MODELS_DIR}/${MODEL_NAME}/"

# Tạo bản sao vào models/codellama-vuln-detector để Phase 1 detector dùng mặc định
echo ""
echo "==> Đang liên kết model mới vào models/codellama-vuln-detector..."
mkdir -p "${LOCAL_MODELS_DIR}/codellama-vuln-detector"
rsync -a "${LOCAL_MODELS_DIR}/${MODEL_NAME}/" "${LOCAL_MODELS_DIR}/codellama-vuln-detector/"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "✅ HOÀN TẤT ĐỒNG BỘ TỪ SERVER VỀ LOCAL!"
echo "  - Model Adapter đã lưu tại:"
echo "    + ${LOCAL_MODELS_DIR}/${MODEL_NAME}"
echo "    + ${LOCAL_MODELS_DIR}/codellama-vuln-detector (sẵn sàng cho Phase 1)"
echo "  - Log huấn luyện đã lưu tại:"
echo "    + ${LOCAL_SERVER_TRAIN_DIR}/"
echo "════════════════════════════════════════════════════════════════════"
