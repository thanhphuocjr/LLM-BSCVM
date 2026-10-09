#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# server_status.sh
# Kiểm tra trạng thái dịch vụ AI v13 và mức sử dụng VRAM GPU 0 từ máy Local (Mac)
# ══════════════════════════════════════════════════════════════════════════════

set -e

SERVER_USER="ai"
SERVER_IP="192.168.91.176"
SERVER_PORT="2225"

echo "════════════════════════════════════════════════════════════════════"
echo "  📊 Đang kiểm tra trạng thái GPU 0 Server..."
echo "  Target: ${SERVER_USER}@${SERVER_IP}:${SERVER_PORT}"
echo "════════════════════════════════════════════════════════════════════"

ssh -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_IP}" "cd ~/server_train && bash service.sh status"
