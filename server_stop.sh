#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# server_stop.sh
# TẮT NGAY LẬP TỨC dịch vụ AI v13 và GIẢI PHÓNG VRAM GPU 0 từ máy Local (Mac)
# Giúp nhường lại tài nguyên GPU 0 cho các dịch vụ khác của bạn.
# ══════════════════════════════════════════════════════════════════════════════

set -e

SERVER_USER="ai"
SERVER_IP="192.168.91.176"
SERVER_PORT="2225"

echo "════════════════════════════════════════════════════════════════════"
echo "  🛑 Đang gửi lệnh TẮT DỊCH VỤ & GIẢI PHÓNG VRAM GPU 0 Server..."
echo "  Target: ${SERVER_USER}@${SERVER_IP}:${SERVER_PORT}"
echo "════════════════════════════════════════════════════════════════════"

ssh -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_IP}" "cd ~/server_train && bash service.sh stop"

echo ""
echo "✅ Dịch vụ AI đã tắt! GPU 0 đã được giải phóng hoàn toàn."
echo "════════════════════════════════════════════════════════════════════"
