#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# server_start.sh
# Khởi chạy dịch vụ AI v13 từ xa trên GPU 0 Server từ máy Local (Mac)
# Server: ai@192.168.91.176 (Port 2225)
# ══════════════════════════════════════════════════════════════════════════════

set -e

SERVER_USER="ai"
SERVER_IP="192.168.91.176"
SERVER_PORT="2225"

echo "════════════════════════════════════════════════════════════════════"
echo "  🚀 Đang gửi lệnh BẬT DỊCH VỤ AI v13 lên GPU 0 Server..."
echo "  Target: ${SERVER_USER}@${SERVER_IP}:${SERVER_PORT}"
echo "════════════════════════════════════════════════════════════════════"

ssh -p "${SERVER_PORT}" "${SERVER_USER}@${SERVER_IP}" "cd ~/server_train && bash service.sh start"

echo ""
echo "✅ Lệnh khởi chạy đã gửi thành công!"
echo "Kiểm tra log: ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP} 'tail -f ~/server_train/api_server.log'"
echo "Hoặc kiểm tra: ./server_status.sh"
echo "════════════════════════════════════════════════════════════════════"
