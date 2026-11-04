#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# service.sh
# Quản lý vòng đời dịch vụ AI v13 trên GPU 0 (Start / Stop / Restart / Status / Logs)
# Usage:
#   bash service.sh start    -> Khởi chạy dịch vụ trên GPU 0
#   bash service.sh stop     -> Dừng dịch vụ ngay & giải phóng VRAM GPU 0
#   bash service.sh restart  -> Khởi động lại dịch vụ
#   bash service.sh status   -> Xem trạng thái dịch vụ và VRAM
#   bash service.sh logs     -> Xem log trực tiếp
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-menu}"

case "$ACTION" in
    start)
        bash "$SCRIPT_DIR/run_api_server.sh"
        ;;
    stop)
        bash "$SCRIPT_DIR/stop_api_server.sh"
        ;;
    restart)
        echo ">>> Đang khởi động lại dịch vụ AI v13..."
        bash "$SCRIPT_DIR/stop_api_server.sh"
        sleep 2
        bash "$SCRIPT_DIR/run_api_server.sh"
        ;;
    status)
        bash "$SCRIPT_DIR/status_api_server.sh"
        ;;
    logs)
        LOG_FILE="$SCRIPT_DIR/api_server.log"
        if [ -f "$LOG_FILE" ]; then
            echo ">>> Đang theo dõi log ($LOG_FILE)... (Nhấn Ctrl+C để thoát)"
            tail -f "$LOG_FILE"
        else
            echo "❌ Chưa có file log ($LOG_FILE)."
        fi
        ;;
    *)
        echo "════════════════════════════════════════════════════════════════════"
        echo "  QUẢN LÝ DỊCH VỤ AI V13 (GPU 0 - RTX 4070 Ti SUPER)"
        echo "════════════════════════════════════════════════════════════════════"
        echo "  Cách sử dụng:"
        echo "    bash service.sh start    : Bật dịch vụ AI (chỉ dùng GPU 0)"
        echo "    bash service.sh stop     : Tắt dịch vụ & giải phóng VRAM GPU 0"
        echo "    bash service.sh restart  : Khởi động lại dịch vụ"
        echo "    bash service.sh status   : Kiểm tra trạng thái & VRAM GPU 0"
        echo "    bash service.sh logs     : Theo dõi log tiến trình"
        echo "════════════════════════════════════════════════════════════════════"
        ;;
esac
