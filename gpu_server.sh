#!/usr/bin/env bash
# Control the GPU inference server (Qwen2.5-Coder-14B v13 detector API) from this Mac.
#
#   ./gpu_server.sh start     start the v13 API on GPU 0      (remote: server_train/service.sh start)
#   ./gpu_server.sh stop      stop it and free GPU 0 VRAM     (remote: service.sh stop)
#   ./gpu_server.sh status    service state + VRAM            (remote: service.sh status)
#   ./gpu_server.sh logs      follow the API log              (remote: service.sh logs)
#   ./gpu_server.sh deploy    rsync server_train/ (code + v12 dataset) to the server
#   ./gpu_server.sh pull      fetch the trained adapter into models/ and the v13 training logs
#
# Override the target per call, e.g.  SERVER_IP=100.122.153.35 ./gpu_server.sh status
set -euo pipefail
cd "$(dirname "$0")"

SERVER_USER="${SERVER_USER:-ai}"
SERVER_IP="${SERVER_IP:-192.168.91.176}"
SERVER_PORT="${SERVER_PORT:-2225}"
REMOTE_DIR="~/server_train"
MODEL_NAME="${MODEL_NAME:-qwen25-v13-14b-detector-final}"
TRAIN_LOGS="${TRAIN_LOGS:-train_v13_*.log}"
TARGET="${SERVER_USER}@${SERVER_IP}"

remote_service() {
  echo ">>> ${TARGET}:${SERVER_PORT}  service.sh $1"
  local tty=()
  [[ "$1" == "logs" ]] && tty=(-t)   # so Ctrl+C also stops the remote tail -f
  ssh ${tty[@]+"${tty[@]}"} -p "$SERVER_PORT" "$TARGET" "cd $REMOTE_DIR && bash service.sh $1"
}

deploy() {
  # Code and the dataset only; checkpoints and weights stay on the server.
  rsync -avz --progress -e "ssh -p $SERVER_PORT" \
    --exclude "__pycache__" --exclude "*.pyc" --exclude "*.log" --exclude "*.pid" \
    --exclude "codellama-*" --exclude "checkpoint-*" --exclude "*.safetensors" --exclude "*.bin" \
    server_train/ "$TARGET:$REMOTE_DIR/"
  local cfg="models/$MODEL_NAME/threshold_config.json"
  if [[ -f "$cfg" ]]; then
    rsync -avz -e "ssh -p $SERVER_PORT" "$cfg" "$TARGET:$REMOTE_DIR/$MODEL_NAME/" || true
  fi
  echo "Deployed. Restart the API to load new code: ./gpu_server.sh stop && ./gpu_server.sh start"
}

pull() {
  # Only the final adapter folder (~1 GB), never the intermediate checkpoint-* folders.
  mkdir -p "models/$MODEL_NAME"
  rsync -avz --progress -e "ssh -p $SERVER_PORT" --include="$TRAIN_LOGS" --exclude="*" \
    "$TARGET:$REMOTE_DIR/" server_train/
  rsync -avz --progress -e "ssh -p $SERVER_PORT" \
    "$TARGET:$REMOTE_DIR/$MODEL_NAME/" "models/$MODEL_NAME/"
  echo "Adapter: models/$MODEL_NAME (read through E2E.config.DETECTOR_ADAPTER_DIR)"
}

case "${1:-}" in
  start | stop | status | logs) remote_service "$1" ;;
  deploy) deploy ;;
  pull) pull ;;
  *) sed -n '2,11p' "$0"; exit 1 ;;
esac
