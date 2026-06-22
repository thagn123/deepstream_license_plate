#!/usr/bin/env bash
set -euo pipefail

# Starts/reuses MediaMTX on host when run from host, or publishes to an
# already-running host RTSP server when run inside ds90.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

VIDEO_DIR="${1:-videos/drive-download-20260616T102510Z-3-001}"
SERVER_NAME="${SERVER_NAME:-mediamtx}"
SERVER_IMAGE="${SERVER_IMAGE:-bluenviron/mediamtx:latest}"

IN_CONTAINER=0
if [ -f /.dockerenv ] || grep -qaE '/docker/|/kubepods/' /proc/1/cgroup 2>/dev/null; then
    IN_CONTAINER=1
fi

if [ "$IN_CONTAINER" = "1" ]; then
    RTSP_HOST="${RTSP_HOST:-rtsp://172.17.0.1:8554}"
    RTSP_READ_HOST="${RTSP_READ_HOST:-$RTSP_HOST}"
else
    RTSP_HOST="${RTSP_HOST:-rtsp://127.0.0.1:8554}"
    RTSP_READ_HOST="${RTSP_READ_HOST:-rtsp://172.17.0.1:8554}"
fi

if [ ! -d "$VIDEO_DIR" ]; then
    echo "[ERROR] Video folder not found: $VIDEO_DIR"
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[ERROR] ffmpeg not found."
    echo "Install it with: apt update && apt install -y ffmpeg"
    exit 1
fi

if ! ffmpeg -version >/dev/null 2>&1; then
    echo "[ERROR] ffmpeg exists but cannot run."
    exit 1
fi

if [ "$IN_CONTAINER" = "0" ]; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "[ERROR] docker not found. MediaMTX is started via Docker in this script."
        exit 1
    fi

    if docker inspect "$SERVER_NAME" >/dev/null 2>&1; then
        if [ "$(docker inspect -f '{{.State.Running}}' "$SERVER_NAME")" != "true" ]; then
            echo "[INFO] Starting existing RTSP server container: $SERVER_NAME"
            docker start "$SERVER_NAME" >/dev/null
        else
            echo "[INFO] RTSP server already running: $SERVER_NAME"
        fi
    else
        echo "[INFO] Creating RTSP server container: $SERVER_NAME"
        docker run -d --network host --name "$SERVER_NAME" "$SERVER_IMAGE" >/dev/null
    fi
else
    echo "[INFO] Running inside container; expecting RTSP server already running on host."
fi

RTSP_HOST_NO_SCHEME="${RTSP_HOST#rtsp://}"
RTSP_HOST_PORT="${RTSP_HOST_NO_SCHEME%%/*}"
RTSP_TCP_HOST="${RTSP_HOST_PORT%%:*}"
RTSP_TCP_PORT="${RTSP_HOST_PORT##*:}"

echo "[INFO] Waiting for RTSP server on ${RTSP_TCP_HOST}:${RTSP_TCP_PORT} ..."
for _ in $(seq 1 20); do
    if timeout 1 bash -c "</dev/tcp/${RTSP_TCP_HOST}/${RTSP_TCP_PORT}" 2>/dev/null; then
        break
    fi
    sleep 0.5
done

if ! timeout 2 bash -c "</dev/tcp/${RTSP_TCP_HOST}/${RTSP_TCP_PORT}" 2>/dev/null; then
    echo "[ERROR] RTSP server is not reachable on ${RTSP_TCP_HOST}:${RTSP_TCP_PORT}"
    if [ "$IN_CONTAINER" = "0" ]; then
        echo "Check logs: docker logs --tail=80 $SERVER_NAME"
    fi
    exit 1
fi

DIR_NAME="$(basename "$VIDEO_DIR")"
FIRST_MP4="$(find "$VIDEO_DIR" -maxdepth 1 -type f -name '*.mp4' | sort | head -n 1)"

echo "============================================================"
echo " RTSP publish host : $RTSP_HOST"
echo " DeepStream host   : $RTSP_READ_HOST"
echo " Video folder      : $VIDEO_DIR"
echo " In container      : $IN_CONTAINER"
echo "============================================================"

if [ -n "$FIRST_MP4" ]; then
    FIRST_NAME="$(basename "$FIRST_MP4" .mp4)"
    FIRST_NAME_SAFE="${FIRST_NAME// /_}"
    echo "[INFO] Example URL for DeepStream inside ds90:"
    echo "  ${RTSP_READ_HOST}/${DIR_NAME}/${FIRST_NAME_SAFE}"
    echo ""
    WORKSPACE_NAME="$(basename "$(pwd)")"
    echo "[INFO] Example DeepStream command:"
    echo "  docker exec -w /workspace/${WORKSPACE_NAME} ds90 python3 src/app_lpr_v2.py \\"
    echo "    ${RTSP_READ_HOST}/${DIR_NAME}/${FIRST_NAME_SAFE} \\"
    echo "    --no-display --pgie-interval 0 --min-stable-votes 2 \\"
    echo "    --square-split-overlap 0.18 --square-split-pad-x 0.16 --square-split-pad-y 0.12"
    echo ""
fi

echo "[INFO] Press Ctrl+C to stop all ffmpeg streams. MediaMTX will remain running."
echo ""

RTSP_HOST="$RTSP_HOST" bash scripts/stream_all.sh "$VIDEO_DIR"
