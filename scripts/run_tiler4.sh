#!/usr/bin/env bash
set -euo pipefail

# Run DeepStream LPR with 4 sources so nvmultistreamtiler renders a 2x2 view.
# Default mode uses the RTSP URLs published by ./run_rtsp.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODE="${1:-rtsp}"                    # rtsp | file
VIDEO_DIR="${VIDEO_DIR:-videos/drive-download-20260616T102510Z-3-001}"
RTSP_READ_HOST="${RTSP_READ_HOST:-rtsp://172.17.0.1:8554}"
OUT_DIR="${OUT_DIR:-outputs/tiler4}"
OUT_FILE="${OUT_FILE:-${OUT_DIR}/tiler4_${MODE}.mp4}"
SAVE_EVENTS="${SAVE_EVENTS:-0}"
DISPLAY_MODE="${DISPLAY_MODE:-1}"    # 0 = no display, 1 = try display
USE_XIMAGESINK="${USE_XIMAGESINK:-1}"

mkdir -p "$OUT_DIR"

if ! docker inspect ds90 >/dev/null 2>&1; then
    echo "[ERROR] Docker container ds90 not found."
    exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' ds90)" != "true" ]; then
    echo "[ERROR] Docker container ds90 is not running."
    exit 1
fi

mapfile -t FILES < <(find "$VIDEO_DIR" -maxdepth 1 -type f -name '*.mp4' | sort | head -n 4)
if [ "${#FILES[@]}" -lt 4 ]; then
    echo "[ERROR] Need at least 4 mp4 files in: $VIDEO_DIR"
    exit 1
fi

DIR_NAME="$(basename "$VIDEO_DIR")"
SOURCES=()

case "$MODE" in
    rtsp)
        echo "[INFO] Mode: RTSP tiler4"
        echo "[INFO] Make sure streams are running in another terminal:"
        echo "  ./run_rtsp.sh"
        echo ""
        if ! docker exec ds90 bash -lc 'timeout 2 bash -c "</dev/tcp/172.17.0.1/8554"' 2>/dev/null; then
            echo "[ERROR] ds90 cannot reach RTSP server at 172.17.0.1:8554"
            echo "Start streams first on host:"
            echo "  ./run_rtsp.sh"
            exit 1
        fi
        for path in "${FILES[@]}"; do
            name="$(basename "$path" .mp4)"
            name_safe="${name// /_}"
            SOURCES+=("${RTSP_READ_HOST}/${DIR_NAME}/${name_safe}")
        done
        ;;
    file|local)
        echo "[INFO] Mode: local file tiler4"
        for path in "${FILES[@]}"; do
            SOURCES+=("$path")
        done
        ;;
    *)
        echo "[ERROR] Unknown mode: $MODE"
        echo "Usage:"
        echo "  ./run_tiler4.sh          # RTSP mode"
        echo "  ./run_tiler4.sh rtsp"
        echo "  ./run_tiler4.sh file"
        exit 1
        ;;
esac

APP_ARGS=(
    python3 src/app_lpr_v2.py
    "${SOURCES[@]}"
    --pgie-interval 0
    --min-stable-votes 2
    --square-split-overlap 0.18
    --square-split-pad-x 0.16
    --square-split-pad-y 0.12
    --output "$OUT_FILE"
)

if [ "$DISPLAY_MODE" != "1" ]; then
    APP_ARGS+=(--no-display)
fi

if [ "$SAVE_EVENTS" = "1" ]; then
    EVENT_DIR="${OUT_DIR}/events_${MODE}"
    APP_ARGS+=(
        --event-output-dir "$EVENT_DIR"
        --event-jsonl "${EVENT_DIR}/events.jsonl"
        --debug-jsonl "${EVENT_DIR}/debug.jsonl"
    )
fi

echo "============================================================"
echo " DeepStream tiler4"
echo " Mode       : $MODE"
echo " Output     : $OUT_FILE"
echo " Save events: $SAVE_EVENTS"
echo " Sources:"
for src in "${SOURCES[@]}"; do
    echo "  - $src"
done
echo "============================================================"

WORKSPACE_NAME="$(basename "$(pwd)")"
docker exec -e USE_XIMAGESINK="$USE_XIMAGESINK" -w "/workspace/${WORKSPACE_NAME}" ds90 "${APP_ARGS[@]}"
