#!/usr/bin/env bash
# Run DeepStream LPR pipeline with Kafka event output.
#
# Prerequisites:
#   docker compose -f docker-compose.kafka-minio.yml up -d   # start Redpanda + MinIO
#   docker start ds90                                         # ensure ds90 is running
#   docker exec -it ds90 pip3 install confluent-kafka         # one-time setup
#
# Usage:
#   ./scripts/run_lpr_with_kafka.sh <video_or_rtsp_url> [extra_app_args...]
#
# Environment overrides:
#   KAFKA_BOOTSTRAP  — default: 172.17.0.1:9092
#   KAFKA_TOPIC      — default: lpr.events.v1
#   EVENT_DIR        — default: outputs/events
#   OUT_FILE         — default: (no video output)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

KAFKA_BOOTSTRAP="${KAFKA_BOOTSTRAP:-172.17.0.1:9092}"
KAFKA_TOPIC="${KAFKA_TOPIC:-lpr.events.v1}"
EVENT_DIR="${EVENT_DIR:-outputs/events}"

VIDEO_OR_RTSP="${1:-}"
if [ -z "$VIDEO_OR_RTSP" ]; then
    echo "[ERROR] Usage: $0 <video_or_rtsp_url> [extra_app_args...]"
    echo ""
    echo "  Example (file):"
    echo "    $0 videos/test5.mp4"
    echo ""
    echo "  Example (RTSP, after ./run_rtsp.sh):"
    echo "    $0 rtsp://172.17.0.1:8554/drive-download-20260616T102510Z-3-001/lpr_230428_003"
    exit 1
fi
shift
EXTRA_ARGS=("$@")

USE_XIMAGESINK="${USE_XIMAGESINK:-1}"

if ! docker inspect ds90 >/dev/null 2>&1; then
    echo "[ERROR] Container ds90 not found. Create/start it first."
    exit 1
fi
if [ "$(docker inspect -f '{{.State.Running}}' ds90)" != "true" ]; then
    echo "[ERROR] Container ds90 is not running. Run: docker start ds90"
    exit 1
fi

mkdir -p "$EVENT_DIR"

echo "============================================================"
echo " LPR DeepStream Pipeline (Kafka events)"
echo " Workspace: $(pwd)"
echo " Kafka    : $KAFKA_BOOTSTRAP → $KAFKA_TOPIC"
echo " Event Dir: $EVENT_DIR"
echo " Source   : $VIDEO_OR_RTSP"
echo " Rendering: $( [ "$USE_XIMAGESINK" = "1" ] && echo "ximagesink" || echo "nveglglessink" )"
echo "============================================================"

WORKSPACE_NAME="$(basename "$(pwd)")"
docker exec -e USE_XIMAGESINK="$USE_XIMAGESINK" -w "/workspace/${WORKSPACE_NAME}" ds90 python3 src/app_lpr_v2.py \
    "$VIDEO_OR_RTSP" \
    --event-output-dir "$EVENT_DIR" \
    --event-jsonl "${EVENT_DIR}/events.jsonl" \
    --debug-jsonl "${EVENT_DIR}/debug.jsonl" \
    --kafka-enable \
    --kafka-bootstrap-server "$KAFKA_BOOTSTRAP" \
    --kafka-topic "$KAFKA_TOPIC" \
    --pgie-interval 0 \
    --min-stable-votes 2 \
    "${EXTRA_ARGS[@]}"
