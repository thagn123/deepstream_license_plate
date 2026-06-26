#!/usr/bin/env bash
# Run media monitor: read LPR event JSONL, upload crops to MinIO, publish
# media results to Kafka topic lpr.media.v1.
#
# Run this on the HOST (outside ds90) so it can reach MinIO at 127.0.0.1:9000.
# The --path-map flag translates container paths (/workspace/last_ds/...)
# to host paths so the monitor can read the saved crop images.
#
# Prerequisites:
#   docker compose -f docker-compose.kafka-minio.yml up -d   # Redpanda + MinIO
#   pip install minio confluent-kafka                        # host deps
#   # OR: run inside ds90 if MinIO is reachable from container
#
# Usage:
#   ./scripts/run_media_monitor_minio.sh            # tail mode (Ctrl-C to stop)
#   ONCE=1 ./scripts/run_media_monitor_minio.sh     # process existing + exit
#
# Environment overrides:
#   MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
#   MINIO_BUCKET, MINIO_PREFIX
#   KAFKA_BOOTSTRAP, KAFKA_MEDIA_TOPIC
#   EVENTS_JSONL, OUTPUT_JSONL
#   HOST_WORKSPACE  — host path matching /workspace/last_ds inside ds90
#   ONCE            — set to 1 for --once mode

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MINIO_ENDPOINT="${MINIO_ENDPOINT:-127.0.0.1:9000}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"
MINIO_BUCKET="${MINIO_BUCKET:-lpr-media}"
MINIO_PREFIX="${MINIO_PREFIX:-lpr}"

KAFKA_BOOTSTRAP="${KAFKA_BOOTSTRAP:-127.0.0.1:9092}"
KAFKA_MEDIA_TOPIC="${KAFKA_MEDIA_TOPIC:-lpr.media.v1}"

EVENT_DIR="${EVENT_DIR:-outputs/events}"
EVENTS_JSONL="${EVENTS_JSONL:-${EVENT_DIR}/events.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-media_results.jsonl}"

# Container path → host path mapping so the monitor can open crop images.
# The DeepStream app saves paths as /workspace/${WORKSPACE_NAME}/outputs/...
# Adjust HOST_WORKSPACE if your local checkout is in a different location.
HOST_WORKSPACE="${HOST_WORKSPACE:-$(pwd)}"
WORKSPACE_NAME="$(basename "$(pwd)")"
PATH_MAP1="/workspace/${WORKSPACE_NAME}=${HOST_WORKSPACE}"
PATH_MAP2="/outputs=/home/thagn/projects/deepstream/outputs"

ONCE_FLAG=()
if [ "${ONCE:-0}" = "1" ]; then
    ONCE_FLAG=(--once)
fi

echo "============================================================"
echo " LPR Media Monitor → MinIO + Kafka"
echo " MinIO  : http://$MINIO_ENDPOINT  bucket=$MINIO_BUCKET  prefix=$MINIO_PREFIX"
echo " Kafka  : $KAFKA_BOOTSTRAP → $KAFKA_MEDIA_TOPIC"
echo " Events : $EVENTS_JSONL"
echo " Output : $OUTPUT_JSONL"
echo " PathMap: $PATH_MAP1, $PATH_MAP2"
echo "============================================================"

.venv/bin/python3 -u tools/media_monitor_kafka.py \
    --events-jsonl  "$EVENTS_JSONL" \
    --output-jsonl  "$OUTPUT_JSONL" \
    --minio-enable \
    --minio-endpoint   "$MINIO_ENDPOINT" \
    --minio-access-key "$MINIO_ACCESS_KEY" \
    --minio-secret-key "$MINIO_SECRET_KEY" \
    --minio-bucket  "$MINIO_BUCKET" \
    --minio-prefix  "$MINIO_PREFIX" \
    --path-map      "$PATH_MAP1" \
    --path-map      "$PATH_MAP2" \
    --media-result-kafka-enable \
    --media-result-kafka-bootstrap "$KAFKA_BOOTSTRAP" \
    --media-result-kafka-topic     "$KAFKA_MEDIA_TOPIC" \
    --upload-retries 3 \
    --upload-retry-delay 2 \
    --replay \
    "${ONCE_FLAG[@]}"
