#!/bin/bash
# ==============================================================================
# HƯỚNG DẪN TRUY CẬP HỆ THỐNG (HOST PORTS)
# - Web Dashboard (LPR Event): http://localhost:8001
# - MinIO S3 API            : http://localhost:9000
# - MinIO S3 Console        : http://localhost:9001 (User/Pass: minioadmin / minioadmin)
# - Redpanda (Kafka Broker) : localhost:9092
# - RTSP Server (MediaMTX)  : rtsp://localhost:8554
# ==============================================================================
set -e

WORKSPACE_DIR="/home/thagn/projects/deepstream/workspace/last_ds_cp"
cd "$WORKSPACE_DIR"

EDGE_PUBLIC_IP="192.168.10.33"
REMOTE_DASHBOARD_HOST="http://localhost:8001"

echo "[1/5] Hạ tầng (Kafka, MinIO)..."
docker compose -f docker-compose.kafka-minio.yml up -d

echo "[2/5] Web Server Dashboard..."
cd "$WORKSPACE_DIR/web_server_kafka"
docker compose up --build -d

echo "[3/5] Forwarder (JSONL → Web)..."
export WEB_API_URL="${REMOTE_DASHBOARD_HOST}/api/upload_json"
nohup ../.venv/bin/python3 -u kafka_consumer.py > consumer.log 2>&1 &
FORWARDER_PID=$!

echo "[4/5] Media Monitor (MinIO & Kafka)..."
cd "$WORKSPACE_DIR"
sudo chmod o+w /home/thagn/projects/deepstream/outputs/events/ || true
sudo rm -f /home/thagn/projects/deepstream/outputs/events/events.jsonl || true
sudo rm -f media_results.jsonl outputs/test_last_2.mp4 || true
export MINIO_ENDPOINT="${EDGE_PUBLIC_IP}:9000"
export EVENTS_JSONL="/home/thagn/projects/deepstream/outputs/events/events.jsonl"
nohup ./scripts/run_media_monitor_minio_kafka.sh > media_monitor.log 2>&1 &
MONITOR_PID=$!

echo "[5/5] DeepStream Pipeline... (Ctrl+C để dừng)"

docker exec ds90 rm -rf /tmp/ds_lpr_v2_runtime_configs

# Khởi động NVIDIA Xorg :2 bên trong container (cần cho nveglglessink / NVIDIA EGL)
docker exec ds90 /usr/local/bin/start-nvidia-display.sh

# DISPLAY=:2: NVIDIA Xorg nội bộ container — đúng EGL path (DRI2), tránh Mesa fallback
docker exec -w /workspace/last_ds_cp ds90 \
    env DISPLAY=:2 \
        GST_REGISTRY=/workspace/.gst_registry.bin \
    python3 /workspace/last_ds_cp/src/app_lpr_v2.py \
    rtsp://127.0.0.1:8554/drive-download-20260616T102510Z-3-001/lpr_230428_005 \
    rtsp://127.0.0.1:8554/drive-download-20260616T102510Z-3-001/lpr_230428_007 \
    rtsp://127.0.0.1:8554/drive-download-20260616T102510Z-3-001/lpr_230428_008 \
    rtsp://127.0.0.1:8554/drive-download-20260616T102510Z-3-001/lpr_230428_009 \
    videos/test1.h264 \
    videos/test3.h264 \
    videos/test4.h264 \
    --output outputs/test_last_2.mp4 \
    --event-output-dir /outputs/events \
    --event-jsonl /outputs/events/events.jsonl \
    --min-plate-width 12 \
    --min-plate-height 4 \
    --save-event-frame \
    --min-stable-votes 2 \
    --pgie-interval 1

echo "[INFO] Đóng hệ thống..."
kill $FORWARDER_PID $MONITOR_PID 2>/dev/null || true
echo "[INFO] Hoàn tất."
