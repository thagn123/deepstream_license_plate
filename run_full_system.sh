#!/bin/bash
# Script tự động khởi chạy toàn bộ hệ thống LPR (DeepStream + Kafka + MinIO + Web)
# Tự động hiển thị màn hình và chỉ gửi các sự kiện biển số xe mới lên dashboard.

set -e

WORKSPACE_DIR="/home/thagn/projects/deepstream/workspace/last_ds_cp"
cd "$WORKSPACE_DIR"

echo "====================================================="
echo "1. Khởi động Hạ tầng (Kafka, Redpanda, MinIO)..."
echo "====================================================="
docker compose -f docker-compose.kafka-minio.yml up -d

echo ""
echo "====================================================="
echo "2. Khởi động Web Server Dashboard..."
echo "====================================================="
cd "$WORKSPACE_DIR/web_server"
docker compose up --build -d

echo ""
echo "====================================================="
echo "3. Khởi chạy Script Forwarder (JSONL -> Web) (chạy ngầm)..."
echo "====================================================="
cd "$WORKSPACE_DIR/web_server"
nohup python3 forward_jsonl_to_web.py \
    --events-jsonl /home/thagn/projects/deepstream/outputs/events/events.jsonl \
    --path-map "/outputs=/home/thagn/projects/deepstream/outputs" > forwarder.log 2>&1 &
FORWARDER_PID=$!

echo ""
echo "====================================================="
echo "4. Khởi chạy Media Monitor (MinIO & Kafka) (chạy ngầm)..."
echo "====================================================="
cd "$WORKSPACE_DIR"
# Đảm bảo quyền ghi vào thư mục events
sudo chmod o+w /home/thagn/projects/deepstream/outputs/events/ || true
nohup ./scripts/run_media_monitor_minio.sh > media_monitor.log 2>&1 &
MONITOR_PID=$!

echo ""
echo "====================================================="
echo "5. Khởi chạy DeepStream Pipeline (Hiển thị GUI trực tiếp)..."
echo "====================================================="
echo "Nhấn Ctrl+C ở terminal này để dừng toàn bộ hệ thống."

# Cài đặt trap để khi bạn nhấn Ctrl+C, script sẽ tự động kill các tiến trình ngầm
trap "echo -e '\n[INFO] Đang đóng hệ thống...'; kill $FORWARDER_PID $MONITOR_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Chạy DeepStream với USE_XIMAGESINK=1 để bypass lỗi MESA và mở màn hình.
# --event-repeat-cooldown-frames 0 (mặc định) đảm bảo chỉ gửi lên khi có xe mới.
docker exec -w /workspace/last_ds_cp -it ds90 python3 /workspace/last_ds_cp/src/app_lpr_v2.py \
    rtsp://127.0.0.1:8554/drive-download-20260616T102510Z-3-001/lpr_230428_001 \
    rtsp://127.0.0.1:8554/drive-download-20260616T102510Z-3-001/lpr_230428_002 \
    rtsp://127.0.0.1:8554/drive-download-20260616T102510Z-3-001/lpr_230428_003 \
    rtsp://127.0.0.1:8554/drive-download-20260616T102510Z-3-001/lpr_230428_006 \
    videos/test1.h264 \
    videos/test3.h264 \
    --output /outputs/test_last.mp4 \
    --event-output-dir /outputs/events \
    --event-jsonl /outputs/events/events.jsonl \
    --save-event-frame \
    --min-stable-votes 2 \
    --pgie-interval 0

# Khi DeepStream chạy xong tự nhiên (không bấm Ctrl+C)
echo "[INFO] Đang đóng hệ thống..."
kill $FORWARDER_PID $MONITOR_PID 2>/dev/null || true
echo "[INFO] Hoàn tất."
