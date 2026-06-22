#!/bin/bash
# stream_all.sh
# Stream toàn bộ video .mp4 trong một thư mục lên RTSP server
# Server mặc định: rtsp://127.0.0.1:8554
# Có thể đổi bằng biến môi trường:
#   RTSP_HOST=rtsp://127.0.0.1:8554 bash rtsp_stream/stream_all.sh videos
# Sử dụng:
#   bash scripts/rtsp_stream/stream_all.sh <thu_muc_video>
#
# Ví dụ:
#   bash scripts/rtsp_stream/stream_all.sh videos
#   bash scripts/rtsp_stream/stream_all.sh /workspace/data/videos_test
#   bash scripts/rtsp_stream/stream_all.sh "/workspace/data/my videos"

RTSP_HOST="${RTSP_HOST:-rtsp://127.0.0.1:8554}"
VIDEO_DIR="${1:?Thiếu thư mục video. Dùng: $0 <thu_muc_video>}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[ERROR] Không tìm thấy ffmpeg."
    echo "Cài bằng:"
    echo "  apt update && apt install -y ffmpeg"
    exit 1
fi

if ! ffmpeg -version >/dev/null 2>&1; then
    echo "[ERROR] ffmpeg có trong PATH nhưng không chạy được."
    echo "Hãy kiểm tra thư viện runtime của ffmpeg, ví dụ lỗi thiếu libFLAC.so.12:"
    echo "  apt update && apt install -y --reinstall libflac12t64 ffmpeg"
    exit 1
fi

if [ ! -d "$VIDEO_DIR" ]; then
    echo "[ERROR] Thư mục không tồn tại: $VIDEO_DIR"
    echo ""
    echo "Nếu bạn đang chạy trong Docker container, folder này có thể chưa được mount vào container."
    echo "Hãy kiểm tra bằng:"
    echo "  ls -lh \"$VIDEO_DIR\""
    echo ""
    echo "Hoặc mount folder video vào container, ví dụ:"
    echo "  -v /home/thagn/Videos/test_rtsp:/videos_rtsp"
    exit 1
fi

RTSP_HOST_NO_SCHEME="${RTSP_HOST#rtsp://}"
RTSP_HOST_PORT="${RTSP_HOST_NO_SCHEME%%/*}"
RTSP_TCP_HOST="${RTSP_HOST_PORT%%:*}"
RTSP_TCP_PORT="${RTSP_HOST_PORT##*:}"

if [ "$RTSP_TCP_HOST" != "$RTSP_TCP_PORT" ] && command -v timeout >/dev/null 2>&1; then
    if ! timeout 3 bash -c "</dev/tcp/${RTSP_TCP_HOST}/${RTSP_TCP_PORT}" 2>/dev/null; then
        echo "[ERROR] Không kết nối được RTSP server: ${RTSP_TCP_HOST}:${RTSP_TCP_PORT}"
        echo "Hãy chạy RTSP server trước, ví dụ:"
        echo "  docker run -d --network host --name mediamtx bluenviron/mediamtx:latest"
        echo ""
        echo "Hoặc đổi RTSP_HOST sang server đang chạy:"
        echo "  RTSP_HOST=rtsp://<ip>:8554 bash $0 \"$VIDEO_DIR\""
        exit 1
    fi
fi

DIR_NAME=$(basename "$VIDEO_DIR")

mapfile -t VIDEO_FILES < <(find "$VIDEO_DIR" -maxdepth 1 -type f -name "*.mp4" | sort)

if [ ${#VIDEO_FILES[@]} -eq 0 ]; then
    echo "[ERROR] Không tìm thấy file .mp4 trong: $VIDEO_DIR"
    exit 1
fi

PIDS=()
URLS=()
FAILED=0

echo "=========================================="
echo " Bắt đầu stream ${#VIDEO_FILES[@]} video"
echo " RTSP server: $RTSP_HOST"
echo " Video dir: $VIDEO_DIR"
echo "=========================================="

for VIDEO_PATH in "${VIDEO_FILES[@]}"; do
    BASENAME=$(basename "$VIDEO_PATH")
    NAME="${BASENAME%.*}"

    # Thay khoảng trắng bằng dấu _
    NAME_SAFE="${NAME// /_}"

    RTSP_URL="${RTSP_HOST}/${DIR_NAME}/${NAME_SAFE}"

    ffmpeg -re -stream_loop -1 -i "$VIDEO_PATH" \
        -c copy \
        -f rtsp \
        -rtsp_transport tcp \
        "$RTSP_URL" \
        -loglevel error &

    PID=$!
    sleep 0.2
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[ERROR] ffmpeg dừng ngay khi publish: $RTSP_URL"
        wait "$PID" 2>/dev/null
        FAILED=$((FAILED + 1))
        continue
    fi

    PIDS+=("$PID")
    URLS+=("$RTSP_URL")

    echo "[->] PID=$PID | $RTSP_URL"
done

if [ ${#PIDS[@]} -eq 0 ]; then
    echo "[ERROR] Không có stream nào chạy được. failed=$FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo " RTSP Stream List"
echo "=========================================="
for URL in "${URLS[@]}"; do
    echo " - $URL"
done
echo "=========================================="
echo ""
echo "Nhấn Ctrl+C để dừng toàn bộ stream."

cleanup() {
    echo ""
    echo "[!] Đang dừng toàn bộ stream..."
    for PID in "${PIDS[@]}"; do
        kill "$PID" 2>/dev/null
    done
    echo "[✓] Đã dừng."
    exit 0
}

trap cleanup INT TERM

wait "${PIDS[@]}"

echo ""
echo "[✓] Tất cả stream đã kết thúc."
