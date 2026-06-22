# RTSP Stream Module

## 1. Mục đích script
Module này cung cấp script `stream_all.sh` để giả lập camera RTSP bằng cách stream toàn bộ các file video `.mp4` trong một thư mục chỉ định lên một RTSP server mặc định (`rtsp://127.0.0.1:8554`). Script hỗ trợ stream vô hạn (loop) theo thời gian thực (real-time).

Lưu ý: script này chỉ publish video lên RTSP server, không tự khởi động RTSP server. Cần chạy RTSP server trước.

Ví dụ chạy MediaMTX bằng Docker trên máy host:
```bash
docker run -d --network host --name mediamtx bluenviron/mediamtx:latest
```

## 2. Cách chạy với folder video trong project
Nếu bạn có một thư mục `videos` nằm trong thư mục gốc của project:
```bash
cd /workspace/last_ds
bash rtsp_stream/stream_all.sh videos
```

Với folder hiện tại của dự án:
```bash
cd /workspace/last_ds
bash rtsp_stream/stream_all.sh videos/drive-download-20260616T102510Z-3-001
```

Nếu muốn publish sang RTSP server khác:
```bash
RTSP_HOST=rtsp://<ip-server>:8554 bash rtsp_stream/stream_all.sh videos/drive-download-20260616T102510Z-3-001
```

## 3. Cách chạy với folder video ngoài project
Bạn có thể trỏ đến một đường dẫn tuyệt đối bất kỳ trên hệ thống hoặc trong container:
```bash
cd /workspace/last_ds
bash rtsp_stream/stream_all.sh /workspace/data/videos_test
```

## 4. Cách chạy với folder có dấu cách
Sử dụng dấu ngoặc kép `""` để bao quanh đường dẫn có chứa dấu cách:
```bash
cd /workspace/last_ds
bash rtsp_stream/stream_all.sh "/workspace/data/my videos"
```

## 5. Ví dụ RTSP URL sinh ra
Định dạng RTSP URL được sinh ra: `rtsp://<HOST>:<PORT>/<ten_thu_muc>/<ten_file_khong_duoi_mp4>` (các dấu cách trong tên file sẽ được thay thế bằng `_`).

Ví dụ với thư mục `/workspace/data/videos_test/` chứa `cam1.mp4`, `cam2.mp4`, `test plate.mp4`, các URL sinh ra sẽ là:
- `rtsp://127.0.0.1:8554/videos_test/cam1`
- `rtsp://127.0.0.1:8554/videos_test/cam2`
- `rtsp://127.0.0.1:8554/videos_test/test_plate`

## 6. Cách test bằng `ffprobe`
Bạn có thể sử dụng `ffprobe` để kiểm tra luồng RTSP có đang phát hay không:
```bash
ffprobe -rtsp_transport tcp rtsp://127.0.0.1:8554/videos_test/cam1
```

## 7. Cách mở bằng VLC
Để xem luồng trên VLC:
1. Mở VLC media player.
2. Chọn `Media` -> `Open Network Stream...` (Phím tắt `Ctrl+N`).
3. Dán đường dẫn URL (ví dụ: `rtsp://127.0.0.1:8554/videos_test/cam1`) vào ô `Please enter a network URL:`.
4. Nhấn `Play`.

## 8. Cách dùng RTSP URL làm input cho DeepStream
Bạn có thể truyền trực tiếp RTSP URL vào app DeepStream của mình:
```bash
python3 src/app_lpr_v2.py rtsp://127.0.0.1:8554/videos_test/cam1 --no-display
```
Nếu app DeepStream hỗ trợ truyền vào nhiều source cùng lúc:
```bash
python3 src/app_lpr_v2.py \
  rtsp://127.0.0.1:8554/videos_test/cam1 \
  rtsp://127.0.0.1:8554/videos_test/cam2 \
  --no-display
```

## 9. Cách dừng toàn bộ stream
Khi đang chạy `stream_all.sh`, bạn chỉ cần nhấn **`Ctrl+C`** trên terminal. Script sẽ tự động dọn dẹp và dừng tất cả các tiến trình `ffmpeg` đang chạy.

## 10. Cách kiểm tra `ffmpeg`
Script yêu cầu phải có `ffmpeg`. Để kiểm tra xem `ffmpeg` đã được cài đặt chưa, dùng lệnh:
```bash
ffmpeg -version
```

## 11. Cách cài `ffmpeg`
Nếu chưa cài đặt `ffmpeg`, bạn có thể cài trên Ubuntu/Debian bằng lệnh:
```bash
apt update && apt install -y ffmpeg
```

## 12. Ghi chú khi chạy trong Docker
Nếu bạn đang chạy ứng dụng trong Docker container và nhận được thông báo lỗi *"Thư mục không tồn tại"*, nguyên nhân có thể là do thư mục video chưa được mount từ máy host vào container.
- Hãy kiểm tra xem thư mục có tồn tại trong container không: `ls -lh /path/to/video/dir`
- Nếu không, hãy mount thư mục vào lúc khởi chạy container bằng tham số `-v`, ví dụ:
  `docker run -v /home/thagn/Videos/test_rtsp:/videos_rtsp ...`
  (Sau đó truyền `/videos_rtsp` vào tham số của `stream_all.sh`)
