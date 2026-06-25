# 🏎️ DeepStream License Plate Recognition (LPR) Pipeline

Một hệ thống tự động nhận diện biển số xe (ALPR) hiệu năng cao, được thiết kế và tối ưu hóa ở mức phần cứng bằng công nghệ NVIDIA DeepStream, CUDA và GStreamer. Hệ thống được triển khai theo kiến trúc Edge-to-Cloud, gửi sự kiện thời gian thực (Real-time Events) qua Kafka tới Dashboard quản lý trung tâm.

---

## 🌟 Tính năng nổi bật

1. **Hiệu năng Real-time:** 
   - Khả năng xử lý tốc độ cao trên phần cứng Edge (NVIDIA Jetson / GPU) với luồng video H.264.
   - Hỗ trợ xử lý đa luồng (Multi-stream) cùng lúc.
   
2. **GPU-Accelerated Blur Filter (`dslaplacian` plugin):**
   - Hỗ trợ lọc bỏ các khung hình rác, bóng mờ (Motion Blur) do xe chạy tốc độ cao.
   - Plugin `dslaplacian` được viết bằng **C++ và CUDA Kernels**, can thiệp trực tiếp vào bộ nhớ VRAM (`Zero-copy memory`), giúp tránh hiện tượng nghẽn cổ chai so với các thư viện xử lý trên CPU. Việc tính toán phương sai Laplacian (Laplacian Variance) cho các pixel diễn ra song song trên GPU.

3. **Kiến trúc AI Module Tiên Tiến:**
   - **PGIE (Primary Inference):** Mô hình YOLOv11s phát hiện các phương tiện (Ô tô, Xe máy).
   - **NVTracker:** NvDCF Tracker giữ vết liên tục của các phương tiện trong suốt vòng đời camera.
   - **SGIE (Secondary Inference):** Mô hình LPR OCR (ONNX 2024) chuyên đọc các ký tự biển số.
   
4. **Cơ chế "Lock Plate" thông minh:**
   - Liên tục theo dõi độ nét của biển số trong từng khung hình. Chỉ giữ lại kết quả có độ nét cao, giúp giảm thiểu đáng kể hiện tượng OCR nhảy chữ hoặc giật lag. Hệ thống cũng hỗ trợ nhận diện biển vuông 2 dòng của Việt Nam.

5. **Event Streaming (Kafka & Dashboard):**
   - Hệ thống được trang bị Web Server Kafka. Khi biển số đạt đủ điều kiện khóa (Locked), một Payload JSON chứa thông tin biển số, tỷ lệ tin cậy, kèm theo hình ảnh cắt (Crop image) sẽ được truyền lên MinIO và Kafka, hiển thị Real-time trên màn hình Dashboard quản lý.

---

## 📂 Cấu trúc Mã Nguồn

Dự án được chia thành các module chức năng rõ ràng:

```text
├── custom_plugins/          # Chứa mã nguồn C++/CUDA của Plugin dslaplacian
│   └── ds_laplacian/        # Lõi GPU xử lý bộ lọc mờ (Makefile, .cu, .cpp)
├── src/                     # Toàn bộ mã nguồn Python nghiệp vụ (Pipeline)
│   ├── app_lpr_v2_new_ocr.py# Điểm khởi chạy (Entry point) chính
│   ├── lpr_config.py        # Cấu hình cứng (Paths, Class IDs)
│   └── lpr/                 # Thư mục Core Logic
│       ├── pipeline.py      # Dựng các node GStreamer (nvstreammux -> YOLO -> Tracker -> OCR)
│       ├── ocr.py           # Bóc tách tensor OCR
│       ├── plate_text.py    # Thuật toán ghép biển 2 dòng & Regex
│       └── probes/          # Các Que thử (Probes) chặn giữa đường ống
│           ├── pgie.py      # Chặn sau YOLO: Vứt bỏ xe rác, khuất lấp
│           ├── metadata.py  # Chốt kết quả Laplacian, Lock Plate và gửi Kafka
│           └── osd.py       # Vẽ Bounding Box & Text lên màn hình
├── web_server_kafka/        # Giao diện Web Dashboard & Kafka Consumer
├── configs/                 # Chứa cấu hình nvinfer (DeepStream TXT Configs)
├── models_converted/        # Nơi lưu trữ các mô hình AI đã chuyển sang TensorRT (.engine)
└── outputs/                 # Kết quả xuất ra (Video, Logs, JSONL)
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy

### 1. Yêu cầu hệ thống
- Hệ điều hành Linux (Ubuntu).
- Card đồ họa NVIDIA hỗ trợ CUDA.
- Docker và NVIDIA Container Toolkit.
- Chạy bên trong container `nvcr.io/nvidia/deepstream:6.4-triton-multiarch` (hoặc bản tương đương).

### 2. Biên dịch (Build) Plugin `dslaplacian`
Mỗi khi triển khai hệ thống lần đầu hoặc sửa đổi code C++, bạn cần Build plugin này:
```bash
cd custom_plugins/ds_laplacian
make clean
make
make install
# Xóa cache gstreamer để cập nhật plugin mới
rm -rf ~/.cache/gstreamer-1.0/
```

### 3. Chạy Toàn Bộ Hệ Thống (Local / Offline)
Chạy kịch bản xử lý nhiều luồng RTSP (hoặc Video thô), hiển thị trực tiếp lên màn hình:
```bash
./run_full_system.sh
```

### 4. Chạy Hệ Thống Kèm Dashboard (Kafka)
Bật toàn bộ hệ sinh thái: Gửi sự kiện lên Web Kafka Dashboard:
```bash
# Trong một terminal: Chạy DeepStream đẩy dữ liệu Kafka
./run_full_system_kafka.sh

# Trong terminal thứ hai: Khởi chạy Giao diện Dashboard
cd web_server_kafka
docker-compose up -d
python3 media_monitor_kafka.py
```
Sau đó mở trình duyệt tại `http://localhost:8000` để xem kết quả Real-time.

---

## 📊 Báo Cáo Kỹ Thuật Chuyên Sâu
Nếu bạn cần tìm hiểu thêm về kiến trúc, thuật toán và kết quả đánh giá (Benchmarking), vui lòng đọc các tài liệu sau đã được đính kèm trong thư mục gốc:
- `REPORT_TECHNICAL_DEEP.md`: Báo cáo kỹ thuật phân tích toàn bộ quy trình, logic xử lý và luồng dữ liệu.
- `REPORT_CUSTOM_PLUGIN.md`: Phân tích chuyên sâu về thuật toán Zero-Copy Memory C++/CUDA của Custom Plugin.
- `SRC_DOCUMENTATION.md`: Bản đồ chi tiết giải thích nhiệm vụ của từng file trong thư mục `src`.
