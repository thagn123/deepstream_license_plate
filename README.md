# 🏎️ DeepStream License Plate Recognition (LPR) Pipeline

Một hệ thống tự động nhận diện biển số xe (ALPR) hiệu năng cao, được thiết kế và tối ưu hóa ở mức phần cứng bằng công nghệ NVIDIA DeepStream (9.0), CUDA và GStreamer. Hệ thống được triển khai theo kiến trúc Edge-to-Cloud, gửi sự kiện thời gian thực (Real-time Events) qua Kafka tới Dashboard quản lý trung tâm.

---

## 🌟 Tính năng nổi bật

1. **Hiệu năng Real-time vượt trội :** 
   - Khả năng xử lý tốc độ cao trên phần cứng Edge/GPU với luồng video H.264 và RTSP.
   - Hỗ trợ xử lý đa luồng (Multi-stream) cùng lúc.

2. **Bộ lọc làm mịn & Căn chỉnh góc nghiêng tăng tốc phần cứng (`dslaplacian` plugin):**
   - **GPU Warp & Blur Filter:** Plugin custom `dslaplacian` được viết bằng **C++ và CUDA Kernels**, can thiệp trực tiếp vào bộ nhớ VRAM (`Zero-copy memory`), giúp tránh hiện tượng nghẽn cổ chai. Phép xoay nghiêng (Perspective Warp), cân bằng tương phản (Contrast Stretch), khử nhiễu (Gaussian Blur) và tính toán phương sai sắc nét (Laplacian Variance) được gom vào **1-pass Fused Kernel** chạy song song trên GPU.
   - **Căn chỉnh dựa vào Ký tự (Character-Based Alignment - `char_est`):** Khắc phục lỗi rò rỉ cản xe (Bumper Leakage) bằng cách phân ngưỡng thích ứng (`adaptiveThreshold`) dò tìm lõi cụm ký tự (`RETR_CCOMP`), tìm hộp xoay nghiêng (`minAreaRect`) và nhân rộng theo tỷ lệ Aspect Ratio chuẩn để khôi phục chính xác 4 góc biển số.
   - **Cơ chế Fallback Đa tầng (Robust Fallback Stack):** 5 cấp độ dự phòng thông minh (`char_est` -> `otsu` -> `canny` -> `canny minAreaRect` -> `scale`) đảm bảo hệ thống bền bỉ, không bao giờ bị crash kể cả khi biển số bị mờ nát hoặc che khuất hoàn toàn.

3. **Kiến trúc AI Module Tiên Tiến:**
   - **PGIE (Primary Inference):** Mô hình YOLOv11s phát hiện các phương tiện (Ô tô, Xe máy) và Biển số.
   - **NVTracker:** NvDCF Tracker giữ vết liên tục của các đối tượng xe và biển số xuyên suốt các khung hình.
   - **SGIE (Secondary Inference):** Mô hình OCR tiên tiến (ONNX 2024 - `lpr_ocr.20240305.onnx`) hỗ trợ nhận dạng trực tiếp cả biển dài 1 dòng và biển vuông 2 dòng của Việt Nam mà không cần phân tách pseudo-objects.

4. **Cơ chế "Lock Plate" & Tiết kiệm Tài nguyên (OCR Throttling):**
   - Theo dõi độ nét và duy trì bầu chọn ký tự (Voting History) để khóa (Lock) biển số có kết quả tin cậy cao nhất.
   - **OCR Throttling:** Đối với các biển số đã khóa, probe `sgie3.py` tự động đổi `class_id` thành `99` ở các frame trung gian, bỏ qua inference SGIE3 trên GPU để tiết kiệm hơn 80% tải GPU, chỉ chạy lại kiểm duyệt định kỳ mỗi `OCR_EVERY_N = 6` frame.

5. **Event Streaming (Kafka & Dashboard):**
   - Tích hợp Kafka Producer gửi sự kiện JSON real-time kèm ảnh crop phương tiện và biển số (lưu trên MinIO S3) trực tiếp về Web Dashboard hiển thị sự kiện thời gian thực.

---

## 📂 Cấu trúc Mã Nguồn

Dự án được chia thành các module chức năng rõ ràng:

```text
├── custom_plugins/          # Chứa mã nguồn C++/CUDA của Plugin dslaplacian
│   └── ds_laplacian/        # Lõi GPU xử lý warp, blur và tính toán Laplacian (C++/CUDA)
├── src/                     # Toàn bộ mã nguồn Python nghiệp vụ (Pipeline)
│   ├── app_lpr_v2.py        # Điểm khởi chạy (Entry point) chính
│   ├── lpr_config.py        # Cấu hình cứng (Paths, Class IDs, thresholds)
│   └── lpr/                 # Thư mục Core Logic
│       ├── pipeline.py      # Dựng và kết nối các node GStreamer
│       ├── ocr.py           # CTC Decoding cho OCR tensor
│       ├── plate_text.py    # Thuật toán ghép biển 2 dòng & Regex chuẩn hóa VN
│       ├── state.py         # Quản lý trạng thái tracking và locked IDs toàn cục
│       └── probes/          # Các Que thử (Probes) chặn giữa đường ống
│           ├── pgie.py      # Chặn sau YOLO: Vứt bỏ xe rác, khuất lấp
│           ├── sgie3.py     # OCR Throttling (chuyển class 13 -> 99 cho biển đã lock)
│           └── metadata.py  # Chốt kết quả Laplacian, Lock Plate, vẽ OSD và gửi Kafka
├── web_server_kafka/        # Giao diện Web Dashboard & Kafka Consumer
├── configs/                 # Chứa cấu hình nvinfer (DeepStream TXT Configs)
├── models/                  # Lưu trữ file mô hình ONNX và TensorRT (.engine)
├── tools/                   # Tiện ích đo lường hiệu năng và media monitor
└── reports/                 # Sơ đồ và ảnh báo cáo trực quan
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy

### 1. Yêu cầu hệ thống
- Hệ điều hành Linux (Ubuntu).
- Card đồ họa NVIDIA hỗ trợ CUDA.
- Docker và NVIDIA Container Toolkit.
- Chạy bên trong container `nvcr.io/nvidia/deepstream:9.0-devel` (hoặc bản tương đương chứa DeepStream 9.0, OpenCV và CUDA).

### 2. Biên dịch (Build) và cài đặt Plugin `dslaplacian`
Mỗi khi sửa đổi code C++/CUDA của Custom Plugin, biên dịch lại và đưa vào thư mục plugin của DeepStream:
```bash
cd custom_plugins/ds_laplacian
make clean
make
# Copy thư viện liên kết động (.so) vào thư mục plugins của DeepStream
cp libnvdsgst_laplacian.so /opt/nvidia/deepstream/deepstream/lib/gst-plugins/
# Xóa cache gstreamer để cập nhật plugin mới
rm -rf ~/.cache/gstreamer-1.0/
```

### 3. Chạy Toàn Bộ Hệ Thống (Local / Offline)
Chạy kịch bản xử lý video thô hoặc luồng camera, hiển thị trực tiếp lên màn hình máy Host (qua kết nối X11):
```bash
# Cấp quyền X server trên máy Host
xhost +local:docker
# Chạy script
./run_full_system.sh
```

### 4. Chạy Hệ Thống Kèm Dashboard (Kafka)
Bật toàn bộ hệ sinh thái: Gửi sự kiện lên Web Kafka Dashboard:
```bash
# Terminal 1: Chạy DeepStream đẩy dữ liệu Kafka & lưu ảnh MinIO
./run_full_system_kafka.sh

# Terminal 2: Khởi chạy Giao diện Dashboard (Web UI & Consumer)
cd web_server_kafka
docker-compose up -d
python3 media_monitor_kafka.py
```
Sau đó mở trình duyệt tại máy Host: `http://localhost:8001` (hoặc console MinIO tại `http://localhost:9001` - user/pass: `minioadmin` / `minioadmin`).

---

## 📊 Báo Cáo Kỹ Thuật Chuyên Sâu
Các tài liệu phân tích kiến trúc, thuật toán và kết quả đánh giá (Benchmarking) nằm trong thư mục gốc:
- [REPORT_TECHNICAL_DEEP.md](file:///home/thagn/projects/deepstream/workspace/last_ds_cp/REPORT_TECHNICAL_DEEP.md): Báo cáo kỹ thuật chi tiết phân tích toàn bộ quy trình, logic xử lý và luồng dữ liệu.
- [technical_report_alignment.md](file:///home/thagn/projects/deepstream/workspace/last_ds_cp/technical_report_alignment.md): Phân tích chuyên sâu về giải thuật căn chỉnh biển số bằng ký tự (`char_est`) và 5 cấp độ Fallback.
- [performance_report.md](file:///home/thagn/projects/deepstream/workspace/last_ds_cp/performance_report.md): Báo cáo thực nghiệm đo đạc hiệu năng và so sánh tốc độ xử lý trước và sau tối ưu.
- [walkthrough.md](file:///home/thagn/projects/deepstream/workspace/last_ds_cp/walkthrough.md): So sánh sai số góc xoay thực tế và lịch sử cập nhật hệ thống.
- [REPORT_CUSTOM_PLUGIN.md](file:///home/thagn/projects/deepstream/workspace/last_ds_cp/REPORT_CUSTOM_PLUGIN.md): Phân tích chuyên sâu về thuật toán Zero-Copy Memory C++/CUDA của Custom Plugin.
- [SRC_DOCUMENTATION.md](file:///home/thagn/projects/deepstream/workspace/last_ds_cp/SRC_DOCUMENTATION.md): Bản đồ chi tiết giải thích nhiệm vụ của từng file trong thư mục `src`.
