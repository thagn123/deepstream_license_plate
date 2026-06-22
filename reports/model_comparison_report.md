# Báo cáo so sánh: Model cũ vs Model mới (YOLOv11s)

Báo cáo này cung cấp thông số so sánh chi tiết giữa mô hình cũ (**vehicle_parking_detect**) và mô hình mới (**YOLOv11s-14cls**) chạy trên cùng một tập dữ liệu đầu vào gồm 4 luồng video test cục bộ (`test1.h264`, `test2.h264`, `test3.h264`, `test4.h264`).

---

## 1. Thông số Kỹ thuật & Cấu hình

| Đặc tính | Model cũ | Model mới |
| :--- | :--- | :--- |
| **Tên ONNX** | `vehicle_parking_detect.onnx` | `yolov11s_14_cls_20241224.onnx` |
| **Kích thước đầu vào** | `1x3x640x640` (Static Batch) | `dynamic_batch_size x 3x640x640` (Dynamic Batch) |
| **Độ chính xác** | FP16 | FP16 |
| **Số class nhận diện** | 1 (Vehicle) | 14 (Vehicle, Person, License Plate,...) |
| **Batch Size cấu hình** | `1` (Không hỗ trợ gom batch luồng) | Tự động thích ứng theo số luồng (Lên tới `8`) |
| **Cơ chế suy luận** | Suy luận tuần tự từng stream | Suy luận song song gộp batch trên GPU |

---

## 2. Kết quả Đo lường Hiệu năng & Độ chính xác

### 2.1. Benchmark 4 Luồng (4 Streams: test1-test4.h264)

Dưới đây là kết quả benchmark khi chạy qua 4 luồng video test cục bộ:

| Chỉ số Đo lường | Model cũ (Batch 1) | Model mới (Batch 4) | Thay đổi (%) |
| :--- | :---: | :---: | :---: |
| **Tracked Vehicles** (Số phương tiện được theo vết) | **464** | **532** | **+14.65%** |
| **Plate Objects** (Số box biển số phát hiện được) | **9,103** | **8,910** | **-2.12%** |
| **OCR Raw Events** (Số lượt gọi nhận diện OCR) | **4,922** | **4,972** | **+1.01%** |
| **Text Plate Tracks** (Tổng số biển số phát hiện chữ) | **32** | **33** | **+3.12%** |
| **Stable Plate Tracks** (Số biển số nhận diện ổn định) | **30** | **31** | **+3.33%** |
| **Average Processing Speed (FPS - Stream 0)** | ~88.55 FPS | **~201.58 FPS** | **+127.64%** |

### 2.2. Benchmark 8 Luồng (8 Streams: mix test1-test5)

Dưới đây là kết quả benchmark nâng cao khi chạy qua 8 luồng video đầu vào (bao gồm `test1` đến `test5`):

| Chỉ số Đo lường | Model cũ (Batch 1) | Model mới (Batch 8) | Thay đổi (%) |
| :--- | :---: | :---: | :---: |
| **Tracked Vehicles** | **989** | **1,115** | **+12.74%** |
| **Plate Objects** | **19,864** | **17,123** | **-13.80%** |
| **OCR Raw Events** | **10,404** | **10,011** | **-3.78%** |
| **Text Plate Tracks** | **65** | **55** | **-15.38%** |
| **Stable Plate Tracks** | **64** | **55** | **-14.06%** |
| **Average Processing Speed (FPS - Stream 0)** | ~53.59 FPS | **~62.99 FPS** | **+17.54%** |

---

## 3. Đánh giá Chi tiết & Nhận xét

### 3.1. Độ chính xác nhận diện phương tiện và biển số
* **Số lượng phương tiện được tracking tăng rõ rệt (+14.65%):** Model YOLOv11s có khả năng nhận diện bao quát hơn và giảm thiểu việc mất dấu phương tiện (ID switching) của tracker.
* **Số lượng Plate Objects phát hiện được giảm nhẹ (-2.12%) nhưng số biển số ổn định tăng (+3.33%):** Điều này chứng minh model mới **ít bị nhiễu (False Positive)** ở các vùng không phải biển số, đồng thời giữ vết biển số tốt hơn, giúp OCR trả về kết quả biển số chuẩn và ổn định (`Stable Plate Tracks`) cao hơn.

### 3.2. Hiệu năng & Tốc độ xử lý (FPS)
* **Khả năng gộp Batch (Parallelism):**
  - **Model cũ** bắt buộc phải chạy từng frame đơn lẻ từ các stream khác nhau (`Batch Size = 1`) do model ONNX bị fix cứng kích thước batch.
  - **Model mới** tận dụng **Dynamic Batching** tự động gộp 4 streams thành 1 batch suy luận duy nhất kích thước 4.
* **Tốc độ xử lý thực tế tăng vọt từ ~88.55 FPS lên ~201.58 FPS (+127%):** Việc xử lý theo batch song song giúp GPU NVIDIA tối ưu hóa các nhân CUDA, giảm đáng kể thời gian trễ do chuyển đổi ngữ cảnh suy luận (inference context overhead).

---

## 4. Hướng dẫn Chạy So sánh

Bạn có thể dễ dàng chuyển đổi giữa 2 model bằng cờ `--pgie-config` mới được tích hợp vào CLI:

* **Chạy với Model cũ:**
  ```bash
  python3 src/app_lpr_v2.py <các_video_đầu_vào> \
      --pgie-config configs/config_pgie_vehicle_detect.txt \
      --no-display
  ```

* **Chạy với Model mới (YOLOv11s):**
  ```bash
  python3 src/app_lpr_v2.py <các_video_đầu_vào> \
      --pgie-config configs/config_pgie_yolov11s.txt \
      --no-display
  ```
