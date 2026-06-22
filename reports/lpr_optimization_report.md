# Báo Cáo: Khắc phục lỗi ONNX Dynamic Batch cho DeepStream LPR

## 1. Hiện tượng lỗi (Bug Description)
Khi chạy pipeline DeepStream với nhiều luồng video cùng lúc (Multi-stream, ví dụ: 3 streams), mô hình PGIE (Vehicle Detection) chỉ nhận diện và vẽ bounding box (bbox) cho luồng đầu tiên (Stream 0). Các luồng còn lại (Stream 1, Stream 2) không hề xuất hiện bất kỳ bounding box nào, mặc dù hệ thống không báo lỗi.

## 2. Nguyên nhân gốc rễ (Root Cause)
Lỗi xuất phát từ cấu trúc tensor của file `vehicle_parking_detect.onnx`.

Khi một mô hình (ví dụ từ PyTorch) được export sang định dạng ONNX, nếu người lập trình không khai báo rõ tham số `dynamic_axes` cho tensor đầu ra (output), ONNX sẽ "ép cứng" (hardcode) kích thước chiều batch (batch dimension) theo giá trị mặc định lúc export (thường là 1).

* **Đầu ra nguyên bản:** `[1, 8400, 6]`
  * `1`: Cố định batch size là 1.
  * `8400`: Số lượng bounding box dự đoán.
  * `6`: Vector chứa [x, y, w, h, objectness, class_prob].

Khi **TensorRT (trong DeepStream)** biên dịch mô hình này với cấu hình `batch-size > 1` (ví dụ `batch-size=3`), nó gặp xung đột:
* Dữ liệu đưa vào có 3 frames (batch=3).
* Tuy nhiên, TensorRT thấy đầu ra ONNX bị ép cứng là `1`, nên nó chỉ cấp phát hoặc xử lý trả về kết quả cho đúng **Slot 0** (frame đầu tiên).
* Các Slot 1 và 2 sẽ không có kết quả hợp lệ (chứa toàn giá trị 0). Do đó, luồng 2 và 3 bị mất sạch bbox.

## 3. Cơ chế khắc phục (The Solution Mechanism)
Để sửa lỗi mà không cần phải export lại mô hình từ mã nguồn huấn luyện (PyTorch), chúng ta can thiệp trực tiếp vào cấu trúc đồ thị (graph) của file ONNX bằng thư viện `onnx` trong Python.

Trong chuẩn định dạng ONNX (Protobuf), kích thước của một chiều (dimension) có thể được định nghĩa bằng một trong hai cách:
1. `dim_value`: Số nguyên cố định (ví dụ: `1`, `8400`).
2. `dim_param`: Tên biến đại diện cho kích thước động (chuỗi, ví dụ: `'batch'`, `'dynamic'`).

### Kịch bản can thiệp (Python Script)
Đoạn code Python dưới đây đã được sử dụng để sửa đổi file ONNX:

```python
import onnx

# 1. Tải mô hình ONNX bị lỗi
model_path = 'models/vehicle_parking_detect.onnx'
new_model_path = 'models/vehicle_parking_detect_dynamic.onnx'
model = onnx.load(model_path)

# 2. Sửa chiều Batch của Input thành Dynamic (phòng hờ)
for i in model.graph.input:
    # Gán chiều đầu tiên (index 0) thành biến chuỗi 'batch'
    i.type.tensor_type.shape.dim[0].dim_param = 'batch'

# 3. Sửa chiều Batch của Output thành Dynamic (Cốt lõi vấn đề)
for o in model.graph.output:
    # Ghi đè dim_value = 1 thành dim_param = 'batch'
    o.type.tensor_type.shape.dim[0].dim_param = 'batch'

# 4. Lưu lại mô hình mới
onnx.save(model, new_model_path)
```

**Điều gì xảy ra sau khi chạy đoạn code này?**
* Chiều đầu tiên của Output không còn là `1` mà chuyển thành chuỗi `'batch'`.
* Shape đầu ra lúc này là: `['batch', 8400, 6]`.

## 4. Kết quả (The Result)
Với file `vehicle_parking_detect_dynamic.onnx` mới:
* TensorRT hiểu rằng chiều đầu tiên là linh hoạt (Dynamic Dimension).
* Nếu bạn thiết lập `batch-size=3` trong `config_pgie.txt`, TensorRT sẽ biên dịch Engine hỗ trợ đầu ra `[3, 8400, 6]`.
* Mảng kết quả TensorRT trả về sẽ có đủ 3 lớp bộ nhớ. Custom Parser C++ (`custom_parser/nvdsinfer_custom_yolov11_flat.cpp`) có thể dễ dàng dùng offset (địa chỉ con trỏ) để bóc tách chính xác mảng `8400x6` cho từng Stream độc lập.
* **Tất cả các luồng (Multi-stream) đều hiển thị Bounding Box đầy đủ và chính xác.**
