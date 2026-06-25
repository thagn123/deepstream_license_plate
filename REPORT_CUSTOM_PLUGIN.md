# Báo Cáo Kỹ Thuật Chuyên Sâu: Plugin `dslaplacian` (GPU-Accelerated)

Tài liệu này giải thích chi tiết từng dòng code, thuật toán và luồng dữ liệu của Custom Plugin `dslaplacian` mà chúng ta đã xây dựng. Đây là tài liệu cốt lõi giúp bạn làm chủ công nghệ GPU và dễ dàng trình bày với hội đồng hoặc đối tác.

---

## 1. Tại sao phải viết Plugin `dslaplacian` bằng C++ / CUDA?
Trong các hệ thống DeepStream thông thường, nếu muốn đánh giá độ nét của biển số, lập trình viên thường dùng Python/OpenCV. Khi đó, GPU phải copy mảng pixel của hình ảnh ngược trở lại RAM cho CPU xử lý. Việc này gây ra **thắt cổ chai PCIe** làm sụt giảm FPS.

**Giải pháp của chúng ta:** Viết một Plugin bằng C++ và CUDA can thiệp trực tiếp vào bộ nhớ VRAM của GPU. Dữ liệu hình ảnh **không bao giờ rời khỏi GPU**. Mọi biến đổi điểm ảnh được xử lý song song bởi hàng ngàn nhân CUDA.

---

## 2. Kiến trúc 3 Bước (Hybrid CPU-GPU)

Quá trình "Tìm 4 góc -> Trải phẳng -> Đo độ sắc nét" không thể chạy hoàn toàn 100% trên GPU vì thuật toán tìm viền (`findContours`) là thuật toán tuần tự. Do đó, mã nguồn được chia làm 3 bước thông minh kết hợp giữa CPU và GPU.

### BƯỚC 1: Lấy vùng ảnh con (GPU -> CPU)
*(File: `laplacian_lib.cu` - Hàm `gpu_crop_and_resize_to_cpu`)*

Thay vì copy toàn bộ khung hình Full HD về CPU, ta dùng 1 CUDA Kernel nhỏ để chỉ copy đúng vùng tọa độ Bounding Box của biển số (ví dụ 150x50 pixel) đưa vào một vùng nhớ đặc biệt gọi là **Mapped Memory** (RAM mà cả CPU và GPU đều truy cập được).
```cpp
// Kernel CUDA: Chạy song song trên GPU để copy từng pixel của biển số
__global__ void crop_resize_kernel(const uint8_t* in_data, int in_pitch, ...) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Tọa độ X của Thread
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Tọa độ Y của Thread
    // ...
    // Copy đúng pixel đó sang buffer của CPU
    out_data[y * out_w + x] = in_data[src_y * in_pitch + src_x];
}
```

### BƯỚC 2: Dò tìm 4 góc và tính Ma trận nghịch đảo (CPU)
*(File: `gstlaplacian.cpp` - Dòng 130-170)*

Lúc này, CPU nhận được một bức ảnh biển số rất nhỏ. CPU sẽ chạy các hàm C++ OpenCV tốc độ cao để tìm viền và tính ma trận trải phẳng.
```cpp
// 1. Chạy Canny Edge Detection để lấy viền
cv::Canny(blur, edges, 50, 150);

// 2. Tìm danh sách các viền (Contours)
std::vector<std::vector<cv::Point>> contours;
cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

// 3. Tìm tứ giác lớn nhất đại diện cho biển số
for (const auto& contour : contours) {
    cv::approxPolyDP(contour, approx, arcLength * 0.02, true);
    if (approx.size() == 4) { /* Đây là biển số */ }
}

// 4. Tính ma trận M-1 (Inverse Perspective Matrix)
cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
cv::Mat M_inv = M.inv();
```
 CPU không tự biến đổi ảnh! Nó chỉ tính ra cái **Ma trận M_inv** rồi ném ngược cái ma trận đó lại cho GPU.

### BƯỚC 3: Trải phẳng, Lọc nhiễu và Đo độ sắc nét (GPU Heavy Duty)
*(File: `laplacian_lib.cu` - Hàm `gpu_warp_equalize_blur_laplacian`)*

Đây là trái tim của hệ thống. GPU nhận ma trận `M_inv` và dùng 2 CUDA Kernels để xử lý hàng vạn pixel cùng một lúc.

**Kernel 3.1: Warp Perspective & Min/Max**
- Mỗi Thread trên GPU tự động nhân ma trận để dời 1 pixel của biển số xéo về vị trí thẳng đứng.
- Đồng thời, chúng sử dụng lệnh `atomicMin` và `atomicMax` để tìm ra điểm sáng nhất và tối nhất của biển số (chuẩn bị cho bước Equalize).

**Kernel 3.2: Stretch (Equalize), Blur và Laplacian (All-in-one)**
Thay vì phải chạy 3 vòng lặp (1 cho độ tương phản, 1 cho làm mờ, 1 cho độ nét), Kernel này gom cả 3 phép toán vào một công thức chạy trong đúng **1 chu kỳ**.
```cpp
__global__ void stretch_blur_laplacian_kernel(...) {
    // 1. Kéo giãn độ tương phản (Contrast Stretching)
    float val = (raw_val - min_val) * 255.0f / (max_val - min_val);

    // 2. Gaussian Blur (Nhân ma trận 3x3 với trọng số chuông)
    float blur_val = val * 0.25f + l * 0.125f + r * 0.125f ...

    // 3. Tính toán phương sai Laplacian (Đạo hàm bậc 2)
    // Tính khoảng cách từ điểm hiện tại đến giá trị trung bình
    float diff = blur_val - mean_val;
    
    // Cộng dồn phương sai vào biến toàn cục một cách an toàn (atomicAdd)
    atomicAdd(out_variance, diff * diff); 
}
```

---

## 3. Cách luân chuyển Dữ liệu vào Pipeline Python
*(File: `gstlaplacian.cpp` - Hàm `gst_laplacian_transform_ip`)*

Làm sao để code Python biết được điểm số mà CUDA vừa tính?
Thay vì tạo ra các cấu trúc rườm rà, chúng ta lợi dụng mảng `misc_obj_info` có sẵn trong bộ nhớ gốc của hệ thống (NvDsObjectMeta).
```cpp
// Lấy kết quả từ GPU (ví dụ được 245.8 điểm)
float variance = gpu_warp_equalize_blur_laplacian(...);

// Ghi đè thẳng vào slot số 0 của metadata biển số đó
obj_meta->misc_obj_info[0] = (int)variance;
```

*(File: `src/lpr/probes/metadata.py` - Phía Python)*

Khi Frame trôi đến Python, Python chỉ việc móc cái túi `misc_obj_info[0]` ra xem điểm:
```python
# Đọc điểm từ plugin C++
lap_score = int(p.misc_obj_info[0])

# Chặn ngay lập tức nếu biển số mờ (Ví dụ: Ngưỡng 150)
if lap_score < 150 and lap_score > 0:
    continue  # Vứt bỏ Bounding Box này, không thèm chạy AI OCR nữa!
```

---

## 4. Cách Build (Biên dịch) Mã Nguồn
Vì Plugin này sử dụng CUDA (đuôi `.cu`), trình biên dịch C++ thông thường (`g++`) không thể hiểu được. Ta phải dùng trình biên dịch siêu cấp của NVIDIA là **`nvcc`** (NVIDIA CUDA Compiler).

Trong thư mục `custom_plugins/ds_laplacian/Makefile`:
1. `nvcc` sẽ dịch file `laplacian_lib.cu` thành file đối tượng phần cứng (`laplacian_lib.o`).
2. `g++` sẽ dịch file cầu nối `gstlaplacian.cpp` thành `gstlaplacian.o`.
3. Cả hai được gộp lại thành thư viện động chia sẻ: `libnvdsgst_laplacian.so`.

**Lệnh Build:**
Chỉ cần chạy lệnh `make` trong thư mục `custom_plugins/ds_laplacian/`. File `.so` sau khi tạo ra sẽ tự động được copy vào `/opt/nvidia/deepstream/deepstream/lib/gst-plugins/` để hệ thống GStreamer toàn cầu nhận diện.

---
**Tổng kết:** Plugin này là một mảnh ghép "nhỏ nhưng có võ", đóng vai trò như một màng lọc vật lý (Hardware Filter), bảo vệ trái tim AI OCR khỏi việc bị quá tải bởi các hình ảnh rác.
