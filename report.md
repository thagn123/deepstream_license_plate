# Báo Cáo: Tối ưu hoá quy trình kiểm thử độ nét biển số (LPR Quality Assessment) bằng GPU

## 1. Mục tiêu
Di chuyển toàn bộ quy trình tiền xử lý nặng (Warp Perspective, Equalization, Gaussian Blur, và tính toán Laplacian Variance) từ Python (OpenCV CPU) sang plugin GStreamer chạy trên GPU (CUDA) để đạt hiệu năng thực tế (real-time) và loại bỏ hoàn toàn các vấn đề đồng bộ hóa (desync) khi lấy ảnh.

## 2. Các vấn đề đã giải quyết
1. **Lỗi Desync giữa Bounding Box và Khung hình:** 
   Việc sử dụng luồng phụ OpenCV để đọc khung hình song song với GStreamer pipeline đã bị loại bỏ. Thay vào đó, bộ đệm Y-plane (Grayscale) của khung hình được trích xuất trực tiếp từ bộ nhớ VRAM của GPU (NvBufSurface) bên trong pipeline, đảm bảo độ chính xác 100% về vị trí cắt ảnh.
2. **Nghẽn cổ chai CPU (CPU Bottleneck):**
   Tính toán ma trận Warp và Laplacian trên CPU bằng Python khiến FPS sụt giảm mạnh mẽ đối với nhiều đối tượng.

## 3. Kiến trúc giải pháp (Hybrid CPU-GPU Plugin)
Do thuật toán dò góc `findContours` là thuật toán tuần tự nên không thể chạy hoàn toàn trên GPU. Chúng tôi đã thiết kế một Custom GStreamer Plugin (`dslaplacian`) sử dụng kiến trúc kết hợp:
1. **Bước 1 (GPU -> CPU):** Một CUDA Kernel nhỏ (`crop_resize_kernel`) sao chép vùng chứa biển số từ NV12 Y-plane sang một bộ đệm nhỏ (150x50) ở bộ nhớ dùng chung (Unified/Pinned Memory) để CPU có thể đọc tức thì.
2. **Bước 2 (CPU):** CPU sử dụng OpenCV C++ (`cv::Canny`, `cv::findContours`, `cv::approxPolyDP`) để dò 4 góc biển số, và tính toán ma trận nghịch đảo góc nhìn (Inverse Perspective Matrix - $M^{-1}$).
3. **Bước 3 (GPU Heavy Load):** Một chuỗi các CUDA Kernels hiệu năng cao đảm nhận phần việc còn lại:
   - `warp_and_minmax_kernel`: Thực hiện trải phẳng biển số (Warp Perspective) thông qua $M^{-1}$ và tìm giá trị Min/Max của ảnh song song.
   - `stretch_blur_laplacian_kernel`: Thực hiện cân bằng sáng (Contrast Stretching/Equalize), Gaussian Blur, và tính điểm Laplacian bằng thuật toán ma trận $3x3$.
4. **Bước 4 (Metadata):** Kết quả độ sắc nét (Laplacian Score) được lưu vào `obj_meta->misc_obj_info[0]` để có thể dễ dàng truy xuất từ bất kỳ module Python hay C++ nào phía sau.

## 4. Kết quả đạt được
- **Hiệu năng:** Điểm số Laplacian được tính trực tiếp từ GPU với độ trễ cực thấp. GPU đã đảm nhiệm thành công toàn bộ các tác vụ biến đổi từng điểm ảnh (Pixel-wise transformations).
- **Độ chính xác:** Các điểm số trả về trong khoảng $(0, 1000)$ tương ứng và chính xác với mô hình đánh giá trên CPU trước đó.
- **Tính khả dụng:** `test_laplacian.py` đã loại bỏ hoàn toàn module OpenCV và việc xử lý ảnh, chỉ cần đọc metadata được tính sẵn từ plugin `dslaplacian`.

## 5. Tích hợp Pipeline và Vị trí lý tưởng
Plugin `dslaplacian` đã được tích hợp thành công vào pipeline LPR chính (`src/lpr/pipeline.py`) với thứ tự: `streammux -> pgie (YOLO) -> tracker -> laplacian -> sgie3 (OCR)`. 

Quyết định đặt plugin **SAU tracker** (thay vì trước tracker) được đưa ra dựa trên 3 lợi ích cốt lõi:
1. **Đánh giá được các box "Shadow Tracking":** Tracker có khả năng dự đoán vị trí biển số ngay cả khi YOLO bị trượt ở một số frame. Đặt sau tracker giúp plugin quét được cả các biển số nội suy này.
2. **Khớp tọa độ mượt với OCR:** Bounding box đã được tracker làm mượt (smooth) tọa độ. Việc tính điểm Laplacian trên vùng này sẽ phản ánh chính xác 100% độ sắc nét của vùng ảnh mà LPRNet (`sgie3`) thực tế sẽ đọc.
3. **Bảo vệ Metadata:** Tránh rủi ro `nvtracker` "làm rơi" hoặc reset custom metadata (`misc_obj_info`) khi nó tái tạo object meta bên trong thuật toán.

## 6. Cấu trúc mã nguồn chính
- `custom_plugins/ds_laplacian/gstlaplacian.cpp`: Thực hiện map buffer, nội suy C++ OpenCV, và chèn metadata.
- `custom_plugins/ds_laplacian/laplacian_lib.cu`: Chứa các nhân CUDA tuỳ chỉnh xử lý ảnh độ trễ thấp.
- `test_laplacian.py`: Script kiểm thử giao tiếp trực tiếp với pipeline và đọc điểm.

*Hoàn thành lúc: 2026-06-25*
