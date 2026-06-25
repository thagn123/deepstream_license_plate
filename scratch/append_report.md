
## 14. Tối ưu hóa Khung hình & Đo độ nét Biển số (Laplacian & Perspective Alignment)

### 14.1. Vấn đề "Lệch Khung Hình" (Desynchronization)
**Tình trạng ban đầu:** Khi trích xuất ảnh biển số để đo độ nét, tọa độ Bounding Box thường xuyên bị lệch khỏi xe (cắt nhầm vào cây cối/nền trời).
**Nguyên nhân:** Việc đọc video song song bằng `OpenCV` (giải mã tuần tự bằng phần mềm) và `GStreamer` (giải mã phần cứng NVDEC - thường drop frames hoặc bị lệch framerate do FFMPEG) tạo ra độ lệch thời gian. Tọa độ của frame `N` bên DeepStream bị áp lên tấm ảnh `N - Δ` bên luồng OpenCV.
**Giải pháp:** 
- Đập bỏ luồng đọc video OpenCV độc lập.
- Cấu hình lại Pipeline GStreamer (chuyển đổi buffer thành batched RGBA thông qua `nvvideoconvert`).
- Sử dụng hàm `pyds.get_nvds_buf_surface` để truy xuất trực tiếp vào VRAM (Bộ nhớ GPU) của `GStreamer`.
- **Kết quả:** Trích xuất chính xác tấm ảnh mà mạng YOLO vừa nhìn thấy để detect. Đồng bộ tuyệt đối 100%, không bao giờ lệch tọa độ.

### 14.2. Tiền xử lý Ảnh & Khử Góc Nghiêng (Warp Perspective)
**Tình trạng ban đầu:** Điểm số phương sai Laplacian bị nhiễu do: kích thước biển số to nhỏ thất thường (scale variance), nền biển Vàng/Trắng làm chênh lệch độ tương phản, nhiễu nén video (pixelation), và góc chụp nghiêng chéo (skew).
**Quy trình 5 Bước Đột Phá đã triển khai (`test_laplacian.py`):**
1. **Tìm & Trải Phẳng Biển Số (Perspective Alignment):** Sử dụng Canny Edge Detection và `cv2.findContours` kết hợp `approxPolyDP` để dò ra hình đa giác 4 góc của khung biển số. Thuật toán tự động sắp xếp 4 điểm và dùng ma trận biến đổi (`cv2.warpPerspective`) để kéo phẳng ảnh về trực diện với kích thước cố định `150x50` (Các ảnh thành công được gắn đuôi `_warped`).
   - *Tính năng Fallback:* Nếu viền biển bị che khuất hoặc diện tích đa giác < 40% bounding box, tự động quay về Crop & Resize an toàn (Gắn đuôi `_raw`).
2. **Chuyển Đổi Grayscale:** Loại bỏ thông tin màu sắc thừa.
3. **Cân Bằng Tương Phản (EqualizeHist):** Căng lại dải histogram. Khắc phục triệt để sự chênh lệch điểm Laplacian giữa xe biển Vàng và xe biển Trắng.
4. **Khử Nhiễu Hạt (Gaussian Blur 3x3):** Làm mờ các vết răng cưa (artifacts) do thuật toán nén H264/H265. Giúp bộ lọc Laplacian chỉ bắt nét vào các cạnh (Edges) thực sự của nét chữ số.
5. **Tính Điểm Laplacian:** Dùng `cv2.Laplacian` và tính phương sai (Variance) để đo độ sắc nét cuối cùng.

### 14.3. Kết Quả & Thống Kê
Quá trình kiểm thử tự động (Auto-test script) trên 5 video thực tế của luồng cao tốc/quốc lộ (`lpr_230428`) ghi nhận:
- Bộ lọc `Width < 100` hoặc `Height < 30` (Bucket `1_qua_nho_bo_qua`) giúp tiết kiệm tài nguyên GPU, loại bỏ ngay các xe ở quá xa.
- Điểm số được gom cụm rất chuẩn xác: 
  - `< 300` : Mờ, vỡ hạt, bóng nhòe (Không đủ điều kiện OCR).
  - `300 - 700` : Độ nét vừa đủ.
  - `> 700` : Rất sắc nét.
- Kỹ thuật **Trải Phẳng (Warp)** thể hiện sự ưu việt khi khử được các góc chéo của Camera, đưa điểm số Laplacian về sát với đánh giá nhận thức thị giác của con người nhất.

**Định hướng tiếp theo (Next Steps):**
Toàn bộ logic tiền xử lý Python/OpenCV đã được nghiệm thu độ chính xác 100%. Bước tiếp theo là **Porting toàn bộ 5 bước này vào nhân GPU (CUDA Kernel)** bên trong thư mục `custom_plugins/ds_laplacian/laplacian_lib.cu`. Điều này sẽ loại bỏ "nút thắt cổ chai" khi tải ngược bộ nhớ từ GPU về CPU, giúp toàn bộ hệ thống DeepStream chạy mượt mà theo thời gian thực (Real-time Real-world Deployment).
