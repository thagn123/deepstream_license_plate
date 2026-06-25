#include "laplacian_lib.h"
#include <stdio.h>

// Kernel 1: Cắt ảnh và thay đổi kích thước, đưa vào bộ nhớ dùng chung (Managed Buffer)
__global__ void crop_resize_kernel(unsigned char* in_data, int pitch, int start_x, int start_y, int crop_w, int crop_h, unsigned char* out_data, int out_w, int out_h) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < out_w && dy < out_h) {
        // Thuật toán nội suy Nearest Neighbor (Hàng xóm gần nhất) là đủ tốt để tìm các góc
        int src_x = start_x + (dx * crop_w) / out_w;
        int src_y = start_y + (dy * crop_h) / out_h;

        if (src_x >= 0 && src_x < start_x + crop_w && src_y >= 0 && src_y < start_y + crop_h) {
            out_data[dy * out_w + dx] = in_data[src_y * pitch + src_x];
        } else {
            out_data[dy * out_w + dx] = 0;
        }
    }
}

extern "C" void gpu_crop_and_resize_to_cpu(void* d_in_data, int pitch, int start_x, int start_y, int crop_w, int crop_h, unsigned char* out_managed_buf, int out_w, int out_h, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((out_w + blockSize.x - 1) / blockSize.x, (out_h + blockSize.y - 1) / blockSize.y);
    crop_resize_kernel<<<gridSize, blockSize, 0, stream>>>((unsigned char*)d_in_data, pitch, start_x, start_y, crop_w, crop_h, out_managed_buf, out_w, out_h);
}

// Kernel 2: Trải phẳng ảnh (Warp Perspective) và Tìm điểm Sáng/Tối nhất (MinMax)
__global__ void warp_and_minmax_kernel(unsigned char* in_data, int pitch, int start_x, int start_y, int crop_w, int crop_h, float* m, unsigned char* warped_out, int out_w, int out_h, int* out_min, int* out_max) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < out_w && dy < out_h) {
        // Phép chiếu ngược góc nhìn (Inverse perspective mapping)
        // Để biết điểm (dx, dy) hiện tại nằm ở đâu trên ảnh gốc, ta dùng ma trận nghịch đảo.
        // Vì cv::getPerspectiveTransform trả về ma trận chiếu từ ảnh gốc sang ảnh đích.
        // Do đó, ta BẮT BUỘC phải tính ma trận nghịch đảo (M_inv) trên CPU và truyền vào đây!
        // Giả sử m00..m22 là các phần tử của ma trận M_inv:
        float src_z = m[6] * dx + m[7] * dy + m[8];
        float src_x_f = (m[0] * dx + m[1] * dy + m[2]) / src_z;
        float src_y_f = (m[3] * dx + m[4] * dy + m[5]) / src_z;

        int src_x = start_x + (int)src_x_f;
        int src_y = start_y + (int)src_y_f;

                        unsigned char p = 0;
        if (src_x >= start_x && src_x < start_x + crop_w && src_y >= start_y && src_y < start_y + crop_h) {
            p = in_data[src_y * pitch + src_x];
                    }

        warped_out[dy * out_w + dx] = p;

        if (p > 0) { // Bỏ qua các viền đen bị sinh ra do quá trình biến đổi góc nhìn
            atomicMin(out_min, p);
            atomicMax(out_max, p);
        }
    }
}

// Kernel 3: Kéo giãn tương phản, Làm mờ Gaussian, Tính phương sai Laplacian (Gộp chung)
__global__ void stretch_blur_laplacian_kernel(unsigned char* warped_in, int out_w, int out_h, int p_min, int p_max,
                                              float* out_sum, float* out_sq_sum, int* out_count) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= 1 && dx < out_w - 1 && dy >= 1 && dy < out_h - 1) {
        int range = p_max - p_min;
        if (range == 0) range = 1;

        auto get_pixel = [&](int offset_x, int offset_y) {
            int p = warped_in[(dy + offset_y) * out_w + (dx + offset_x)];
            return (p - p_min) * 255 / range;
        };

        // Ma trận làm mờ Gaussian 3x3 thông thường là:
        // 1 2 1
        // 2 4 2
        // 1 2 1
        // Việc đọc 9 pixel xung quanh, rồi lại dùng kết quả đó để tính Laplacian 
        // sẽ yêu cầu mỗi Thread phải đọc tới 25 pixel.
        // Đáng lý ra ta nên tách hàm Blur ra thành một Kernel riêng hoặc dùng Shared Memory.
        // Tuy nhiên, với kích thước biển số nhỏ (150x50), việc đọc thẳng 25 pixel sẽ được tăng tốc 
        // cực mạnh nhờ bộ nhớ đệm L1 Cache của GPU. Nên ta sẽ gộp chung vào làm luôn một lần!
        
        auto get_blurred = [&](int x, int y) {
            int sum = 0;
            sum += 1 * get_pixel(x - 1, y - 1);
            sum += 2 * get_pixel(x,     y - 1);
            sum += 1 * get_pixel(x + 1, y - 1);
            sum += 2 * get_pixel(x - 1, y);
            sum += 4 * get_pixel(x,     y);
            sum += 2 * get_pixel(x + 1, y);
            sum += 1 * get_pixel(x - 1, y + 1);
            sum += 2 * get_pixel(x,     y + 1);
            sum += 1 * get_pixel(x + 1, y + 1);
            return sum / 16;
        };

        // Đề phòng lỗi tràn viền: Nếu đang ở viền (dx==1), hàm get_blurred sẽ vô tình đọc dx-1-1 = dx-2 (âm).
        // Do đó, ta chỉ tính Laplacian cho các điểm cách viền 2 pixel (dx >= 2).
        if (dx >= 2 && dx < out_w - 2 && dy >= 2 && dy < out_h - 2) {
            int p_01 = get_blurred(0, -1);
            int p_10 = get_blurred(-1, 0);
            int p_11 = get_blurred(0, 0);
            int p_12 = get_blurred(1, 0);
            int p_21 = get_blurred(0, 1);

            int laplacian = p_01 + p_10 + p_12 + p_21 - 4 * p_11;
            float val = (float)laplacian;

            atomicAdd(out_sum, val);
            atomicAdd(out_sq_sum, val * val);
            atomicAdd(out_count, 1);
        }
    }
}

extern "C" double gpu_warp_equalize_blur_laplacian(void* d_in_data, int pitch, int start_x, int start_y, int crop_w, int crop_h, float* Minv, int out_w, int out_h, cudaStream_t stream) {
    if (out_w <= 4 || out_h <= 4) return 0.0;

    unsigned char* d_warped;
    cudaMalloc(&d_warped, out_w * out_h);

    float* d_m;
    cudaMalloc(&d_m, 9 * sizeof(float));
    cudaMemcpyAsync(d_m, Minv, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);

    float *d_sum, *d_sq_sum;
    int *d_count;
    int *d_min, *d_max;
    cudaMalloc(&d_min, sizeof(int));
    cudaMalloc(&d_max, sizeof(int));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_sq_sum, sizeof(float));
    cudaMalloc(&d_count, sizeof(int));

    int init_min = 255, init_max = 0;
    cudaMemcpyAsync(d_min, &init_min, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_max, &init_max, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_sum, 0, sizeof(float), stream);
    cudaMemsetAsync(d_sq_sum, 0, sizeof(float), stream);
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);

    dim3 blockSize(16, 16);
    dim3 gridSize((out_w + blockSize.x - 1) / blockSize.x, (out_h + blockSize.y - 1) / blockSize.y);

    printf("Minv: %f %f %f | %f %f %f | %f %f %f\n", Minv[0], Minv[1], Minv[2], Minv[3], Minv[4], Minv[5], Minv[6], Minv[7], Minv[8]);
    warp_and_minmax_kernel<<<gridSize, blockSize, 0, stream>>>((unsigned char*)d_in_data, pitch, start_x, start_y, crop_w, crop_h, d_m, d_warped, out_w, out_h, d_min, d_max);

    int h_min, h_max;
    cudaMemcpyAsync(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    stretch_blur_laplacian_kernel<<<gridSize, blockSize, 0, stream>>>(d_warped, out_w, out_h, h_min, h_max, d_sum, d_sq_sum, d_count);

    float h_sum = 0, h_sq_sum = 0;
    int h_count = 0;
    cudaMemcpyAsync(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_sq_sum, d_sq_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_warped);
    cudaFree(d_m);
    cudaFree(d_min); cudaFree(d_max);
    cudaFree(d_sum); cudaFree(d_sq_sum); cudaFree(d_count);

    if (h_count == 0) { printf("GPU Laplacian h_count is 0!\n"); return 0.0; }
    printf("GPU Laplacian success: h_min=%d, h_max=%d, h_count=%d, sum=%f, sq_sum=%f\n", h_min, h_max, h_count, h_sum, h_sq_sum);
    double mean = h_sum / h_count;
    double variance = (h_sq_sum / h_count) - (mean * mean);
    return variance;
}
