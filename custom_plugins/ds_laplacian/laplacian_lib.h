#ifndef _LAPLACIAN_LIB_H_
#define _LAPLACIAN_LIB_H_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Pre-allocated GPU buffer context — create once per plugin instance, reuse every frame
struct LaplGpuCtx {
    unsigned char* d_warped;   // OUT_W * OUT_H bytes
    float*         d_m;        // 9 floats — inverse perspective matrix
    float*         d_sum;
    float*         d_sq_sum;
    int*           d_count;
    int*           d_min;
    int*           d_max;
    cudaStream_t   stream;
};

LaplGpuCtx* lapl_gpu_ctx_create(int out_w, int out_h);
void         lapl_gpu_ctx_destroy(LaplGpuCtx* ctx);

// Crop + resize Y-plane → managed/device buffer (CPU-accessible managed memory)
void gpu_crop_and_resize_to_cpu(void* d_in_data, int pitch,
                                 int start_x, int start_y, int crop_w, int crop_h,
                                 unsigned char* out_managed_buf,
                                 int out_w, int out_h, cudaStream_t stream);

// Warp + contrast stretch + Gaussian blur + Laplacian variance — uses pre-allocated ctx buffers
// Minv[9]: row-major 3×3 inverse perspective matrix (output pixel → source crop pixel)
double gpu_warp_equalize_blur_laplacian_ctx(void* d_in_data, int pitch,
                                             int start_x, int start_y, int crop_w, int crop_h,
                                             float* Minv, int out_w, int out_h,
                                             LaplGpuCtx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* _LAPLACIAN_LIB_H_ */
