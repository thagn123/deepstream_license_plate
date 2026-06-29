#ifndef _LAPLACIAN_LIB_H_
#define _LAPLACIAN_LIB_H_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Crop + resize Y-plane → managed/device buffer (CPU-accessible managed memory)
void gpu_crop_and_resize_to_cpu(void* d_in_data, int pitch,
                                 int start_x, int start_y, int crop_w, int crop_h,
                                 unsigned char* out_managed_buf,
                                 int out_w, int out_h, cudaStream_t stream);


// Warp (inverse perspective) + contrast stretch + Gaussian blur + Laplacian variance
// Minv[9]: row-major 3x3 inverse perspective matrix (maps output pixel → source crop pixel)
double gpu_warp_equalize_blur_laplacian(void* d_in_data, int pitch,
                                         int start_x, int start_y, int crop_w, int crop_h,
                                         float* Minv,
                                         int out_w, int out_h, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* _LAPLACIAN_LIB_H_ */
