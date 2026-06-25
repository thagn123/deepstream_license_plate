#ifndef _LAPLACIAN_LIB_H_
#define _LAPLACIAN_LIB_H_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Crop and resize the bounding box from NV12 Y-plane to a CPU-accessible managed buffer
void gpu_crop_and_resize_to_cpu(void* d_in_data, int pitch, int start_x, int start_y, int crop_w, int crop_h, unsigned char* out_managed_buf, int out_w, int out_h, cudaStream_t stream);

// Perform Warp, Equalize, Blur, Laplacian on the GPU entirely using the Inverse Perspective Matrix (Minv)
double gpu_warp_equalize_blur_laplacian(void* d_in_data, int pitch, int start_x, int start_y, int crop_w, int crop_h, float* Minv, int out_w, int out_h, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* _LAPLACIAN_LIB_H_ */
