import re
with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'r') as f:
    content = f.read()

# Replace warp_and_minmax_kernel signature
content = re.sub(
    r'__global__ void warp_and_minmax_kernel\(unsigned char\* in_data, int pitch, int start_x, int start_y, int crop_w, int crop_h, \n                                       float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22,\n                                       unsigned char\* warped_out, int out_w, int out_h, int\* out_min, int\* out_max\)',
    '__global__ void warp_and_minmax_kernel(unsigned char* in_data, int pitch, int start_x, int start_y, int crop_w, int crop_h, float* m, unsigned char* warped_out, int out_w, int out_h, int* out_min, int* out_max)',
    content
)

# Replace usage inside kernel
content = content.replace('m20 * dx + m21 * dy + m22', 'm[6] * dx + m[7] * dy + m[8]')
content = content.replace('m00 * dx + m01 * dy + m02', 'm[0] * dx + m[1] * dy + m[2]')
content = content.replace('m10 * dx + m11 * dy + m12', 'm[3] * dx + m[4] * dy + m[5]')

# In gpu_warp_equalize_blur_laplacian, allocate d_m
alloc_m = """    float* d_m;
    cudaMalloc(&d_m, 9 * sizeof(float));
    cudaMemcpyAsync(d_m, Minv, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);

    int *d_min, *d_max;"""
content = content.replace('    int *d_min, *d_max;', alloc_m)

# Fix kernel call
call_pattern = r'warp_and_minmax_kernel<<<gridSize, blockSize, 0, stream>>>\(\(unsigned char\*\)d_in_data, pitch, start_x, start_y, crop_w, crop_h, \n                                                               Minv\[0\], Minv\[1\], Minv\[2\], Minv\[3\], Minv\[4\], Minv\[5\], Minv\[6\], Minv\[7\], Minv\[8\], \n                                                               d_warped, out_w, out_h, d_min, d_max\);'
content = re.sub(call_pattern, 'warp_and_minmax_kernel<<<gridSize, blockSize, 0, stream>>>((unsigned char*)d_in_data, pitch, start_x, start_y, crop_w, crop_h, d_m, d_warped, out_w, out_h, d_min, d_max);', content)

# Free d_m
content = content.replace('cudaFree(d_warped);', 'cudaFree(d_warped);\n    cudaFree(d_m);')

# Remove debug code
content = re.sub(r'        if \(dx == 0 && dy == 0\) \{\n            printf\(.*?\);\n        \}\n', '', content)

# Remove if (p==0) p=1
content = content.replace('if (p == 0) p = 1;\n', '')

with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'w') as f:
    f.write(content)
