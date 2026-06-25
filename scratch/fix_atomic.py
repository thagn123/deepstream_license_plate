import re
with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'r') as f:
    content = f.read()

# Replace double with float for sums
content = content.replace('double* out_sum, double* out_sq_sum', 'float* out_sum, float* out_sq_sum')
content = content.replace('double val = (double)laplacian;', 'float val = (float)laplacian;')

alloc_sum = """    float *d_sum, *d_sq_sum;
    int *d_count;
    cudaMalloc(&d_min, sizeof(int));
    cudaMalloc(&d_max, sizeof(int));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_sq_sum, sizeof(float));
    cudaMalloc(&d_count, sizeof(int));

    int init_min = 255, init_max = 0;
    cudaMemcpyAsync(d_min, &init_min, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_max, &init_max, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_sum, 0, sizeof(float), stream);
    cudaMemsetAsync(d_sq_sum, 0, sizeof(float), stream);"""

content = re.sub(r'    int \*d_min, \*d_max;.*cudaMemsetAsync\(d_sq_sum, 0, sizeof\(double\), stream\);', alloc_sum, content, flags=re.DOTALL)

read_sum = """    float h_sum = 0, h_sq_sum = 0;
    int h_count = 0;
    cudaMemcpyAsync(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_sq_sum, d_sq_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);"""

content = re.sub(r'    double h_sum = 0, h_sq_sum = 0;\n    int h_count = 0;\n    cudaMemcpyAsync\(&h_sum, d_sum, sizeof\(double\), cudaMemcpyDeviceToHost, stream\);\n    cudaMemcpyAsync\(&h_sq_sum, d_sq_sum, sizeof\(double\), cudaMemcpyDeviceToHost, stream\);', read_sum, content)

with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'w') as f:
    f.write(content)
