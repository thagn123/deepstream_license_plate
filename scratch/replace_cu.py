import re

with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'r') as f:
    content = f.read()

# Add printf to check if kernel 2 runs and count
content = content.replace('if (h_count == 0) return 0.0;', 'if (h_count == 0) { printf("GPU Laplacian h_count is 0!\\n"); return 0.0; }\n    printf("GPU Laplacian success: h_min=%d, h_max=%d, h_count=%d, sum=%f, sq_sum=%f\\n", h_min, h_max, h_count, h_sum, h_sq_sum);')

with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'w') as f:
    f.write(content)
