import re
with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'r') as f:
    content = f.read()

content = content.replace('    float *d_sum, *d_sq_sum;\n    int *d_count;', '    float *d_sum, *d_sq_sum;\n    int *d_count;\n    int *d_min, *d_max;')

with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'w') as f:
    f.write(content)
