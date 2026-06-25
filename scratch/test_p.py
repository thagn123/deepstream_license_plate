import re
with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'r') as f:
    content = f.read()

content = content.replace('p = in_data[src_y * pitch + src_x];', 'p = in_data[src_y * pitch + src_x];\n            if (p == 0) p = 1;')

with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'w') as f:
    f.write(content)
