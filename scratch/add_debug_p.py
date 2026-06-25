import re
with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'r') as f:
    content = f.read()

debug_code = """        if (dx == 0 && dy == 0) {
            printf("Kernel [0,0]: p=%d, in_data val=%d\\n", p, in_data[src_y * pitch + src_x]);
        }"""
content = content.replace('if (dx == 0 && dy == 0) {', debug_code + '\n        if (dx == 0 && dy == 0) {')

with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'w') as f:
    f.write(content)
