import re
with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'r') as f:
    content = f.read()

debug_code = """        if (dx == 0 && dy == 0) {
            printf("Kernel [0,0]: src_x_f=%f, src_y_f=%f, src_x=%d, src_y=%d, start_x=%d, start_y=%d, crop_w=%d, crop_h=%d\\n", src_x_f, src_y_f, src_x, src_y, start_x, start_y, crop_w, crop_h);
        }"""
content = content.replace('unsigned char p = 0;', debug_code + '\n        unsigned char p = 0;')

with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'w') as f:
    f.write(content)
