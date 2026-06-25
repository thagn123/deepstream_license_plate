import re
with open('custom_plugins/ds_laplacian/gstlaplacian.cpp', 'r') as f:
    content = f.read()

content = content.replace('double variance = gpu_warp_equalize_blur_laplacian(d_y_plane, pitch, crop_x, crop_y, crop_w, crop_h, Minv_ptr, out_w, out_h, 0);', 'double variance = gpu_warp_equalize_blur_laplacian(d_y_plane, pitch, crop_x, crop_y, crop_w, crop_h, Minv_ptr, out_w, out_h, 0);\n          static int p_cnt = 0;\n          if(p_cnt++ < 10) printf("C++ Variance: %f\\n", variance);')

with open('custom_plugins/ds_laplacian/gstlaplacian.cpp', 'w') as f:
    f.write(content)
