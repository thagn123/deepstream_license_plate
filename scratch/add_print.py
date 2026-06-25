import re
with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'r') as f:
    content = f.read()

content = content.replace('warp_and_minmax_kernel<<<gridSize, blockSize, 0, stream>>>', 'printf("Minv: %f %f %f | %f %f %f | %f %f %f\\n", Minv[0], Minv[1], Minv[2], Minv[3], Minv[4], Minv[5], Minv[6], Minv[7], Minv[8]);\n    warp_and_minmax_kernel<<<gridSize, blockSize, 0, stream>>>')

with open('custom_plugins/ds_laplacian/laplacian_lib.cu', 'w') as f:
    f.write(content)
