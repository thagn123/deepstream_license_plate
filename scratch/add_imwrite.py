import re

with open('custom_plugins/ds_laplacian/gstlaplacian.cpp', 'r') as f:
    content = f.read()

# Add imwrite
content = content.replace('cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);', 'cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);\n          static int debug_cnt = 0;\n          if (debug_cnt++ < 5) {\n              cv::imwrite("outputs/laplacian_samples/debug_crop_" + std::to_string(debug_cnt) + ".jpg", cpu_img);\n          }')

with open('custom_plugins/ds_laplacian/gstlaplacian.cpp', 'w') as f:
    f.write(content)
