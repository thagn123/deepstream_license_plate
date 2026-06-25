import re

with open('custom_plugins/ds_laplacian/gstlaplacian.cpp', 'r') as f:
    content = f.read()

# Add OpenCV include
if '#include <opencv2/opencv.hpp>' not in content:
    content = content.replace('#include "laplacian_lib.h"', '#include "laplacian_lib.h"\n#include <opencv2/opencv.hpp>\n#include <vector>\n#include <algorithm>')

# Add init logic
init_logic = """static void
gst_laplacian_init (GstLaplacian * laplacian)
{
  laplacian->class_id = -1; // Process all by default
  cudaMallocManaged(&laplacian->cpu_crop_buf, 150 * 50);
}

static void
gst_laplacian_finalize (GObject * object)
{
  GstLaplacian *laplacian = GST_LAPLACIAN (object);
  if (laplacian->cpu_crop_buf) {
      cudaFree(laplacian->cpu_crop_buf);
      laplacian->cpu_crop_buf = nullptr;
  }
  G_OBJECT_CLASS (gst_laplacian_parent_class)->finalize (object);
}
"""
# Replace init
content = re.sub(r'static void\ngst_laplacian_init[^{]+{[^}]+}', init_logic, content)

# Add finalize to class_init
class_init_add = """
  gobject_class->set_property = gst_laplacian_set_property;
  gobject_class->get_property = gst_laplacian_get_property;
  gobject_class->finalize = gst_laplacian_finalize;
"""
content = content.replace('  gobject_class->set_property = gst_laplacian_set_property;\n  gobject_class->get_property = gst_laplacian_get_property;', class_init_add)

# Replace the inner transform logic
transform_logic = """      // Ensure valid crop size
      if (crop_w > 10 && crop_h > 10) {
          int out_w = 150;
          int out_h = 50;

          // 1. Crop and resize to CPU Managed Memory
          gpu_crop_and_resize_to_cpu(d_y_plane, pitch, crop_x, crop_y, crop_w, crop_h, laplacian->cpu_crop_buf, out_w, out_h, 0);
          cudaStreamSynchronize(0);

          // 2. OpenCV Contour Finding
          cv::Mat cpu_img(out_h, out_w, CV_8UC1, laplacian->cpu_crop_buf);
          cv::Mat blur, edges;
          cv::GaussianBlur(cpu_img, blur, cv::Size(5, 5), 0);
          cv::Canny(blur, edges, 50, 150);
          
          std::vector<std::vector<cv::Point>> contours;
          cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
          
          // Sort contours by area
          std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
              return cv::contourArea(a) > cv::contourArea(b);
          });
          
          std::vector<cv::Point> plate_contour;
          double min_area = 0.4 * (out_w * out_h);
          
          for (size_t i = 0; i < std::min((size_t)5, contours.size()); i++) {
              double peri = cv::arcLength(contours[i], true);
              std::vector<cv::Point> approx;
              cv::approxPolyDP(contours[i], approx, 0.05 * peri, true);
              
              if (approx.size() == 4 && cv::contourArea(approx) > min_area) {
                  plate_contour = approx;
                  break;
              }
          }
          
          float Minv_ptr[9];
          
          if (!plate_contour.empty()) {
              // Sort 4 points
              std::vector<cv::Point2f> src_pts(4);
              for(int i=0; i<4; i++) src_pts[i] = plate_contour[i];
              
              // Find top-left, top-right, bottom-right, bottom-left
              std::sort(src_pts.begin(), src_pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
              cv::Point2f tl = src_pts[0].y < src_pts[1].y ? src_pts[0] : src_pts[1];
              cv::Point2f bl = src_pts[0].y > src_pts[1].y ? src_pts[0] : src_pts[1];
              cv::Point2f tr = src_pts[2].y < src_pts[3].y ? src_pts[2] : src_pts[3];
              cv::Point2f br = src_pts[2].y > src_pts[3].y ? src_pts[2] : src_pts[3];
              
              std::vector<cv::Point2f> ordered_pts = {tl, tr, br, bl};
              std::vector<cv::Point2f> dst_pts = {
                  cv::Point2f(0, 0),
                  cv::Point2f(out_w - 1, 0),
                  cv::Point2f(out_w - 1, out_h - 1),
                  cv::Point2f(0, out_h - 1)
              };
              
              // M maps ordered_pts -> dst_pts. (from 150x50 crop to 150x50 warped)
              // Wait, the ordered_pts are coordinates IN THE 150x50 RESIZED IMAGE!
              // But we need M to map from original crop_w x crop_h to out_w x out_h!
              // To fix this, we scale ordered_pts back to the original crop size!
              for(int i=0; i<4; i++) {
                  ordered_pts[i].x = ordered_pts[i].x * crop_w / (float)out_w;
                  ordered_pts[i].y = ordered_pts[i].y * crop_h / (float)out_h;
              }
              
              cv::Mat M = cv::getPerspectiveTransform(ordered_pts, dst_pts);
              cv::Mat Minv = M.inv();
              
              for(int i=0; i<3; i++) {
                  for(int j=0; j<3; j++) {
                      Minv_ptr[i*3 + j] = Minv.at<double>(i, j);
                  }
              }
          } else {
              // Fallback identity mapping
              Minv_ptr[0] = crop_w / (float)out_w; Minv_ptr[1] = 0; Minv_ptr[2] = 0;
              Minv_ptr[3] = 0; Minv_ptr[4] = crop_h / (float)out_h; Minv_ptr[5] = 0;
              Minv_ptr[6] = 0; Minv_ptr[7] = 0; Minv_ptr[8] = 1.0f;
          }

          // 3. GPU heavy processing
          double variance = gpu_warp_equalize_blur_laplacian(d_y_plane, pitch, crop_x, crop_y, crop_w, crop_h, Minv_ptr, out_w, out_h, 0);
          
          // Attach result
          NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
          double *meta_data = (double *)g_malloc0(sizeof(double));
          *meta_data = variance;
          
          user_meta->user_meta_data = (void *)meta_data;
          user_meta->base_meta.meta_type = nvds_get_user_meta_type((gchar*)"Laplacian.Score");
          user_meta->base_meta.release_func = [](gpointer data, gpointer user_data) {
              NvDsUserMeta *meta = (NvDsUserMeta *)data;
              if (meta->user_meta_data) {
                  g_free(meta->user_meta_data);
                  meta->user_meta_data = nullptr;
              }
          };
          
          nvds_add_user_meta_to_obj(obj_meta, user_meta);
      }"""

content = re.sub(r'      // Ensure valid crop size\n      if \(crop_w > 2 && crop_h > 2\) \{.*?\n      \}', transform_logic, content, flags=re.DOTALL)

with open('custom_plugins/ds_laplacian/gstlaplacian.cpp', 'w') as f:
    f.write(content)
