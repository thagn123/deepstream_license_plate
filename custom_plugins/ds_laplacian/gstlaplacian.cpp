#include "gstlaplacian.h"
#include <string.h>
#include <sys/time.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#include "laplacian_lib.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

#define PACKAGE "dslaplacian"
#define VERSION "1.0"

GST_DEBUG_CATEGORY_STATIC (gst_laplacian_debug);
#define GST_CAT_DEFAULT gst_laplacian_debug

#define GST_TYPE_LAPLACIAN (gst_laplacian_get_type())

enum
{
  PROP_0,
  PROP_CLASS_ID
};

static void gst_laplacian_set_property (GObject * object, guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_laplacian_get_property (GObject * object, guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_laplacian_finalize (GObject * object);
static GstFlowReturn gst_laplacian_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf);

G_DEFINE_TYPE (GstLaplacian, gst_laplacian, GST_TYPE_BASE_TRANSFORM);

static void
gst_laplacian_class_init (GstLaplacianClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;


  gobject_class->set_property = gst_laplacian_set_property;
  gobject_class->get_property = gst_laplacian_get_property;
  gobject_class->finalize = gst_laplacian_finalize;


  g_object_class_install_property (gobject_class, PROP_CLASS_ID,
      g_param_spec_int ("class-id", "Class ID",
          "Object class ID to filter for Laplacian variance computation (-1 for all)",
          -1, 255, -1,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_details_simple (gstelement_class,
      "Laplacian Variance",
      "Filter/Video",
      "Computes laplacian variance of specific object bounding boxes",
      "Custom Plugin");

  gst_element_class_add_pad_template (gstelement_class,
      gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
          gst_caps_from_string ("video/x-raw(memory:NVMM), format=(string)NV12")));
  gst_element_class_add_pad_template (gstelement_class,
      gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
          gst_caps_from_string ("video/x-raw(memory:NVMM), format=(string)NV12")));

  gstbasetransform_class->transform_ip = GST_DEBUG_FUNCPTR (gst_laplacian_transform_ip);

  GST_DEBUG_CATEGORY_INIT (gst_laplacian_debug, "dslaplacian", 0, "Laplacian plugin");
}

static void
gst_laplacian_init (GstLaplacian * laplacian)
{
  laplacian->class_id = -1; // Mặc định xử lý tất cả các class (không lọc)
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


static void
gst_laplacian_set_property (GObject * object, guint prop_id, const GValue * value, GParamSpec * pspec)
{
  GstLaplacian *laplacian = GST_LAPLACIAN (object);

  switch (prop_id) {
    case PROP_CLASS_ID:
      laplacian->class_id = g_value_get_int (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_laplacian_get_property (GObject * object, guint prop_id, GValue * value, GParamSpec * pspec)
{
  GstLaplacian *laplacian = GST_LAPLACIAN (object);

  switch (prop_id) {
    case PROP_CLASS_ID:
      g_value_set_int (value, laplacian->class_id);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static GstFlowReturn
gst_laplacian_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstLaplacian *laplacian = GST_LAPLACIAN (btrans);
  GstMapInfo in_map_info;
  NvBufSurface *surface = NULL;

  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    GST_ERROR_OBJECT (laplacian, "Failed to map gst buffer");
    return GST_FLOW_ERROR;
  }

  surface = (NvBufSurface *) in_map_info.data;
  
  // Yêu cầu bộ nhớ của surface phải nằm trên GPU (NVMM)
  if (surface->memType != NVBUF_MEM_CUDA_DEVICE && 
      surface->memType != NVBUF_MEM_CUDA_UNIFIED) {
        GST_WARNING_OBJECT(laplacian, "Surface is not in GPU memory.");
        gst_buffer_unmap(inbuf, &in_map_info);
        return GST_FLOW_OK;
  }

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    gst_buffer_unmap(inbuf, &in_map_info);
    return GST_FLOW_OK;
  }

  // Duyệt qua từng khung hình (frame) trong batch
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    guint batch_id = frame_meta->batch_id;
    
    // Thuộc tính của kênh Y (Giả định định dạng ảnh là NV12)
    void* d_y_plane = surface->surfaceList[batch_id].dataPtr;
    int pitch = surface->surfaceList[batch_id].pitch;
    int img_width = surface->surfaceList[batch_id].width;
    int img_height = surface->surfaceList[batch_id].height;

    // Duyệt qua từng đối tượng (object) được phát hiện trong khung hình
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) (l_obj->data);
      
      // Lọc theo class_id nếu được thiết lập
      if (laplacian->class_id != -1 && obj_meta->class_id != laplacian->class_id) {
          continue;
      }

      int crop_x = (int)obj_meta->rect_params.left;
      int crop_y = (int)obj_meta->rect_params.top;
      int crop_w = (int)obj_meta->rect_params.width;
      int crop_h = (int)obj_meta->rect_params.height;
      
      // Kiểm tra để đảm bảo tọa độ không bị tràn viền ảnh
      if (crop_x < 0) crop_x = 0;
      if (crop_y < 0) crop_y = 0;
      if (crop_x + crop_w > img_width) crop_w = img_width - crop_x;
      if (crop_y + crop_h > img_height) crop_h = img_height - crop_y;

      // Đảm bảo kích thước cắt (crop) hợp lệ
      if (crop_w > 10 && crop_h > 10) {
          int out_w = 150;
          int out_h = 50;

          // 1. Cắt ảnh và thay đổi kích thước, đưa vào bộ nhớ CPU Managed Memory
          gpu_crop_and_resize_to_cpu(d_y_plane, pitch, crop_x, crop_y, crop_w, crop_h, laplacian->cpu_crop_buf, out_w, out_h, 0);
          cudaStreamSynchronize(0);

          // 2. Dùng OpenCV tìm đường viền (Contour)
          cv::Mat cpu_img(out_h, out_w, CV_8UC1, laplacian->cpu_crop_buf);
          cv::Mat blur, edges;
          cv::GaussianBlur(cpu_img, blur, cv::Size(5, 5), 0);
          cv::Canny(blur, edges, 50, 150);
          
          std::vector<std::vector<cv::Point>> contours;
          cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
          static int debug_cnt = 0;
          if (debug_cnt++ < 5) {
              cv::imwrite("outputs/laplacian_samples/debug_crop_" + std::to_string(debug_cnt) + ".jpg", cpu_img);
          }
          
          // Sắp xếp các đường viền theo diện tích giảm dần
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
              // Sắp xếp 4 điểm tọa độ
              std::vector<cv::Point2f> src_pts(4);
              for(int i=0; i<4; i++) src_pts[i] = plate_contour[i];
              
              // Xác định góc trên-trái, trên-phải, dưới-phải, dưới-trái
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
              
              // Ma trận M chiếu ordered_pts -> dst_pts (từ ảnh crop 150x50 sang ảnh phẳng 150x50).
              // Chú ý: ordered_pts hiện đang là tọa độ TRONG ẢNH ĐÃ RESIZE 150x50!
              // Nhưng ta cần ma trận M chiếu từ ảnh gốc (crop_w x crop_h) sang (out_w x out_h)!
              // Để sửa lỗi này, ta cần phóng to tọa độ ordered_pts trả về kích thước gốc ban đầu.
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
              // Nếu không tìm thấy, dùng ma trận đơn vị làm phương án dự phòng
              Minv_ptr[0] = crop_w / (float)out_w; Minv_ptr[1] = 0; Minv_ptr[2] = 0;
              Minv_ptr[3] = 0; Minv_ptr[4] = crop_h / (float)out_h; Minv_ptr[5] = 0;
              Minv_ptr[6] = 0; Minv_ptr[7] = 0; Minv_ptr[8] = 1.0f;
          }

          // 3. Xử lý các phép toán nặng trên GPU
          double variance = gpu_warp_equalize_blur_laplacian(d_y_plane, pitch, crop_x, crop_y, crop_w, crop_h, Minv_ptr, out_w, out_h, 0);
          static int p_cnt = 0;
          if(p_cnt++ < 10) printf("C++ Variance: %f\n", variance);
          
obj_meta->misc_obj_info[0] = (int)variance;
      }
    }
  }

  gst_buffer_unmap (inbuf, &in_map_info);
  return GST_FLOW_OK;
}

static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, "dslaplacian", GST_RANK_NONE, GST_TYPE_LAPLACIAN);
}

// Đăng ký Plugin vào hệ thống GStreamer
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_laplacian,
    "DeepStream Laplacian Variance Plugin",
    plugin_init, "1.0", "Proprietary", "DeepStream", "http://nvidia.com")
