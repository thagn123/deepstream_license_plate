#include "gstlaplacian.h"
#include <string.h>
#include <sys/time.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#include "laplacian_lib.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

#define PACKAGE "dslaplacian"
#define VERSION "1.0"

GST_DEBUG_CATEGORY_STATIC (gst_laplacian_debug);
#define GST_CAT_DEFAULT gst_laplacian_debug

#define GST_TYPE_LAPLACIAN (gst_laplacian_get_type())

enum { PROP_0, PROP_CLASS_ID };

/* ══════════════════════════════════════════════════════════════════════
 * OpenCV CPU context — khởi tạo lazy tại frame đầu tiên
 * ══════════════════════════════════════════════════════════════════════ */
// No CPU context required anymore since we don't use CLAHE.
struct LaplCpuCtx {
};

/* ══════════════════════════════════════════════════════════════════════
 * Forward declarations
 * ══════════════════════════════════════════════════════════════════════ */
static void         gst_laplacian_set_property (GObject*, guint, const GValue*, GParamSpec*);
static void         gst_laplacian_get_property (GObject*, guint, GValue*, GParamSpec*);
static void         gst_laplacian_finalize     (GObject*);
static GstFlowReturn gst_laplacian_transform_ip(GstBaseTransform*, GstBuffer*);

G_DEFINE_TYPE (GstLaplacian, gst_laplacian, GST_TYPE_BASE_TRANSFORM);

/* ══════════════════════════════════════════════════════════════════════
 * GLib/GStreamer boilerplate
 * ══════════════════════════════════════════════════════════════════════ */
static void
gst_laplacian_class_init (GstLaplacianClass* klass)
{
    GObjectClass*          gobject_class        = (GObjectClass*)klass;
    GstElementClass*       gstelement_class     = (GstElementClass*)klass;
    GstBaseTransformClass* gstbasetransform_class = (GstBaseTransformClass*)klass;

    gobject_class->set_property = gst_laplacian_set_property;
    gobject_class->get_property = gst_laplacian_get_property;
    gobject_class->finalize     = gst_laplacian_finalize;

    g_object_class_install_property (gobject_class, PROP_CLASS_ID,
        g_param_spec_int ("class-id", "Class ID",
            "Object class ID to filter (-1 for all)",
            -1, 255, -1,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_set_details_simple (gstelement_class,
        "Laplacian Variance", "Filter/Video",
        "Perspective-align + Laplacian variance for license plates",
        "Custom Plugin");

    gst_element_class_add_pad_template (gstelement_class,
        gst_pad_template_new ("src",  GST_PAD_SRC,  GST_PAD_ALWAYS,
            gst_caps_from_string ("video/x-raw(memory:NVMM), format=(string)NV12")));
    gst_element_class_add_pad_template (gstelement_class,
        gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
            gst_caps_from_string ("video/x-raw(memory:NVMM), format=(string)NV12")));

    gstbasetransform_class->transform_ip = GST_DEBUG_FUNCPTR (gst_laplacian_transform_ip);

    GST_DEBUG_CATEGORY_INIT (gst_laplacian_debug, "dslaplacian", 0, "Laplacian plugin");
}

static void
gst_laplacian_init (GstLaplacian* laplacian)
{
    laplacian->class_id          = -1;
    laplacian->cpu_initialized   = FALSE;
    laplacian->cpu_ctx           = nullptr;

    // CPU-accessible managed memory buffer 1000×500 (max crop resolution)
    cudaMallocManaged(&laplacian->cpu_detect_buf, 1000 * 500);
}

static void
gst_laplacian_finalize (GObject* object)
{
    GstLaplacian* laplacian = GST_LAPLACIAN (object);

    if (laplacian->cpu_detect_buf) {
        cudaFree(laplacian->cpu_detect_buf);
        laplacian->cpu_detect_buf = nullptr;
    }
    if (laplacian->cpu_ctx) {
        delete static_cast<LaplCpuCtx*>(laplacian->cpu_ctx);
        laplacian->cpu_ctx = nullptr;
    }
    G_OBJECT_CLASS (gst_laplacian_parent_class)->finalize (object);
}

static void
gst_laplacian_set_property (GObject* object, guint prop_id,
                             const GValue* value, GParamSpec* pspec)
{
    GstLaplacian* laplacian = GST_LAPLACIAN (object);
    switch (prop_id) {
        case PROP_CLASS_ID: laplacian->class_id = g_value_get_int(value); break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec); break;
    }
}

static void
gst_laplacian_get_property (GObject* object, guint prop_id,
                             GValue* value, GParamSpec* pspec)
{
    GstLaplacian* laplacian = GST_LAPLACIAN (object);
    switch (prop_id) {
        case PROP_CLASS_ID: g_value_set_int(value, laplacian->class_id); break;
        default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec); break;
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Helper: sắp xếp 4 điểm → TL, TR, BR, BL
 * ══════════════════════════════════════════════════════════════════════ */
static void sort_corners(std::vector<cv::Point2f>& pts)
{
    std::sort(pts.begin(), pts.end(),
              [](const cv::Point2f& a, const cv::Point2f& b){ return a.x < b.x; });
    cv::Point2f tl = pts[0].y < pts[1].y ? pts[0] : pts[1];
    cv::Point2f bl = pts[0].y < pts[1].y ? pts[1] : pts[0];
    cv::Point2f tr = pts[2].y < pts[3].y ? pts[2] : pts[3];
    cv::Point2f br = pts[2].y < pts[3].y ? pts[3] : pts[2];
    pts = {tl, tr, br, bl};
}

/* ══════════════════════════════════════════════════════════════════════
 * Helper: giao điểm 2 đoạn thẳng [x1,y1,x2,y2]
 * ══════════════════════════════════════════════════════════════════════ */
static bool line_intersect(const cv::Vec4i& la, const cv::Vec4i& lb, cv::Point2f& pt)
{
    float a1 = la[3]-la[1], b1 = la[0]-la[2], c1 = a1*la[0] + b1*la[1];
    float a2 = lb[3]-lb[1], b2 = lb[0]-lb[2], c2 = a2*lb[0] + b2*lb[1];
    float det = a1*b2 - a2*b1;
    if (std::abs(det) < 1e-6f) return false;
    pt.x = (c1*b2 - c2*b1) / det;
    pt.y = (a1*c2 - a2*c1) / det;
    return true;
}

/* ══════════════════════════════════════════════════════════════════════
 * Tìm 4 góc biển số trong ảnh edge (DET_W × DET_H).
 * Trả về true nếu tìm thấy và đã ghi Minv_ptr.
 *
 * Chiến lược (theo thứ tự ưu tiên):
 *   1. findContours → convexHull → approxPolyDP (epsilon 0.02..0.10)
 *   2. HoughLinesP → top/bot/left/right → 4 giao điểm
 *   3. minAreaRect trên toàn bộ điểm Canny
 *   Fallback: scale thuần (identity warp)
 * ══════════════════════════════════════════════════════════════════════ */
static bool find_plate_corners(const cv::Mat& gray_padded,
                                float Minv_ptr[9])
{
    int pw = gray_padded.cols;
    int ph = gray_padded.rows;

    if (pw < 10 || ph < 4) {
        return false;
    }

    // Border check helper
    auto is_border_quad = [&](const std::vector<cv::Point2f>& pts) {
        int on_border = 0;
        for (const auto& p : pts) {
            if (p.x <= 1 || p.x >= pw - 2 || p.y <= 1 || p.y >= ph - 2)
                on_border++;
        }
        return on_border >= 3;
    };
    auto is_border_quad_array = [&](const cv::Point2f pts[4]) {
        int on_border = 0;
        for (int i = 0; i < 4; i++) {
            if (pts[i].x <= 1 || pts[i].x >= pw - 2 || pts[i].y <= 1 || pts[i].y >= ph - 2)
                on_border++;
        }
        return on_border >= 3;
    };

    auto _valid_plate_quad_local = [&](const std::vector<cv::Point2f>& pts) {
        if (pts.size() != 4) return false;
        std::vector<cv::Point2f> spts = pts;
        sort_corners(spts);
        float top_w  = cv::norm(spts[1] - spts[0]);
        float bottom_w  = cv::norm(spts[2] - spts[3]);
        float left_h = cv::norm(spts[3] - spts[0]);
        float right_h= cv::norm(spts[2] - spts[1]);
        float w_avg  = (top_w + bottom_w) / 2.0f;
        float h_avg  = (left_h + right_h) / 2.0f;
        if (w_avg < 5.0f || h_avg < 3.0f)
            return false;
        float ratio = w_avg / h_avg;
        return 1.0f <= ratio && ratio <= 7.5f;
    };

    auto _valid_plate_quad_array = [&](const cv::Point2f pts[4]) {
        std::vector<cv::Point2f> spts(4);
        for (int i = 0; i < 4; i++) spts[i] = pts[i];
        return _valid_plate_quad_local(spts);
    };

    static const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    double min_area = 0.10 * pw * ph;

    std::vector<cv::Point2f> plate_pts;
    bool found = false;

    // ── Strategy 1: Character-based Estimation (Adaptive Threshold + RETR_CCOMP) ──
    {
        cv::Mat thresh_adapt;
        cv::adaptiveThreshold(gray_padded, thresh_adapt, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

        std::vector<std::vector<cv::Point>> cnts_cc;
        std::vector<cv::Vec4i> hierarchy_cc;
        cv::findContours(thresh_adapt, cnts_cc, hierarchy_cc, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

        if (!cnts_cc.empty() && !hierarchy_cc.empty()) {
            std::vector<cv::Point> char_points;
            std::vector<cv::Rect> char_boxes;

            for (size_t i = 0; i < cnts_cc.size(); ++i) {
                cv::Rect rect = cv::boundingRect(cnts_cc[i]);
                double area = cv::contourArea(cnts_cc[i]);
                
                float h_ratio = rect.height / (float)ph;
                float w_ratio = rect.width / (float)pw;
                float aspect = rect.width > 0 ? rect.height / (float)rect.width : 0.0f;

                if (h_ratio > 0.15f && h_ratio < 0.85f &&
                    w_ratio > 0.02f && w_ratio < 0.35f &&
                    aspect > 0.5f && aspect < 6.0f) {
                    
                    if (rect.x > 2 && rect.y > 2 &&
                        (rect.x + rect.width) < pw - 3 &&
                        (rect.y + rect.height) < ph - 3) {
                        
                        float solidity = rect.width * rect.height > 0 ? area / (float)(rect.width * rect.height) : 0.0f;
                        if (solidity > 0.15f) {
                            char_boxes.push_back(rect);
                            for (const auto& pt : cnts_cc[i]) {
                                char_points.push_back(pt);
                            }
                        }
                    }
                }
            }

            if (char_boxes.size() >= 2) {
                cv::RotatedRect rr = cv::minAreaRect(char_points);
                
                float char_w = rr.size.width;
                float char_h = rr.size.height;
                float text_aspect = std::max(char_w, char_h) / std::max(std::min(char_w, char_h), 1.0f);
                if (char_w < char_h) {
                    text_aspect = char_h / std::max(char_w, 1.0f);
                }

                float scale_w = 1.25f;
                float scale_h = 1.65f;
                if (text_aspect <= 2.0f) {
                    scale_w = 1.35f;
                    scale_h = 1.25f;
                }

                cv::Point2f center = rr.center;
                cv::Size2f expanded_size(rr.size.width * scale_w, rr.size.height * scale_h);
                
                expanded_size.width = std::min(expanded_size.width, (float)pw * 0.98f);
                expanded_size.height = std::min(expanded_size.height, (float)ph * 0.98f);

                cv::RotatedRect expanded_rr(center, expanded_size, rr.angle);
                cv::Point2f corners[4];
                expanded_rr.points(corners);

                plate_pts.resize(4);
                for (int i = 0; i < 4; ++i) {
                    plate_pts[i].x = std::max(0.0f, std::min(corners[i].x, (float)pw - 1.0f));
                    plate_pts[i].y = std::max(0.0f, std::min(corners[i].y, (float)ph - 1.0f));
                }
                found = true;
            }
        }
    }

    // ── Strategy 2: Otsu ──
    cv::Mat thresh;
    if (!found) {
        cv::Mat blur_otsu;
        cv::GaussianBlur(gray_padded, blur_otsu, cv::Size(3, 3), 0);
        cv::threshold(blur_otsu, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // Check border white count to invert if needed
        double border_sum = 0;
        for (int x = 0; x < pw; x++) {
            border_sum += thresh.at<uchar>(0, x) + thresh.at<uchar>(1, x);
            border_sum += thresh.at<uchar>(ph - 1, x) + thresh.at<uchar>(ph - 2, x);
        }
        for (int y = 0; y < ph; y++) {
            border_sum += thresh.at<uchar>(y, 0) + thresh.at<uchar>(y, 1);
            border_sum += thresh.at<uchar>(y, pw - 1) + thresh.at<uchar>(y, pw - 2);
        }
        double total_border = (2 * pw * 2) + (2 * ph * 2);
        if (border_sum / 255.0 > total_border * 0.5) {
            cv::bitwise_not(thresh, thresh);
        }

        cv::Mat processed;
        cv::morphologyEx(thresh, processed, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(processed, processed, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(processed, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::sort(cnts.begin(), cnts.end(), [](const auto& a, const auto& b) {
            return cv::contourArea(a) > cv::contourArea(b);
        });

        for (const auto& cnt : cnts) {
            double area = cv::contourArea(cnt);
            if (area < min_area) break;
            std::vector<cv::Point> hull;
            cv::convexHull(cnt, hull);
            double peri = cv::arcLength(hull, true);
            const double eps_list[] = {0.02, 0.04, 0.06, 0.08, 0.10};
            for (double eps : eps_list) {
                std::vector<cv::Point> approx;
                cv::approxPolyDP(hull, approx, eps * peri, true);
                if (approx.size() == 4) {
                    std::vector<cv::Point2f> pts(4);
                    for (int i = 0; i < 4; i++) pts[i] = cv::Point2f(approx[i]);
                    if (!is_border_quad(pts) && _valid_plate_quad_local(pts)) {
                        plate_pts = pts;
                        found = true;
                        break;
                    }
                }
            }
            if (found) break;

            // Fallback to minAreaRect for this Otsu contour
            cv::RotatedRect rr = cv::minAreaRect(cnt);
            cv::Point2f corners[4];
            rr.points(corners);
            if (!is_border_quad_array(corners) && _valid_plate_quad_array(corners)) {
                plate_pts.resize(4);
                for (int i = 0; i < 4; i++) plate_pts[i] = corners[i];
                found = true;
                break;
            }
        }
    }

    // ── Strategy 2: Canny Contours ──
    if (!found) {
        cv::Mat blur_canny, edges, closed;
        cv::GaussianBlur(gray_padded, blur_canny, cv::Size(3, 3), 0);
        cv::Canny(blur_canny, edges, 50, 150);
        cv::morphologyEx(edges, closed, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(closed, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::sort(cnts.begin(), cnts.end(), [](const auto& a, const auto& b) {
            return cv::contourArea(a) > cv::contourArea(b);
        });

        for (const auto& cnt : cnts) {
            double area = cv::contourArea(cnt);
            if (area < min_area) break;
            std::vector<cv::Point> hull;
            cv::convexHull(cnt, hull);
            double peri = cv::arcLength(hull, true);
            const double eps_list[] = {0.02, 0.04, 0.06, 0.08, 0.10};
            for (double eps : eps_list) {
                std::vector<cv::Point> approx;
                cv::approxPolyDP(hull, approx, eps * peri, true);
                if (approx.size() == 4) {
                    std::vector<cv::Point2f> pts(4);
                    for (int i = 0; i < 4; i++) pts[i] = cv::Point2f(approx[i]);
                    if (!is_border_quad(pts) && _valid_plate_quad_local(pts)) {
                        plate_pts = pts;
                        found = true;
                        break;
                    }
                }
            }
            if (found) break;

            // Fallback to minAreaRect for this Canny contour
            cv::RotatedRect rr = cv::minAreaRect(cnt);
            cv::Point2f corners[4];
            rr.points(corners);
            if (!is_border_quad_array(corners) && _valid_plate_quad_array(corners)) {
                plate_pts.resize(4);
                for (int i = 0; i < 4; i++) plate_pts[i] = corners[i];
                found = true;
                break;
            }
        }

        // ── Strategy 3: minAreaRect on all Canny edges ──
        if (!found) {
            std::vector<cv::Point> edge_pts;
            cv::findNonZero(edges, edge_pts);
            if (edge_pts.size() >= 4) {
                cv::RotatedRect rr = cv::minAreaRect(edge_pts);
                cv::Point2f corners[4];
                rr.points(corners);
                if (!is_border_quad_array(corners) && _valid_plate_quad_array(corners)) {
                    plate_pts.resize(4);
                    for (int i = 0; i < 4; i++) plate_pts[i] = corners[i];
                    found = true;
                }
            }
        }
    }

    // ── Tính M → Minv ─────────────────────────────────────────────────
    if (found && !plate_pts.empty()) {
        sort_corners(plate_pts);

        std::vector<cv::Point2f> dst_pts = {
            {0.f,           0.f          },
            {(float)(OUT_W-1), 0.f       },
            {(float)(OUT_W-1), (float)(OUT_H-1)},
            {0.f,           (float)(OUT_H-1)}
        };

        cv::Mat M     = cv::getPerspectiveTransform(plate_pts, dst_pts);
        cv::Mat Minv  = M.inv();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Minv_ptr[i*3+j] = (float)Minv.at<double>(i,j);
        return true;
    }

    // ── Fallback: scale thuần ─────────────────────────────────────────
    Minv_ptr[0] = pw / (float)OUT_W; Minv_ptr[1] = 0; Minv_ptr[2] = 0;
    Minv_ptr[3] = 0; Minv_ptr[4] = ph / (float)OUT_H; Minv_ptr[5] = 0;
    Minv_ptr[6] = 0; Minv_ptr[7] = 0;                 Minv_ptr[8] = 1.f;
    return false;
}

/* ══════════════════════════════════════════════════════════════════════
 * transform_ip — được gọi mỗi frame
 * ══════════════════════════════════════════════════════════════════════ */
static GstFlowReturn
gst_laplacian_transform_ip (GstBaseTransform* btrans, GstBuffer* inbuf)
{
    GstLaplacian* laplacian = GST_LAPLACIAN (btrans);
    GstMapInfo    in_map_info;
    NvBufSurface* surface = nullptr;

    if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
        GST_ERROR_OBJECT (laplacian, "Failed to map gst buffer");
        return GST_FLOW_ERROR;
    }
    surface = (NvBufSurface*)in_map_info.data;

    if (surface->memType != NVBUF_MEM_CUDA_DEVICE &&
        surface->memType != NVBUF_MEM_CUDA_UNIFIED) {
        GST_WARNING_OBJECT (laplacian, "Surface not in GPU memory.");
        gst_buffer_unmap (inbuf, &in_map_info);
        return GST_FLOW_OK;
    }

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
    if (!batch_meta) {
        gst_buffer_unmap (inbuf, &in_map_info);
        return GST_FLOW_OK;
    }

    // ── Khởi tạo OpenCV CPU objects lần đầu tiên ─────────────────────
    if (!laplacian->cpu_initialized) {
        laplacian->cpu_initialized = TRUE;
    }

    // ── Duyệt frame ───────────────────────────────────────────────────
    for (NvDsMetaList* l_frame = batch_meta->frame_meta_list;
         l_frame != nullptr; l_frame = l_frame->next)
    {
        NvDsFrameMeta* frame_meta = (NvDsFrameMeta*)l_frame->data;
        guint  batch_id  = frame_meta->batch_id;

        void* d_y_plane  = surface->surfaceList[batch_id].dataPtr;
        int   pitch      = surface->surfaceList[batch_id].pitch;
        int   img_width  = surface->surfaceList[batch_id].width;
        int   img_height = surface->surfaceList[batch_id].height;

        // ── Duyệt object ──────────────────────────────────────────────
        for (NvDsMetaList* l_obj = frame_meta->obj_meta_list;
             l_obj != nullptr; l_obj = l_obj->next)
        {
            NvDsObjectMeta* obj_meta = (NvDsObjectMeta*)l_obj->data;
            if (laplacian->class_id != -1 && obj_meta->class_id != laplacian->class_id)
                continue;

            int cx = (int)obj_meta->rect_params.left;
            int cy = (int)obj_meta->rect_params.top;
            int cw = (int)obj_meta->rect_params.width;
            int ch = (int)obj_meta->rect_params.height;

            if (cx < 0) cx = 0;
            if (cy < 0) cy = 0;
            if (cx + cw > img_width)  cw = img_width  - cx;
            if (cy + ch > img_height) ch = img_height - cy;
            if (cw <= 10 || ch <= 4) continue;

            // Mở rộng bbox 30% mỗi phía
            int pad_x = std::max(8,  (int)(cw * 0.30f));
            int pad_y = std::max(6,  (int)(ch * 0.30f));
            int px    = std::max(0,  cx - pad_x);
            int py    = std::max(0,  cy - pad_y);
            int pw    = std::min(img_width  - px, cw + 2*pad_x);
            int ph    = std::min(img_height - py, ch + 2*pad_y);

            // Giới hạn kích thước crop tối đa để tránh tràn bộ đệm
            int copy_w = std::min(pw, 1000);
            int copy_h = std::min(ph, 500);

            /* ── Bước 1: Crop padded region trên GPU -> cpu_detect_buf (Unified Memory) ở độ phân giải gốc ── */
            gpu_crop_and_resize_to_cpu(d_y_plane, pitch, px, py, copy_w, copy_h,
                                       laplacian->cpu_detect_buf, copy_w, copy_h, 0);
            cudaStreamSynchronize(0); // Chờ GPU hoàn thành ghi vào Unified Memory

            /* ── Bước 2: CPU wrapping + Corner Detection ────────────── */
            cv::Mat cpu_in(copy_h, copy_w, CV_8UC1, laplacian->cpu_detect_buf);

            /* ── Bước 3: CPU Otsu / Canny / minAreaRect → Minv ─ */
            float Minv_ptr[9];
            find_plate_corners(cpu_in, Minv_ptr);

            /* ── Bước 4: GPU warp + Laplacian variance ───────────────── */
            double variance = gpu_warp_equalize_blur_laplacian(
                d_y_plane, pitch, px, py, pw, ph,
                Minv_ptr, OUT_W, OUT_H, 0);

            obj_meta->misc_obj_info[0] = (int)variance;
        }
    }

    gst_buffer_unmap (inbuf, &in_map_info);
    return GST_FLOW_OK;
}

/* ══════════════════════════════════════════════════════════════════════
 * Plugin registration
 * ══════════════════════════════════════════════════════════════════════ */
static gboolean
plugin_init (GstPlugin* plugin)
{
    return gst_element_register (plugin, "dslaplacian", GST_RANK_NONE, GST_TYPE_LAPLACIAN);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR,
    nvdsgst_laplacian,
    "DeepStream Laplacian Variance Plugin",
    plugin_init, "1.0", "Proprietary", "DeepStream", "http://nvidia.com")
