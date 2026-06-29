#ifndef _GST_LAPLACIAN_H_
#define _GST_LAPLACIAN_H_

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

#define GST_TYPE_LAPLACIAN (gst_laplacian_get_type())
#define GST_LAPLACIAN(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_LAPLACIAN,GstLaplacian))
#define GST_LAPLACIAN_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_LAPLACIAN,GstLaplacianClass))

typedef struct _GstLaplacian GstLaplacian;
typedef struct _GstLaplacianClass GstLaplacianClass;

#define DET_W 300   // chiều rộng thumbnail dùng để detect corner
#define DET_H 100   // chiều cao thumbnail dùng để detect corner
#define OUT_W 150   // chiều rộng output warp biển số
#define OUT_H  50   // chiều cao output warp biển số

struct _GstLaplacian
{
  GstBaseTransform base_trans;
  gint  class_id;

  /* CPU-accessible managed buffer (DET_W × DET_H) — input cho pipeline OpenCV */
  unsigned char* cpu_detect_buf;

  /* Flag + context cho CPU */
  gboolean cpu_initialized;
  void*    cpu_ctx;   /* trỏ tới LaplCpuCtx* (C++ only, khai báo trong .cpp) */
};

struct _GstLaplacianClass
{
  GstBaseTransformClass parent_class;
};

GType gst_laplacian_get_type(void);

G_END_DECLS

#endif /* _GST_LAPLACIAN_H_ */
