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

struct _GstLaplacian
{
  GstBaseTransform base_trans;
  gint class_id; // Class ID to filter objects (e.g. 13 for license plate)
  unsigned char* cpu_crop_buf; // Managed memory buffer for CPU contour finding
};

struct _GstLaplacianClass
{
  GstBaseTransformClass parent_class;
};

GType gst_laplacian_get_type(void);

G_END_DECLS

#endif /* _GST_LAPLACIAN_H_ */
