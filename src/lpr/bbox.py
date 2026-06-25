import math
import lpr_config as config

def _bbox_tuple(obj_meta) -> tuple:
    r = obj_meta.rect_params
    return (int(r.left), int(r.top), int(r.width), int(r.height))
def _bbox_iou(a, b):
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0: return 0.0
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0
def _bbox_center_distance(a, b):
    cx1, cy1 = a[0] + a[2]/2.0, a[1] + a[3]/2.0
    cx2, cy2 = b[0] + b[2]/2.0, b[1] + b[3]/2.0
    return ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
def _smooth_bbox(prev, new, alpha):
    return (
        int(alpha * new[0] + (1 - alpha) * prev[0]),
        int(alpha * new[1] + (1 - alpha) * prev[1]),
        int(alpha * new[2] + (1 - alpha) * prev[2]),
        int(alpha * new[3] + (1 - alpha) * prev[3]),
    )
def _bbox_xyxy_from_tuple(bbox: tuple) -> tuple:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)
