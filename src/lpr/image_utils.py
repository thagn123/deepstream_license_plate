import math
import cv2


def _crop_plate_from_frame(frame_image, bbox: tuple, pad_ratio: float = 0.15, min_width: int = 100):
    if frame_image is None:
        return None, bbox, 0.0, False
    fh, fw = frame_image.shape[:2]
    x, y, w, h = bbox
    pad_x = max(2, int(w * pad_ratio))
    pad_y = max(2, int(h * pad_ratio))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, 0, 0), 0.0, True

    raw = frame_image[y1:y2, x1:x2]
    crop = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR) if raw.shape[2] == 4 else raw.copy()
    ch, cw = crop.shape[:2]
    if cw < min_width and cw > 0:
        scale = max(2, int(math.ceil(min_width / float(cw))))
        crop = cv2.resize(crop, (cw * scale, ch * scale), interpolation=cv2.INTER_CUBIC)

    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        sharp = 0.0
    size_score = min(1.0, crop.shape[1] / 160.0) * 0.45 + min(1.0, crop.shape[0] / 48.0) * 0.20
    sharp_score = min(1.0, sharp / 120.0) * 0.25
    clipped = x1 == 0 or y1 == 0 or x2 == fw or y2 == fh
    clip_score = 0.10 if not clipped else 0.0
    quality = max(0.0, min(1.0, size_score + sharp_score + clip_score))
    return crop, (x1, y1, x2 - x1, y2 - y1), quality, clipped
