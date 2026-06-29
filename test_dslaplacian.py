#!/usr/bin/env python3
"""
test_dslaplacian.py — Detect biển số + hiển thị ảnh BEFORE/AFTER ds_laplacian.

Cách dùng:
  python3 test_dslaplacian.py anh1.jpg anh2.jpg anh3.jpg
  python3 test_dslaplacian.py anh.jpg --conf 0.05
"""

import sys
import os
import argparse
import warnings
import numpy as np
import cv2

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH  = os.path.join(_SCRIPT_DIR, "models", "yolov11s_14_cls_20241224.onnx")
_LP_CLASS_ID = 13
_INPUT_SIZE  = 640
_LABELS = [
    "seater_12_16","bus","car","club_cart","human","moto","moto_rider",
    "shuttle_bus_5_7","truck","bike","cyclist","shuttle_bus_18","head","license_plate",
]
OUT_W, OUT_H = 150, 50   # kích thước chuẩn của dslaplacian


# ══════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════

def _letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_y, pad_x = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def _nms(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    kept = []
    while len(order):
        i = order[0]; kept.append(i)
        if len(order) == 1: break
        ix1=np.maximum(x1[i],x1[order[1:]]); iy1=np.maximum(y1[i],y1[order[1:]])
        ix2=np.minimum(x2[i],x2[order[1:]]); iy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order=order[1:][iou<=iou_thresh]
    return kept


def detect(img_bgr, conf_thresh):
    import onnxruntime as ort
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lb, scale, pad_x, pad_y = _letterbox(img_rgb)
    inp = np.transpose(lb.astype(np.float32)/255.0, (2,0,1))[np.newaxis]
    sess = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0]  # [8400,6]

    scores = out[:,4]; class_ids = np.round(out[:,5]).astype(int)
    mask = (scores >= conf_thresh) & (class_ids == _LP_CLASS_ID)
    out_lp, sc_lp = out[mask], scores[mask]
    if len(out_lp) == 0:
        return []

    h_orig, w_orig = img_bgr.shape[:2]
    boxes = out_lp[:,:4].copy()
    boxes[:,[0,2]] = (boxes[:,[0,2]] - pad_x) / scale
    boxes[:,[1,3]] = (boxes[:,[1,3]] - pad_y) / scale
    boxes = np.clip(boxes, 0, [w_orig,h_orig,w_orig,h_orig])
    kept = _nms(boxes, sc_lp)

    results = []
    for i in kept:
        x1,y1,x2,y2 = boxes[i]
        results.append({
            "x":int(x1),"y":int(y1),"w":int(x2-x1),"h":int(y2-y1),
            "conf":float(sc_lp[i]),
        })
    return results


# ══════════════════════════════════════════════════════════════════════
# Corner detection helpers
# ══════════════════════════════════════════════════════════════════════

def _sort_corners(pts):
    """Sắp xếp 4 điểm → TL, TR, BR, BL."""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    pts = pts[pts[:, 0].argsort()]
    tl = pts[0] if pts[0, 1] < pts[1, 1] else pts[1]
    bl = pts[1] if pts[0, 1] < pts[1, 1] else pts[0]
    tr = pts[2] if pts[2, 1] < pts[3, 1] else pts[3]
    br = pts[3] if pts[2, 1] < pts[3, 1] else pts[2]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _line_intersect(l1, l2):
    """Giao điểm của 2 đoạn thẳng (dạng [x1,y1,x2,y2])."""
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2
    a1,b1,c1 = y2-y1, x1-x2, (y2-y1)*x1+(x1-x2)*y1
    a2,b2,c2 = y4-y3, x3-x4, (y4-y3)*x3+(x3-x4)*y3
    det = a1*b2 - a2*b1
    if abs(det) < 1e-6:
        return None
    x = (c1*b2 - c2*b1) / det
    y = (a1*c2 - a2*c1) / det
    return np.array([x, y], dtype=np.float32)


def _hough_corners(lines, img_w, img_h):
    """Từ HoughLinesP tìm 4 biên (top/bot/left/right) → 4 giao điểm."""
    h_lines, v_lines = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 25 or angle > 155:
            h_lines.append(((y1 + y2) / 2.0, l[0]))
        elif 65 < angle < 115:
            v_lines.append(((x1 + x2) / 2.0, l[0]))

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    h_lines.sort(key=lambda x: x[0])
    v_lines.sort(key=lambda x: x[0])
    top, bot   = h_lines[0][1],  h_lines[-1][1]
    left, right = v_lines[0][1], v_lines[-1][1]

    tl = _line_intersect(top, left)
    tr = _line_intersect(top, right)
    br = _line_intersect(bot, right)
    bl = _line_intersect(bot, left)
    if any(p is None for p in [tl, tr, br, bl]):
        return None

    margin = max(img_w, img_h) * 0.6
    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    if (corners < -margin).any() or (corners[:, 0] > img_w + margin).any() \
            or (corners[:, 1] > img_h + margin).any():
        return None
    return corners


def _valid_plate_quad(pts):
    """True nếu 4 điểm tạo thành tứ giác có tỉ lệ hợp lệ của biển số."""
    spts = _sort_corners(pts)
    top_w  = np.linalg.norm(spts[1] - spts[0])
    bot_w  = np.linalg.norm(spts[2] - spts[3])
    left_h = np.linalg.norm(spts[3] - spts[0])
    right_h= np.linalg.norm(spts[2] - spts[1])
    w_avg  = (top_w + bot_w) / 2.0
    h_avg  = (left_h + right_h) / 2.0
    if w_avg < 5 or h_avg < 3:
        return False
    ratio = w_avg / h_avg
    # Biển xe hơi VN ~4.27:1, xe máy ~1.46:1, cho phép dải rộng
    return 1.0 <= ratio <= 7.5


def _quad_from_mask(mask, min_area):
    """Tìm quadrilateral tốt nhất trong binary mask. None nếu không tìm được."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(cnt) < min_area:
            break
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        for eps in [0.02, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15]:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                if _valid_plate_quad(pts):
                    return pts
    return None


def _find_plate_corners(gray_padded, orig_bbox_w, orig_bbox_h):
    """
    Tìm 4 góc biển số trong padded crop ở độ phân giải gốc (Native Resolution)
    bằng cách dò tìm và gom cụm các ký tự (adaptive threshold + RETR_CCOMP),
    hoặc fallback về các phương án contour/scale thông thường.
    """
    pw = gray_padded.shape[1]
    ph = gray_padded.shape[0]

    if orig_bbox_w < 10 or orig_bbox_h < 4:
        return None, 'scale'

    # 1. Adaptive Thresholding to isolate characters
    block_size = 11
    thresh = cv2.adaptiveThreshold(
        gray_padded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 2
    )

    # Find all contours with hierarchy
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None and len(cnts) > 0:
        hierarchy = hierarchy[0]
        char_points = []
        char_boxes = []

        for c_idx, cnt in enumerate(cnts):
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            h_ratio = ch / float(ph)
            w_ratio = cw / float(pw)
            aspect = ch / float(cw) if cw > 0 else 0

            # Characters must not touch the crop boundaries and have character-like shapes
            if 0.15 < h_ratio < 0.85 and 0.02 < w_ratio < 0.35 and 0.5 < aspect < 6.0:
                if cx > 2 and cy > 2 and (cx + cw) < pw - 3 and (cy + ch) < ph - 3:
                    area = cv2.contourArea(cnt)
                    solidity = area / (cw * ch) if (cw * ch) > 0 else 0
                    if solidity > 0.15:
                        char_boxes.append((cx, cy, cw, ch))
                        for pt in cnt:
                            char_points.append(pt[0])

        # If we have enough character candidates, we calculate the estimated plate bounding box
        if len(char_boxes) >= 2:
            char_points = np.array(char_points)
            rr = cv2.minAreaRect(char_points)
            
            # Determine scaling factors based on aspect ratio of the text region
            char_w, char_h = rr[1][0], rr[1][1]
            text_aspect = max(char_w, char_h) / max(min(char_w, char_h), 1.0)
            if char_w < char_h:
                # Rotated by ~90 degrees or height is larger, handle accordingly
                text_aspect = char_h / max(char_w, 1.0)

            if text_aspect > 2.0:
                # 1-line long plate
                scale_w = 1.25
                scale_h = 1.65
            else:
                # 2-line square plate
                scale_w = 1.35
                scale_h = 1.25

            center = rr[0]
            size = rr[1]
            angle = rr[2]

            expanded_size = (size[0] * scale_w, size[1] * scale_h)
            # Clip expanded size to not exceed crop dimensions
            expanded_size = (min(expanded_size[0], pw * 0.98), min(expanded_size[1], ph * 0.98))

            expanded_rr = (center, expanded_size, angle)
            expanded_corners = cv2.boxPoints(expanded_rr).astype(np.float32)

            # Clamp corners to crop image bounds
            clamped_corners = []
            for p in expanded_corners:
                px_val = np.clip(p[0], 0, pw - 1)
                py_val = np.clip(p[1], 0, ph - 1)
                clamped_corners.append([px_val, py_val])

            return _sort_corners(np.array(clamped_corners)), 'char_est'

    # Fallback to standard Otsu/Canny external contour
    # Border check helper
    def is_border_quad(pts):
        on_border = 0
        for p in pts:
            if p[0] <= 1 or p[0] >= pw - 2 or p[1] <= 1 or p[1] >= ph - 2:
                on_border += 1
        return on_border >= 3

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    min_area = 0.10 * pw * ph

    # ── Strategy 2: Otsu ──
    blur_otsu = cv2.GaussianBlur(gray_padded, (3, 3), 0)
    _, thresh_otsu = cv2.threshold(blur_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    border_pixels = np.sum(thresh_otsu[0:2, :]) + np.sum(thresh_otsu[-2:, :]) + np.sum(thresh_otsu[:, 0:2]) + np.sum(thresh_otsu[:, -2:])
    total_border = (2 * pw * 2) + (2 * ph * 2)
    if border_pixels / 255 > total_border * 0.5:
        thresh_otsu = cv2.bitwise_not(thresh_otsu)

    processed = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            break
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        for eps in [0.02, 0.04, 0.06, 0.08, 0.10]:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                if not is_border_quad(pts) and _valid_plate_quad(pts):
                    return pts, 'otsu'
        
        # Fallback to minAreaRect on this contour
        rr = cv2.minAreaRect(cnt)
        corners = cv2.boxPoints(rr).astype(np.float32)
        if not is_border_quad(corners) and _valid_plate_quad(corners):
            return corners, 'otsu_minar'

    # ── Strategy 3: Canny Contours ──
    blur_canny = cv2.GaussianBlur(gray_padded, (3, 3), 0)
    edges = cv2.Canny(blur_canny, 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            break
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        for eps in [0.02, 0.04, 0.06, 0.08, 0.10]:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                if not is_border_quad(pts) and _valid_plate_quad(pts):
                    return pts, 'contour'
        
        # Fallback to minAreaRect on this contour
        rr = cv2.minAreaRect(cnt)
        corners = cv2.boxPoints(rr).astype(np.float32)
        if not is_border_quad(corners) and _valid_plate_quad(corners):
            return corners, 'minar'

    # ── Strategy 4: minAreaRect on all Canny edges ──
    edge_pts = cv2.findNonZero(edges)
    if edge_pts is not None and len(edge_pts) >= 4:
        rr = cv2.minAreaRect(edge_pts)
        corners = cv2.boxPoints(rr).astype(np.float32)
        if not is_border_quad(corners) and _valid_plate_quad(corners):
            return corners, 'minar'

    return None, 'scale'


# ══════════════════════════════════════════════════════════════════════
# dslaplacian pipeline (tái tạo laplacian_lib.cu + gstlaplacian.cpp)
# ══════════════════════════════════════════════════════════════════════

def run_laplacian(gray_padded_crop, orig_bbox_w, orig_bbox_h):
    """
    Trả về (warped_stretched, variance, align_mode, corners_in_padded_crop).
    align_mode: 'contour' | 'hough' | 'minar' | 'scale'
    corners: float32 array 4×2 trong tọa độ padded crop, hoặc None.
    """
    cw, ch = gray_padded_crop.shape[1], gray_padded_crop.shape[0]
    dst = np.array([[0,0],[OUT_W-1,0],[OUT_W-1,OUT_H-1],[0,OUT_H-1]], dtype=np.float32)

    corners, align_mode = _find_plate_corners(gray_padded_crop, orig_bbox_w, orig_bbox_h)

    if corners is not None:
        src = _sort_corners(corners)
        M   = cv2.getPerspectiveTransform(src, dst)
    else:
        src = np.array([[0,0],[cw,0],[cw,ch],[0,ch]], dtype=np.float32)
        M   = cv2.getPerspectiveTransform(src, dst)

    # Warp từ padded crop → OUT_W×OUT_H
    warped = cv2.warpPerspective(gray_padded_crop, M, (OUT_W, OUT_H),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Contrast stretch (bỏ pixel đen viền warp)
    mask = warped > 0
    if mask.any():
        mn, mx = int(warped[mask].min()), int(warped[mask].max())
        warped  = np.clip((warped.astype(np.int32) - mn) * 255 // max(mx - mn, 1),
                          0, 255).astype(np.uint8)

    # Gaussian 3×3 → Laplacian 4-connected → variance
    kG      = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16.0
    blurred = cv2.filter2D(warped, -1, kG, borderType=cv2.BORDER_REPLICATE)
    kL      = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    lap     = cv2.filter2D(blurred.astype(np.float32), -1, kL, borderType=cv2.BORDER_REPLICATE)
    interior = lap[2:-2, 2:-2].flatten()
    mean     = interior.mean()
    variance = float((interior**2).mean() - mean**2)

    return warped, variance, align_mode, corners


# ══════════════════════════════════════════════════════════════════════
# Tạo ảnh kết quả
# ══════════════════════════════════════════════════════════════════════

def _text(img, txt, xy, scale=0.5, color=(255,255,255), thickness=1):
    cv2.putText(img,txt,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thickness+2)
    cv2.putText(img,txt,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness)


def build_result_image(img_bgr, detections, plate_results, img_name):
    """
    Layout mỗi ảnh:
      Hàng 1: ảnh gốc thu nhỏ với bboxes
      Hàng 2+: BEFORE (padded crop + corner overlay) | AFTER (warp phẳng)
    """
    PANEL_H  = 180    # chiều cao panel biển số
    BEFORE_W = 320    # chiều rộng BEFORE
    AFTER_W  = OUT_W * PANEL_H // OUT_H   # 150*180/50 = 540

    # ── Ảnh overview ──────────────────────────────────────────────────
    ov_h = 300
    ov_w = int(img_bgr.shape[1] * ov_h / img_bgr.shape[0])
    ov   = cv2.resize(img_bgr, (ov_w, ov_h))
    sx   = img_bgr.shape[1] / ov_w
    sy   = img_bgr.shape[0] / ov_h
    for d in detections:
        x1,y1 = int(d["x"]/sx), int(d["y"]/sy)
        x2,y2 = int((d["x"]+d["w"])/sx), int((d["y"]+d["h"])/sy)
        cv2.rectangle(ov,(x1,y1),(x2,y2),(0,255,0),2)
        _text(ov,f"LP {d['conf']:.2f}",(x1,max(y1-4,12)),0.45,(0,255,0))

    total_w = max(ov_w, BEFORE_W + AFTER_W + 4)
    if ov_w < total_w:
        pad = np.zeros((ov_h, total_w-ov_w, 3), np.uint8)
        ov  = np.hstack([ov,pad])

    hdr = np.zeros((28, total_w, 3), np.uint8)
    _text(hdr, img_name, (8,20), 0.55, (200,220,255))

    rows = [hdr, ov]

    _ALIGN_COLOR = {'contour':(0,220,60), 'hough':(0,200,255),
                     'minar':(0,140,255), 'scale':(60,60,220)}
    _ALIGN_TEXT  = {'contour':"4-corner contour",
                     'hough'  :"Hough lines",
                     'minar'  :"minAreaRect",
                     'scale'  :"scale (fallback)"}

    if not plate_results:
        empty = np.zeros((60, total_w, 3), np.uint8)
        _text(empty, "Khong phat hien bien so", (10, 38), 0.65, (60,80,255), 2)
        rows.append(empty)
    else:
        for idx, (d, warped, variance, align_mode, corners, px, py, pw, ph) in enumerate(plate_results, 1):
            border_color = _ALIGN_COLOR.get(align_mode, (60,60,220))

            # ── BEFORE: padded crop + corner overlay ──────────────────
            padded_bgr = img_bgr[py:py+ph, px:px+pw].copy()

            if corners is not None:
                sorted_c = _sort_corners(corners)
                # Vẽ tứ giác nối 4 góc
                poly_pts = sorted_c.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(padded_bgr, [poly_pts], True, (0, 255, 0), 2)
                # Vẽ 4 góc với màu riêng: TL=đỏ, TR=vàng, BR=tím, BL=cam
                corner_colors = [(0,0,255), (0,255,255), (255,0,255), (0,128,255)]
                for pt, col in zip(sorted_c, corner_colors):
                    cv2.circle(padded_bgr, (int(pt[0]), int(pt[1])), 5, col, -1)
                    cv2.circle(padded_bgr, (int(pt[0]), int(pt[1])), 6, (0,0,0), 1)

            before = cv2.resize(padded_bgr, (BEFORE_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(before, (0,0), (BEFORE_W-1, PANEL_H-1), (0,200,255), 2)
            _text(before, "BEFORE (padded crop)", (6, 22), 0.52, (255,255,255))
            _text(before, f"bbox {d['w']}x{d['h']} px", (6, 44), 0.42, (200,200,200))
            _text(before, f"conf={d['conf']:.2f}", (6, PANEL_H-32), 0.42, (200,200,200))
            align_label = _ALIGN_TEXT.get(align_mode, align_mode)
            _text(before, f"align: {align_label}", (6, PANEL_H-12), 0.42, border_color)

            # ── AFTER: warped 150x50 → phóng to với INTER_LINEAR ──────
            after_gray = cv2.resize(warped, (AFTER_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
            after = cv2.cvtColor(after_gray, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(after, (0,0), (AFTER_W-1, PANEL_H-1), border_color, 3)
            _text(after, "AFTER (dslaplacian warp)", (6, 22), 0.52, (255,255,255))
            sharp = "SAC NET" if variance > 100 else ("TRUNG BINH" if variance > 30 else "MO")
            sc = (0,220,60) if variance > 100 else ((0,200,200) if variance > 30 else (60,80,255))
            _text(after, f"variance = {int(variance)}", (6, PANEL_H-32), 0.55, (0,200,255))
            _text(after, sharp, (6, PANEL_H-8), 0.6, sc, 2)

            sep = np.full((PANEL_H, 4, 3), 60, np.uint8)
            row = np.hstack([before, sep, after])

            if row.shape[1] < total_w:
                row = np.hstack([row, np.zeros((PANEL_H, total_w-row.shape[1], 3), np.uint8)])
            rows.append(row)

    divider = np.full((6, total_w, 3), 40, np.uint8)
    rows.append(divider)
    return np.vstack(rows)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def process_image(img_path, conf_thresh):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [ERROR] Không đọc được: {img_path}")
        return None, []

    detections = detect(img, conf_thresh)
    plate_results = []
    fh, fw = img.shape[:2]

    for d in detections:
        x = max(0, d["x"]); y = max(0, d["y"])
        w = min(d["w"], fw-x); h = min(d["h"], fh-y)
        if w < 10 or h < 4:
            continue

        # Mở rộng bbox 30% mỗi phía — mirror gstlaplacian.cpp
        pad_x = max(8, int(w * 0.30))
        pad_y = max(6, int(h * 0.30))
        px = max(0, x - pad_x)
        py = max(0, y - pad_y)
        pw = min(fw - px, w + 2 * pad_x)
        ph = min(fh - py, h + 2 * pad_y)

        gray_padded = cv2.cvtColor(img[py:py+ph, px:px+pw], cv2.COLOR_BGR2GRAY)
        warped, variance, align_mode, corners = run_laplacian(gray_padded, w, h)
        d.update({"x": x, "y": y, "w": w, "h": h})
        plate_results.append((d, warped, variance, align_mode, corners, px, py, pw, ph))

    return img, plate_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--out",  default="result_laplacian.jpg")
    args = parser.parse_args()

    all_blocks = []
    summary    = []

    print(f"\n{'Ảnh':<30} {'Biển số':>12}  {'Variance':>10}  {'Kết quả'}")
    print("─" * 65)

    for img_path in args.images:
        name = os.path.basename(img_path)
        img, plate_results = process_image(img_path, args.conf)
        if img is None:
            continue

        block = build_result_image(img, [r[0] for r in plate_results], plate_results, name)
        all_blocks.append(block)

        if plate_results:
            for i, (d, _, var, align, *_rest) in enumerate(plate_results, 1):
                sharp = "SẮC NÉT" if var > 100 else ("TRUNG BÌNH" if var > 30 else "MỜ")
                det_txt = f"{i} (conf={d['conf']:.2f})"
                print(f"  {name:<28} {det_txt:>12}  {int(var):>10}  {sharp}  [{align}]")
                summary.append((name, det_txt, int(var), sharp))
        else:
            print(f"  {name:<28} {'không detect':>12}  {'—':>10}  —")
            summary.append((name, "không detect", None, "—"))

    print("─" * 65)

    if not all_blocks:
        print("[ERROR] Không có ảnh nào xử lý được.")
        sys.exit(1)

    max_w = max(b.shape[1] for b in all_blocks)
    padded = []
    for b in all_blocks:
        if b.shape[1] < max_w:
            b = np.hstack([b, np.zeros((b.shape[0], max_w-b.shape[1], 3), np.uint8)])
        padded.append(b)

    final = np.vstack(padded)
    cv2.imwrite(args.out, final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n[OUTPUT] → {args.out}  ({final.shape[1]}x{final.shape[0]})")


if __name__ == "__main__":
    main()
