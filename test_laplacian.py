"""
test_laplacian.py — Chạy pipeline DeepStream + dslaplacian và lưu ảnh before/after alignment.

Pipeline: uridecodebin → nvstreammux → nvinfer(PGIE) → dslaplacian → tiler → fakesink

Với mỗi biển số tìm được, lưu:
  - BEFORE: crop gốc từ frame (BGR, bounding box thực)
  - AFTER:  ảnh sau warp perspective 150x50 (đúng logic của laplacian_lib.cu)

Output: outputs/laplacian_samples/
  plate_NNN_SAC_NET_before.jpg   — crop gốc
  plate_NNN_SAC_NET_after.jpg    — sau align + contrast stretch
  plate_NNN_SAC_NET_compare.jpg  — ghép ngang before | after
  summary.jpg                    — tổng hợp tất cả

Cách chạy (trong container ds90):
  python3 test_laplacian.py [video_path]
"""

import sys
import os
import cv2
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import pyds

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "videos/drive-download-20260616T102510Z-3-001/lpr_230428_002.mp4"

OUTPUT_DIR     = "outputs/laplacian_samples"
MAX_SAVES      = 3           # tổng số biển số lưu
SAVE_EVERY_N   = 3           # cách nhau ít nhất N frames để tránh trùng

OUT_W, OUT_H = 150, 50       # giống C++ plugin

# ── Alignment logic (tái tạo đúng laplacian_lib.cu + gstlaplacian.cpp) ───────

def _sort_corners_cpp(pts: np.ndarray) -> np.ndarray:
    """Sort 4 điểm: sort by x → tl/bl từ 2 trái, tr/br từ 2 phải (giống C++)."""
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
    """Từ HoughLinesP tìm 4 biên (top/bot horizontal, left/right vertical) → 4 góc."""
    h_lines, v_lines = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        mid_y = (y1 + y2) / 2.0
        mid_x = (x1 + x2) / 2.0
        if angle < 25 or angle > 155:
            h_lines.append((mid_y, l[0]))
        elif 65 < angle < 115:
            v_lines.append((mid_x, l[0]))

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    h_lines.sort(key=lambda x: x[0])
    v_lines.sort(key=lambda x: x[0])

    top, bot  = h_lines[0][1],  h_lines[-1][1]
    left, right = v_lines[0][1], v_lines[-1][1]

    tl = _line_intersect(top,  left)
    tr = _line_intersect(top,  right)
    br = _line_intersect(bot,  right)
    bl = _line_intersect(bot,  left)
    if any(p is None for p in [tl, tr, br, bl]):
        return None

    margin = max(img_w, img_h) * 0.6
    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    if (corners < -margin).any() or (corners[:, 0] > img_w + margin).any() \
            or (corners[:, 1] > img_h + margin).any():
        return None
    return corners


def align_plate_cpp_logic(gray_crop: np.ndarray):
    """
    Tái tạo đúng gstlaplacian.cpp (v2):
    1. Resize crop → 300x100 → det_img
    2. CLAHE + GaussianBlur 5x5 → Canny(30,120)
    3. morphologyEx CLOSE (3x3)
    4. Tìm contour 4 góc / Hough lines / minAreaRect trên det_img
    5. Scale tọa độ về crop gốc → PerspectiveTransform → 150x50
    6. Contrast stretch (min-max)
    """
    DET_W, DET_H = 300, 100
    cw, ch = gray_crop.shape[1], gray_crop.shape[0]
    
    det_img = cv2.resize(gray_crop, (DET_W, DET_H), interpolation=cv2.INTER_NEAREST)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enh   = clahe.apply(det_img)
    blur  = cv2.GaussianBlur(enh, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    plate_pts = []

    # ── Strategy 1: contour ──
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    min_area = 0.10 * DET_W * DET_H

    for cnt in cnts:
        if cv2.contourArea(cnt) < min_area:
            break
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        for eps in [0.02, 0.04, 0.06, 0.08, 0.10]:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                plate_pts = approx.reshape(4, 2).astype(np.float32)
                break
        if len(plate_pts) > 0:
            break

    # ── Strategy 2: Hough lines ──
    if len(plate_pts) == 0:
        min_len = max(8, int(DET_W * 0.25))
        gap     = max(4, int(DET_W * 0.12))
        lines   = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                   threshold=max(15, int(DET_W * 0.25)),
                                   minLineLength=min_len, maxLineGap=gap)
        if lines is not None:
            corners = _hough_corners(lines, DET_W, DET_H)
            if corners is not None:
                plate_pts = corners

    # ── Strategy 3: minAreaRect ──
    if len(plate_pts) == 0:
        edge_pts = cv2.findNonZero(edges)
        if edge_pts is not None and len(edge_pts) >= 4:
            rr = cv2.minAreaRect(edge_pts)
            plate_pts = cv2.boxPoints(rr).astype(np.float32)

    dst_pts = np.array([[0,0],[OUT_W-1,0],[OUT_W-1,OUT_H-1],[0,OUT_H-1]], dtype=np.float32)

    if len(plate_pts) > 0:
        ordered = _sort_corners_cpp(plate_pts)
        src_orig = ordered.copy()
        src_orig[:, 0] *= cw / DET_W    # scale từ 300x100 → crop gốc
        src_orig[:, 1] *= ch / DET_H
        M = cv2.getPerspectiveTransform(src_orig, dst_pts)
        contour_found = True
    else:
        src_orig = np.array([[0,0],[cw,0],[cw,ch],[0,ch]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_orig, dst_pts)
        contour_found = False

    warped = cv2.warpPerspective(gray_crop, M, (OUT_W, OUT_H),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Contrast stretch
    mask = warped > 0
    if mask.any():
        p_min, p_max = int(warped[mask].min()), int(warped[mask].max())
        rng = max(p_max - p_min, 1)
        warped = np.clip((warped.astype(np.int32) - p_min) * 255 // rng, 0, 255).astype(np.uint8)

    thumb = cv2.resize(gray_crop, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
    return thumb, warped, contour_found, edges


def _make_compare(crop_bgr, thumb, warped, edges, contour_found, lap_score, w, h, frame_num):
    """Tạo ảnh so sánh 4 panel: crop gốc | thumb 150x50 | canny | after warp+stretch."""
    PH = 130
    PW_ORIG = max(60, int(w * PH / max(h, 1)))
    PW_PROC = OUT_W * PH // OUT_H   # 390 px

    def panel(img, title, sub="", color=(255,255,255)):
        c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        c = cv2.resize(c, (c.shape[1], PH), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(c, (0,0), (c.shape[1]-1, PH-1), (0,200,255), 2)
        cv2.putText(c, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0),       3)
        cv2.putText(c, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color,         1)
        if sub:
            cv2.putText(c, sub, (5, PH-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        return c

    warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    border = (0,220,60) if contour_found else (60,60,220)
    cv2.rectangle(warped_bgr, (1,1), (OUT_W-2, OUT_H-2), border, 2)

    sharp = "SAC NET" if lap_score > 100 else ("TRUNG BINH" if lap_score > 30 else "MO")
    sharp_color = (0,220,60) if lap_score > 100 else ((0,200,200) if lap_score > 30 else (60,80,255))
    align_txt = "align=YES" if contour_found else "align=NO (scale)"

    p0 = panel(cv2.resize(crop_bgr, (PW_ORIG, PH)),
               "BEFORE (crop goc)", f"{w}x{h} px")
    p1 = panel(cv2.resize(thumb, (PW_PROC, PH), cv2.INTER_NEAREST),
               "Resize 150x50",    "nearest-neighbor (=thumb CUDA)")
    p2 = panel(cv2.resize(edges, (PW_PROC, PH), cv2.INTER_NEAREST),
               "Canny edges",      align_txt)
    p3 = panel(cv2.resize(warped_bgr, (PW_PROC, PH), cv2.INTER_NEAREST),
               "AFTER warp+stretch", f"var={int(lap_score)}  {sharp}", sharp_color)

    total_w = PW_ORIG + PW_PROC * 3
    header = np.zeros((24, total_w, 3), np.uint8)
    cv2.putText(header,
                f"frame={frame_num}  bbox={w}x{h}  lap_var={int(lap_score)}  {align_txt}",
                (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,220,255), 1)

    return np.vstack([header, np.hstack([p0, p1, p2, p3])])


# ── Globals cho probe ──────────────────────────────────────────────────────────

_save_count   = 0
_last_frame   = -SAVE_EVERY_N
_saved_strips = []


def sink_pad_buffer_probe(pad, info, u_data):
    global _save_count, _last_frame, _saved_strips

    gst_buffer = info.get_buffer()
    if not gst_buffer or _save_count >= MAX_SAVES:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_num = frame_meta.frame_num

        try:
            frame_rgba = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_bgr  = cv2.cvtColor(np.array(frame_rgba, copy=True, order='C'),
                                       cv2.COLOR_RGBA2BGR)
        except Exception:
            frame_bgr = None

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            if _save_count >= MAX_SAVES:
                break
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.class_id == 13 and frame_bgr is not None:
                if frame_num - _last_frame >= SAVE_EVERY_N:
                    lap_score = float(obj_meta.misc_obj_info[0])

                    fh, fw = frame_bgr.shape[:2]
                    x = max(0, int(obj_meta.rect_params.left))
                    y = max(0, int(obj_meta.rect_params.top))
                    w = min(int(obj_meta.rect_params.width),  fw - x)
                    h = min(int(obj_meta.rect_params.height), fh - y)

                    if w < 12 or h < 4:
                        try: l_obj = l_obj.next
                        except StopIteration: break
                        continue

                    crop_bgr  = frame_bgr[y:y+h, x:x+w]
                    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                    thumb, warped, contour_found, edges = align_plate_cpp_logic(crop_gray)

                    sharp  = "SAC_NET" if lap_score > 100 else ("TRUNG_BINH" if lap_score > 30 else "MO")
                    n      = _save_count
                    prefix = os.path.join(OUTPUT_DIR, f"plate_{n:03d}_{sharp}")

                    cv2.imwrite(f"{prefix}_before.jpg", crop_bgr)
                    cv2.imwrite(f"{prefix}_after.jpg",  warped)
                    compare = _make_compare(crop_bgr, thumb, warped, edges,
                                            contour_found, lap_score, w, h, frame_num)
                    cv2.imwrite(f"{prefix}_compare.jpg", compare)

                    _saved_strips.append(compare)
                    _save_count += 1
                    _last_frame  = frame_num

                    align_lbl = "align=YES" if contour_found else "align=NO "
                    print(f"[{n:03d}] f={frame_num:5d}  {w:3d}x{h:2d}  "
                          f"var={int(lap_score):6d}  {sharp:12s}  {align_lbl}")

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def _save_summary():
    if not _saved_strips:
        return
    max_w = max(s.shape[1] for s in _saved_strips)
    rows = []
    for s in _saved_strips:
        if s.shape[1] < max_w:
            pad = np.zeros((s.shape[0], max_w - s.shape[1], 3), np.uint8)
            s = np.hstack([s, pad])
        rows.append(s)
    path = os.path.join(OUTPUT_DIR, "summary.jpg")
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\n[SUMMARY] {_save_count} biển số → {path}")


# ── GStreamer pipeline ─────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Gst.init(None)

    pipeline   = Gst.Pipeline()
    source     = Gst.ElementFactory.make("uridecodebin",      "source")
    nvvidconv  = Gst.ElementFactory.make("nvvideoconvert",    "nvvidconv-pre")
    streammux  = Gst.ElementFactory.make("nvstreammux",       "streammux")
    pgie       = Gst.ElementFactory.make("nvinfer",           "pgie")
    laplacian  = Gst.ElementFactory.make("dslaplacian",       "laplacian")
    tiler      = Gst.ElementFactory.make("nvmultistreamtiler","tiler")
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert",    "nvvidconv-rgba")
    caps_gpu   = Gst.ElementFactory.make("capsfilter",        "caps-gpu")
    sink       = Gst.ElementFactory.make("fakesink",          "sink")

    for el in [source, nvvidconv, streammux, pgie, laplacian,
               tiler, nvvidconv2, caps_gpu, sink]:
        if not el:
            print(f"[ERROR] Không tạo được GStreamer element")
            sys.exit(1)
        pipeline.add(el)

    source.set_property("uri", f"file://{os.path.abspath(VIDEO_PATH)}")
    nvvidconv.set_property("nvbuf-memory-type",  3)
    nvvidconv2.set_property("nvbuf-memory-type", 3)
    streammux.set_property("width",  1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 33000)
    streammux.set_property("nvbuf-memory-type", 3)
    pgie.set_property("config-file-path", "configs/config_pgie_yolov11s.txt")
    laplacian.set_property("class-id", 13)
    tiler.set_property("rows", 1)
    tiler.set_property("columns", 1)
    tiler.set_property("width",  1920)
    tiler.set_property("height", 1080)
    caps_gpu.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    sink.set_property("sync", 0)

    def on_pad_added(dec, pad, _):
        if not pad.get_current_caps():
            return
        if pad.get_current_caps().get_structure(0).get_name().startswith("video"):
            sinkpad = nvvidconv.get_static_pad("sink")
            if not sinkpad.is_linked():
                pad.link(sinkpad)

    source.connect("pad-added", on_pad_added, None)
    nvvidconv.get_static_pad("src").link(streammux.get_request_pad("sink_0"))
    streammux.link(pgie)
    pgie.link(laplacian)
    laplacian.link(tiler)
    tiler.link(nvvidconv2)
    nvvidconv2.link(caps_gpu)
    caps_gpu.link(sink)

    # Probe sau caps_gpu — laplacian đã xong nên misc_obj_info[0] đã có giá trị
    caps_gpu.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER,
                                              sink_pad_buffer_probe, 0)

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()

    def bus_call(bus, msg, loop):
        t = msg.type
        if t == Gst.MessageType.EOS:
            print("\n[INFO] EOS — kết thúc video.")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[ERROR] {err}: {dbg}")
            loop.quit()
        return True

    bus.connect("message", bus_call, loop)

    print(f"[INFO] Video  : {VIDEO_PATH}")
    print(f"[INFO] Output : {OUTPUT_DIR}/")
    print(f"[INFO] Lưu tối đa {MAX_SAVES} biển số, cách {SAVE_EVERY_N} frame/lần")
    print(f"[INFO] Mỗi biển: *_before.jpg | *_after.jpg | *_compare.jpg")
    print("[INFO] Pipeline đang chạy... (Ctrl+C để dừng sớm)\n")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[INFO] Dừng sớm.")
    finally:
        pipeline.set_state(Gst.State.NULL)
        _save_summary()


if __name__ == "__main__":
    main()
