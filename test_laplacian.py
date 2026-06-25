import sys
import gi
import ctypes
import os
import cv2
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import pyds

if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
else:
    VIDEO_PATH = "videos/drive-download-20260616T102510Z-3-001/lpr_230428_002.mp4"

OUTPUT_DIR = "outputs/laplacian_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

save_counters = {
    "1_qua_nho_bo_qua": 0,
    "2_du_to_nhung_mo_hoac_vo_hat": 0,
    "3_du_to_vua_net": 0,
    "4_du_to_rat_net": 0
}

def align_plate(plate_crop):
    """
    Cố gắng tìm 4 góc của biển số và trải phẳng (Warp Perspective).
    Nếu không tìm được 4 góc, trả về ảnh resize bình thường.
    """
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    plate_contour = None
    min_area = 0.4 * (plate_crop.shape[0] * plate_crop.shape[1])
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > min_area:
            plate_contour = approx
            break
            
    if plate_contour is not None:
        pts = plate_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        dst = np.array([[0, 0], [150 - 1, 0], [150 - 1, 50 - 1], [0, 50 - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(gray, M, (150, 50))
        return warped, True
    
    resized = cv2.resize(gray, (150, 50), interpolation=cv2.INTER_AREA)
    return resized, False

def sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer: return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.class_id == 13:
                w = int(obj_meta.rect_params.width)
                h = int(obj_meta.rect_params.height)
                
                cv2_score = float(obj_meta.misc_obj_info[0])
                
                if w < 100 or h < 30:
                    category = "1_qua_nho_bo_qua"
                elif cv2_score < 300:
                    category = "2_du_to_nhung_mo_hoac_vo_hat"
                elif cv2_score < 700:
                    category = "3_du_to_vua_net"
                else:
                    category = "4_du_to_rat_net"

                if save_counters.get(category, 0) < 30:
                    print(f"[{category}] Biển số {w}x{h}: Điểm Laplacian từ GPU = {cv2_score:.1f}")
                    save_counters[category] = save_counters.get(category, 0) + 1

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK

def main(args):
    Gst.init(None)
    pipeline = Gst.Pipeline()

    source = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    source.set_property('uri', f"file://{os.path.abspath(VIDEO_PATH)}")
    
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")
    nvvidconv.set_property('nvbuf-memory-type', 3)

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 33000)
    streammux.set_property('nvbuf-memory-type', 3)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', "configs/config_pgie_yolov11s.txt")
    
    # Dùng tiler để chống kẹt pipeline và chuyển đổi buffer batch -> 2D
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    tiler.set_property("width", 1920)
    tiler.set_property("height", 1080)
    tiler.set_property("rows", 1)
    tiler.set_property("columns", 1)
    
    nvvidconv_rgba = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv_rgba")
    nvvidconv_rgba.set_property('nvbuf-memory-type', 3)
    
    caps_gpu = Gst.ElementFactory.make("capsfilter", "caps_gpu")
    caps_gpu.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    
    sink = Gst.ElementFactory.make("fakesink", "fake-output")
    laplacian = Gst.ElementFactory.make("dslaplacian", "laplacian")
    laplacian.set_property("class-id", 13)
    sink.set_property("sync", 0)

    for el in [source, nvvidconv, streammux, pgie, laplacian, tiler, nvvidconv_rgba, caps_gpu, sink]:
        pipeline.add(el)

    def on_pad_added(decodebin, pad, data):
        caps = pad.get_current_caps()
        if not caps: return
        if caps.get_structure(0).get_name().startswith("video"):
            pad.link(nvvidconv.get_static_pad("sink"))
    
    source.connect("pad-added", on_pad_added, None)

    nvvidconv_src = nvvidconv.get_static_pad("src")
    streammux_sink = streammux.get_request_pad("sink_0")
    nvvidconv_src.link(streammux_sink)
    streammux.link(pgie)
    pgie.link(laplacian)
    laplacian.link(tiler)
    tiler.link(nvvidconv_rgba)
    nvvidconv_rgba.link(caps_gpu)
    caps_gpu.link(sink)

    probe_pad = caps_gpu.get_static_pad("src")
    probe_pad.add_probe(Gst.PadProbeType.BUFFER, sink_pad_buffer_probe, 0)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    loop = GLib.MainLoop()
    
    def bus_call(bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("Đã đọc xong video.")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}: {debug}")
            loop.quit()
        return True

    bus.connect("message", bus_call, loop)
    
    print("Pipeline chạy... Sẽ tự thoát khi xong.")
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except BaseException:
        pass
    
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
