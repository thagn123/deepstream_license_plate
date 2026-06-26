"""
sgie3.py — Simplified SGIE3 sink probe for lpr_ocr.20240305.onnx.

The new OCR model handles 2-line (square) plates natively via the '_' token,
so no TOP/BOTTOM pseudo-object splitting is needed.
This probe only records plate bounding boxes and updates the FPS counter.
"""

import pyds
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import lpr_config as config
from lpr import state


def sgie3_sink_pad_buffer_probe_new(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        sid = frame_meta.source_id
        if state.perf_data is not None:
            state.perf_data.update_fps("stream{}".format(sid))

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if (obj_meta.class_id == config.LP_CLASS_ID
                    and obj_meta.unique_component_id == config.PGIE_UNIQUE_ID):
                if obj_meta.object_id in state.locked_plate_ids:
                    if frame_meta.frame_num % 5 != 0:
                        obj_meta.class_id = 99
                else:
                    r = obj_meta.rect_params
                    state.plate_rects[(sid, obj_meta.object_id)] = (
                        r.left, r.top, r.width, r.height
                    )

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
