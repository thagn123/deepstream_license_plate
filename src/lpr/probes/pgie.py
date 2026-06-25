import pyds
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import lpr_config as config
from lpr import state
from lpr.osd_utils import _hide_text


_PGIE_KEEP = None


def _remove_obj(frame_meta, obj_meta):
    try:
        pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
    except Exception:
        obj_meta.rect_params.border_width = 0
        _hide_text(obj_meta)


def pgie_src_pad_buffer_probe(pad, info, u_data):
    """Keep only PGIE vehicle/LP objects before nvtracker.

    Drops objects that are:
    - Not in the keep-class set (non-vehicle, non-LP classes)
    - Vehicles smaller than min_vehicle_width × min_vehicle_height (too small/far)
    """
    global _PGIE_KEEP
    if _PGIE_KEEP is None:
        _PGIE_KEEP = config.VEHICLE_CLASS_IDS | {
            config.LP_CLASS_ID,
            config.LP_TOP_CLASS_ID,
            config.LP_BOT_CLASS_ID,
        }

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    min_w = state.min_vehicle_width
    min_h = state.min_vehicle_height
    if state.min_vehicle_width_ratio > 0:
        min_w = int(state.min_vehicle_width_ratio * state.muxer_width)
    if state.min_vehicle_height_ratio > 0:
        min_h = int(state.min_vehicle_height_ratio * state.muxer_height)

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
            try:
                next_obj = l_obj.next
            except StopIteration:
                next_obj = None

            if obj_meta.unique_component_id == config.PGIE_UNIQUE_ID:
                cls = obj_meta.class_id
                if cls not in _PGIE_KEEP:
                    _remove_obj(frame_meta, obj_meta)
                elif cls in config.VEHICLE_CLASS_IDS and (min_w > 0 or min_h > 0):
                    w = obj_meta.rect_params.width
                    h = obj_meta.rect_params.height
                    if (min_w > 0 and w < min_w) or (min_h > 0 and h < min_h):
                        _remove_obj(frame_meta, obj_meta)

            l_obj = next_obj

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
