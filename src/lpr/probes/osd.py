import pyds
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import lpr_config as config
from lpr import state as lpr_state
from lpr.osd_utils import _display_id, _hide_text, _place_label, _add_fps_overlay, _update_continuous_fps
from lpr.cleanup import _cleanup_history


def osd_sink_pad_buffer_probe(pad, info, u_data):
    lpr_state.osd_probe_frame += 1
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    all_visible_keys = set()
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        sid = frame_meta.source_id
        _update_continuous_fps(sid)
        lpr_state.perf_data.update_fps(f"stream{sid}")

        t_rows = max(1, lpr_state.tiler_rows)
        t_cols = max(1, lpr_state.tiler_cols)
        tile_w = config.TILER_WIDTH / t_cols
        tile_h = config.TILER_HEIGHT / t_rows

        visible_vehicle_keys = set()

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.unique_component_id == config.SGIE3_UNIQUE_ID:
                obj_meta.rect_params.border_width = 0
                obj_meta.rect_params.has_bg_color = 0
                _hide_text(obj_meta)
            else:
                cid = obj_meta.class_id
                if obj_meta.unique_component_id == config.PGIE_UNIQUE_ID:
                    r = obj_meta.rect_params
                    cx = r.left + r.width / 2.0
                    cy = r.top + r.height / 2.0
                    col = max(0, min(t_cols - 1, int(cx / tile_w)))
                    row = max(0, min(t_rows - 1, int(cy / tile_h)))
                    obj_sid = row * t_cols + col

                    if cid in config.VEHICLE_CLASS_IDS and obj_meta.object_id != config.UNTRACKED_OBJECT_ID:
                        visible_vehicle_keys.add((obj_sid, obj_meta.object_id))

                    if cid in (config.LP_TOP_CLASS_ID, config.LP_BOT_CLASS_ID):
                        obj_meta.rect_params.border_width = 0
                        obj_meta.rect_params.has_bg_color = 0
                        _hide_text(obj_meta)
                        try:
                            l_obj = l_obj.next
                        except StopIteration:
                            break
                        continue

                    color = config._CLASS_COLORS[cid % len(config._CLASS_COLORS)]
                    label = _class_label_str(cid)

                    if cid in (config.LP_CLASS_ID, 99):
                        best_vid = obj_meta.object_id
                        shown_text = ""
                        for key, vs in lpr_state.vehicle_states.items():
                            if vs.source_id == obj_sid and vs.best_plate_object_id == obj_meta.object_id:
                                best_vid = vs.vehicle_tracker_id
                                shown_text = vs.best_plate_text_stable or vs.display_plate_text
                                break

                        display_id = _display_id(obj_sid, best_vid)
                        sid_str = f"stream{obj_sid}"
                        stream_fps = lpr_state.perf_data.get_current_fps(sid_str)
                        source_uri = lpr_state.source_uri_by_id.get(obj_sid, "")
                        is_forced_lq = config.FORCE_LQ_RTSP and "rtsp://" in source_uri
                        
                        is_low_quality = is_forced_lq or (0 < stream_fps < 15)
                        quality_str = "LQ" if is_low_quality else "HQ"
                        
                        display_text = f"Plate: {shown_text} [{quality_str}] #{display_id}" if shown_text else f"Plate [{quality_str}] #{display_id}"
                    else:
                        display_id = _display_id(obj_sid, obj_meta.object_id)
                        display_text = f"{label} #{display_id}"

                    r = obj_meta.rect_params
                    r.border_width = 2 if cid in (config.LP_CLASS_ID, 99) else 3
                    r.border_color.set(color[0], color[1], color[2], color[3])
                    r.has_bg_color = 0

                    if display_text:
                        obj_meta.text_params.display_text = display_text
                        _place_label(obj_meta, 18)
                        obj_meta.text_params.font_params.font_name = "Serif"
                        obj_meta.text_params.font_params.font_size = 11
                        obj_meta.text_params.font_params.font_color.set(color[0], color[1], color[2], 1.0)
                        obj_meta.text_params.set_bg_clr = 1
                        obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.65)
                    else:
                        _hide_text(obj_meta)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        _add_fps_overlay(batch_meta, frame_meta)
        all_visible_keys.update(visible_vehicle_keys)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    lpr_state.cleanup_counter += 1
    if lpr_state.cleanup_counter % config._CLEANUP_INTERVAL == 0:
        _cleanup_history(all_visible_keys)

    return Gst.PadProbeReturn.OK


def _class_label_str(class_id: int) -> str:
    if 0 <= class_id < len(config.VEHICLE_LABELS):
        return config.VEHICLE_LABELS[class_id].replace("_", " ").title()
    return "cls{}".format(class_id)
