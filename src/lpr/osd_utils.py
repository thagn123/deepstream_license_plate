import time
import pyds
from lpr import state


def _display_id(source_id: int, obj_id: int) -> int:
    key = (source_id, obj_id)
    if key not in state.short_id_map:
        state.short_id_map[key] = state.next_short_id
        state.next_short_id += 1
    return state.short_id_map[key]


def _hide_text(obj_meta):
    try:
        obj_meta.text_params.display_text = ""
        obj_meta.text_params.set_bg_clr = 0
        obj_meta.text_params.font_params.font_size = 1
        obj_meta.text_params.font_params.font_color.set(0.0, 0.0, 0.0, 0.0)
    except Exception:
        pass


def _place_label(obj_meta, y_pad: int = 18):
    try:
        r = obj_meta.rect_params
        obj_meta.text_params.x_offset = max(0, int(r.left))
        obj_meta.text_params.y_offset = max(0, int(r.top) - y_pad)
    except Exception:
        pass


def _update_continuous_fps(source_id: int):
    now = time.perf_counter()
    key = f"stream{source_id}"
    fps_data = state.fps_overlay_state.get(key)
    if fps_data is None:
        state.fps_overlay_state[key] = {"last_ts": now, "fps": 0.0}
        return

    dt = now - fps_data["last_ts"]
    fps_data["last_ts"] = now
    if dt <= 0.0:
        return

    instant_fps = 1.0 / dt
    prev_fps = fps_data.get("fps", 0.0)
    if prev_fps <= 0.0:
        fps_data["fps"] = instant_fps
    else:
        fps_data["fps"] = (prev_fps * (1.0 - state.fps_overlay_alpha)) + (instant_fps * state.fps_overlay_alpha)


def _add_fps_overlay(batch_meta, frame_meta):
    if state.fps_overlay_state:
        fps_text = " | ".join(
            f"{stream_id}: {data.get('fps', 0.0):.1f} FPS"
            for stream_id, data in sorted(state.fps_overlay_state.items())
        )
    else:
        fps_text = "FPS: calculating..."

    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    display_meta.num_labels = 1
    text_params = display_meta.text_params[0]
    text_params.display_text = fps_text
    text_params.x_offset = 14
    text_params.y_offset = 14
    text_params.font_params.font_name = "Serif"
    text_params.font_params.font_size = 14
    text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    text_params.set_bg_clr = 1
    text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.65)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
