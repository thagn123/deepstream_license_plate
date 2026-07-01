import lpr_config as config
from lpr import state


def _cleanup_history(active_ids: set):
    for key in active_ids:
        state.object_last_seen[key] = state.osd_probe_frame
    stale_cutoff = state.osd_probe_frame - config._STATE_STALE_AFTER_FRAMES
    stale_keys = {key for key, last_seen in state.object_last_seen.items() if last_seen < stale_cutoff}
    stale_oids = {oid for (_, oid) in stale_keys}
    for key in stale_keys:
        state.object_last_seen.pop(key, None)
        state.vehicle_states.pop(key, None)
        state.short_id_map.pop(key, None)
        state.plate_history.pop(key, None)
        state.ocr_frame_cache.pop(key, None)
        state.split_ocr.pop(key, None)
        state.plate_text_seen.pop(key, None)
    state.locked_plate_ids -= stale_oids
