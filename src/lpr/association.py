from lpr.bbox import _bbox_tuple
from lpr.meta_utils import _is_vehicle_obj, _safe_parent


def _associate_plate_to_vehicle(plate_meta, vehicles_list):
    parent = _safe_parent(plate_meta)
    if parent and _is_vehicle_obj(parent):
        return parent.object_id, "parent", 1.0

    px, py, pw, ph = _bbox_tuple(plate_meta)
    pcx = px + pw / 2
    pcy = py + ph / 2

    best_vid = -1
    best_score = 0.0

    for v in vehicles_list:
        vx, vy, vw, vh = _bbox_tuple(v)
        if vx <= pcx <= vx + vw and vy <= pcy <= vy + vh:
            overlap_w = max(0, min(px + pw, vx + vw) - max(px, vx))
            overlap_h = max(0, min(py + ph, vy + vh) - max(py, vy))
            overlap_area = overlap_w * overlap_h
            plate_area = pw * ph
            containment = overlap_area / plate_area if plate_area > 0 else 0

            v_bottom_half = vy + vh / 2
            v_score = 1.0 if pcy >= v_bottom_half else 0.5

            score = containment * v_score
            if score > best_score:
                best_score = score
                best_vid = v.object_id

    if best_vid != -1:
        return best_vid, "geometry", best_score

    return -1, "none", 0.0
