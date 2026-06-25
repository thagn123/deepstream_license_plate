import lpr_config as config
from lpr.bbox import _bbox_tuple, _bbox_iou
from lpr.meta_utils import _is_vehicle_obj, _safe_parent


def _geometry_associate_plate(plate_bbox: tuple, vehicles: dict):
    if not vehicles:
        return None
    px, py, pw, ph = plate_bbox
    pcx = px + pw / 2.0
    pcy = py + ph / 2.0
    best_key = None
    best_score = -1.0
    for track_key, data in vehicles.items():
        vx, vy, vw, vh = data["bbox"]
        if vw <= 0 or vh <= 0:
            continue
        contains_center = vx <= pcx <= vx + vw and vy <= pcy <= vy + vh
        ix1 = max(px, vx)
        iy1 = max(py, vy)
        ix2 = min(px + pw, vx + vw)
        iy2 = min(py + ph, vy + vh)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        plate_area = max(1, pw * ph)
        containment = inter / plate_area
        vertical_pos = max(0.0, min(1.0, (pcy - vy) / vh))
        score = containment * 2.0 + (1.0 if contains_center else 0.0) + vertical_pos * 0.25
        if score > best_score:
            best_score = score
            best_key = track_key
    return best_key if best_score >= 0.75 else None


def _resolve_vehicle_track_key(obj_meta, sid: int, vehicles: dict):
    parent = _safe_parent(obj_meta)
    if _is_vehicle_obj(parent):
        return (sid, parent.object_id)
    grandparent = _safe_parent(parent) if parent is not None else None
    if _is_vehicle_obj(grandparent):
        return (sid, grandparent.object_id)
    return _geometry_associate_plate(_bbox_tuple(obj_meta), vehicles)


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
