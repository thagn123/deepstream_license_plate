import lpr_config as config
from lpr import state


def _pseudo_parent_lookup(sid: int, frame_num: int, class_id: int, left: int, top: int, height: int):
    key = (sid, frame_num, class_id, left, top, height)
    parent_id = state.pseudo_parent_map.get(key)
    if parent_id is not None:
        return parent_id

    for (ksid, kframe, kclass, kleft, ktop, kheight), value in state.pseudo_parent_map.items():
        if ksid != sid or kframe != frame_num or kclass != class_id:
            continue
        if abs(kleft - left) <= 2 and abs(ktop - top) <= 2 and abs(kheight - height) <= 2:
            return value
    return None


def _safe_parent(obj_meta):
    try:
        return obj_meta.parent
    except Exception:
        return None


def _is_vehicle_obj(obj_meta) -> bool:
    return (
        obj_meta is not None
        and obj_meta.unique_component_id == config.PGIE_UNIQUE_ID
        and obj_meta.class_id in config.VEHICLE_CLASS_IDS
    )


def _class_label(class_id: int) -> str:
    if 0 <= class_id < len(config.VEHICLE_LABELS):
        return config.VEHICLE_LABELS[class_id]
    return "cls{}".format(class_id)
