import os
import cv2
import pyds
import json
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import lpr_config as config
from lpr import state as lpr_state
from lpr.models import VehicleTrackState
from lpr.bbox import _bbox_tuple, _bbox_iou, _bbox_center_distance, _smooth_bbox
from lpr.meta_utils import _class_label
from lpr.osd_utils import _display_id
from lpr.image_utils import _crop_plate_from_frame
from lpr.ocr import _read_lpr_text
from lpr.plate_text import (
    _correct_vn_plate, _plate_quality_score, _plate_pattern_score,
    _should_replace_stable_text, _stable_plate,
    _plate_history_stats,
)
from lpr.association import _associate_plate_to_vehicle
from lpr.events import _debug_event, _emit_event


def metadata_src_pad_buffer_probe(pad, info, u_data):
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
        frame_num = frame_meta.frame_num
        pts = getattr(frame_meta, "buf_pts", 0)
        if not pts:
            pts = getattr(gst_buffer, "pts", 0)

        frame_image = None
        if lpr_state.event_output_dir or lpr_state.save_event_frame:
            try:
                frame_image = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            except Exception:
                frame_image = None

        vehicles = []
        plates = []

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if (
                obj_meta.unique_component_id == config.PGIE_UNIQUE_ID
                and obj_meta.class_id in config.VEHICLE_CLASS_IDS
                and obj_meta.object_id != config.UNTRACKED_OBJECT_ID
            ):
                vehicles.append(obj_meta)
            elif obj_meta.class_id in (config.LP_CLASS_ID, 99) and obj_meta.unique_component_id == config.PGIE_UNIQUE_ID:
                plates.append(obj_meta)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # ── Update vehicle track states ───────────────────────────────────────
        for v in vehicles:
            vid = v.object_id
            track_key = (sid, vid)
            if track_key not in lpr_state.vehicle_states:
                lpr_state.vehicle_states[track_key] = VehicleTrackState(
                    display_id=_display_id(sid, vid),
                    source_id=sid,
                    vehicle_tracker_id=vid,
                    first_seen_frame=frame_num,
                    vehicle_class=v.class_id,
                    vehicle_class_name=_class_label(v.class_id).replace("_", " ").title(),
                )

            vs = lpr_state.vehicle_states[track_key]
            vs.last_seen_frame = frame_num
            vs.vehicle_confidence = float(v.confidence)

            raw_v_bbox = _bbox_tuple(v)
            vs.vehicle_bbox_raw = raw_v_bbox

            if vs.vehicle_bbox == (0, 0, 0, 0):
                vs.vehicle_bbox = raw_v_bbox
            else:
                iou = _bbox_iou(vs.vehicle_bbox, raw_v_bbox)
                cdist = _bbox_center_distance(vs.vehicle_bbox, raw_v_bbox)
                max_jump = lpr_state.bbox_max_center_jump_ratio * max(raw_v_bbox[2], raw_v_bbox[3])
                if iou < lpr_state.bbox_reset_iou or cdist > max_jump:
                    vs.vehicle_bbox = raw_v_bbox
                else:
                    vs.vehicle_bbox = _smooth_bbox(vs.vehicle_bbox, raw_v_bbox, lpr_state.bbox_smooth_alpha)
            vs.last_bbox_update_frame = frame_num

        # ── Source Quality Assessment ─────────────────────────────────────────
        sid_str = f"stream{frame_meta.source_id}"
        stream_fps = lpr_state.perf_data.get_current_fps(sid_str)
        source_uri = lpr_state.source_uri_by_id.get(frame_meta.source_id, "")
        
        is_low_quality_source = False
        is_forced_lq = config.FORCE_LQ_RTSP and "rtsp://" in source_uri
        
        if is_forced_lq or (0 < frame_meta.source_frame_width < 1280) or (0 < frame_meta.source_frame_height < 720) or (0 < stream_fps < 15):
            is_low_quality_source = True

        # ── Process plates ────────────────────────────────────────────────────
        frame_best_plates = {}
        frame_plate_seen = {}
        for p in plates:
            # Chỉ dùng đánh giá chất lượng của cả luồng video
            is_low_quality = is_low_quality_source

            # Dynamic Confidence Filter
            target_conf = 0.15 if is_low_quality else 0.25
            if p.confidence > 0.0 and p.confidence < target_conf:
                continue

            if p.rect_params.width < lpr_state.min_plate_width or p.rect_params.height < lpr_state.min_plate_height:
                continue

            # Dynamic Laplacian Filter
            if not getattr(lpr_state, 'disable_laplacian', False):
                lap_score = int(p.misc_obj_info[0])
                target_lap = 50 if is_low_quality else 150
                if lap_score < target_lap and lap_score > 0:
                    continue
            lpr_state.metrics["plate_objects"] += 1

            vid, assoc_method, assoc_score = _associate_plate_to_vehicle(p, vehicles)
            if vid == -1:
                vid = p.object_id
                if (sid, vid) not in lpr_state.vehicle_states:
                    lpr_state.vehicle_states[(sid, vid)] = VehicleTrackState(
                        display_id=_display_id(sid, vid), source_id=sid, vehicle_tracker_id=vid,
                        first_seen_frame=frame_num, vehicle_class=-1, vehicle_class_name="Unknown Vehicle",
                    )
                assoc_method = "none"
                assoc_score = 0.0

            track_key = (sid, vid)
            
            if vid == 434:
                crop_img = _crop_plate_from_frame(frame_image, p.rect_params)
                if crop_img is not None:
                    os.makedirs("outputs/debug_434", exist_ok=True)
                    cv2.imwrite(f"outputs/debug_434/plate_434_frame_{frame_num}.jpg", crop_img)

            _p_area = p.rect_params.width * p.rect_params.height
            if track_key not in frame_plate_seen or _p_area > frame_plate_seen[track_key][1]:
                frame_plate_seen[track_key] = (p, _p_area)

            full_text, full_conf = _read_lpr_text(p, config.SGIE3_UNIQUE_ID)
            best_cand_text = _correct_vn_plate(full_text)
            best_cand_conf = full_conf

            if not best_cand_text:
                continue

            best_cand_score = _plate_quality_score(
                best_cand_text, best_cand_conf,
                p.rect_params.width, p.rect_params.height, assoc_score
            )

            if lpr_state.debug_jsonl_path:
                _debug_event("plate_ocr", {
                    "frame_num": frame_num,
                    "source_id": sid,
                    "plate_object_id": p.object_id,
                    "text": best_cand_text,
                    "conf": round(best_cand_conf, 3),
                    "score": round(max(best_cand_score, 0.0), 3),
                    "display_text": lpr_state.vehicle_states.get(track_key, VehicleTrackState()).display_plate_text,
                    "stable_text": lpr_state.vehicle_states.get(track_key, VehicleTrackState()).best_plate_text_stable,
                })

            if best_cand_score <= 0.0:
                continue
            lpr_state.metrics["ocr_raw_events"] += 1

            if track_key not in frame_best_plates or best_cand_score > frame_best_plates[track_key]['score']:
                frame_best_plates[track_key] = {
                    'p': p, 'score': best_cand_score, 'text': best_cand_text, 'conf': best_cand_conf,
                    'assoc_method': assoc_method, 'assoc_score': assoc_score,
                }

        # ── Update best plate state per vehicle ───────────────────────────────
        for track_key, b in frame_best_plates.items():
            vs = lpr_state.vehicle_states[track_key]
            p = b['p']
            vs.best_plate_object_id = p.object_id
            vs.association_method = b['assoc_method']
            vs.association_score = b['assoc_score']

            raw_p_bbox = _bbox_tuple(p)
            vs.plate_bbox_raw = raw_p_bbox

            if vs.best_plate_bbox == (0, 0, 0, 0):
                vs.best_plate_bbox = raw_p_bbox
            else:
                iou = _bbox_iou(vs.best_plate_bbox, raw_p_bbox)
                cdist = _bbox_center_distance(vs.best_plate_bbox, raw_p_bbox)
                max_jump = lpr_state.bbox_max_center_jump_ratio * max(raw_p_bbox[2], raw_p_bbox[3])
                if iou < lpr_state.bbox_reset_iou * 1.5 or cdist > max_jump * 0.8:
                    vs.best_plate_bbox = raw_p_bbox
                else:
                    vs.best_plate_bbox = _smooth_bbox(vs.best_plate_bbox, raw_p_bbox, lpr_state.bbox_smooth_alpha)

            display_pattern_score = _plate_pattern_score(vs.display_plate_text)
            candidate_pattern_score = _plate_pattern_score(b["text"])
            display_candidate_better = (
                not vs.display_plate_text
                or b["score"] >= vs.display_plate_score
                or candidate_pattern_score > display_pattern_score
            )
            if display_candidate_better:
                vs.best_plate_text_raw = b['text']
                vs.display_plate_text = b['text']
                vs.display_plate_score = b['score']
                vs.ocr_confidence = b['conf']

            stable_text = _stable_plate(track_key, b['text'], b['conf'], p.rect_params.width, p.rect_params.height, b['assoc_score'])
            stable_votes, stable_score = _plate_history_stats(track_key, stable_text) if stable_text else (0, 0.0)

            if b['score'] > vs.best_score:
                vs.best_plate_object_id = p.object_id
                vs.association_method = b['assoc_method']
                vs.association_score = b['assoc_score']
                vs.best_plate_text_raw = b['text']
                vs.ocr_confidence = b['conf']

            vs.last_pts = pts

            if not stable_text or stable_score <= 0.0:
                continue

            current_text = vs.best_plate_text_stable
            text_changed = stable_text != current_text
            candidate_better = _should_replace_stable_text(
                current_text, vs.best_score, vs.best_votes,
                stable_text, stable_score, stable_votes,
            )

            if not candidate_better:
                continue

            if text_changed and current_text:
                vs.plate_text_switches += 1

            vs.best_plate_text_stable = stable_text
            vs.display_plate_text = stable_text
            vs.display_plate_score = max(vs.display_plate_score, stable_score)
            vs.best_votes = stable_votes
            vs.best_score = stable_score if text_changed else max(vs.best_score, stable_score)
            lpr_state.plate_text_seen[track_key] = {
                "text": vs.best_plate_text_stable, "stable": True,
                "score": vs.best_score, "votes": vs.best_votes,
            }

            if vs.best_votes >= lpr_state.ocr_lock_min_votes and vs.best_score >= lpr_state.ocr_lock_min_score:
                lpr_state.locked_plate_ids.add(p.object_id)

            # ── Anti-spam emit decision ───────────────────────────────────────
            if vs.association_method == "none" or vs.vehicle_class == -1:
                continue

            event_key = (vs.source_id, vs.vehicle_tracker_id, stable_text)
            frames_since_event = frame_num - vs.last_event_frame

            if lpr_state.emit_duplicates:
                emit_now = True
            elif event_key in lpr_state.emitted_event_keys:
                emit_now = (
                    lpr_state.event_repeat_cooldown_frames > 0
                    and frames_since_event >= lpr_state.event_repeat_cooldown_frames
                )
            else:
                if not vs.last_emitted_plate_text:
                    emit_now = True
                else:
                    stable_pattern_score = _plate_pattern_score(stable_text)
                    emitted_pattern_score = _plate_pattern_score(vs.last_emitted_plate_text)
                    emit_now = (
                        stable_score >= vs.last_emitted_score + 0.5
                        or stable_pattern_score > emitted_pattern_score
                    )

            if emit_now:
                if lpr_state.event_output_dir and frame_image is not None:
                    p_bgr, _, _, _ = _crop_plate_from_frame(frame_image, vs.best_plate_bbox, 0.0)
                    if p_bgr is not None:
                        fname = f"{sid}_{vs.vehicle_tracker_id}_{frame_num}_plate.jpg"
                        vs.crop_plate_path = os.path.abspath(os.path.join(lpr_state.event_output_dir, fname))
                        cv2.imwrite(vs.crop_plate_path, p_bgr)

                    if vs.vehicle_bbox != (0, 0, 0, 0):
                        v_bgr, _, _, _ = _crop_plate_from_frame(frame_image, vs.vehicle_bbox, 0.0)
                        if v_bgr is not None:
                            fname = f"{sid}_{vs.vehicle_tracker_id}_{frame_num}_vehicle.jpg"
                            vs.crop_vehicle_path = os.path.abspath(os.path.join(lpr_state.event_output_dir, fname))
                            cv2.imwrite(vs.crop_vehicle_path, v_bgr)

                    if lpr_state.save_event_frame:
                        fname = f"{sid}_{vs.vehicle_tracker_id}_{frame_num}_frame.jpg"
                        vs.frame_path = os.path.abspath(os.path.join(lpr_state.event_output_dir, fname))
                        full_bgr = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR) if frame_image.shape[2] == 4 else frame_image
                        cv2.imwrite(vs.frame_path, full_bgr)

                lpr_state.emitted_event_keys.add(event_key)
                vs.last_event_frame = frame_num
                vs.last_emitted_plate_text = vs.best_plate_text_stable
                vs.last_emitted_score = vs.best_score
                _emit_event(vs, frame_num)

        # ── Fallback: plates seen but no valid OCR — still track bbox ─────────
        for track_key, (p_seen, _) in frame_plate_seen.items():
            if track_key not in frame_best_plates:
                vs_s = lpr_state.vehicle_states.get(track_key)
                if vs_s is not None:
                    vs_s.best_plate_object_id = p_seen.object_id
                    raw_p_bbox = _bbox_tuple(p_seen)
                    if vs_s.best_plate_bbox == (0, 0, 0, 0):
                        vs_s.best_plate_bbox = raw_p_bbox
                    else:
                        iou = _bbox_iou(vs_s.best_plate_bbox, raw_p_bbox)
                        cdist = _bbox_center_distance(vs_s.best_plate_bbox, raw_p_bbox)
                        max_jump = lpr_state.bbox_max_center_jump_ratio * max(raw_p_bbox[2], raw_p_bbox[3])
                        if iou < lpr_state.bbox_reset_iou * 1.5 or cdist > max_jump * 0.8:
                            vs_s.best_plate_bbox = raw_p_bbox
                        else:
                            vs_s.best_plate_bbox = _smooth_bbox(vs_s.best_plate_bbox, raw_p_bbox, lpr_state.bbox_smooth_alpha)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
