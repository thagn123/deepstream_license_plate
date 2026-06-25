import sys
import json
from lpr import state as lpr_state


def _debug_event(event_type: str, payload: dict):
    if not lpr_state.debug_jsonl_path:
        return
    try:
        row = dict(payload)
        row["event"] = event_type
        with open(lpr_state.debug_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _build_lpr_event(vs, frame_num: int) -> dict:
    import datetime
    return {
        "event_id": f"{vs.source_id}_{vs.vehicle_tracker_id}_{frame_num}",
        "event_type": "license_plate_recognized",
        "schema_version": "1.0",
        "source_id": vs.source_id,
        "source_uri": lpr_state.source_uri_by_id.get(vs.source_id, ""),
        "frame_num": frame_num,
        "pts": vs.last_pts,
        "created_at": datetime.datetime.now().isoformat(),
        "vehicle": {
            "tracker_id": vs.vehicle_tracker_id,
            "display_id": vs.display_id,
            "class_id": vs.vehicle_class,
            "class_name": vs.vehicle_class_name,
            "confidence": round(vs.vehicle_confidence, 4),
            "bbox": list(vs.vehicle_bbox),
        },
        "plate": {
            "object_id": vs.best_plate_object_id,
            "bbox": list(vs.best_plate_bbox),
            "text_raw": vs.best_plate_text_raw,
            "text_stable": vs.best_plate_text_stable,
            "score": round(vs.best_score, 4),
            "votes": int(vs.best_votes),
            "ocr_confidence": round(vs.ocr_confidence, 4),
            "ocr_backend": lpr_state.ocr_backend,
            "stable": bool(vs.best_plate_text_stable),
        },
        "association": {
            "method": vs.association_method,
            "score": round(vs.association_score, 4),
        },
        "media": {
            "plate_image_path": vs.crop_plate_path,
            "vehicle_image_path": vs.crop_vehicle_path,
            "frame_image_path": vs.frame_path,
            "plate_image_url": "",
            "vehicle_image_url": "",
            "frame_image_url": "",
        },
    }


def _kafka_delivery_cb(err, msg):
    if err is not None:
        key_str = msg.key().decode("utf-8", "replace") if msg.key() else ""
        sys.stderr.write(
            f"[WARN] Kafka delivery failed: {err} "
            f"(topic={msg.topic()} key={key_str})\n"
        )


def _emit_event(vs, frame_num: int):
    is_valid_vehicle = vs.association_method != "none" and vs.vehicle_class != -1

    if not is_valid_vehicle and lpr_state.debug_jsonl_path:
        try:
            with open(lpr_state.debug_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "event": "unassociated_plate",
                    "source_id": vs.source_id,
                    "frame_num": frame_num,
                    "plate_object_id": vs.best_plate_object_id,
                    "plate_bbox": list(vs.best_plate_bbox),
                    "raw_text": vs.best_plate_text_raw,
                }) + "\n")
        except Exception:
            pass

    if not is_valid_vehicle:
        return
    if not lpr_state.event_jsonl_path and not lpr_state.kafka_enabled:
        return

    event = _build_lpr_event(vs, frame_num)
    event_json = json.dumps(event, ensure_ascii=False)

    if lpr_state.event_jsonl_path:
        try:
            with open(lpr_state.event_jsonl_path, "a", encoding="utf-8") as f:
                f.write(event_json + "\n")
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to write event JSONL: {e}\n")

    if lpr_state.kafka_enabled and lpr_state.kafka_producer is not None:
        try:
            kafka_key = f"{vs.source_id}:{vs.vehicle_tracker_id}".encode("utf-8")
            lpr_state.kafka_producer.produce(
                lpr_state.kafka_topic,
                key=kafka_key,
                value=event_json.encode("utf-8"),
                on_delivery=_kafka_delivery_cb,
            )
            lpr_state.kafka_producer.poll(0)
        except Exception as e:
            sys.stderr.write(f"[WARN] Kafka produce failed (event still saved locally): {e}\n")
