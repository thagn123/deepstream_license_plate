#!/usr/bin/env python3
import os
import json
import time
import requests
import sys

from confluent_kafka import Consumer, KafkaError

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_MEDIA_TOPIC", "lpr.media.v1")
WEB_API_URL = os.environ.get("WEB_API_URL", "http://localhost:8000/api/upload_json")
API_KEY = os.environ.get("API_KEY", "secret_key_nx_01")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "DeepStream-LPR-V2")
DEVICE_ID = os.environ.get("DEVICE_ID", "Jetson-Xavier-NX-01")

def bbox_csv(value) -> str:
    if not value:
        return ""
    try:
        return ",".join(str(int(float(v))) for v in value)
    except Exception:
        return ""

def process_message(msg_value):
    try:
        data = json.loads(msg_value)
    except Exception as e:
        print(f"[WARN] Failed to parse JSON: {e}")
        return

    # Check if raw_event is present
    raw_event = data.get("raw_event", {})
    if not raw_event:
        print(f"[WARN] No raw_event found in message. Skipping.")
        return

    plate = raw_event.get("plate") or {}
    vehicle = raw_event.get("vehicle") or {}
    plate_text = plate.get("text_stable") or plate.get("text_raw") or ""
    source_id = str(raw_event.get("source_id", "0"))
    source_uri = raw_event.get("source_uri") or ""

    event_id = data.get("event_id") or raw_event.get("event_id") or f"ds-{source_id}-{raw_event.get('frame_num', int(time.time()))}"

    payload = {
        "event_id": event_id,
        "project_name": PROJECT_NAME,
        "device_id": DEVICE_ID,
        "camera_id": f"source-{source_id}",
        "camera_name": source_uri or f"RTSP Source {source_id}",
        "event_type": "license_plate_detected",
        "message": f"Nhan dien bien so {plate_text}" if plate_text else "Nhan dien bien so",
        "plate_text": plate_text,
        "confidence": plate.get("ocr_confidence"),
        "object_id": str(plate.get("object_id")) if plate.get("object_id") is not None else None,
        "track_id": str(vehicle.get("tracker_id")) if vehicle.get("tracker_id") is not None else None,
        "bbox": bbox_csv(plate.get("bbox")),
        "model_name": "LPRNet",
        "model_version": "DeepStream",
        "raw_metadata": json.dumps(raw_event, ensure_ascii=False),
        "timestamp": raw_event.get("created_at"),
        "plate_image_url": data.get("plate_image_url"),
        "vehicle_image_url": data.get("vehicle_image_url"),
        "frame_image_url": data.get("frame_image_url"),
    }

    try:
        resp = requests.post(
            WEB_API_URL,
            json=payload,
            headers={"X-API-Key": API_KEY},
            timeout=10.0
        )
        ok = 200 <= resp.status_code < 300
        status = "OK" if ok else "FAIL"
        note = ""
        try:
            body = resp.json()
            note = body.get("status") or body.get("detail") or ""
        except Exception:
            note = resp.text[:120]
        
        print(f"[{status}] {event_id} -> {resp.status_code} {note}")
    except Exception as e:
        print(f"[FAIL] {event_id} -> exception: {e}")

def main():
    conf = {
        'bootstrap.servers': KAFKA_BOOTSTRAP,
        'group.id': 'test-group-4',
        'auto.offset.reset': 'earliest'
    }

    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_TOPIC])

    print(f"[INFO] Started Kafka Consumer connecting to {KAFKA_BOOTSTRAP}")
    print(f"[INFO] Subscribed to topic: {KAFKA_TOPIC}")
    print(f"[INFO] Web API URL: {WEB_API_URL}")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(msg.error())
                    break
            
            val = msg.value().decode('utf-8')
            process_message(val)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
