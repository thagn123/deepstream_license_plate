#!/usr/bin/env python3
"""
test_client.py — Simulates edge devices sending heartbeats and LPR events.

Usage:
  python3 test_client.py
  python3 test_client.py --host http://localhost:8000 --api-key secret_key_nx_01 --device-id Jetson-Xavier-NX-01
"""

import argparse
import io
import json
import random
import string
import time
import uuid
from datetime import datetime, timezone

import requests
from PIL import Image, ImageDraw, ImageFont

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--host", default="http://localhost:8000")
parser.add_argument("--api-key", default="secret_key_nx_01")
parser.add_argument("--device-id", default="Jetson-Xavier-NX-01")
parser.add_argument("--project", default="DeepStream-LPR-V2")
parser.add_argument("--interval", type=float, default=3.0)
args = parser.parse_args()

HEADERS = {"X-API-Key": args.api_key}
CAMERAS = [
    ("cam-01", "Gate A - Entrance"),
    ("cam-02", "Gate B - Exit"),
    ("cam-03", "Parking Lot North"),
]
EVENT_TYPES = ["license_plate_detected", "vehicle_detected", "error"]
PLATE_PREFIXES = ["30F", "51G", "29A", "36B", "43C", "92L", "77H"]


def _random_plate() -> str:
    prefix = random.choice(PLATE_PREFIXES)
    nums = "".join(random.choices(string.digits, k=3))
    suf = "".join(random.choices(string.digits, k=2))
    return f"{prefix}-{nums}.{suf}"


def _get_font(size: int):
    font_paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "arial.ttf"
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _make_plate_image(plate: str) -> bytes:
    # 320x80 white plate with blue top bar and thick border
    img = Image.new("RGB", (320, 80), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Black border
    draw.rectangle([(0, 0), (319, 79)], outline=(0, 0, 0), width=4)
    # Blue top line
    draw.rectangle([(4, 4), (315, 12)], fill=(0, 102, 204))
    # Big text
    font = _get_font(32)
    draw.text((160, 48), plate, fill=(0, 0, 0), anchor="mm", font=font)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _make_vehicle_image(track_id: int) -> bytes:
    # 640x360 nice sky blue background with a stylized red car drawing
    img = Image.new("RGB", (640, 360), color=(224, 242, 254))
    draw = ImageDraw.Draw(img)
    # Draw red car body
    draw.rectangle([(160, 150), (480, 250)], fill=(239, 68, 68), outline=(0, 0, 0), width=3)
    # Draw car top cabin
    draw.rectangle([(220, 90), (420, 150)], fill=(191, 219, 254), outline=(0, 0, 0), width=3)
    # Draw black wheels
    draw.ellipse([(200, 230), (260, 290)], fill=(31, 41, 55))
    draw.ellipse([(380, 230), (440, 290)], fill=(31, 41, 55))
    # Text
    font = _get_font(24)
    draw.text((320, 40), f"VEHICLE TRACK #{track_id}", fill=(17, 24, 39), anchor="mm", font=font)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _make_frame_image(cam_name: str, plate: str) -> bytes:
    # 1280x720 scene showing gray road, blue sky, green grass, and highlighted boxes
    img = Image.new("RGB", (1280, 720), color=(209, 213, 219)) # Gray pavement
    draw = ImageDraw.Draw(img)
    # Blue sky
    draw.rectangle([(0, 0), (1279, 180)], fill=(186, 230, 253))
    # Green grass
    draw.rectangle([(0, 180), (1279, 320)], fill=(187, 247, 208))
    # Yellow lane divider
    draw.line([(0, 520), (1280, 520)], fill=(253, 224, 71), width=10)
    # Car box
    draw.rectangle([(300, 260), (980, 640)], outline=(99, 102, 241), width=8) # Indigo vehicle bbox
    # Plate box
    draw.rectangle([(540, 520), (740, 580)], fill=(255, 255, 255), outline=(16, 185, 129), width=5) # Emerald plate bbox
    # Text overlay
    font = _get_font(40)
    draw.text((640, 80), f"CAMERA: {cam_name.upper()}", fill=(30, 41, 59), anchor="mm", font=font)
    draw.text((640, 140), f"PLATE DETECTED: {plate}", fill=(16, 185, 129), anchor="mm", font=font)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def send_heartbeat(fps: float, gpu_temp: float):
    payload = {
        "device_id": args.device_id,
        "project_name": args.project,
        "hostname": f"{args.device_id.lower()}.local",
        "ip_address": f"192.168.1.{random.randint(10, 50)}",
        "fps": round(fps, 2),
        "gpu_temp": round(gpu_temp, 1),
    }
    try:
        r = requests.post(f"{args.host}/api/heartbeat", json=payload, headers=HEADERS, timeout=5)
        print(f"[HB] fps={fps:.1f} temp={gpu_temp:.1f}°C -> {r.status_code}")
    except Exception as e:
        print(f"[HB] error: {e}")


def send_lpr_event(cam_id: str, cam_name: str, plate: str, confidence: float):
    event_id = f"{args.device_id}-{cam_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    track_id = random.randint(1, 9999)

    plate_img = _make_plate_image(plate)
    vehicle_img = _make_vehicle_image(track_id)
    frame_img = _make_frame_image(cam_name, plate)

    raw_meta = {
        "stream_id": cam_id,
        "frame_number": random.randint(1000, 99999),
        "ntp_timestamp": time.time_ns(),
        "objects": [
            {
                "track_id": track_id,
                "class_id": 2,
                "class_label": "car",
                "bbox": [random.randint(100, 400)] * 4,
                "license_plate": {
                    "text": plate,
                    "confidence": confidence,
                    "ocr_method": "LPRNet",
                },
            }
        ],
    }

    files = {
        "plate_image": (f"plate_{uuid.uuid4().hex[:8]}.jpg", plate_img, "image/jpeg"),
        "vehicle_image": (f"vehicle_{uuid.uuid4().hex[:8]}.jpg", vehicle_img, "image/jpeg"),
        "frame_image": (f"frame_{uuid.uuid4().hex[:8]}.jpg", frame_img, "image/jpeg"),
    }
    data = {
        "event_id": event_id,
        "project_name": args.project,
        "device_id": args.device_id,
        "camera_id": cam_id,
        "camera_name": cam_name,
        "event_type": "license_plate_detected",
        "message": f"Đã nhận diện biển số {plate}",
        "plate_text": plate,
        "confidence": confidence,
        "track_id": track_id,
        "model_name": "LPRNet-Vietnamese",
        "model_version": "v2.1",
        "raw_metadata": json.dumps(raw_meta, ensure_ascii=False),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        r = requests.post(f"{args.host}/api/upload", data=data, files=files, headers=HEADERS, timeout=10)
        print(f"[LPR] {plate} cam={cam_id} conf={confidence:.2f} -> {r.status_code} {r.json().get('status', '')}")
    except Exception as e:
        print(f"[LPR] error: {e}")


def send_error_event(cam_id: str, cam_name: str):
    event_id = f"{args.device_id}-{cam_id}-ERR-{uuid.uuid4().hex[:8]}"
    data = {
        "event_id": event_id,
        "project_name": args.project,
        "device_id": args.device_id,
        "camera_id": cam_id,
        "camera_name": cam_name,
        "event_type": "error",
        "message": random.choice([
            "Pipeline stalled: no frames received for 10s",
            "GPU memory warning: 85% utilization",
            "RTSP stream reconnecting...",
            "NvInfer: inference timeout on batch",
        ]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        r = requests.post(f"{args.host}/api/upload", data=data, headers=HEADERS, timeout=5)
        print(f"[ERR] {cam_name} -> {r.status_code}")
    except Exception as e:
        print(f"[ERR] error: {e}")


# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"[SIM] Starting simulator → {args.host}")
print(f"[SIM] Device: {args.device_id} | Project: {args.project}")
print(f"[SIM] Interval: {args.interval}s")

tick = 0
fps_base = 28.0
temp_base = 55.0

while True:
    tick += 1
    fps = max(10.0, fps_base + random.uniform(-3, 3))
    temp = max(40.0, min(85.0, temp_base + random.uniform(-2, 4)))

    send_heartbeat(fps, temp)

    cam_id, cam_name = random.choice(CAMERAS)
    choice = random.choices(
        ["lpr", "vehicle", "error"],
        weights=[0.7, 0.2, 0.1],
    )[0]

    if choice == "lpr":
        plate = _random_plate()
        confidence = random.uniform(0.72, 0.99)
        send_lpr_event(cam_id, cam_name, plate, confidence)
    elif choice == "error":
        send_error_event(cam_id, cam_name)
    else:
        # vehicle detected (no plate)
        event_id = f"{args.device_id}-{cam_id}-VEH-{uuid.uuid4().hex[:8]}"
        data = {
            "event_id": event_id,
            "project_name": args.project,
            "device_id": args.device_id,
            "camera_id": cam_id,
            "camera_name": cam_name,
            "event_type": "vehicle_detected",
            "message": f"Phát hiện phương tiện tại {cam_name}",
            "track_id": random.randint(1, 9999),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            r = requests.post(f"{args.host}/api/upload", data=data, headers=HEADERS, timeout=5)
            print(f"[VEH] {cam_name} -> {r.status_code}")
        except Exception as e:
            print(f"[VEH] error: {e}")

    delay = args.interval + random.uniform(-1, 2)
    time.sleep(max(0.5, delay))
