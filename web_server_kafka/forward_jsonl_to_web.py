#!/usr/bin/env python3
"""
Forward DeepStream LPR JSONL events and crop images to the web dashboard.

This script bridges src/app_lpr_v2.py output:
  outputs/events/events.jsonl + media.*_image_path

to web_server/main.py:
  POST /api/upload with plate_image, vehicle_image, frame_image.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def parse_path_map(items: list[str]) -> list[tuple[str, str]]:
    result = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--path-map must be SRC=DST, got: {item!r}")
        src, dst = item.split("=", 1)
        result.append((src, dst))
    return result


def resolve_path(path: str, path_map: list[tuple[str, str]]) -> str:
    if not path:
        return ""
    if os.path.isfile(path):
        return path
    for src, dst in path_map:
        if path.startswith(src):
            candidate = dst + path[len(src):]
            if os.path.isfile(candidate):
                return candidate
    return path


def replay_existing(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                yield line


def tail_follow(path: Path, poll_interval: float):
    while not path.exists():
        time.sleep(poll_interval)
    with path.open("r", encoding="utf-8") as fh:
        fh.seek(0, 2)
        while True:
            line = fh.readline()
            if line:
                yield line.rstrip("\n")
            else:
                time.sleep(poll_interval)


def bbox_csv(value) -> str:
    if not value:
        return ""
    try:
        return ",".join(str(int(float(v))) for v in value)
    except Exception:
        return ""


def event_to_upload(event: dict, args, path_map: list[tuple[str, str]]):
    plate = event.get("plate") or {}
    vehicle = event.get("vehicle") or {}
    media = event.get("media") or {}

    plate_text = plate.get("text_stable") or plate.get("text_raw") or ""
    source_id = str(event.get("source_id", "0"))
    source_uri = event.get("source_uri") or ""

    data = {
        "event_id": event.get("event_id") or f"ds-{source_id}-{event.get('frame_num', int(time.time()))}",
        "project_name": args.project_name,
        "device_id": args.device_id,
        "camera_id": f"source-{source_id}",
        "camera_name": source_uri or f"RTSP Source {source_id}",
        "event_type": "license_plate_detected",
        "message": f"Nhan dien bien so {plate_text}" if plate_text else "Nhan dien bien so",
        "plate_text": plate_text,
        "confidence": plate.get("ocr_confidence") or "",
        "object_id": plate.get("object_id") or "",
        "track_id": vehicle.get("tracker_id") or "",
        "bbox": bbox_csv(plate.get("bbox")),
        "model_name": "LPRNet",
        "model_version": "DeepStream",
        "raw_metadata": json.dumps(event, ensure_ascii=False),
        "timestamp": event.get("created_at") or "",
    }

    file_fields = {
        "plate_image": media.get("plate_image_path", ""),
        "vehicle_image": media.get("vehicle_image_path", ""),
        "frame_image": media.get("frame_image_path", ""),
    }

    files = {}
    opened = []
    missing = []
    for field, raw_path in file_fields.items():
        path = resolve_path(raw_path, path_map)
        if not path:
            continue
        if not os.path.isfile(path):
            missing.append(path)
            continue
        fh = open(path, "rb")
        opened.append(fh)
        files[field] = (os.path.basename(path), fh, "image/jpeg")

    return data, files, opened, missing


def post_event(event: dict, args, path_map: list[tuple[str, str]]) -> bool:
    if event.get("event_type") != "license_plate_recognized":
        return True

    data, files, opened, missing = event_to_upload(event, args, path_map)
    try:
        resp = requests.post(
            f"{args.host.rstrip('/')}/api/upload",
            data=data,
            files=files,
            headers={"X-API-Key": args.api_key},
            timeout=args.timeout,
        )
    finally:
        for fh in opened:
            fh.close()

    ok = 200 <= resp.status_code < 300
    status = "OK" if ok else "FAIL"
    note = ""
    try:
        body = resp.json()
        note = body.get("status") or body.get("detail") or ""
    except Exception:
        note = resp.text[:120]

    missing_note = f" missing_files={len(missing)}" if missing else ""
    print(f"[{status}] {data['event_id']} -> {resp.status_code} {note}{missing_note}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward DeepStream JSONL events to web_server /api/upload")
    parser.add_argument("--events-jsonl", default="../outputs/events/events.jsonl")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--api-key", default="secret_key_nx_01")
    parser.add_argument("--device-id", default="Jetson-Xavier-NX-01")
    parser.add_argument("--project-name", default="DeepStream-LPR-V2")
    parser.add_argument("--path-map", action="append", default=[], metavar="SRC=DST")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--replay", action="store_true", help="Process existing lines before tailing")
    parser.add_argument("--once", action="store_true", help="Process existing lines and exit")
    args = parser.parse_args()

    try:
        path_map = parse_path_map(args.path_map)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    events_path = Path(args.events_jsonl)
    print(f"[INFO] events-jsonl: {events_path}")
    print(f"[INFO] web upload : {args.host.rstrip('/')}/api/upload")

    processed = 0
    failed = 0

    def handle(line: str):
        nonlocal processed, failed
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[WARN] bad JSON: {exc}", file=sys.stderr)
            return
        if event.get("event_type") != "license_plate_recognized":
            return
        processed += 1
        if not post_event(event, args, path_map):
            failed += 1

    if args.replay or args.once:
        for line in replay_existing(events_path):
            handle(line)

    if not args.once:
        try:
            for line in tail_follow(events_path, args.poll_interval):
                handle(line)
        except KeyboardInterrupt:
            pass

    print(f"[DONE] processed={processed} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
