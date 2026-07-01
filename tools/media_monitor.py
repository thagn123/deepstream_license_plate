#!/usr/bin/env python3
"""
media_monitor.py — Tail-follow LPR event JSONL, upload media to MinIO/HTTP, publish results.

For each LPR event that has media.*_image_path set:
  - mock   : verify file exists, record status (no upload).
  - minio  : upload to MinIO/S3-compatible object store via --minio-enable.
  - http   : HTTP POST multipart via --upload-url.

Results written to --output-jsonl and optionally published to a Kafka topic.

Usage (mock):
  python3 tools/media_monitor.py \
    --events-jsonl outputs/events/events.jsonl \
    --output-jsonl outputs/events/media_results.jsonl \
    --mock --replay --once

Usage (MinIO + Kafka result):
  python3 tools/media_monitor.py \
    --events-jsonl outputs/events/events.jsonl \
    --output-jsonl outputs/events/media_results.jsonl \
    --minio-enable \
    --minio-endpoint 127.0.0.1:9000 \
    --minio-access-key minioadmin \
    --minio-secret-key minioadmin \
    --minio-bucket lpr-media \
    --minio-prefix lpr \
    --path-map /workspace/last_ds=/home/thagn/projects/deepstream/workspace/last_ds \
    --media-result-kafka-enable \
    --media-result-kafka-topic lpr.media.v1
"""

import sys
import os
import json
import time
import datetime
import argparse


SCHEMA_VERSION = "1.0"


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


# ── Path mapping ──────────────────────────────────────────────────────────────

def _resolve_path(path: str, path_map: list) -> str:
    """Translate a container-side path to a host-side path using --path-map rules.

    path_map is a list of (src_prefix, dst_prefix) tuples tried in order.
    Returns the first mapped path that exists as a file; falls back to original.
    """
    if not path:
        return path
    if os.path.isfile(path):
        return path
    for src, dst in path_map:
        if path.startswith(src):
            candidate = dst + path[len(src):]
            if os.path.isfile(candidate):
                return candidate
    return path


# ── Upload backends ───────────────────────────────────────────────────────────

def _upload_file_mock(event_id: str, field: str, path: str) -> dict:
    exists = os.path.isfile(path)
    return {
        "field": field,
        "path": path,
        "url": "",
        "status": "success" if exists else "failed",
        "error": "" if exists else f"file not found: {path}",
        "retry_count": 0,
    }


def _minio_public_url(endpoint: str, secure: bool, bucket: str, key: str) -> str:
    scheme = "https" if secure else "http"
    return f"{scheme}://{endpoint}/{bucket}/{key}"


def _upload_file_minio(event_id: str, field: str, path: str,
                        minio_client, bucket: str, object_key: str,
                        endpoint: str, secure: bool,
                        public_endpoint: str = "") -> dict:
    try:
        minio_client.fput_object(bucket, object_key, path, content_type="image/jpeg")
        url = _minio_public_url(public_endpoint or endpoint, secure, bucket, object_key)
        return {"field": field, "path": path, "url": url, "status": "success", "error": "", "retry_count": 0}
    except FileNotFoundError:
        return {"field": field, "path": path, "url": "", "status": "failed",
                "error": f"file not found: {path}", "retry_count": 0}
    except Exception as e:
        return {"field": field, "path": path, "url": "", "status": "failed",
                "error": str(e), "retry_count": 0}


def _upload_file_http(event_id: str, field: str, path: str,
                       upload_url: str, timeout: int = 10) -> dict:
    try:
        import urllib.request
        with open(path, "rb") as fh:
            data = fh.read()
        boundary = "----LPRMediaMonitor"
        fname = os.path.basename(path)
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")
        req = urllib.request.Request(
            upload_url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "X-Event-Id": event_id,
                "X-Media-Field": field,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read().decode("utf-8", errors="replace")
        try:
            resp_json = json.loads(resp_body)
            url = resp_json.get("url", resp_json.get("path", ""))
        except Exception:
            url = ""
        return {"field": field, "path": path, "url": url, "status": "success", "error": "", "retry_count": 0}
    except FileNotFoundError:
        return {"field": field, "path": path, "url": "", "status": "failed",
                "error": f"file not found: {path}", "retry_count": 0}
    except Exception as e:
        return {"field": field, "path": path, "url": "", "status": "failed",
                "error": str(e), "retry_count": 0}


def _upload_with_retry(upload_fn, retries: int, retry_delay: float) -> dict:
    """Call upload_fn up to `retries` times; return first success or last failure."""
    last_result = None
    for attempt in range(max(1, retries)):
        result = upload_fn()
        if result["status"] == "success":
            result["retry_count"] = attempt
            return result
        last_result = result
        if attempt < retries - 1:
            time.sleep(retry_delay)
    last_result["retry_count"] = retries - 1
    return last_result


# ── Event processing ──────────────────────────────────────────────────────────

def _process_event(event: dict, args, minio_client=None, path_map=None) -> dict:
    event_id  = event.get("event_id", "")
    source_id = str(event.get("source_id", "0"))
    media     = event.get("media", {})
    path_map  = path_map or []

    # field_path_key → (result_url_key, minio object filename)
    fields = {
        "plate_image_path":   ("plate_image_url",   "plate.jpg"),
        "vehicle_image_path": ("vehicle_image_url",  "vehicle.jpg"),
        "frame_image_path":   ("frame_image_url",    "frame.jpg"),
    }

    result_urls  = {"plate_image_url": "", "vehicle_image_url": "", "frame_image_url": ""}
    upload_results  = []
    max_retry_count = 0

    for path_key, (url_key, obj_suffix) in fields.items():
        raw_path = media.get(path_key, "")
        if not raw_path:
            continue

        path = _resolve_path(raw_path, path_map)
        if path != raw_path:
            print(f"[PATH-MAP] {raw_path}\n          → {path}")

        if args.mock:
            r = _upload_file_mock(event_id, path_key, path)

        elif args.minio_enable and minio_client is not None:
            object_key = f"{args.minio_prefix}/{source_id}/{event_id}/{obj_suffix}"

            def _do_minio(p=path, pk=path_key, ok=object_key):
                return _upload_file_minio(
                    event_id, pk, p, minio_client,
                    args.minio_bucket, ok, args.minio_endpoint, args.minio_secure,
                    public_endpoint=getattr(args, "minio_public_endpoint", "") or "",
                )

            r = _upload_with_retry(_do_minio, args.upload_retries, args.upload_retry_delay)

        elif args.upload_url:
            def _do_http(p=path, pk=path_key):
                return _upload_file_http(event_id, pk, p, args.upload_url)

            r = _upload_with_retry(_do_http, args.upload_retries, args.upload_retry_delay)

        else:
            continue

        upload_results.append(r)
        result_urls[url_key] = r.get("url", "")
        max_retry_count = max(max_retry_count, r.get("retry_count", 0))

    # Determine aggregate status
    if not upload_results:
        overall_status = "pending"
    else:
        n_ok   = sum(1 for r in upload_results if r["status"] == "success")
        n_fail = sum(1 for r in upload_results if r["status"] == "failed")
        if n_fail == 0:
            overall_status = "success"
        elif n_ok == 0:
            overall_status = "failed"
        else:
            overall_status = "partial"

    return {
        "event_id":         event_id,
        "schema_version":   SCHEMA_VERSION,
        "plate_image_url":  result_urls["plate_image_url"],
        "vehicle_image_url": result_urls["vehicle_image_url"],
        "frame_image_url":  result_urls["frame_image_url"],
        "upload_status":    overall_status,
        "retry_count":      max_retry_count,
        "error_message":    "; ".join(r["error"] for r in upload_results if r.get("error")),
        "created_at":       _now_iso(),
        "_uploads":         upload_results,
    }


# ── JSONL tailing ─────────────────────────────────────────────────────────────

def _tail_follow(path: str, poll_interval: float):
    """Yield lines appended to path, blocking forever until Ctrl-C."""
    while not os.path.exists(path):
        time.sleep(poll_interval)
    with open(path, "r", encoding="utf-8") as fh:
        fh.seek(0, 2)  # skip existing content; remove to start from beginning
        while True:
            line = fh.readline()
            if line:
                yield line.rstrip("\n")
            else:
                time.sleep(poll_interval)


def _replay_existing(path: str):
    """Yield all existing lines (for --replay / --once mode)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                yield line


def _write_result(output_jsonl: str, result: dict):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
        with open(output_jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to write media result: {e}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LPR media monitor — tail event JSONL and upload media files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input / output ────────────────────────────────────────────────────────
    parser.add_argument("--events-jsonl",  required=True,
                        help="Path to LPR events JSONL to tail")
    parser.add_argument("--output-jsonl",  required=True,
                        help="Path to write media result records")
    parser.add_argument("--poll-interval", type=float, default=1.0,
                        help="Seconds between file polls in tail mode (default: 1)")
    parser.add_argument("--replay", action="store_true",
                        help="Process existing events from the beginning, then tail")
    parser.add_argument("--once",   action="store_true",
                        help="Process existing events and exit (no tail)")

    # ── Upload mode ───────────────────────────────────────────────────────────
    parser.add_argument("--mock",       action="store_true",
                        help="Mock: verify files exist locally, no upload")
    parser.add_argument("--upload-url", default="",
                        help="HTTP endpoint for multipart file upload")
    parser.add_argument("--upload-retries",     type=int,   default=3,
                        help="Upload attempts per file (default: 3)")
    parser.add_argument("--upload-retry-delay", type=float, default=2.0,
                        help="Seconds between retries (default: 2)")

    # ── MinIO ─────────────────────────────────────────────────────────────────
    parser.add_argument("--minio-enable",     action="store_true",
                        help="Upload files to MinIO / S3-compatible store")
    parser.add_argument("--minio-endpoint",   default="localhost:9000",
                        help="MinIO endpoint host:port for internal connection (default: localhost:9000)")
    parser.add_argument("--minio-public-endpoint", default="",
                        help="MinIO public host:port for generated image URLs (default: same as --minio-endpoint)")
    parser.add_argument("--minio-access-key", default="minioadmin")
    parser.add_argument("--minio-secret-key", default="minioadmin")
    parser.add_argument("--minio-bucket",     default="lpr-media",
                        help="Bucket name (default: lpr-media)")
    parser.add_argument("--minio-secure",     action="store_true",
                        help="Use HTTPS for MinIO (default: HTTP)")
    parser.add_argument("--minio-prefix",     default="lpr",
                        help="Object key prefix (default: lpr). "
                             "Key pattern: {prefix}/{source_id}/{event_id}/plate.jpg")

    # ── Path mapping ──────────────────────────────────────────────────────────
    parser.add_argument("--path-map", action="append", default=[], metavar="SRC=DST",
                        help="Translate path prefix from event to local filesystem. "
                             "Example: --path-map /workspace/last_ds=/home/user/ds "
                             "(repeatable, first match wins)")

    # ── Media result Kafka ────────────────────────────────────────────────────
    parser.add_argument("--media-result-kafka-enable",    action="store_true",
                        help="Publish media_result records to Kafka")
    parser.add_argument("--media-result-kafka-bootstrap", default="localhost:9092",
                        help="Kafka bootstrap server (default: localhost:9092)")
    parser.add_argument("--media-result-kafka-topic",     default="lpr.media.v1",
                        help="Kafka topic for media results (default: lpr.media.v1)")
    parser.add_argument("--media-result-kafka-client-id", default="ds-media-monitor",
                        help="Kafka client ID")

    args = parser.parse_args()

    # Validate: exactly one upload mode required
    if not args.mock and not args.minio_enable and not args.upload_url:
        sys.stderr.write(
            "[ERROR] Specify one upload mode: --mock | --minio-enable | --upload-url\n"
        )
        sys.exit(1)

    # ── Parse path map ────────────────────────────────────────────────────────
    path_map = []
    for pm in args.path_map:
        if "=" not in pm:
            sys.stderr.write(f"[ERROR] --path-map must be SRC=DST, got: {pm!r}\n")
            sys.exit(1)
        src, dst = pm.split("=", 1)
        path_map.append((src, dst))
        print(f"[INFO] Path map: {src!r} → {dst!r}")

    # ── Init MinIO client ─────────────────────────────────────────────────────
    minio_client = None
    if args.minio_enable:
        try:
            from minio import Minio
            minio_client = Minio(
                args.minio_endpoint,
                access_key=args.minio_access_key,
                secret_key=args.minio_secret_key,
                secure=args.minio_secure,
            )
            if not minio_client.bucket_exists(args.minio_bucket):
                minio_client.make_bucket(args.minio_bucket)
                print(f"[INFO] MinIO: bucket created — {args.minio_bucket}")
            else:
                print(f"[INFO] MinIO: bucket exists — {args.minio_bucket}")
            print(
                f"[INFO] MinIO ready: endpoint={args.minio_endpoint} "
                f"bucket={args.minio_bucket} prefix={args.minio_prefix}"
            )
        except ImportError:
            sys.stderr.write(
                "[ERROR] --minio-enable requires the minio package.\n"
                "  Install inside ds90: pip3 install minio\n"
                "  Or on host:          pip install minio\n"
            )
            sys.exit(1)
        except Exception as e:
            sys.stderr.write(f"[ERROR] MinIO init failed: {e}\n")
            sys.exit(1)

    # ── Init media-result Kafka producer ──────────────────────────────────────
    result_kafka_producer = None
    if args.media_result_kafka_enable:
        try:
            from confluent_kafka import Producer as _KP
            result_kafka_producer = _KP({
                "bootstrap.servers": args.media_result_kafka_bootstrap,
                "client.id":         args.media_result_kafka_client_id,
            })
            print(
                f"[INFO] Media-result Kafka: {args.media_result_kafka_bootstrap}"
                f" → topic={args.media_result_kafka_topic}"
            )
        except ImportError:
            sys.stderr.write(
                "[ERROR] --media-result-kafka-enable requires confluent-kafka.\n"
                "  Install: pip3 install confluent-kafka\n"
            )
            sys.exit(1)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Media-result Kafka init failed: {e}\n")
            sys.exit(1)

    # ── Banner ────────────────────────────────────────────────────────────────
    mode_str = ("mock" if args.mock
                else f"minio://{args.minio_endpoint}/{args.minio_bucket}" if args.minio_enable
                else args.upload_url)
    print(f"[INFO] LPR media monitor starting")
    print(f"[INFO]   events-jsonl  : {args.events_jsonl}")
    print(f"[INFO]   output-jsonl  : {args.output_jsonl}")
    print(f"[INFO]   upload mode   : {mode_str}")
    print(f"[INFO]   retries       : {args.upload_retries} × {args.upload_retry_delay}s delay")
    print(f"[INFO]   poll-interval : {args.poll_interval}s")

    # ── Processing ────────────────────────────────────────────────────────────
    processed = 0
    failed    = 0

    def _result_kafka_delivery(err, msg):
        if err is not None:
            key = msg.key().decode("utf-8", "replace") if msg.key() else ""
            sys.stderr.write(f"[WARN] Media-result Kafka delivery failed: {err} (key={key})\n")

    def handle_line(line: str):
        nonlocal processed, failed
        if not line.strip():
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"[WARN] Bad JSON line: {e}\n")
            return

        # Only handle main LPR events; skip debug/summary records
        if event.get("event_type") != "license_plate_recognized":
            return

        result = _process_event(event, args, minio_client, path_map)
        _write_result(args.output_jsonl, result)
        processed += 1
        if result["upload_status"] in ("failed", "partial"):
            failed += 1

        # Publish media result to Kafka (metadata only — no binary)
        if result_kafka_producer is not None:
            try:
                result_kafka_producer.produce(
                    args.media_result_kafka_topic,
                    key=result["event_id"].encode("utf-8"),
                    value=json.dumps(result, ensure_ascii=False).encode("utf-8"),
                    on_delivery=_result_kafka_delivery,
                )
                result_kafka_producer.poll(0)
            except Exception as e:
                sys.stderr.write(f"[WARN] Media-result Kafka produce failed: {e}\n")

        # Human-readable one-liner per event
        summary_parts = []
        for r in result.get("_uploads", []):
            icon  = "✓" if r["status"] == "success" else "✗"
            label = r["field"].replace("_image_path", "")
            tail  = f" → {r['url']}" if r.get("url") else f": {r['path']}"
            if r.get("error"):
                tail += f" [{r['error']}]"
            summary_parts.append(f"{icon} {label}{tail}")
        tag = result["upload_status"].upper()
        print(f"[{tag}] {result['event_id']} | {' | '.join(summary_parts) or 'no media'}")

    if args.replay or args.once:
        for line in _replay_existing(args.events_jsonl):
            handle_line(line)

    if not args.once:
        try:
            for line in _tail_follow(args.events_jsonl, args.poll_interval):
                handle_line(line)
        except KeyboardInterrupt:
            pass

    if result_kafka_producer is not None:
        try:
            remaining = result_kafka_producer.flush(timeout=5)
            if remaining:
                sys.stderr.write(
                    f"[WARN] Media-result Kafka flush: {remaining} msgs still in queue\n"
                )
        except Exception as e:
            sys.stderr.write(f"[WARN] Media-result Kafka flush error: {e}\n")

    print(f"\n[DONE] processed={processed} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
