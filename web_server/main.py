import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import (
    Depends, FastAPI, File, Form, Header, HTTPException, Query,
    UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database import get_db, init_db
from models import Device, Event, EventImage

# ── Environment ───────────────────────────────────────────────────────────────
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "7"))
HEARTBEAT_TIMEOUT = int(os.environ.get("HEARTBEAT_TIMEOUT_SECONDS", "60"))
THUMB_MAX_WIDTH = 300
THUMB_QUALITY = 80

_raw_whitelist = os.environ.get("API_KEY_WHITELIST", "Jetson-Xavier-NX-01=secret_key_nx_01")
API_KEY_WHITELIST: dict[str, str] = {}
for pair in _raw_whitelist.split(","):
    pair = pair.strip()
    if "=" in pair:
        dev, key = pair.split("=", 1)
        API_KEY_WHITELIST[dev.strip()] = key.strip()

ORIGINALS_DIR = UPLOAD_DIR / "originals"
THUMBS_DIR = UPLOAD_DIR / "thumbs"

# Create at import time so StaticFiles mount doesn't fail
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
THUMBS_DIR.mkdir(parents=True, exist_ok=True)


# ── WebSocket Manager ─────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, message: dict):
        payload = json.dumps(message, ensure_ascii=False, default=str)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


manager = ConnectionManager()


# ── Background retention task ─────────────────────────────────────────────────
async def _retention_loop():
    from database import AsyncSessionLocal
    while True:
        await asyncio.sleep(3600)
        cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(EventImage).where(
                        and_(
                            EventImage.is_available == True,
                            EventImage.created_at < cutoff,
                        )
                    )
                )
                images = result.scalars().all()
                for img in images:
                    if img.file_path and Path(img.file_path).exists():
                        try:
                            Path(img.file_path).unlink()
                        except OSError:
                            pass
                    img.is_available = False
                    img.deleted_at = datetime.now(timezone.utc)
                await db.commit()
        except Exception as exc:
            print(f"[retention] error: {exc}")


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()
    task = asyncio.create_task(_retention_loop())
    yield
    task.cancel()


app = FastAPI(title="Edge Event Monitor", lifespan=lifespan)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Auth helper ───────────────────────────────────────────────────────────────
def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if not x_api_key or x_api_key not in API_KEY_WHITELIST.values():
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


# ── Image helper ──────────────────────────────────────────────────────────────
def _save_image(data: bytes, fname: str) -> tuple[str, str, str]:
    uid = uuid.uuid4().hex
    ext = Path(fname).suffix or ".jpg"
    orig_name = f"{uid}{ext}"
    orig_path = ORIGINALS_DIR / orig_name
    orig_path.write_bytes(data)

    thumb_name = f"{uid}_thumb.jpg"
    thumb_path = THUMBS_DIR / thumb_name

    try:
        with Image.open(orig_path) as im:
            ratio = THUMB_MAX_WIDTH / im.width if im.width > THUMB_MAX_WIDTH else 1
            new_size = (int(im.width * ratio), int(im.height * ratio))
            thumb = im.resize(new_size, Image.LANCZOS)
            thumb = thumb.convert("RGB")
            thumb.save(thumb_path, "JPEG", quality=THUMB_QUALITY)
    except Exception:
        thumb_path.write_bytes(data)

    return (
        f"/uploads/originals/{orig_name}",
        f"/uploads/thumbs/{thumb_name}",
        str(orig_path),
    )


# ── POST /api/upload ──────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_event(
    event_id: str = Form(...),
    project_name: str = Form(...),
    device_id: str = Form(...),
    camera_id: Optional[str] = Form(default=None),
    camera_name: Optional[str] = Form(default=None),
    event_type: str = Form(...),
    message: Optional[str] = Form(default=None),
    plate_text: Optional[str] = Form(default=None),
    confidence: Optional[float] = Form(default=None),
    object_id: Optional[int] = Form(default=None),
    track_id: Optional[int] = Form(default=None),
    bbox: Optional[str] = Form(default=None),
    fps: Optional[float] = Form(default=None),
    model_name: Optional[str] = Form(default=None),
    model_version: Optional[str] = Form(default=None),
    raw_metadata: Optional[str] = Form(default=None),
    timestamp: Optional[str] = Form(default=None),
    plate_image: Optional[UploadFile] = File(default=None),
    vehicle_image: Optional[UploadFile] = File(default=None),
    frame_image: Optional[UploadFile] = File(default=None),
    _key: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(select(Event).where(Event.event_id == event_id))
    if existing.scalar_one_or_none():
        return JSONResponse({"status": "duplicate", "event_id": event_id})

    ts = datetime.now(timezone.utc)
    if timestamp:
        try:
            ts = datetime.fromisoformat(timestamp)
        except ValueError:
            pass

    event = Event(
        event_id=event_id,
        project_name=project_name,
        device_id=device_id,
        camera_id=camera_id,
        camera_name=camera_name,
        event_type=event_type,
        message=message,
        plate_text=plate_text,
        confidence=confidence,
        object_id=object_id,
        track_id=track_id,
        bbox=bbox,
        fps=fps,
        model_name=model_name,
        model_version=model_version,
        raw_metadata=raw_metadata,
        timestamp=ts,
        created_at=datetime.now(timezone.utc),
    )
    db.add(event)
    await db.flush()

    ws_images: dict = {}
    for file_field, img_type in [
        (plate_image, "plate"),
        (vehicle_image, "vehicle"),
        (frame_image, "frame"),
    ]:
        if file_field and file_field.filename:
            data = await file_field.read()
            if data:
                img_url, thumb_url, file_path = _save_image(data, file_field.filename)
                ei = EventImage(
                    event_id=event_id,
                    image_type=img_type,
                    image_url=img_url,
                    thumb_url=thumb_url,
                    file_path=file_path,
                    is_available=True,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(ei)
                ws_images[f"{img_type}_image_url"] = thumb_url

    await db.commit()

    await manager.broadcast({
        "type": "event_created",
        "data": {
            "event_id": event_id,
            "project_name": project_name,
            "device_id": device_id,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "event_type": event_type,
            "message": message,
            "plate_text": plate_text,
            "confidence": confidence,
            "timestamp": ts.isoformat(),
            "images": ws_images,
        },
    })

    return JSONResponse({"status": "created", "event_id": event_id})


# ── POST /api/heartbeat ───────────────────────────────────────────────────────
@app.post("/api/heartbeat")
async def heartbeat(
    payload: dict,
    _key: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
):
    device_id = payload.get("device_id", "")
    project_name = payload.get("project_name", "")
    hostname = payload.get("hostname")
    ip_address = payload.get("ip_address")
    fps = float(payload.get("fps", 0.0))
    gpu_temp = float(payload.get("gpu_temp", 0.0))

    now = datetime.now(timezone.utc)

    result = await db.execute(
        select(Device).where(
            and_(Device.device_id == device_id, Device.project_name == project_name)
        )
    )
    device = result.scalar_one_or_none()
    if device is None:
        device = Device(
            device_id=device_id,
            project_name=project_name,
            hostname=hostname,
            ip_address=ip_address,
            last_fps=fps,
            last_gpu_temp=gpu_temp,
            last_seen_at=now,
            created_at=now,
        )
        db.add(device)
    else:
        device.hostname = hostname or device.hostname
        device.ip_address = ip_address or device.ip_address
        device.last_fps = fps
        device.last_gpu_temp = gpu_temp
        device.last_seen_at = now

    await db.commit()

    await manager.broadcast({
        "type": "device_heartbeat",
        "data": {
            "device_id": device_id,
            "project_name": project_name,
            "hostname": hostname,
            "ip_address": ip_address,
            "status": "online",
            "fps": fps,
            "gpu_temp": gpu_temp,
            "last_seen_at": now.isoformat(),
        },
    })

    return {"status": "ok"}


# ── GET /api/events ───────────────────────────────────────────────────────────
@app.get("/api/events")
async def list_events(
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0),
    project_name: Optional[str] = Query(default=None),
    device_id: Optional[str] = Query(default=None),
    camera_id: Optional[str] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
    plate_text: Optional[str] = Query(default=None),
    from_time: Optional[str] = Query(default=None),
    to_time: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    q = select(Event).options(selectinload(Event.images)).order_by(desc(Event.timestamp))

    if project_name:
        q = q.where(Event.project_name == project_name)
    if device_id:
        q = q.where(Event.device_id == device_id)
    if camera_id:
        q = q.where(Event.camera_id == camera_id)
    if event_type:
        q = q.where(Event.event_type == event_type)
    if plate_text:
        q = q.where(Event.plate_text.ilike(f"%{plate_text}%"))
    if from_time:
        try:
            q = q.where(Event.timestamp >= datetime.fromisoformat(from_time))
        except ValueError:
            pass
    if to_time:
        try:
            q = q.where(Event.timestamp <= datetime.fromisoformat(to_time))
        except ValueError:
            pass

    total_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(total_q)).scalar_one()

    result = await db.execute(q.offset(offset).limit(limit))
    events = result.scalars().all()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [_event_to_dict(e) for e in events],
    }


# ── GET /api/events/{event_id} ────────────────────────────────────────────────
@app.get("/api/events/{event_id}")
async def get_event(event_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Event).options(selectinload(Event.images)).where(Event.event_id == event_id)
    )
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return _event_to_dict(event, include_raw=True)


# ── GET /api/devices ──────────────────────────────────────────────────────────
@app.get("/api/devices")
async def list_devices(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Device).order_by(desc(Device.last_seen_at)))
    devices = result.scalars().all()
    now = datetime.now(timezone.utc)
    return [_device_to_dict(d, now) for d in devices]


# ── GET /api/stats ────────────────────────────────────────────────────────────
@app.get("/api/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    total_events = (await db.execute(select(func.count(Event.id)))).scalar_one()

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=HEARTBEAT_TIMEOUT)
    devices_result = await db.execute(select(Device))
    devices = devices_result.scalars().all()
    online = sum(1 for d in devices if d.last_seen_at.replace(tzinfo=timezone.utc) >= cutoff)

    per_device = await db.execute(
        select(Event.device_id, func.count(Event.id).label("count"))
        .group_by(Event.device_id)
        .order_by(desc("count"))
    )

    return {
        "total_events": total_events,
        "total_devices": len(devices),
        "online_devices": online,
        "events_per_device": [
            {"device_id": row.device_id, "count": row.count}
            for row in per_device
        ],
    }


# ── GET /api/projects ─────────────────────────────────────────────────────────
@app.get("/api/projects")
async def list_projects(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Event.project_name).distinct().order_by(Event.project_name)
    )
    return [row[0] for row in result]


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── WS /ws ────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Serve index.html ──────────────────────────────────────────────────────────
from fastapi.responses import FileResponse


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _event_to_dict(event: Event, include_raw: bool = False) -> dict:
    images = []
    for img in (event.images or []):
        images.append({
            "id": img.id,
            "image_type": img.image_type,
            "image_url": img.image_url,
            "thumb_url": img.thumb_url,
            "is_available": img.is_available,
            "deleted_at": img.deleted_at.isoformat() if img.deleted_at else None,
        })
    d = {
        "event_id": event.event_id,
        "project_name": event.project_name,
        "device_id": event.device_id,
        "camera_id": event.camera_id,
        "camera_name": event.camera_name,
        "event_type": event.event_type,
        "message": event.message,
        "plate_text": event.plate_text,
        "confidence": event.confidence,
        "object_id": event.object_id,
        "track_id": event.track_id,
        "bbox": event.bbox,
        "fps": event.fps,
        "model_name": event.model_name,
        "model_version": event.model_version,
        "timestamp": event.timestamp.isoformat() if event.timestamp else None,
        "created_at": event.created_at.isoformat() if event.created_at else None,
        "images": images,
    }
    if include_raw:
        d["raw_metadata"] = event.raw_metadata
    return d


def _device_to_dict(device: Device, now: datetime) -> dict:
    last_seen = device.last_seen_at
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)
    online = (now - last_seen).total_seconds() <= HEARTBEAT_TIMEOUT
    return {
        "id": device.id,
        "device_id": device.device_id,
        "project_name": device.project_name,
        "hostname": device.hostname,
        "ip_address": device.ip_address,
        "last_fps": device.last_fps,
        "last_gpu_temp": device.last_gpu_temp,
        "last_seen_at": last_seen.isoformat(),
        "created_at": device.created_at.isoformat() if device.created_at else None,
        "status": "online" if online else "offline",
    }
