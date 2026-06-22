# AI Implementation Prompt: Edge Event Monitoring Dashboard

Use this prompt to instruct an AI assistant to implement Phase 1 and Phase 2 of the Edge Event Monitoring Dashboard.

---

## Task Overview
Implement a lightweight, highly visual, real-time Web Server and Frontend Dashboard designed to receive, store, and display event messages and cropping images sent from edge devices (e.g., NVIDIA Jetson running DeepStream pipelines).

---

## Technical Stack & Directory Structure
Implement the application in Python using **FastAPI** for the backend, **SQLite** with **SQLAlchemy** for the database, and **Pillow** for image processing. The frontend must be built using **Vanilla HTML, CSS (Premium Dark Mode), and Javascript (WebSockets)** without external JS frameworks (like React or Vue) to keep it simple, lightweight, and fast.

Create the project in a directory named `web_server` with the following structure:
```text
web_server/
├── requirements.txt
├── database.py         # SQLAlchemy configuration & Session
├── models.py           # SQLAlchemy Database models (devices, events, event_images)
├── main.py             # FastAPI App, API Routes, WebSocket Manager, Background Tasks
├── test_client.py      # Simulator script to send events & heartbeats
├── static/
│   ├── index.html      # Main Dashboard layout
│   ├── css/
│   │   └── styles.css  # Premium Dark Mode stylesheet
│   └── js/
│       └── app.js      # WebSocket connection, Live Feed management, search & filters
└── data/               # SQLite database location (mounted volume)
```

---

## Component Details

### 1. Database Schema (`models.py` & `database.py`)
Use SQLite. Implement three tables with foreign keys and the following fields:
*   **`devices`**:
    *   `id`: Integer (PK)
    *   `device_id`: String (Indexed)
    *   `project_name`: String (Indexed)
    *   `hostname`: String (Nullable)
    *   `ip_address`: String (Nullable)
    *   `last_fps`: Float (Default: 0.0)
    *   `last_gpu_temp`: Float (Default: 0.0)
    *   `last_seen_at`: DateTime
    *   `created_at`: DateTime (Default: current timestamp)
    *   *Constraint*: `UNIQUE(device_id, project_name)`
*   **`events`**:
    *   `id`: Integer (PK)
    *   `event_id`: String (Unique, Indexed) -> Used to deduplicate incoming events.
    *   `project_name`: String
    *   `device_id`: String
    *   `camera_id`: String
    *   `camera_name`: String
    *   `event_type`: String (e.g., `license_plate_detected`, `vehicle_detected`, `error`)
    *   `message`: String
    *   `plate_text`: String (Nullable)
    *   `confidence`: Float (Nullable)
    *   `object_id`: Integer (Nullable)
    *   `track_id`: Integer (Nullable)
    *   `bbox`: String (Nullable, format: "xmin,ymin,xmax,ymax")
    *   `fps`: Float (Nullable)
    *   `model_name`: String (Nullable)
    *   `model_version`: String (Nullable)
    *   `raw_metadata`: String (JSON text, Nullable)
    *   `timestamp`: DateTime (Indexed, DESC)
    *   `created_at`: DateTime (Default: current timestamp)
*   **`event_images`**:
    *   `id`: Integer (PK)
    *   `event_id`: String (FK referencing `events.event_id`)
    *   `image_type`: String (e.g., `plate`, `vehicle`, `frame`)
    *   `image_url`: String (HTTP path for the dashboard)
    *   `thumb_url`: String (HTTP path for the dashboard thumbnail)
    *   `file_path`: String (Physical path on disk)
    *   `is_available`: Boolean (Default: True)
    *   `deleted_at`: DateTime (Nullable)
    *   `created_at`: DateTime (Default: current timestamp)

**Indexes**:
Create the following indexes to ensure high performance under continuous loads:
*   `idx_events_timestamp` on `events(timestamp DESC)`
*   `idx_events_search` on `events(project_name, device_id, timestamp DESC)`
*   `idx_events_plate` on `events(plate_text)`
*   `idx_event_images_event_id` on `event_images(event_id)`
*   `idx_devices_last_seen` on `devices(last_seen_at)`

---

### 2. FastAPI Backend (`main.py`)

#### Security (API Key Authentication)
Implement authentication for the ingest endpoints (`/api/upload` and `/api/heartbeat`). Read `API_KEY_WHITELIST` environment variable (format: `device_id1=key1,device_id2=key2`). If the incoming request has a missing or invalid header `X-API-Key`, return `401 Unauthorized`.

#### Image Uploading & Thumbnail Processing
In the `POST /api/upload` endpoint:
*   Read files `plate_image`, `vehicle_image`, and `frame_image`.
*   Save the originals under `uploads/originals/` directory.
*   Generate a thumbnail for each image using **Pillow** with a maximum width of 300px, keeping the aspect ratio, and compressed at 80% JPEG quality. Save it under `uploads/thumbs/`.
*   De-duplicate based on `event_id`. If `event_id` already exists in database, skip database insertion and return `200 OK` (to prevent duplicate events due to networking retries).

#### Endpoints
*   `POST /api/upload`: Receives event data and image uploads. Broadcasts a standardized `event_created` JSON to all WebSocket clients.
*   `POST /api/heartbeat`: Receives device stats (`device_id`, `project_name`, `hostname`, `ip_address`, `fps`, `gpu_temp`). Upserts the `devices` table, and broadcasts a `device_heartbeat` JSON.
*   `GET /api/events`: Fetches events supporting limit, offset, and filters (`project_name`, `device_id`, `camera_id`, `event_type`, `plate_text`, and date range `from_time`/`to_time`).
*   `GET /api/events/{event_id}`: Returns full details of a specific event (including `raw_metadata`).
*   `GET /api/devices`: Fetches list of devices. Calculates online status: if `now - last_seen_at <= HEARTBEAT_TIMEOUT_SECONDS` (default: 60s), status is "online", otherwise "offline".
*   `GET /api/stats`: Returns daily statistics (total events, online devices, event count per device).
*   `GET /health`: Basic health check.
*   `WS /ws`: WebSocket endpoint. Implement a simple WebSocket Connection Manager to broadcast events to all connected clients.

#### WebSocket Message Standards
Broadcast messages in this JSON structure:
`{"type": "<message_type>", "data": <payload>}`
*   `type` can be `event_created`, `device_heartbeat`, or `device_offline`.

#### Background Retention Task
Run an hourly async background task using FastAPI's startup event context:
*   Query `event_images` where `created_at` is older than `RETENTION_DAYS` (default: 7 days) and `is_available` is True.
*   Delete the original image files from disk to save space. Leave thumbnails intact.
*   Update database record: set `is_available = False` and `deleted_at = datetime.now()`.

---

### 3. Frontend Dashboard (`static/index.html`, `static/css/styles.css`, `static/js/app.js`)
Create a responsive, modern, dark-themed UI that looks premium and high-tech (use dark glassmorphism effects, harmonized emerald/indigo/rose accent colors, and nice transitions).

#### Layout Structure
*   **Header**: Shows application title, current time, and a summary status bar (total events, active/total devices).
*   **Sidebar / Left Panel**: Displays the list of monitored Edge Devices showing `hostname`, `project_name`, `ip_address`, `fps`, `gpu_temp` (with a colored temperature indicator), and a pulsing green indicator for "online" or a gray indicator for "offline".
*   **Main Center Panel (Live Event Feed)**:
    *   Displays cards of incoming events from the WebSocket in real time.
    *   Cards should slide or fade in.
    *   **Color-coding**: Draw a left border or color indicator on event cards depending on `event_type`:
        *   `license_plate_detected` -> Emerald green (`#10b981`)
        *   `vehicle_detected` -> Indigo purple (`#6366f1`)
        *   `error` -> Rose red (`#f43f5e`)
        *   Other -> Slate gray (`#64748b`)
    *   Each card displays: Time, Device, Camera, Event Type, Message, Plate Text (if available, styled like a mini license plate), and the thumbnail image.
*   **Top Bar controls**:
    *   Search/Filter form: Filter by Project, Device, Plate Text, Event Type.
    *   Feed Control buttons:
        *   `Pause Feed`: Temporarily stops adding new events to the DOM (buffers them in memory or pauses rendering so the user can read).
        *   `Resume Feed`: Resumes rendering and appends any buffered events.
        *   `Clear View`: Clears the current feed display.
*   **Event Detail Drawer (Right side)**:
    *   Sliding drawer (transitions from the right edge) when clicking on an event card.
    *   Displays:
        *   Large view of the cropped image (with a fallback message *"Ảnh gốc đã được dọn dẹp"* if `is_available` is False).
        *   Full metadata table.
        *   **Raw JSON**: Pretty-printed JSON block of the `raw_metadata` field in a scrollable, styled `<pre><code>` block for developers to debug DeepStream payloads.

---

### 4. Simulator Script (`test_client.py`)
Write a standalone Python script to simulate edge devices.
*   It should read a local image (or generate a placeholder image using Pillow with random text on it).
*   Periodically (every 2 to 5 seconds):
    *   Send heartbeats to `/api/heartbeat` with changing CPU/GPU temps and FPS.
    *   Generate random LPR events (e.g., license plates like "30F-123.45", "51G-999.99" with random confidence and metadata).
    *   Upload images to `/api/upload` with correct header `X-API-Key`.
*   Support command-line arguments: `--host` (default: `http://localhost:8000`), `--api-key` (default: `secret_key_nx_01`), `--device-id` (default: `Jetson-Xavier-NX-01`).

---

## Docker & Compose Setup
Write a `Dockerfile` and a `docker-compose.yml` to launch the server:
*   Set environment variables: `DATABASE_URL=sqlite:////app/data/events.db`, `UPLOAD_DIR=/app/uploads`, `API_KEY_WHITELIST=Jetson-Xavier-NX-01=secret_key_nx_01`, `RETENTION_DAYS=7`, `HEARTBEAT_TIMEOUT_SECONDS=60`.
*   Define volume mounts for database data (`/app/data`) and uploads (`/app/uploads`).
*   Ensure directories are automatically created if they don't exist.

---

## Instructions for implementation
Please write all the files outlined above. Ensure you follow clean coding practices, handle file uploads asynchronously in FastAPI without blocking, write robust SQLite query execution, and design an exceptionally polished Frontend UI.
