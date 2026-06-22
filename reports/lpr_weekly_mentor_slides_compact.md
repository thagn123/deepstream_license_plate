---
marp: true
theme: default
paginate: true
size: 16:9
---

# Weekly Report - DeepStream LPR Pipeline

**Muc tieu trong tuan**

Build pipeline nhan dien bien so xe tu video/RTSP, tao metadata co the gui server:

```text
xe nao -> tracker ID nao -> bien so nao -> frame/source nao -> anh crop nao
```

![Pipeline](assets/pipeline_architecture_diagram.jpg)

---

# 1. Tong quan pipeline da build

```text
Video/RTSP
  -> PGIE detect xe + bien
  -> nvtracker gan ID xe
  -> LPRNet OCR bien so
  -> metadata probe truoc tiler
  -> stable vote + chon OCR tot nhat
  -> crop plate/vehicle
  -> JSONL + Kafka metadata
  -> media_monitor upload MinIO
```

**Ket qua chinh**

- Hien thi bbox xe/bien + OCR tren frame.
- Event metadata co `vehicle`, `plate`, `association`, `media`.
- Kafka topic `lpr.events.v1` nhan metadata.
- MinIO luu anh crop plate/vehicle.
- Kafka topic `lpr.media.v1` nhan URL anh.

---

# 2. Build duoc gi trong code

**Core app:** `src/app_lpr_v2.py`

| Thanh phan | Cong dung |
|---|---|
| `VehicleTrackState` | gom metadata theo tung xe |
| `metadata_src_pad_buffer_probe()` | xu ly metadata/event/crop truoc tiler |
| `sgie3_sink_pad_buffer_probe()` | split bien vuong top/bottom truoc OCR |
| `_correct_vn_plate()` | sua OCR theo format bien Viet Nam |
| `_stable_plate()` | vote de lay bien on dinh |
| `_build_lpr_event()` | tao event JSON server-ready |
| `_emit_event()` | ghi JSONL va gui Kafka |

**Media service:** `tools/media_monitor.py`

| Thanh phan | Cong dung |
|---|---|
| `_process_event()` | doc event, upload anh, tao media result |
| `_upload_file_minio()` | upload crop len MinIO |
| `_resolve_path()` | map path container/host |
| `_upload_with_retry()` | retry khi upload loi |

---

# 3. Van de 1 - OCR raw bi nhieu ky tu sai

![Plate crops](assets/plate_crop_contact_sheet.jpg)

**Kho khan ban dau**

LPRNet hay nham cac ky tu giong nhau:

```text
O <-> 0, S <-> 5, B <-> 8, D <-> 0
```

Vi du:

```text
3OH1234S -> 30H12345
```

**Fix**

Nhom ham:

```text
_correct_vn_plate()
_plate_pattern_score()
_plate_quality_score()
```

Logic:

```text
2 ky tu dau: so tinh/thanh
1-2 ky tu tiep: chu series
4-5 ky tu cuoi: so
```

**Ket qua**

Khong cham diem truc tiep text raw nua, ma normalize theo format bien Viet Nam truoc.

---

# 4. Van de 2 - Bien vuong 2 dong OCR sai

![Square plate](assets/annotated_square_plate_osd.jpg)

**Kho khan ban dau**

Bien vuong co bbox nhung OCR:

- mat dong dau,
- doc sai thu tu,
- hoac them ky tu rac giua hai dong.

Vi du:

```text
Sai : 30HT00644
Dung: 30H00644
```

**Fix**

Trong `sgie3_sink_pad_buffer_probe()`:

```text
neu aspect ratio bien < threshold
-> tao pseudo-object dong tren
-> tao pseudo-object dong duoi
-> LPRNet OCR tung dong
-> _square_join_variants() ghep lai
```

**Ket qua**

Bien vuong stream `lpr_230428_003` doc duoc:

```text
30H00644
```

---

# 5. Van de 3 - OCR nhay theo frame va spam event

![OSD scene](assets/frame_osd_square_plate.jpg)

**Kho khan ban dau**

Xe chay nhanh, bien nho, motion blur:

```text
Frame 1: 30L0053
Frame 2: 30L0052
Frame 3: 30L0052
```

Neu gui tung frame len server thi bi spam va nhieu OCR rac.

**Fix**

Nhom ham:

```text
_stable_plate()
_plate_similar_enough()
_should_replace_stable_text()
```

Co che:

- vote theo `(source_id, vehicle_tracker_id)`,
- gom OCR gan giong nhau bang Levenshtein,
- chi thay stable text neu candidate moi thuc su tot hon.

**Chong spam**

```python
event_key = (source_id, vehicle_tracker_id, stable_text)
```

Mac dinh moi xe + bien chi emit mot lan.

---

# 6. Van de 4 - Multi-stream mat bbox

**Kho khan ban dau**

Khi chay 3-4 stream:

```text
stream 0: co bbox
stream 1/2/3: it hoac mat bbox
```

**Root cause**

Model `vehicle_parking_detect.onnx` output hardcode:

```text
[1, 8400, 6]
```

Trong graph con co `Reshape` hardcode batch=1:

```text
[1, 64, -1]
[1, 4, 16, 8400]
```

**Fix hien tai**

```python
streammux.set_property("batch-size", num_sources)
pgie.set_property("batch-size", 1)
```

**Trade-off**

- Dung detection moi stream.
- Hieu suat thap hon batching that.
- Huong toi uu: re-export ONNX dynamic batch dung cach.

---

# 7. Metadata server-ready + Kafka/MinIO

![Vehicle crop](assets/crop_square_vehicle_30H00644.jpg)

**Event JSON**

```json
{
  "source_id": 0,
  "frame_num": 176,
  "vehicle": {
    "tracker_id": "...",
    "class_name": "Car",
    "bbox": [...]
  },
  "plate": {
    "text_stable": "30H00644",
    "ocr_confidence": 0.9542
  },
  "association": {
    "method": "geometry",
    "score": 1.0
  },
  "media": {
    "plate_image_path": "...",
    "vehicle_image_path": "..."
  }
}
```

**Flow server**

```text
DeepStream -> Kafka lpr.events.v1
media_monitor -> MinIO -> Kafka lpr.media.v1
```

---

# 8. Tinh trang hien tai

**Da hoan thanh**

- LPRNet-only OCR pipeline.
- Xe/bien/tracker metadata.
- Bien vuong top/bottom OCR.
- OCR correction theo format bien Viet Nam.
- Stable vote + drift guard.
- Bbox smoothing.
- Crop plate/vehicle tu frame goc truoc tiler.
- Anti-spam event.
- Kafka metadata producer.
- MinIO upload + media result Kafka.
- RTSP scripts + Redpanda/MinIO compose.

**Da test**

- Single source OK.
- 3-source OK voi PGIE batch=1.
- Kafka `lpr.events.v1` nhan metadata.
- MinIO co anh crop.
- Kafka `lpr.media.v1` nhan URL anh.

---

# 9. Van de dang gap va huong tiep theo

**1. PGIE batch=1**

- Hien dung va on dinh.
- Chua toi uu throughput.
- Can re-export ONNX dynamic batch that.

**2. Media monitor van doc JSONL**

Hien tai:

```text
DeepStream -> JSONL + Kafka
media_monitor -> doc JSONL -> MinIO
```

Huong tiep:

```text
media_monitor consume Kafka lpr.events.v1 truc tiep
```

**3. Bien nho / nghien / motion blur**

- Van co the miss OCR trong realtime.
- Huong tiep: deskew/rectify crop, sharpen crop, model OCR tot hon cho bien nghien.

**4. RTSP loop ngan reset tracker ID**

- Vote history reset theo tracker.
- Huong tiep: persistence ngan han theo plate similarity + region + time window.

---

# Cau ket thuc voi mentor

Tuan nay em da dua pipeline tu muc demo detect/OCR len thanh pipeline gan production:

```text
dung xe -> dung bien -> dung tracker -> dung frame/source -> metadata + anh crop gui server
```

Hien tai pipeline da chay duoc end-to-end voi Kafka va MinIO. Phan con lai chu yeu la toi uu performance multi-stream va cai thien OCR cho bien nho/nghien trong realtime.

