# Bao Cao Tuan - DeepStream LPR Pipeline

Muc tieu file nay: dung lam khung trinh bay voi mentor. Flow trinh bay nen di theo:

```text
1. Tuan nay em build duoc gi
2. No giai quyet van de gi
3. Cac bug/khó khan gap phai va cach fix
4. Tinh trang hien tai
5. Van de con ton tai / huong tiep theo
```

Anh minh hoa dang co trong:

```text
reports/assets/
```

---

## 1. Tuan nay em da build duoc gi

### 1.1. Pipeline LPR chay tren DeepStream 9.0

File chinh:

```text
src/app_lpr_v2.py
```

Pipeline hien tai:

```text
Video/RTSP
  -> uridecodebin
  -> nvstreammux
  -> PGIE vehicle + plate detector
  -> nvtracker
  -> SGIE LPRNet OCR
  -> metadata probe truoc tiler
  -> tiler
  -> nvosd
  -> display/output
```

Build duoc:

- Detect xe va bien so.
- Gan tracker ID cho tung xe.
- OCR bien so bang LPRNet.
- Hien thi bbox xe/bien va text OCR len frame.
- Crop anh bien so va xe tu frame goc.
- Tao event metadata server-ready.
- Gui metadata len Kafka.
- Upload anh crop len MinIO thong qua media monitor.

Anh nen dung:

```text
reports/assets/frame_osd_square_plate.jpg
reports/assets/annotated_square_plate_osd.jpg
reports/assets/pipeline_architecture_diagram.jpg
```

---

## 2. Cac thanh phan code chinh va cong dung

### 2.1. `VehicleTrackState`

Vi tri:

```text
src/app_lpr_v2.py:172
```

Cong dung:

`VehicleTrackState` la state trung tam cho moi xe. Moi xe duoc key boi:

```python
(source_id, vehicle_tracker_id)
```

No luu:

- `source_id`: stream/camera nao.
- `vehicle_tracker_id`: tracker ID cua xe.
- `vehicle_bbox`: bbox xe da lam muot.
- `best_plate_bbox`: bbox bien so tot nhat.
- `best_plate_text_raw`: OCR moi nhat/tot nhat.
- `best_plate_text_stable`: OCR da on dinh.
- `crop_plate_path`, `crop_vehicle_path`: anh crop de upload.
- `last_emitted_plate_text`: bien da gui server gan nhat.

Van de giai quyet:

Truoc do OCR chi la text roi rac theo frame. Backend can biet:

```text
Bien nay thuoc xe nao?
Tracker ID nao?
Frame nao?
Source nao?
Anh crop o dau?
```

State nay gom tat ca thong tin do lai theo tung xe.

---

### 2.2. `_read_lpr_text(obj_meta, gie_unique_id)`

Vi tri:

```text
src/app_lpr_v2.py:320
```

Cong dung:

Doc output tensor cua LPRNet tu DeepStream object metadata.

LPRNet tra ve:

- `tf_op_layer_ArgMax`: chuoi index ky tu.
- `tf_op_layer_Max`: confidence tung ky tu.

Ham nay decode index thanh text bien so va tinh confidence trung binh.

Van de giai quyet:

Truoc do pipeline co nhieu OCR backend/YOLO-char parser phuc tap. Hien tai da dua ve mot backend duy nhat:

```text
LPRNet OCR only
```

Giup pipeline gon hon, nhanh hon, it duplicate OCR hon.

---

### 2.3. `_correct_vn_plate(raw)`

Vi tri:

```text
src/app_lpr_v2.py:481
```

Cong dung:

Sua loi ky tu OCR theo format bien so Viet Nam.

Code dung cac mapping:

```python
_LETTER_TO_DIGIT = {'B':'8','D':'0','G':'6','I':'1','O':'0','S':'5'}
_DIGIT_TO_LETTER = {'8':'B','0':'D','6':'G','5':'S','7':'T'}
```

Logic:

```text
2 ky tu dau: uu tien chu so tinh/thanh
1-2 ky tu tiep: uu tien chu series
4-5 ky tu cuoi: uu tien chu so
```

Vi du trinh bay:

```text
OCR raw        : 3OH1234S
Sau correction: 30H12345
```

Van de giai quyet:

OCR raw hay bi nham cac ky tu co hinh dang giong nhau. Neu cham diem raw truc tiep thi co the loai nham bien dung. Ham nay dua OCR ve dang hop ly truoc khi cham diem.

---

### 2.4. `_plate_pattern_score(text)`

Vi tri:

```text
src/app_lpr_v2.py:401
```

Cong dung:

Cham diem text dua tren pattern bien Viet Nam:

```text
2 so tinh + 1/2 chu series + 4/5 so cuoi
```

Vi du pattern hop le:

```text
30H00644
30G09090
15D03116
```

Van de giai quyet:

Khong phai text nao LPRNet tra ve cung nen tin. Ham nay giup loai text sai format.

---

### 2.5. `_plate_quality_score(text, conf, width, height, association_score)`

Vi tri:

```text
src/app_lpr_v2.py:435
```

Cong dung:

Tinh diem tong hop cho mot OCR candidate:

```python
score = pattern_score
score += conf * 2.0
score += size_factor
score += association_score
```

No ket hop:

- Text co dung format khong.
- OCR confidence.
- Bbox bien co du lon/ro khong.
- Bien co gan duoc voi xe khong.

Van de giai quyet:

Trong mot frame co the co nhieu candidate OCR. Ham nay chon candidate fit nhat voi bien so thuc te thay vi lay ket qua moi nhat.

---

### 2.6. `sgie3_sink_pad_buffer_probe()`

Vi tri:

```text
src/app_lpr_v2.py:916
```

Cong dung:

Probe truoc SGIE LPRNet. No lam 2 viec:

1. Ghi lai bbox bien so goc.
2. Neu bien la bien vuong, tao pseudo-object cho dong tren va dong duoi.

Logic:

```python
if ar < _square_plate_ar_threshold:
    # split thanh top crop va bottom crop
```

Van de giai quyet:

Bien vuong 2 dong OCR ca bien mot lan de bi:

- mat dong tren,
- doc sai thu tu,
- doc them ky tu rac giua hai dong.

Bang cach chia top/bottom truoc OCR, LPRNet doc tung phan ro hon.

Anh nen dung:

```text
reports/assets/crop_square_plate_30H00644_zoom.jpg
reports/assets/annotated_square_plate_osd.jpg
```

---

### 2.7. `_square_join_variants(top, bot)`

Vi tri:

```text
src/app_lpr_v2.py:448
```

Cong dung:

Ghep OCR dong tren va dong duoi cua bien vuong thanh bien so day du.

Bug cu:

```text
top = 30HT
bot = 00644
ket qua xau = 30HT00644
```

Fix:

Neu 3 ky tu dau cua top da la prefix hop le:

```text
30H
```

thi tao them candidate:

```text
30H00644
```

Van de giai quyet:

LPRNet doi khi doc khoang trong giua 2 dong thanh ky tu `T`. Ham nay tao candidate da trim de candidate dung co co hoi thang khi cham diem.

---

### 2.8. `_stable_plate(track_key, raw_text, conf, width, height, assoc_score)`

Vi tri:

```text
src/app_lpr_v2.py:582
```

Cong dung:

Luu lich su OCR gan day theo tung xe, sau do vote de chon bien on dinh.

No lam:

- Normalize OCR bang `_correct_vn_plate`.
- Tinh score bang `_plate_quality_score`.
- Luu vao history toi da 30 candidates.
- Gom nhom cac bien gan giong nhau bang Levenshtein distance.
- Chon bien co vote/score tot nhat.

Van de giai quyet:

OCR bi nhay theo frame do:

- motion blur,
- bien nho,
- xe di chuyen,
- RTSP fps thap.

Thay vi gui tung OCR raw len server, ham nay chi tra ve text stable.

---

### 2.9. `_should_replace_stable_text(...)`

Vi tri:

```text
src/app_lpr_v2.py:554
```

Cong dung:

Quyet dinh co nen thay bien stable cu bang bien moi hay khong.

Fix quan trong:

```python
if major_change and new_pattern < current_pattern:
    return False
```

Van de giai quyet:

Neu bien da stable roi, mot vai frame blur co the tao OCR moi sai nhung lap lai nhieu lan. Ham nay chan viec thay stable text bang candidate co pattern kem hon, dac biet khi province/series bi doi manh.

---

### 2.10. `_associate_plate_to_vehicle(plate_meta, vehicles_list)`

Vi tri:

```text
src/app_lpr_v2.py:1039
```

Cong dung:

Gan bien so vao dung xe.

Thu tu:

1. Neu DeepStream co parent object thi dung parent.
2. Neu khong co parent thi fallback geometry:
   - bien nam trong bbox xe,
   - uu tien nua duoi bbox xe,
   - tinh overlap score.

Van de giai quyet:

Backend can biet OCR thuoc xe nao. Neu chi co text bien so ma khong co xe/tracker ID thi khong du de gui event server.

---

### 2.11. `metadata_src_pad_buffer_probe()`

Vi tri:

```text
src/app_lpr_v2.py:1177
```

Cong dung:

Day la probe trung tam tao metadata/event.

No lam:

1. Lay `source_id`, `frame_num`, `pts`.
2. Lay frame RGBA goc bang `pyds.get_nvds_buf_surface`.
3. Tach object thanh vehicles, plates, plate_parts.
4. Cap nhat `VehicleTrackState`.
5. Doc OCR LPRNet.
6. Chon candidate OCR tot nhat.
7. Stable vote.
8. Crop plate/vehicle.
9. Anti-spam event.
10. Goi `_emit_event`.

Van de giai quyet:

Day la noi gom tat ca logic tu AI output thanh event server-ready.

---

### 2.12. `_build_lpr_event(state, frame_num)`

Vi tri:

```text
src/app_lpr_v2.py:1076
```

Cong dung:

Tao JSON event gom:

```json
{
  "source_id": 0,
  "source_uri": "...",
  "frame_num": 167,
  "pts": 16550000000,
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

Van de giai quyet:

Chuyen ket qua LPR thanh schema backend co the dung.

---

### 2.13. `_emit_event(state, frame_num)`

Vi tri:

```text
src/app_lpr_v2.py:1130
```

Cong dung:

Gui event ra 2 kenh:

1. Ghi local:

```text
outputs/events/events.jsonl
```

2. Gui Kafka neu bat:

```bash
--kafka-enable
--kafka-bootstrap-server 172.17.0.1:9092
```

Kafka topic:

```text
lpr.events.v1
```

Van de giai quyet:

Server khong can doc truc tiep DeepStream memory. No nhan metadata qua Kafka.

---

### 2.14. `osd_sink_pad_buffer_probe()`

Vi tri:

```text
src/app_lpr_v2.py:1527
```

Cong dung:

Chi lo hien thi:

- an object SGIE OCR raw,
- ve bbox xe,
- ve bbox bien,
- hien thi text bien len bbox bien,
- khong lam logic metadata/event.

Van de giai quyet:

Tach OSD display ra khoi metadata logic. Display co the update lien tuc, nhung event server chi emit khi stable.

---

## 3. Media Monitor - Kafka va MinIO

File:

```text
tools/media_monitor.py
```

### 3.1. `_resolve_path(path, path_map)`

Vi tri:

```text
tools/media_monitor.py:50
```

Cong dung:

Map path trong container sang path tren host.

Vi du:

```text
/workspace/last_ds/outputs/events/a.jpg
```

thanh:

```text
/home/thagn/projects/deepstream/workspace/last_ds/outputs/events/a.jpg
```

Van de giai quyet:

DeepStream app chay trong container, media monitor co the chay tren host. Neu khong map path thi monitor khong tim thay file anh.

---

### 3.2. `_upload_file_minio(...)`

Vi tri:

```text
tools/media_monitor.py:87
```

Cong dung:

Upload crop image len MinIO/S3:

```python
minio_client.fput_object(bucket, object_key, path)
```

Object key:

```text
lpr/{source_id}/{event_id}/plate.jpg
lpr/{source_id}/{event_id}/vehicle.jpg
```

Van de giai quyet:

Khong gui binary anh qua Kafka. Kafka chi gui metadata, anh de MinIO quan ly.

---

### 3.3. `_upload_with_retry(...)`

Vi tri:

```text
tools/media_monitor.py:141
```

Cong dung:

Neu upload that bai, thu lai theo:

```bash
--upload-retries
--upload-retry-delay
```

Van de giai quyet:

Mang/server co the loi tam thoi. Retry giup upload on dinh hon ma khong lam chet DeepStream app.

---

### 3.4. `_process_event(event, args, minio_client, path_map)`

Vi tri:

```text
tools/media_monitor.py:158
```

Cong dung:

Doc mot event LPR, upload cac anh co trong:

```text
media.plate_image_path
media.vehicle_image_path
media.frame_image_path
```

Sau do tao media result:

```json
{
  "event_id": "...",
  "plate_image_url": "...",
  "vehicle_image_url": "...",
  "upload_status": "success"
}
```

Van de giai quyet:

Backend can URL anh sau khi upload, khong chi local path trong container.

---

## 4. Cac bug lon va cach fix

### Bug 1 - OCR raw nhieu ky tu sai

Kho khan:

```text
OCR doc nham O/0, S/5, B/8, D/0...
```

Fix:

```text
_correct_vn_plate
_plate_pattern_score
_plate_quality_score
```

Ket qua:

Text OCR duoc normalize theo format bien Viet Nam truoc khi cham diem.

---

### Bug 2 - Bien vuong khong doc du 2 dong

Kho khan:

Bien vuong co bbox nhung LPRNet doc sai/khong doc dong tren.

Fix:

```text
sgie3_sink_pad_buffer_probe
_square_join_variants
```

Ket qua:

Bien vuong `30H00644` doc du hon sau khi split top/bottom.

Anh:

```text
reports/assets/crop_square_plate_30H00644_zoom.jpg
reports/assets/annotated_square_plate_osd.jpg
```

---

### Bug 3 - OCR nhay theo frame va tao event rac

Kho khan:

Xe di chuyen, bien nho, motion blur lam OCR thay doi giua cac frame.

Fix:

```text
_stable_plate
_plate_similar_enough
_should_replace_stable_text
```

Ket qua:

Chi cap nhat stable text khi candidate moi du tot.

---

### Bug 4 - Gui qua nhieu event len server

Kho khan:

Neu moi frame deu gui, server se nhan nhieu message trung lap.

Fix:

Trong `metadata_src_pad_buffer_probe`:

```python
event_key = (source_id, vehicle_tracker_id, stable_text)
```

Mac dinh moi key chi emit mot lan.

Ket qua:

Mot xe + mot bien -> mot event chinh, giam spam server.

---

### Bug 5 - Multi-stream mat bbox o stream sau

Kho khan:

Chay 3-4 source thi source dau co bbox, source sau mat bbox.

Root cause:

ONNX detector output hardcode:

```text
[1, 8400, 6]
```

Va trong graph co `Reshape` hardcode batch=1.

Fix hien tai:

```python
streammux.set_property("batch-size", num_sources)
pgie.set_property("batch-size", 1)
```

Ket qua:

Tat ca stream co detection dung, doi lai throughput thap hon batch that.

---

### Bug 6 - Crop anh sai/toa do sai khi dung tiler

Kho khan:

Sau tiler, frame da bi ghep va scale. Crop anh tu do de bi mo/sai toa do.

Fix:

Chuyen metadata/crop probe len truoc tiler:

```text
SGIE -> nvvideoconvert -> caps_gpu -> metadata_src_pad_buffer_probe -> tiler
```

Ket qua:

Crop bien/xe tu frame goc tung source.

---

### Bug 7 - Kafka/MinIO production integration

Kho khan:

Backend can metadata va anh, nhung khong nen nhet anh binary vao Kafka.

Fix:

```text
DeepStream app -> Kafka lpr.events.v1
media_monitor -> MinIO upload -> Kafka lpr.media.v1
```

Ket qua da test:

- Kafka `lpr.events.v1` nhan metadata OCR.
- MinIO bucket `lpr-media` co anh crop.
- Kafka `lpr.media.v1` nhan URL anh.

---

## 5. Tinh trang hien tai

Da hoan thanh:

- LPRNet-only OCR pipeline.
- Detect xe/bien + tracker.
- OCR bien ngang va bien vuong.
- VN plate correction.
- Stable voting va drift guard.
- Bbox smoothing.
- Metadata event server-ready.
- Crop plate/vehicle/frame optional.
- Anti-spam event.
- Kafka metadata producer.
- MinIO media upload.
- Kafka media result producer.
- RTSP streaming script.
- Redpanda + MinIO docker compose.

Tinh trang test:

- `py_compile` OK cho app va media monitor.
- Single source chay OK.
- 3-source chay OK khi PGIE batch=1.
- Kafka/MinIO da test end-to-end.

---

## 6. Van de dang gap / gioi han hien tai

### 6.1. PGIE batch=1 anh huong hieu suat

Hien tai dung batch=1 de dam bao detection dung moi stream.

Anh huong:

- Chay dung va on dinh.
- Khong tan dung duoc dynamic batch TensorRT.

Huong giai quyet:

Re-export ONNX tu PyTorch voi dynamic batch that, khong chi patch metadata ONNX.

---

### 6.2. Media monitor van doc JSONL lam input

Hien tai:

```text
DeepStream -> events.jsonl + Kafka
media_monitor -> doc events.jsonl -> MinIO -> Kafka media result
```

Van de:

JSONL van la cau noi local.

Huong tiep theo:

Them mode:

```text
media_monitor consume Kafka lpr.events.v1
```

Khi do JSONL chi con de debug.

---

### 6.3. Bien qua nho / motion blur van co the miss OCR

Nguyen nhan:

- RTSP fps thap.
- Xe re trai/phai lam bien nghien.
- Bien qua nho.
- Motion blur.

Da co:

- min plate width/height filter.
- OCR confidence.
- stable vote.
- square split.

Huong tiep:

- them rectification/deskew cho crop bien,
- model OCR tot hon cho bien nghien,
- adaptive crop padding/sharpening,
- giam interval, tang chat luong stream.

---

### 6.4. RTSP loop reset tracker ID

Van de:

Voi clip loop ngan, nvtracker ID bi reset moi loop, lam vote history reset.

Huong tiep:

Them short-term persistence theo:

```text
source_id + plate similarity + bbox region + time window
```

de noi lai vote giua cac loop ngan.

---

## 7. Cach noi voi mentor ngan gon

Co the trinh bay nhu sau:

> Tuan nay em da build duoc pipeline LPR end-to-end tren DeepStream. Ban dau pipeline chi detect/OCR de hien thi, nhung backend can metadata day du: bien thuoc xe nao, tracker ID nao, source/frame nao, anh crop o dau. Em da them `VehicleTrackState` de gom metadata theo tung xe, them correction OCR theo format bien Viet Nam, xu ly rieng bien vuong bang split top/bottom, stable voting de tranh OCR rac theo frame, va anti-spam event de moi xe/bien chi gui mot lan. Sau do em tich hop Kafka de gui metadata va MinIO de luu anh crop. Hien tai pipeline chay duoc single/multi source, Kafka/MinIO da test end-to-end. Van de lon con lai la model detector ONNX hardcode batch=1 nen dang ep PGIE batch=1 de dam bao dung; neu muon toi uu throughput thi can re-export ONNX dynamic batch that.

