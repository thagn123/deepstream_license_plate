# Báo Cáo Kỹ Thuật Chuyên Sâu: DeepStream LPR Pipeline

**Ngày:** 2026-06-25  
**Phạm vi:** Toàn bộ hệ thống nhận dạng biển số xe (License Plate Recognition) xây dựng trên NVIDIA DeepStream SDK

---

## Mục Lục

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Kiến Trúc Pipeline GStreamer](#2-kiến-trúc-pipeline-gstreamer)
3. [Nguồn Đầu Vào & nvstreammux](#3-nguồn-đầu-vào--nvstreammux)
4. [PGIE — Phát Hiện Đối Tượng (YOLOv11s)](#4-pgie--phát-hiện-đối-tượng-yolov11s)
5. [Python Probe PGIE — Lọc Trước Tracker](#5-python-probe-pgie--lọc-trước-tracker)
6. [Custom Plugin dslaplacian — Đo Độ Nét GPU](#6-custom-plugin-dslaplacian--đo-độ-nét-gpu)
7. [nvtracker — Theo Dõi Đối Tượng](#7-nvtracker--theo-dõi-đối-tượng)
8. [SGIE3 Sink Probe — Phân Tách Biển Số Vuông](#8-sgie3-sink-probe--phân-tách-biển-số-vuông)
9. [SGIE3 — Nhận Diện Ký Tự OCR (nvinfer)](#9-sgie3--nhận-diện-ký-tự-ocr-nvinfer)
10. [OCR Decode — Thuật Toán CTC](#10-ocr-decode--thuật-toán-ctc)
11. [Metadata Probe — Xử Lý & Tổng Hợp Kết Quả](#11-metadata-probe--xử-lý--tổng-hợp-kết-quả)
12. [Hệ Thống Plate Text Processing](#12-hệ-thống-plate-text-processing)
13. [Vehicle–Plate Association](#13-vehicleplate-association)
14. [BBox Smoothing — Làm Mượt Tọa Độ](#14-bbox-smoothing--làm-mượt-tọa-độ)
15. [Cơ Chế Phát Sự Kiện (Event Emission)](#15-cơ-chế-phát-sự-kiện-event-emission)
16. [Quản Lý Trạng Thái (State Management)](#16-quản-lý-trạng-thái-state-management)
17. [Hệ Thống Cấu Hình & CLI](#17-hệ-thống-cấu-hình--cli)
18. [Phân Tích Điểm Nghẽn Hiệu Suất](#18-phân-tích-điểm-nghẽn-hiệu-suất)
19. [Kết Quả Đánh Giá & So Sánh Model](#19-kết-quả-đánh-giá--so-sánh-model)

---

## 1. Tổng Quan Hệ Thống

### Mục tiêu

Hệ thống nhận một hoặc nhiều luồng video (file MP4, RTSP) làm đầu vào, tự động phát hiện xe cộ, phát hiện biển số, nhận diện ký tự trên biển số, rồi phát ra **sự kiện có cấu trúc** (JSON) mỗi khi đọc được một biển số ổn định.

### Đầu vào — Đầu ra

| Đầu vào | Đầu ra |
|---|---|
| File MP4, RTSP stream, HTTP video | JSON event per biển số |
| Nhiều nguồn đồng thời (multi-source) | Ảnh crop biển số (JPEG) |
| Bất kỳ loại xe thương mại nào | Ảnh crop xe (JPEG) |
| Độ phân giải bất kỳ (chuẩn hóa về 1920×1080) | Kafka message (tuỳ chọn) |

### Công nghệ cốt lõi

- **NVIDIA DeepStream SDK** — framework xử lý video AI real-time dựa trên GStreamer
- **TensorRT (TRT)** — inference engine của NVIDIA, tối ưu hoá mô hình AI cho GPU
- **Python + PyDS** — Python bindings cho DeepStream metadata API, dùng để viết probe functions
- **OpenCV / CUDA** — xử lý ảnh trong C++ plugin và Python post-processing
- **GStreamer** — multimedia framework quản lý pipeline, buffer, và luồng dữ liệu

---

## 2. Kiến Trúc Pipeline GStreamer

### Sơ đồ luồng dữ liệu

```
[uridecodebin] ──┐
[uridecodebin] ──┤
       ...       │
[uridecodebin] ──┘
                 │  NV12/NVMM
                 ▼
         [nvstreammux]          ← chuẩn hoá về 1920×1080, gộp batch
                 │  batch frames (NVMM)
                 ▼
           [nvinfer PGIE]       ← YOLOv11s, phát hiện 14 class
                 │
           [PGIE Probe]         ← (Python) lọc non-vehicle, drop xe nhỏ
                 │
           [nvtracker]          ← NvDCF, gán object_id ổn định
                 │
          [dslaplacian]         ← (C++/CUDA) đo độ nét biển số
                 │
        [SGIE3 Sink Probe]      ← (Python) phân tách biển vuông → pseudo-objects
                 │
         [nvinfer SGIE3]        ← OCR model (LPRNet / New OCR 2024)
                 │
         [nvvideoconvert]       ← NVMM → RGBA (để đọc pixel trong Python)
                 │
          [capsfilter]          ← video/x-raw(memory:NVMM), format=RGBA
                 │
         [Metadata Probe]       ← (Python) OCR decode, vote, emit event
                 │
    [nvmultistreamtiler]        ← ghép multi-source thành 1 khung hiển thị
                 │
           [nvdsosd]            ← vẽ bbox, text lên frame
                 │
             [tee]              ─────────────────────────────┐
              │                                              │
     [queue-display]                                  [queue-file]
              │                                              │
    [nveglglessink]                          [nvvideoconvert → h264 → mp4]
    (hoặc fakesink)
```

### Vị trí gắn probe

| Probe | Vị trí | Loại |
|---|---|---|
| `pgie_src_pad_buffer_probe` | Pad SRC của nvinfer PGIE | `BUFFER` |
| `sgie3_sink_pad_buffer_probe` | Pad SINK của nvinfer SGIE3 | `BUFFER` |
| `metadata_src_pad_buffer_probe` | Pad SRC của capsfilter RGBA | `BUFFER` |
| `osd_sink_pad_buffer_probe` | Pad SINK của nvdsosd | `BUFFER` |

Các probe chạy **đồng bộ trong GStreamer pipeline thread** — nếu probe chậm thì cả pipeline bị block.

---

## 3. Nguồn Đầu Vào & nvstreammux

### uridecodebin

Mỗi video source được tạo một phần tử `uridecodebin` riêng. DeepStream tự động chọn decoder:

- **File MP4/H264/H265** → NVDEC (hardware decode trên GPU, zero-copy sang NVMM)
- **RTSP** → RTP depay → jitterbuffer → NVDEC
- **Fallback** → FFMPEG software decode (nếu không có NVDEC support)

Callback `cb_newpad` được gọi khi decoder tạo ra pad video, tự động link tới sinkpad của nvstreammux.

```python
source.connect("pad-added", cb_newpad, sinkpad)
```

### nvstreammux

**Vai trò:** Nhận nhiều luồng video từ nhiều nguồn, đồng bộ và gộp thành một batch tensor duy nhất gửi xuống PGIE.

**Cấu hình:**
```
width  = 1920          # resize tất cả input về 1920×1080
height = 1080
batch-size = num_sources
batched-push-timeout = 33000 µs  # ≈ 30fps budget
live-source = 0 (file) hoặc 1 (RTSP)
```

**Hoạt động:**
- Scale mỗi frame về 1920×1080 bằng CUDA (zero-copy trong NVMM)
- Gộp N frame của N nguồn thành 1 batch tensor NxCxHxW
- Gán `source_id` (0..N-1) cho mỗi frame trong batch để phân biệt nguồn
- Gán `frame_num` tăng dần per-source

**Kết quả:** Sau muxer, tất cả bbox đều ở không gian tọa độ 1920×1080, bất kể độ phân giải gốc của video.

---

## 4. PGIE — Phát Hiện Đối Tượng (YOLOv11s)

### Model

| Tham số | Giá trị |
|---|---|
| Kiến trúc | YOLOv11s |
| Input size | 640×640 (3 channel) |
| Số class | 14 |
| Precision | FP16 (TRT) |
| Batch size | = num_sources (động) |
| Interval | 0 (chạy mỗi frame) |

**File config:** `configs/config_pgie_yolov11s.txt`  
**Engine:** `models_converted/yolov11s_14_cls_20241224/model.onnx_b{N}_gpu0_fp16.engine`

### 14 Class phát hiện

| Class ID | Tên | Vai trò |
|---|---|---|
| 0 | seater_12_16 | Xe khách 12-16 chỗ |
| 1 | bus | Xe buýt |
| 2 | car | Ô tô con |
| 3 | club_cart | Xe golf / mini |
| 4 | human | Người (bị lọc bỏ) |
| 5 | moto | Xe máy (bị lọc bỏ) |
| 6 | moto_rider | Người lái xe máy (bị lọc bỏ) |
| 7 | shuttle_bus_5_7 | Xe shuttle nhỏ |
| 8 | truck | Xe tải |
| 9 | bike | Xe đạp (bị lọc bỏ) |
| 10 | cyclist | Người đạp xe (bị lọc bỏ) |
| 11 | shuttle_bus_18 | Xe shuttle lớn |
| 12 | head | Đầu người (bị lọc bỏ) |
| **13** | **license_plate** | **Biển số xe** |

**VEHICLE_CLASS_IDS = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11}** — nhưng sau probe PGIE chỉ còn: `{0,1,2,3,7,8,11}` (các class phương tiện thực sự) + class 13 (biển số).

### Custom Parser

YOLOv11 không output theo format chuẩn YOLO của DeepStream, nên cần:
```
custom-lib-path=.../libnvds_infercustom_yolov11_flat.so
parse-bbox-func-name=NvDsInferParseCustomYoloV11Flat
```

Parser C++ này nhận raw output tensor và chuyển về `NvDsInferObjectDetectionInfo` mà DeepStream hiểu.

### NMS (Non-Maximum Suppression)

```ini
cluster-mode=2              # NMS clustering
pre-cluster-threshold=0.25  # confidence score threshold
nms-iou-threshold=0.45      # IoU threshold
topk=300                    # giữ tối đa 300 bbox per frame
```

Sau NMS, mỗi bbox được ghi vào `NvDsObjectMeta` trong batch metadata của frame.

### Chọn batch size động

```python
is_static_b1 = (onnx_name == "vehicle_parking_detect.onnx")
pgie_batch_size = 1 if is_static_b1 else num_sources
```

Với model YOLOv11s thì `pgie_batch_size = num_sources`. Engine TRT tương ứng được chọn tự động theo batch size.

---

## 5. Python Probe PGIE — Lọc Trước Tracker

**File:** `src/lpr/probes/pgie.py`  
**Vị trí:** Sau PGIE, **trước** nvtracker

### Mục đích

Loại bỏ các đối tượng không cần theo dõi **trước khi** đưa vào tracker — tiết kiệm tài nguyên tracker và giảm false positive.

### Logic lọc

```python
_PGIE_KEEP = config.VEHICLE_CLASS_IDS | {LP_CLASS_ID, LP_TOP_CLASS_ID, LP_BOT_CLASS_ID}
# = {0,1,2,3,5,6,7,8,9,10,11,13,14,15}
```

Với mỗi object trong frame:

1. **Class filter:** Nếu class không thuộc `_PGIE_KEEP` → xoá (`human`, `head`, `bike`, `cyclist` bị loại)
2. **Size filter (chỉ áp dụng cho vehicle class):**

```python
min_w = state.min_vehicle_width   # mặc định 60px
min_h = state.min_vehicle_height  # mặc định 40px

# Override bằng ratio nếu được cấu hình:
if state.min_vehicle_width_ratio > 0:
    min_w = int(state.min_vehicle_width_ratio * state.muxer_width)
if state.min_vehicle_height_ratio > 0:
    min_h = int(state.min_vehicle_height_ratio * state.muxer_height)

if w < min_w or h < min_h:
    _remove_obj(frame_meta, obj_meta)  # xe quá nhỏ/xa → bỏ
```

### Lý do lọc theo kích thước

Xe quá nhỏ (xa) có biển số quá nhỏ để OCR chính xác. Lọc ngay tại probe PGIE giúp:
- Tracker không phải cấp phát track_id cho xe sẽ không bao giờ được OCR
- SGIE3 không phải xử lý biển số không thể đọc được
- Giảm số lượng entry trong `vehicle_states` dict

### Ratio-based threshold (tính năng mới)

Cho phép ngưỡng tự động scale theo muxer resolution:

```
--min-vehicle-width-ratio 0.031  →  59px @ 1920  /  39px @ 1280  /  19px @ 640
--min-vehicle-height-ratio 0.037 →  39px @ 1080  /  26px @ 720   /  17px @ 480
```

---

## 6. Custom Plugin dslaplacian — Đo Độ Nét GPU

**File nguồn:** `custom_plugins/ds_laplacian/`  
**Output:** `.so` → `gst-plugins/libnvdsgst_laplacian.so`  
**Vị trí trong pipeline:** Sau nvtracker, trước SGIE3

### Tại sao cần plugin này?

Không phải mọi biển số được phát hiện đều rõ nét. Các biển số mờ (xe ở xa, chuyển động nhanh, nén video mạnh) nếu đưa vào OCR sẽ tạo ra kết quả nhiễu, tốn GPU. Plugin này hoạt động như **hardware filter** — đo độ sắc nét ngay trên VRAM, loại bỏ biển số mờ **trước khi** SGIE3 inference.

### Kiến trúc 3 bước Hybrid CPU-GPU

**Bước 1 — Crop nhỏ từ GPU sang Mapped Memory:**

CUDA kernel chạy song song, chỉ copy vùng bounding box của biển số (ví dụ 150×50px) từ VRAM sang Mapped Memory (vùng RAM được cả CPU và GPU truy cập). Không copy toàn bộ frame Full HD.

```cpp
__global__ void crop_resize_kernel(const uint8_t* in_data, int in_pitch, ...) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    out_data[y * out_w + x] = in_data[src_y * in_pitch + src_x];
}
```

**Bước 2 — Tìm 4 góc biển số (CPU, thuật toán tuần tự):**

CPU nhận ảnh nhỏ, chạy pipeline OpenCV:
1. `cv::Canny(blur, edges, 50, 150)` — phát hiện cạnh
2. `cv::findContours(...)` — tìm tất cả viền
3. `approxPolyDP(contour, approx, arcLength * 0.02, true)` — xấp xỉ đa giác
4. Tìm tứ giác 4 điểm lớn nhất → đây là khung biển số
5. `cv::getPerspectiveTransform(src_pts, dst_pts)` → tính ma trận M
6. `M_inv = M.inv()` → ném ngược lại cho GPU

Nếu không tìm được tứ giác (viền bị che, diện tích < 40% bbox) → fallback sang crop thẳng (đánh dấu `_raw`).

**Bước 3 — Warp + Equalize + Blur + Laplacian (GPU, tất cả trong 2 kernel):**

**Kernel 3.1: Warp Perspective & Min/Max**
- Mỗi thread GPU nhân ma trận M_inv để dịch chuyển 1 pixel về vị trí thẳng đứng
- Đồng thời dùng `atomicMin` / `atomicMax` để tìm giá trị pixel min/max cho bước equalize

**Kernel 3.2: Stretch + Blur + Laplacian (all-in-one):**
```cpp
__global__ void stretch_blur_laplacian_kernel(...) {
    // 1. Contrast Stretching (thay thế EqualizeHist)
    float val = (raw_val - min_val) * 255.0f / (max_val - min_val);

    // 2. Gaussian Blur 3×3 (khử nhiễu nén video)
    float blur_val = val*0.25f + neighbors*0.125f + corners*0.0625f;

    // 3. Laplacian variance (đo độ sắc nét)
    float diff = blur_val - mean_val;
    atomicAdd(out_variance, diff * diff);
}
```

Ba phép toán gộp thành **1 kernel pass** — tiết kiệm memory bandwidth GPU so với 3 pass riêng lẻ.

### Truyền điểm số về Python

Plugin ghi kết quả vào slot `misc_obj_info[0]` trong `NvDsObjectMeta` của biển số:

```cpp
obj_meta->misc_obj_info[0] = (int)variance;  // ví dụ: 245
```

Python probe đọc lại:

```python
lap_score = int(p.misc_obj_info[0])
if lap_score < 150 and lap_score > 0:
    continue  # biển số mờ → bỏ qua, không chạy OCR
```

### Phân vùng điểm số Laplacian

| Điểm | Ý nghĩa |
|---|---|
| 0 | Plugin không chạy / không tính được (bỏ qua, không lọc) |
| 1–149 | Mờ, nhiễu, không đủ nét → skip OCR |
| 150–300 | Nét vừa đủ → cho qua |
| 300–700 | Nét tốt |
| > 700 | Rất sắc nét |

---

## 7. nvtracker — Theo Dõi Đối Tượng

### Cấu hình

```
ll-lib-file = libnvds_nvmultiobjecttracker.so  (NvDCF)
ll-config-file = config_tracker_NvDCF_noReid.yml
tracker-width  = 640
tracker-height = 384
gpu-id = 0
```

**Thuật toán:** NvDCF (NVIDIA Discriminative Correlation Filter) — không dùng Re-ID.

### Vai trò

- Nhận object detections từ PGIE (đã lọc qua probe)
- Gán `object_id` (track ID) ổn định cho mỗi xe/biển qua nhiều frame
- Dự đoán vị trí khi PGIE bỏ qua 1 frame (frame skip với pgie-interval > 0)
- Xử lý occlusion (che khuất tạm thời)

### Điều quan trọng về track ID

`object_id` là **định danh duy nhất** trong suốt vòng đời của 1 xe trong frame. Mọi metadata (bbox history, plate text votes, event emit state) đều được key bởi `(source_id, object_id)`. Khi tracker mất track → new track = new object_id = history bị reset.

---

## 8. SGIE3 Sink Probe — Phân Tách Biển Số Vuông

**File:** `src/lpr/probes/sgie3.py`  
**Vị trí:** Pad SINK của SGIE3, **trước** khi SGIE3 inference chạy

### Vấn đề biển số 2 dòng

Biển số Việt Nam có 2 loại:
- **Biển ngang (1 dòng):** Aspect ratio W/H ≈ 2.5–4.0 (rộng hơn cao)
- **Biển vuông (2 dòng):** Aspect ratio W/H < 1.7 (gần vuông)

OCR model được train để nhận diện **1 dòng**. Đưa biển 2 dòng vào trực tiếp → OCR đọc cả 2 dòng cùng lúc → kết quả sai.

### Giải pháp: Tạo pseudo-objects

Với mỗi biển có `AR < square_plate_ar_threshold (1.7)`, probe tạo ra **6 pseudo-objects** bổ sung:

**3 cấu hình split** (để thử nhiều tỉ lệ phân chia):

| Config | Top fraction | Bottom fraction |
|---|---|---|
| 1 | 0.44 + overlap/2 | 0.56 + overlap/2 |
| 2 | 0.50 + overlap/2 | 0.50 + overlap/2 |
| 3 | 0.58 + overlap/2 | 0.58 + overlap/2 |

Với `square_split_overlap = 0.12`:
- Config 1: top=50%, bot=62% (chồng lên nhau ở giữa)
- Config 2: top=56%, bot=56%
- Config 3: top=64%, bot=64%

**Với mỗi config:** tạo 1 `LP_TOP_CLASS_ID` (class 14) và 1 `LP_BOT_CLASS_ID` (class 15), mỗi cái được padding thêm theo chiều ngang (`pad_x = 12%`) và dọc (`pad_y = 8%`).

**Tổng:** 3 configs × 2 halves = **6 pseudo-objects per biển vuông**.

### Ghi nhớ quan hệ cha-con

```python
state.pseudo_parent_map[(sid, fnum, class_id, int(left), int(top), int(height))] = parent_plate_id
```

Key gồm 6 phần tử để tìm lại plate cha khi metadata probe cần ghép kết quả top+bot.

### Khoá biển đã ổn định

Nếu biển đã có đủ votes và score cao (`locked_plate_ids`), probe đổi `class_id = 99` để SGIE3 bỏ qua — tránh inference không cần thiết cho biển đã biết.

---

## 9. SGIE3 — Nhận Diện Ký Tự OCR (nvinfer)

### Hai model OCR

#### LPRNet (baseline18) — Legacy

| Tham số | Giá trị |
|---|---|
| Input | 3×48×96 (RGB float) |
| Network type | 100 (custom CTC) |
| Precision | FP16 |
| Batch size | 8 |
| Output layer | `tf_op_layer_ArgMax` (indices), `tf_op_layer_Max` (confidences) |
| Scale factor | 1/255 = 0.00392... |
| Source | `us_lprnet_baseline18_deployable.onnx` |

#### New OCR 2024

| Tham số | Giá trị |
|---|---|
| Input | 3×64×128 (BGR float, model normalize nội bộ) |
| Network type | 100 (custom CTC) |
| Precision | FP16 |
| Batch size | 8 |
| Output layer | cùng tên |
| Scale factor | 1.0 (raw pixel, model tự normalize với ImageNet mean/std) |
| Color format | BGR (1) |
| Source | `lpr_ocr.20240305.onnx` |

### Hoạt động của nvinfer (Secondary GIE)

```ini
process-mode=2          # secondary mode (chạy trên detected objects)
operate-on-gie-id=1     # chỉ nhận output từ PGIE (unique_id=1)
operate-on-class-ids=13;14;15  # chỉ xử lý biển số (LP, LP_TOP, LP_BOT)
interval=0              # chạy mỗi frame
secondary-reinfer-interval=0   # reinfer mỗi frame cho mỗi track
output-tensor-meta=1    # ghi raw tensor vào metadata (để Python decode)
```

SGIE3 crop từng biển số từ frame (resize về 48×96 hoặc 64×128), chạy TRT inference, ghi output tensor vào `NvDsUserMeta` → `NvDsInferTensorMeta` của từng object.

**Quan trọng:** `output-tensor-meta=1` — SGIE3 không decode text mà ghi raw tensor indices, để Python decode theo thuật toán CTC tùy chỉnh.

---

## 10. OCR Decode — Thuật Toán CTC

**File:** `src/lpr/ocr.py`  
**Hàm chính:** `_read_lpr_text(obj_meta, gie_unique_id)`

### CTC là gì?

CTC (Connectionist Temporal Classification) là thuật toán decode cho OCR model. Model output một chuỗi indices (dài hơn chuỗi ký tự thực), CTC decode loại bỏ:
1. Ký tự "blank" (index = len(alphabet))
2. Ký tự lặp lại liên tiếp

**Alphabet:** `"0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"` (35 ký tự, không có O vì dễ nhầm với 0)

### Quy trình decode

```python
def _decode_lpr_indices(indices: list) -> str:
    result = []
    prev = -1
    blank_id = len(config._LPR_CHARS)  # = 35
    for idx in indices:
        if idx != blank_id and idx != prev:  # không blank, không lặp
            if 0 <= idx < len(config._LPR_CHARS):
                result.append(config._LPR_CHARS[idx])
        prev = idx
    return "".join(result)
```

### Tính confidence

Confidence = trung bình các confidence của ký tự **thực sự** (loại bỏ blank và repeat):

```python
valid_confs = []
prev = -1
for idx, c in zip(indices, confs):
    if idx != blank_id and idx != prev:
        valid_confs.append(c)
    prev = idx
conf = sum(valid_confs) / len(valid_confs)
```

### Đọc từ tensor metadata

```python
l_user = obj_meta.obj_user_meta_list
# tìm NvDsInferTensorMeta có unique_id == SGIE3_UNIQUE_ID (4)
# tìm layer tên "tf_op_layer_ArgMax" → indices (int32)
# tìm layer tên "tf_op_layer_Max" → confidences (float32)
ptr = ctypes.cast(pyds.get_ptr(idx_layer.buffer), ctypes.POINTER(ctypes.c_int32))
indices = [ptr[j] for j in range(n)]
```

---

## 11. Metadata Probe — Xử Lý & Tổng Hợp Kết Quả

**File:** `src/lpr/probes/metadata.py`  
**Vị trí:** Pad SRC của capsfilter (sau SGIE3, trước tiler)  
**Đây là probe nặng nhất — chạy mọi frame, O(vehicles × plates × candidates)**

### 11.1 Phân loại đối tượng trong frame

```python
vehicles    = []  # PGIE objects thuộc VEHICLE_CLASS_IDS, đã có tracker ID
plates      = []  # PGIE objects class_id == 13 (LP_CLASS_ID) hoặc 99 (locked)
plate_parts = []  # PGIE objects class_id == 14 (LP_TOP) hoặc 15 (LP_BOT)
```

### 11.2 Cập nhật Vehicle Track State

Mỗi xe được quản lý bởi một `VehicleTrackState` object, key là `(sid, object_id)`:

```python
vs = lpr_state.vehicle_states[(sid, vid)]
vs.last_seen_frame = frame_num
vs.vehicle_confidence = float(v.confidence)
```

**BBox smoothing cho xe:**
```python
iou = _bbox_iou(vs.vehicle_bbox, raw_v_bbox)
cdist = _bbox_center_distance(vs.vehicle_bbox, raw_v_bbox)
max_jump = bbox_max_center_jump_ratio * max(w, h)  # 0.5 × max(width, height)

if iou < bbox_reset_iou or cdist > max_jump:
    vs.vehicle_bbox = raw_v_bbox  # reset: xe nhảy quá mạnh (traffic cut, occlusion)
else:
    vs.vehicle_bbox = EMA(vs.vehicle_bbox, raw_v_bbox, alpha=0.4)  # smoothing
```

### 11.3 Đánh Giá Chất Lượng Nguồn & Lọc Động (Dynamic Quality Routing)

Hệ thống triển khai cơ chế tự động đánh giá chất lượng video thời gian thực dựa trên 2 yếu tố: Độ phân giải luồng (Spatial) và Tốc độ khung hình (Temporal - FPS). Nếu nguồn thuộc dạng chất lượng thấp (LQ), hệ thống sẽ linh hoạt nới lỏng ngưỡng nhận diện YOLO và màng lọc Laplacian để tăng tối đa tỷ lệ bắt mờ.

```python
# ── Source Quality Assessment ─────────────────────────────────────────
sid_str = f"stream{frame_meta.source_id}"
stream_fps = lpr_state.perf_data.get_current_fps(sid_str)
source_uri = lpr_state.source_uri_by_id.get(frame_meta.source_id, "")

is_low_quality_source = False
# Ép RTSP thành LQ trong lúc test (nếu cần), hoặc tự động đánh giá qua độ phân giải & FPS
is_forced_lq = config.FORCE_LQ_RTSP and "rtsp://" in source_uri

if is_forced_lq or (0 < frame_meta.source_frame_width < 1280) or (0 < frame_meta.source_frame_height < 720) or (0 < stream_fps < 15):
    is_low_quality_source = True

for p in plates:
    is_low_quality = is_low_quality_source

    # Lọc Confidence động
    target_conf = 0.15 if is_low_quality else 0.25
    if p.confidence > 0.0 and p.confidence < target_conf:
        continue

    # Lọc Laplacian động
    if not getattr(lpr_state, 'disable_laplacian', False):
        lap_score = int(p.misc_obj_info[0])
        target_lap = 50 if is_low_quality else 150
        if lap_score < target_lap and lap_score > 0:
            continue
```
* Ý nghĩa:
  * Camera nét (High Quality): Siết chặt YOLO Conf >= 0.25 và Laplacian >= 150 để OCR xử lý mượt mà, tránh rác.
  * Camera mờ (Low Quality): Hạ chuẩn YOLO Conf >= 0.15 và Laplacian >= 50, tận dụng đối đa recall để không bỏ sót sự kiện.

### 11.4 Thu thập OCR text từ plate-parts (biển vuông)

```python
for part in plate_parts:
    parent_id = _pseudo_parent_lookup(sid, frame_num, class_id, left, top, height)
    text, conf = _ocr_mod._read_lpr_text(part, SGIE3_UNIQUE_ID)
    norm = _correct_vn_plate(text)
    plate_part_texts[parent_id]["top" or "bot"].append((norm, conf))
```

6 pseudo-objects per biển vuông → 6 OCR readings → ghép top+bot thành candidates.

### 11.4 Lọc plate và tạo candidate list

Với mỗi plate `p` trong `plates`:

1. **Confidence filter:** `p.confidence < min_plate_conf (0.05)` → skip
2. **Size filter:** `width < 20px` hoặc `height < 6px` → skip  
3. **Laplacian filter:** `misc_obj_info[0] < 150` (mờ) → skip
4. **OCR decode:** `full_text, full_conf = _read_lpr_text(p, SGIE3_UNIQUE_ID)`
5. **Candidate generation:**

```python
candidates = []
_add_candidate(full_raw_text, full_conf)  # toàn biển từ full plate

# Từ các split top+bot (biển vuông):
for t_txt, t_conf in tops:
    for b_txt, b_conf in bots:
        for joined in _square_join_variants(t_txt, b_txt):  # ghép nhiều cách
            _add_candidate(joined, (t_conf + b_conf) / 2.0)
    _add_candidate(t_txt, t_conf)  # top riêng
for b_txt, b_conf in bots:
    _add_candidate(b_txt, b_conf)  # bot riêng
```

6. **Chọn best candidate:**

```python
for cand, conf in candidates:
    score = _plate_quality_score(cand, conf, width, height, assoc_score)
    if score > best_cand_score: ...
```

---

## 12. Hệ Thống Plate Text Processing

**File:** `src/lpr/plate_text.py`

### 12.1 `_correct_vn_plate(raw)` — Sửa lỗi nhận dạng

Biển số VN có cấu trúc cứng: `[tỉnh][chữ cái series][số]`. OCR hay nhầm:
- Số → chữ: `0→D, 1→K, 5→S, 8→B`
- Chữ → số: `B→8, O→0, S→5, I→1`

Hàm thử mọi `series_len` (1 hoặc 2 chữ cái series), sửa từng vị trí theo quy tắc:

```python
for idx in range(min(2, len(chars))):
    chars[idx] = fix(chars[idx], want_digit=True)   # 2 số đầu = mã tỉnh
for idx in range(2, suffix_start):
    chars[idx] = fix(chars[idx], want_digit=False)  # series = chữ cái
for idx in range(suffix_start, len(chars)):
    chars[idx] = fix(chars[idx], want_digit=True)   # suffix = số
```

Trả về variant có `_plate_pattern_score` cao nhất.

### 12.2 `_plate_pattern_score(text)` — Đánh giá format

Cho điểm dựa trên việc text có khớp format biển VN không:

| Điều kiện | Điểm |
|---|---|
| Không đúng format (< 7 ký tự, prefix không phải số) | 0.0 |
| Đúng format cơ bản | 5.0 |
| Series 1 chữ cái | +0.50 |
| Series 2 chữ cái | +0.35 |
| Suffix 5 chữ số | +0.60 |
| Suffix 4 chữ số | +0.25 |

**Score ≥ 5.0** = hợp lệ format. Score 0.0 = không hợp lệ.

### 12.3 `_plate_quality_score(text, conf, w, h, assoc_score)` — Điểm tổng hợp

```python
score = _plate_pattern_score(text)        # format score (5.0–6.1)
if score <= 0: return 0.0                 # format không hợp lệ → loại ngay

score += conf * 2.0                       # OCR confidence (0–2.0)
size_factor = min(1.0, w*h / 10000.0)    # kích thước (0–1.0, saturate ở 100×100)
score += size_factor
score += max(0.0, association_score)      # xe–biển association (0–1.0)
```

**Range thực tế:** 7.0–9.5. Điểm cao = format đúng + OCR tin cậy + biển lớn + gắn chắc với xe.

### 12.4 `_stable_plate(track_key, text, conf, w, h, assoc_score)` — Voting

Duy trì `plate_history[track_key]` — danh sách tối đa **30 readings** gần nhất.

**Thuật toán clustering:**
1. Đếm số lần xuất hiện của mỗi text (votes)
2. Gộp các text "gần giống nhau" vào cùng cluster (`_plate_similar_enough`)
3. Cluster "gần giống" = cùng prefix tỉnh/series, suffix khác nhau ≤ 1 edit distance

```python
strong_single = (c >= 1 and sc >= _SINGLE_VOTE_ACCEPT_SCORE and pattern_score >= 5.5)
if c >= state.min_stable_votes or strong_single:
    # cluster này đủ điều kiện "stable"
```

**Hai ngưỡng chấp nhận:**
- `min_stable_votes` (default 2): cần ít nhất 2 readings đồng ý
- `_SINGLE_VOTE_ACCEPT_SCORE = 8.0`: reading đơn lẻ nhưng điểm rất cao cũng được chấp nhận

### 12.5 `_should_replace_stable_text(...)` — Cập nhật biển đã ổn định

Khi đã có stable text, điều kiện để thay thế bằng candidate mới:

| Điều kiện | Quyết định |
|---|---|
| Chưa có stable text | Thay thế ngay |
| Text giống nhau, score/votes cao hơn | Cập nhật score/votes |
| Text khác nhau, major change (khác tỉnh hoặc similarity < 55%), format mới kém hơn | Giữ nguyên |
| Text khác, đủ thêm votes (`+ 2 + 2 nếu major`) | Thay thế |
| Text khác, score cao hơn nhiều (`+ 1.25 + 1.0 nếu major`) | Thay thế |
| Text khác, format rõ ràng tốt hơn và score cao hơn `0.5` | Thay thế |

**Mục đích:** Ngăn stable plate bị đổi do 1 reading nhiễu. Yêu cầu bằng chứng đủ mạnh.

### 12.6 OCR Lock

```python
if vs.best_votes >= ocr_lock_min_votes (3) and vs.best_score >= ocr_lock_min_score (4.0):
    lpr_state.locked_plate_ids.add(p.object_id)
```

Một khi biển được "lock", SGIE3 probe đổi class thành 99 → SGIE3 **không inference nữa** cho biển này. Tiết kiệm GPU inference cho biển đã đọc chắc chắn.

---

## 13. Vehicle–Plate Association

**File:** `src/lpr/association.py`

### Vấn đề

PGIE phát hiện xe và biển số độc lập. Cần biết biển số nào thuộc xe nào.

### Phương pháp 1: Parent link (ưu tiên)

DeepStream tracker có thể thiết lập quan hệ cha-con trong metadata. Nếu plate có parent là vehicle:

```python
parent = _safe_parent(plate_meta)
if parent and _is_vehicle_obj(parent):
    return parent.object_id, "parent", 1.0
```

### Phương pháp 2: Geometry association (fallback)

Khi không có parent link, tìm xe chứa tâm biển số:

```python
for v in vehicles_list:
    # Kiểm tra tâm biển số (pcx, pcy) có nằm trong bbox xe không
    contains_center = vx <= pcx <= vx+vw and vy <= pcy <= vy+vh

    # Tính containment = phần trăm diện tích biển nằm trong xe
    containment = intersection_area / plate_area

    # Vị trí dọc: biển ở nửa dưới xe → score cao hơn (biển thường ở đầu/đuôi)
    v_score = 1.0 if pcy >= vy + vh/2 else 0.5

    score = containment * v_score
```

**Trả về xe có score cao nhất.** Nếu score < ngưỡng → method = "none", không associate.

### Phương pháp 3: No association

Nếu không tìm được xe nào chứa biển → tạo `VehicleTrackState` orphan với `vehicle_class = -1`. Biển này sẽ không emit event (bị filter ở bước anti-spam).

---

## 14. BBox Smoothing — Làm Mượt Tọa Độ

**Mục đích:** YOLO detector cho ra bbox bị jitter (dao động nhỏ) giữa các frame. Smoothing giúp:
- OSD hiển thị bbox ổn định, không rung
- Crop ảnh sự kiện chính xác hơn
- Giảm ảnh hưởng của 1 frame detection kém

**Công thức EMA (Exponential Moving Average):**

```python
def _smooth_bbox(old, new, alpha=0.4):
    return tuple(alpha * n + (1 - alpha) * o for o, n in zip(old, new))
# alpha=0.4: 40% bbox mới + 60% bbox cũ
```

**Reset conditions** (không smooth, nhận ngay bbox mới):

| Điều kiện | Ý nghĩa |
|---|---|
| `IoU < 0.2` | Bbox nhảy quá xa → xe mới hoặc track switch |
| `center_distance > 0.5 × max(w, h)` | Tâm dịch hơn 50% kích thước bbox |

Cả xe và biển số đều được smooth riêng biệt với cùng tham số.

---

## 15. Cơ Chế Phát Sự Kiện (Event Emission)

### Điều kiện emit

```python
# Không emit nếu:
if vs.association_method == "none":    continue   # không biết xe nào
if vs.vehicle_class == -1:             continue   # xe orphan

# Event key = (source_id, vehicle_tracker_id, plate_text)
event_key = (vs.source_id, vs.vehicle_tracker_id, stable_text)
```

**Logic anti-spam:**

| Trường hợp | emit_now |
|---|---|
| `emit_duplicates=True` (debug mode) | Luôn True |
| Key đã emit, `event_repeat_cooldown_frames > 0`, đủ frames | True (repeat) |
| Key đã emit, không cooldown | False |
| Key chưa emit, chưa emit lần nào | True |
| Key chưa emit, plate cũ tốt hơn | False |
| Key chưa emit, plate mới tốt hơn pattern | True |
| Key chưa emit, plate mới score cao hơn 0.5 | True |

### Khi emit

1. **Crop ảnh biển số** từ frame hiện tại (nếu `event_output_dir` được set):
   ```python
   p_bgr, _, _, _ = _crop_plate_from_frame(frame_image, vs.best_plate_bbox, 0.0)
   cv2.imwrite(f"{sid}_{vid}_{frame_num}_plate.jpg", p_bgr)
   ```

2. **Crop ảnh xe:**
   ```python
   cv2.imwrite(f"{sid}_{vid}_{frame_num}_vehicle.jpg", v_bgr)
   ```

3. **Crop toàn frame** (nếu `save_event_frame=True`):
   ```python
   cv2.imwrite(f"{sid}_{vid}_{frame_num}_frame.jpg", full_bgr)
   ```

4. **Ghi JSON event** vào `event_jsonl` file

5. **Gửi Kafka** (nếu `kafka_enabled=True`):
   ```python
   state.kafka_producer.produce(topic, key=plate_text, value=json.dumps(event))
   ```

### Cấu trúc JSON event

```json
{
  "source_id": 0,
  "vehicle_tracker_id": 42,
  "frame_num": 1234,
  "pts": 41166700000,
  "plate_text": "51B12345",
  "ocr_confidence": 0.87,
  "best_votes": 3,
  "best_score": 8.94,
  "vehicle_class": 8,
  "vehicle_class_name": "Truck",
  "vehicle_confidence": 0.91,
  "association_method": "geometry",
  "association_score": 0.85,
  "plate_bbox": [x, y, w, h],
  "vehicle_bbox": [x, y, w, h],
  "crop_plate_path": "/path/to/plate.jpg",
  "crop_vehicle_path": "/path/to/vehicle.jpg"
}
```

---

## 16. Quản Lý Trạng Thái (State Management)

**File:** `src/lpr/state.py`

Tất cả trạng thái runtime được lưu trong module-level variables (singleton pattern):

| Dict/Set | Key | Value | Kích thước ước tính |
|---|---|---|---|
| `vehicle_states` | `(sid, oid)` | `VehicleTrackState` | N_tracked_objects |
| `plate_history` | `(sid, oid)` | `list[{text, score}]` tối đa 30 | N_tracked_objects |
| `pseudo_parent_map` | `(sid, fnum, cls, x, y, h)` | `parent_plate_id` | N_frames × N_square_plates × 6 |
| `emitted_event_keys` | `(sid, vid, text)` | — | N_unique_events |
| `locked_plate_ids` | `plate.object_id` | — | N_locked_plates |
| `plate_text_seen` | `(sid, oid)` | `{text, stable, score, votes}` | N_tracked_objects |
| `source_uri_by_id` | `source_id` | URI string | N_sources |

### Reset giữa các lần chạy

Hàm `pipeline.run()` reset toàn bộ state trước khi chạy:

```python
state.vehicle_states    = {}
state.plate_history     = {}
state.pseudo_parent_map = {}
state.emitted_event_keys = set()
state.locked_plate_ids  = set()
# ... tất cả các dict khác
```

### VehicleTrackState — Trạng thái 1 xe

```
display_id          : short ID để hiển thị
vehicle_class       : 0-11 (loại xe)
vehicle_bbox        : (x,y,w,h) đã smooth
vehicle_bbox_raw    : bbox gốc từ detector
best_plate_bbox     : (x,y,w,h) bbox biển đã smooth
best_plate_text_raw : text best candidate (chưa stable)
best_plate_text_stable : text đã stable qua voting
display_plate_text  : text hiển thị trên OSD
best_votes          : số votes của stable text
best_score          : score tổng hợp cao nhất
ocr_confidence      : confidence OCR
last_emitted_plate_text : text đã emit lần cuối
last_event_frame    : frame số của event cuối
association_method  : "parent" / "geometry" / "none"
association_score   : 0.0–1.0
```

---

## 17. Hệ Thống Cấu Hình & CLI

### File config tĩnh

| File | Mục đích |
|---|---|
| `configs/config_pgie_yolov11s.txt` | PGIE model, threshold, NMS |
| `configs/config_sgie_lprnet.txt` | SGIE3 OCR LPRNet |
| `configs/config_sgie_lpr_ocr_2024.txt` | SGIE3 OCR New 2024 |
| `configs/ds_tracker_config.txt` | nvtracker dimensions |
| `configs/config_tracker_NvDCF_noReid.yml` | NvDCF algorithm params |

### Runtime config override

Pipeline tạo bản copy tạm trong `/tmp/ds_lpr_v2_runtime_configs/` để inject các tham số runtime (batch-size, interval...) mà không sửa file gốc:

```python
sgie3_overrides = {"interval": "0", "secondary-reinfer-interval": "0"}
sgie3_config = _runtime_config_path(config.SGIE3_CONFIG_PATH, sgie3_overrides)
```

### Tham số CLI quan trọng

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--no-display` | — | Dùng fakesink, không render |
| `--pgie-interval N` | 0 | Skip N-1 frame giữa PGIE inferences |
| `--min-vehicle-width N` | 60 | Lọc xe nhỏ (px) |
| `--min-vehicle-height N` | 40 | Lọc xe thấp (px) |
| `--min-vehicle-width-ratio f` | 0.0 | Lọc xe theo tỉ lệ muxer width |
| `--min-vehicle-height-ratio f` | 0.0 | Lọc xe theo tỉ lệ muxer height |
| `--min-plate-conf f` | 0.05 | Ngưỡng confidence biển số |
| `--min-plate-width N` | 20 | Kích thước tối thiểu biển (px) |
| `--min-plate-height N` | 6 | Chiều cao tối thiểu biển (px) |
| `--min-stable-votes N` | 2 | Số votes để stable |
| `--event-cooldown-frames N` | 60 | Min frames giữa 2 events cùng xe |
| `--event-output-dir dir` | — | Thư mục lưu ảnh crop |
| `--event-jsonl path` | — | File ghi JSON events |
| `--save-event-frame` | — | Lưu cả frame gốc |
| `--bbox-smooth-alpha f` | 0.4 | EMA alpha cho bbox smoothing |
| `--bbox-reset-iou f` | 0.2 | Reset smooth khi IoU < threshold |
| `--square-plate-ar-threshold f` | 1.7 | AR < ngưỡng → xử lý biển vuông |
| `--ocr-every-n-frames N` | 6 | Tham số throttle (hiện không hoạt động) |
| `--ocr-min-conf f` | 0.0 | Min confidence để vote |
| `--emit-duplicates` | — | Debug: emit mọi lần stable text cập nhật |
| `--kafka-enable` | — | Bật Kafka producer |
| `--kafka-bootstrap-server` | localhost:9092 | Kafka server |

---

## 18. Phân Tích Điểm Nghẽn Hiệu Suất

### Dữ liệu thực đo

- **15 video** ~33MB/video (khoảng 60–120 giây mỗi video)
- **Wall time LPRNet:** 4–7 giây/video (nhanh hơn realtime 10–20×)
- Thống kê tiêu biểu per-video: `plate_objects=188–566`, `ocr_raw_events=35–75`

### Bottleneck #1 (CRITICAL): OCR_EVERY_N là dead code

`state.OCR_EVERY_N = 6` được đặt và in ra log `"[INFO] OCR throttle : every 6 frames"` nhưng **không được kiểm tra ở bất kỳ đâu** trong `metadata.py`, `sgie3.py`, hay bất kỳ file probe nào.

Hậu quả: Python metadata probe chạy full logic cho mỗi plate mỗi frame. Với `plate_objects=566` trên video dài ~100 giây @ 30fps, đây là hàng chục nghìn lần gọi `_read_lpr_text` + `_correct_vn_plate` + `_plate_quality_score` + voting logic.

### Bottleneck #2 (SIGNIFICANT): Metadata probe block GStreamer thread

`metadata_src_pad_buffer_probe` chạy đồng bộ trong GStreamer pipeline thread. Mọi microsecond trong probe = thời gian pipeline bị treo. Các operation nặng:

| Operation | Chi phí |
|---|---|
| `_read_lpr_text` per plate | Python ctypes + list comprehension trên tensor |
| `_correct_vn_plate` | Thử nhiều variants, mỗi cái chạy regex + string ops |
| `_plate_quality_score` → `_plate_pattern_score` | Gọi mỗi candidate |
| `_stable_plate` | Sort + clustering O(N²) trên plate_history |
| `cv2.imwrite` (khi có event) | Đồng bộ I/O trong pipeline thread |

### Bottleneck #3 (MODERATE): pseudo_parent_map tăng không giới hạn

Mỗi frame có biển vuông → thêm 6 entries vào `state.pseudo_parent_map`. Dict này **không bao giờ được dọn** trong vòng đời 1 video:

```
1000 frames × 2 biển vuông × 6 entries = 12,000 entries/video
```

Dict lớn hơn → Python hash lookup chậm dần → metadata probe chậm dần về cuối video.

### Bottleneck #4 (MODERATE): secondary-reinfer-interval=0

SGIE3 inference chạy trên **mọi plate được track, mọi frame**. Với 5 xe mỗi mang 1 biển, 30fps → 150 OCR inferences/giây. Thực tế với batch_size=8, các inference được gộp thành batch 8, nhưng vẫn là tải GPU đáng kể.

Cách đúng để throttle GPU inference là tăng `secondary-reinfer-interval` trong SGIE3 config (ví dụ =5 → chỉ reinfer mỗi 5 frame per track). `OCR_EVERY_N` trong Python không ảnh hưởng GPU.

### Bottleneck #5 (MINOR): Startup overhead

Mỗi video chạy độc lập → mỗi lần deserialize TRT engine từ disk:
- PGIE engine: ~0.3–0.5s
- SGIE3 engine: ~0.3–0.5s

Tổng ~0.6–1.0s overhead per video. Không ảnh hưởng deployment liên tục, nhưng đáng kể khi batch eval.

### Bottleneck #6 (MINOR): SGIE3 batch-size=8 chưa được tận dụng

`batch-size=8` nhưng với 1 nguồn video, mỗi frame thường có 1–3 biển → 5–7 slot batch trống. Không phải vấn đề trong single-source, nhưng với multi-source deployment, batch mới thực sự được lấp đầy.

---

## 19. Kết Quả Đánh Giá & So Sánh Model

### Tóm tắt so sánh (15 video, 2026-06-25)

| Tiêu chí | LPRNet (baseline18) | New OCR 2024 | Kết luận |
|---|---|---|---|
| Tổng events | 131 | **169** (+29%) | New OCR vượt trội |
| Biển VN hợp lệ (regex) | 88.5% | **100%** | New OCR vượt trội |
| OCR confidence mean | **0.862** | 0.807 | LPRNet cao hơn |
| Vote stability | 1.07 | **1.15** | New OCR nhỉnh hơn |
| Association score | 0.889 | **0.953** | New OCR vượt trội |
| Unique plate texts | **119** | 113 | LPRNet đa dạng hơn |
| Tốc độ (sau fix) | ~5s/video | **~4–5s/video** | Tương đương |

### Bug quan trọng đã fix: TRT engine rebuild

New OCR thiếu `model-engine-file=` trong config → mỗi lần chạy rebuild TRT engine từ ONNX mất 44–126 giây. Sau khi thêm 1 dòng config:

```ini
model-engine-file=/workspace/last_ds_cp/models/lpr_ocr.20240305.onnx_b8_gpu0_fp16.engine
```

Thời gian xử lý giảm từ **57s/video → 4–5s/video** (giảm 11×).

### Nhận xét về LPRNet vs New OCR 2024

**LPRNet** hay thêm 1 ký tự `0` thừa vào biển:
- `15F00365` (LPRNet) vs `15F0365` (New OCR) — cùng 1 biển, LPRNet có 0 thừa
- `15F01157` vs `15F0157`, `15B03551` vs `15B0351`

New OCR 2024 output 7 ký tự (62.7%) vs LPRNet 8 ký tự (77.1%). Nhiều khả năng New OCR đúng hơn với biển chuẩn VN (2 số tỉnh + 1 chữ series + 4–5 số), nhưng cần ground truth để xác nhận.

---

## Phụ Lục: Sơ Đồ Luồng Xử Lý Chi Tiết 1 Frame

```
Frame N đến từ nvstreammux (1920×1080 NVMM batch)
│
├─ [nvinfer PGIE] chạy YOLOv11s inference
│   → Output: list NvDsObjectMeta {class, confidence, bbox}
│
├─ [PGIE Probe] (Python, đồng bộ)
│   ├─ Loại bỏ: human, head, bike, cyclist, moto
│   └─ Loại bỏ: xe W < 60px hoặc H < 40px
│
├─ [nvtracker] NvDCF
│   └─ Gán/duy trì object_id cho mỗi detection
│
├─ [dslaplacian] (C++/CUDA, asynchronous GPU kernels)
│   └─ Mỗi bbox class=13: đo Laplacian variance → ghi misc_obj_info[0]
│
├─ [SGIE3 Sink Probe] (Python, đồng bộ)
│   ├─ Ghi plate_rects[(sid, oid)] = bbox
│   ├─ Biển vuông (AR < 1.7):
│   │   └─ Tạo 6 pseudo LP_TOP/LP_BOT objects với tọa độ khác nhau
│   │       ghi pseudo_parent_map[(sid,fnum,cls,x,y,h)] = parent_id
│   └─ Biển đã locked (object_id in locked_plate_ids):
│       └─ Đổi class_id = 99, SGIE3 sẽ bỏ qua
│
├─ [nvinfer SGIE3] (GPU TRT inference)
│   ├─ Crop từng bbox class 13/14/15 từ frame
│   ├─ Resize về 48×96 (LPRNet) hoặc 64×128 (New OCR)
│   ├─ Gộp batch tối đa 8 crops
│   ├─ TRT inference → output tensor (indices + confidences)
│   └─ Ghi NvDsInferTensorMeta vào obj_user_meta_list của mỗi object
│
├─ [nvvideoconvert] NVMM NV12 → NVMM RGBA (để đọc pixel từ Python)
├─ [capsfilter] format constraint
│
├─ [Metadata Probe] (Python, đồng bộ — HEAVY)
│   ├─ (Optional) get_nvds_buf_surface() → frame_image nếu cần crop
│   ├─ Phân loại: vehicles / plates / plate_parts
│   │
│   ├─ FOR mỗi vehicle:
│   │   ├─ Tạo/cập nhật VehicleTrackState
│   │   └─ Smooth vehicle bbox (EMA)
│   │
│   ├─ FOR mỗi plate_part (LP_TOP/LP_BOT):
│   │   ├─ Lookup parent từ pseudo_parent_map
│   │   ├─ _read_lpr_text() → CTC decode → (text, conf)
│   │   └─ _correct_vn_plate(text) → plate_part_texts[parent_id]
│   │
│   ├─ FOR mỗi plate (LP):
│   │   ├─ Filter: confidence, size, laplacian
│   │   ├─ _associate_plate_to_vehicle() → (vid, method, score)
│   │   ├─ _read_lpr_text() → full plate OCR
│   │   ├─ Build candidates list (full + top/bot joins)
│   │   ├─ _plate_quality_score() cho mỗi candidate
│   │   └─ Lưu best candidate per track_key
│   │
│   ├─ FOR mỗi track_key có best plate:
│   │   ├─ Smooth plate bbox (EMA)
│   │   ├─ Cập nhật display_plate_text
│   │   ├─ _stable_plate() → voting + clustering → stable_text
│   │   ├─ _should_replace_stable_text() → có nên cập nhật stable không
│   │   ├─ OCR lock check (votes ≥ 3 và score ≥ 4.0)
│   │   └─ Anti-spam emit decision
│   │       └─ Nếu emit:
│   │           ├─ cv2.imwrite(plate_crop)
│   │           ├─ cv2.imwrite(vehicle_crop)
│   │           ├─ Ghi JSON event
│   │           └─ Kafka produce (nếu enabled)
│   │
│   └─ Fallback: cập nhật plate_bbox cho plate không có OCR text
│
├─ [nvmultistreamtiler] ghép sources
├─ [nvdsosd] vẽ bbox + text
└─ → Display / File sink
```

---

*Báo cáo này tổng hợp từ: phân tích source code trực tiếp, đo lường thực nghiệm trên 15 video, và kết quả từ REPORT.md + REPORT_CUSTOM_PLUGIN.md. Mọi số liệu phản ánh trạng thái hệ thống tại 2026-06-25.*
