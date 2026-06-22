#!/usr/bin/env python3
"""
app_lpr_v2.py — DeepStream edge profile: all-class tracking + plate OCR.

Pipeline:
  uridecodebin(s)
    → nvstreammux
    → nvinfer [PGIE]     gie-id=1  detect original 14 classes
    → nvtracker          assign stable IDs to PGIE objects
    → nvinfer [SGIE]     gie-id=4  LPRNet OCR on PGIE license_plate boxes
    → nvvideoconvert
    → capsfilter RGBA
    → [metadata event probe]
    → nvmultistreamtiler
    → nvdsosd
    → tee ──► [fakesink / display sink]
         └──► [nvv4l2h264enc → qtmux → filesink]   (if --output is specified)

Usage:
  python3 src/app_lpr_v2.py videos/test5.mp4
  python3 src/app_lpr_v2.py videos/test5.mp4 --output outputs/out.mp4
  python3 src/app_lpr_v2.py videos/test5.mp4 --no-display
"""

import sys
import os
import re
import json
import math
import time
import ctypes
import configparser
from collections import deque, Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

from common.platform_info import PlatformInfo
from common.bus_call import bus_call
from common.FPS import PERF_DATA

# pyrefly: ignore [missing-import]
import pyds

# ── Config paths ──────────────────────────────────────────────────────────────
PGIE_CONFIG_PATH    = os.path.join(PROJECT_ROOT, "configs", "config_pgie_yolov11s.txt")
TRACKER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "ds_tracker_config.txt")
SGIE3_CONFIG_PATH   = os.path.join(PROJECT_ROOT, "configs", "config_sgie_lprnet.txt")
RUNTIME_CONFIG_DIR  = os.path.join("/tmp", "ds_lpr_v2_runtime_configs")

MUXER_BATCH_TIMEOUT_USEC = 33000
MUXER_WIDTH = 1920
MUXER_HEIGHT = 1080
TILER_WIDTH = 1280
TILER_HEIGHT = 720

# ── GIE unique IDs (must match gie-unique-id in config) ────────────────────
PGIE_UNIQUE_ID  = 1
SGIE1_UNIQUE_ID = 2   # plate detector on tracked vehicle crops
SGIE2_UNIQUE_ID = 3   # LPRNet fallback/reference on SGIE1 plate boxes
SGIE3_UNIQUE_ID = 4   # LPRNet OCR on full/split plate boxes
UNTRACKED_OBJECT_ID = (1 << 64) - 1

VEHICLE_CLASS_IDS = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11}
LP_CLASS_ID     = 13
LP_TOP_CLASS_ID = 14  # Pseudo-class: top half of square plate
LP_BOT_CLASS_ID = 15  # Pseudo-class: bottom half of square plate

# ── Class labels ──────────────────────────────────────────────────────────────
VEHICLE_LABELS = [
    "seater_12_16", "bus", "car", "club_cart", "human",
    "moto", "moto_rider", "shuttle_bus_5_7", "truck", "bike",
    "cyclist", "shuttle_bus_18", "head", "license_plate",
]

# Bbox colors per class (R, G, B, A) — 0..1
_CLASS_COLORS = [
    (0.2, 0.8, 1.0, 1.0),   # 0  seater_12_16  cyan
    (1.0, 0.5, 0.0, 1.0),   # 1  bus            orange
    (0.2, 0.6, 1.0, 1.0),   # 2  car            blue
    (1.0, 1.0, 0.0, 1.0),   # 3  club_cart      yellow
    (1.0, 0.2, 0.2, 1.0),   # 4  human          red
    (0.8, 0.2, 0.8, 1.0),   # 5  moto           purple
    (0.9, 0.5, 0.9, 1.0),   # 6  moto_rider     light purple
    (0.0, 1.0, 0.8, 1.0),   # 7  shuttle_bus_5_7
    (1.0, 0.3, 0.3, 1.0),   # 8  truck          pink-red
    (0.4, 1.0, 0.4, 1.0),   # 9  bike           light green
    (0.3, 0.3, 1.0, 1.0),   # 10 cyclist        violet
    (1.0, 0.6, 0.0, 1.0),   # 11 shuttle_bus_18 amber
    (0.7, 0.7, 0.7, 1.0),   # 12 head           gray (hidden)
    (0.0, 1.0, 0.0, 1.0),   # 13 license_plate  green
]

# ── LPR character set (CTC decode) ───────────────────────────────────────────
_LPR_CHARS = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
_LPR_LAYER_NAME = "tf_op_layer_ArgMax"
_lpr_layer_warning_printed = False

# ── Temporal smoothing and track state ────────────────────────────────────────
_short_id_map: dict = {}    # (source_id, object_id) → small display id
_plate_history: dict = {}   # legacy fallback: (source_id, object_id) → deque[str]

_PLATE_HISTORY_LEN = 15
_PLATE_MIN_VOTES   = 3
_PLATE_REPLACE_VOTE_MARGIN = 2
_SINGLE_VOTE_ACCEPT_SCORE = 8.0
_STABLE_REPLACE_SCORE_MARGIN = 1.25
_STATE_STALE_AFTER_FRAMES = 450

_next_short_id = 1
_cleanup_counter = 0
_CLEANUP_INTERVAL = 150

# Maps pseudo-object location → parent LP object_id.
_pseudo_parent_map: dict = {}

# OCR results from split pseudo-objects, keyed by (source_id, parent_lp_object_id).
_split_ocr: dict = {}

# Plate bounding boxes captured before SGIE3, in original frame coordinates.
_plate_rects: dict = {}

# ── OCR backend configuration ─────────────────────────────────────────────────
_ocr_backend: str = 'lprnet'
_OCR_EVERY_N: int = 6                 # run OCR bookkeeping every N frames per track
_OCR_MIN_CONF: float = 0.0            # minimum confidence to vote in _stable_plate
_save_crops_dir: str = None           # path to save plate crops (None = disabled)
_ocr_frame_cache: dict = {}           # (sid, oid) → (text, conf, frame_num)
_osd_probe_frame: int = 0             # monotonic counter; frame_meta.frame_num = 0 post-tiler
_tiler_rows: int = 1
_tiler_cols: int = 1
_save_crop_seq: dict = {}             # (sid, oid) → int, per-track crop save count
_object_last_seen: dict = {}
_plate_text_seen: dict = {}
_debug_jsonl_path: str = None
_metrics = Counter()
_square_plate_ar_threshold: float = 1.7
_square_split_overlap: float = 0.12
_square_split_pad_x: float = 0.12
_square_split_pad_y: float = 0.08
_min_plate_conf: float = 0.05
_min_plate_width: int = 20
_min_plate_height: int = 6
_bbox_smooth_alpha: float = 0.4
_bbox_reset_iou: float = 0.2
_bbox_max_center_jump_ratio: float = 0.5
_event_output_dir: str = None
_event_jsonl_path: str = None
_event_cooldown_frames: int = 60
_min_stable_votes: int = 2
_source_uri_by_id: dict = {}
_save_event_frame: bool = False
_emitted_event_keys: set = set()
_emit_duplicates: bool = False
_event_repeat_cooldown_frames: int = 0
_kafka_enabled: bool = False
_kafka_producer = None
_kafka_topic: str = "lpr.events.v1"
_fps_overlay_state: dict = {}
_fps_overlay_alpha: float = 0.2

perf_data = None


import datetime

@dataclass
class VehicleTrackState:
    display_id: int = 0
    source_id: int = 0
    vehicle_tracker_id: int = 0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    last_event_frame: int = 0
    last_pts: int = 0
    vehicle_class: int = -1
    vehicle_class_name: str = ""
    vehicle_confidence: float = 0.0
    vehicle_bbox: tuple = (0, 0, 0, 0)
    vehicle_bbox_raw: tuple = (0, 0, 0, 0)
    best_plate_object_id: int = 0
    best_plate_bbox: tuple = (0, 0, 0, 0)
    plate_bbox_raw: tuple = (0, 0, 0, 0)
    last_bbox_update_frame: int = 0
    plate_bbox_candidates: deque = field(default_factory=lambda: deque(maxlen=30))
    best_plate_text_raw: str = ""
    best_plate_text_stable: str = ""
    display_plate_text: str = ""
    display_plate_score: float = 0.0
    best_votes: int = 0
    best_score: float = 0.0
    ocr_confidence: float = 0.0
    plate_text_switches: int = 0
    last_emitted_plate_text: str = ""
    last_emitted_score: float = 0.0
    association_method: str = ""
    association_score: float = 0.0
    crop_plate_path: str = ""
    crop_vehicle_path: str = ""
    frame_path: str = ""
    
_vehicle_states: dict = {}


# ── Config helpers ────────────────────────────────────────────────────────────

_OLD_ROOTS = (
    "/workspace/ds_lpr_v2",
    "/workspace/new_ds_lpr",
    "/workspace/las_ds",
    "/workspace/last_ds",
    "/workspace/last_ds_cp",
)
# Replace only value-side occurrences (after '='), so comments are untouched.
_PATH_SUB_RE = re.compile(
    r"(=\s*)(" + "|".join(
        re.escape(p) for p in sorted(_OLD_ROOTS, key=len, reverse=True)
    ) + r")(/|$)"
)

def _apply_property_overrides(text: str, overrides: dict) -> str:
    if not overrides:
        return text

    lines = text.splitlines()
    seen = set()
    key_patterns = {
        key: re.compile(r"^(\s*)(" + re.escape(key) + r")(\s*=\s*)(.*)$")
        for key in overrides
    }

    for idx, line in enumerate(lines):
        for key, pattern in key_patterns.items():
            match = pattern.match(line)
            if match:
                lines[idx] = "{}{}{}{}".format(match.group(1), key, match.group(3), overrides[key])
                seen.add(key)

    missing = [key for key in overrides if key not in seen]
    if missing:
        insert_at = None
        for idx, line in enumerate(lines):
            if line.strip() == "[property]":
                insert_at = idx + 1
                break
        if insert_at is None:
            insert_at = len(lines)
        additions = ["{}={}".format(key, overrides[key]) for key in missing]
        lines[insert_at:insert_at] = additions

    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _runtime_config_path(path: str, property_overrides: dict = None) -> str:
    os.makedirs(RUNTIME_CONFIG_DIR, exist_ok=True)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = _PATH_SUB_RE.sub(r"\g<1>" + PROJECT_ROOT + r"\3", text)
    text = _apply_property_overrides(text, property_overrides or {})
    out_path = os.path.join(RUNTIME_CONFIG_DIR, os.path.basename(path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def _pgie_engine_path_for_batch(batch_size: int) -> str:
    requested = max(1, int(batch_size))
    
    # Defaults
    engine_path = os.path.join(PROJECT_ROOT, "models", "vehicle_parking_detect.onnx_b1_gpu0_fp16.engine")
    
    try:
        config = configparser.ConfigParser()
        config.read(PGIE_CONFIG_PATH)
        if config.has_option("property", "model-engine-file"):
            configured_path = config.get("property", "model-engine-file")
            # Replace docker workspace paths with PROJECT_ROOT
            for old_root in _OLD_ROOTS:
                if configured_path == old_root or configured_path.startswith(old_root + os.sep):
                    configured_path = configured_path.replace(old_root, PROJECT_ROOT, 1)
            engine_path = configured_path
    except Exception:
        pass

    # Find the batch size part in the engine path (e.g. "_b1_")
    engine_dir = os.path.dirname(engine_path)
    engine_filename = os.path.basename(engine_path)
    
    match = re.search(r"_b(\d+)_", engine_filename)
    if not match:
        # If no batch size suffix in template filename, just return the path
        return engine_path
        
    template_batch = int(match.group(1))
    
    # Check if a preferred batch (1, 4, 8) matches the request and exists
    prefix_part = engine_filename[:match.start()]
    suffix_part = engine_filename[match.end():]

    preferred_batches = (1, 4, 8)
    for batch in preferred_batches:
        if requested <= batch:
            pref_filename = re.sub(r"_b\d+_", f"_b{batch}_", engine_filename)
            pref_path = os.path.join(engine_dir, pref_filename)
            if os.path.exists(pref_path):
                return pref_path
            # Also check with .onnx suffix to the prefix
            pref_filename_onnx = re.sub(r"_b\d+_", f"_b{batch}_", engine_filename.replace(prefix_part, prefix_part + ".onnx", 1))
            pref_path_onnx = os.path.join(engine_dir, pref_filename_onnx)
            if os.path.exists(pref_path_onnx):
                return pref_path_onnx
            break
            
    # Check exact match
    exact_filename = re.sub(r"_b\d+_", f"_b{requested}_", engine_filename)
    exact_path = os.path.join(engine_dir, exact_filename)
    if os.path.exists(exact_path):
        return exact_path
    exact_filename_onnx = re.sub(r"_b\d+_", f"_b{requested}_", engine_filename.replace(prefix_part, prefix_part + ".onnx", 1))
    exact_path_onnx = os.path.join(engine_dir, exact_filename_onnx)
    if os.path.exists(exact_path_onnx):
        return exact_path_onnx
        
    # Search directory for any files matching the pattern
    pattern = re.compile(re.escape(prefix_part) + r"(?:\.onnx)?_b(\d+)_" + re.escape(suffix_part))
    
    candidates = []
    try:
        for name in os.listdir(engine_dir):
            m = pattern.match(name)
            if not m:
                continue
            batch = int(m.group(1))
            if batch >= requested:
                candidates.append((batch, os.path.join(engine_dir, name)))
    except OSError:
        candidates = []
        
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
        
    return exact_path


# ── LPR Helpers ───────────────────────────────────────────────────────────────

def _decode_lpr_indices(indices: list) -> str:
    result = []
    prev = -1
    blank_id = len(_LPR_CHARS)
    for idx in indices:
        if idx != blank_id and idx != prev:
            if 0 <= idx < len(_LPR_CHARS):
                result.append(_LPR_CHARS[idx])
        prev = idx
    return "".join(result)


def _read_lpr_text(obj_meta, gie_unique_id: int = SGIE2_UNIQUE_ID) -> tuple:
    global _lpr_layer_warning_printed
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            if tensor_meta.unique_id == gie_unique_id:
                found_target = False
                idx_layer = None
                max_layer = None
                for i in range(tensor_meta.num_output_layers):
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                    if layer.layerName == _LPR_LAYER_NAME:
                        found_target = True
                        idx_layer = layer
                    elif layer.layerName == "tf_op_layer_Max":
                        max_layer = layer
                if found_target and idx_layer:
                    ptr = ctypes.cast(pyds.get_ptr(idx_layer.buffer), ctypes.POINTER(ctypes.c_int32))
                    n = idx_layer.dims.d[0] if idx_layer.dims.numDims == 1 else idx_layer.dims.d[1] if idx_layer.dims.numDims == 2 else 24
                    indices = [ptr[j] for j in range(n)]
                    
                    conf = 0.55
                    if max_layer:
                        conf_ptr = ctypes.cast(pyds.get_ptr(max_layer.buffer), ctypes.POINTER(ctypes.c_float))
                        n_conf = max_layer.dims.d[0] if max_layer.dims.numDims == 1 else max_layer.dims.d[1] if max_layer.dims.numDims == 2 else 24
                        confs = [conf_ptr[j] for j in range(n_conf)]
                        
                        valid_confs = []
                        blank_id = len(_LPR_CHARS)
                        prev = -1
                        for idx, c in zip(indices, confs):
                            if idx != blank_id and idx != prev:
                                valid_confs.append(c)
                            prev = idx
                        if valid_confs:
                            conf = sum(valid_confs) / len(valid_confs)
                            
                    text = _decode_lpr_indices(indices)
                    if text:
                        return text, conf
                    return "", 0.0
                    
                if not found_target and not _lpr_layer_warning_printed:
                    available = [pyds.get_nvds_LayerInfo(tensor_meta, i).layerName for i in range(tensor_meta.num_output_layers)]
                    sys.stderr.write("[WARN] LPRNet: layer '{}' not found.\n".format(_LPR_LAYER_NAME))
                    _lpr_layer_warning_printed = True
        try:
            l_user = l_user.next
        except StopIteration:
            break
    return "", 0.0


# ── Position-aware VN plate OCR correction ───────────────────────────────────
_LETTER_TO_DIGIT = {'B':'8','D':'0','G':'6','I':'1','O':'0',
                    'P':'9','S':'5','T':'7','Z':'2','M':'1',
                    'A':'4','E':'3','H':'4'}
_DIGIT_TO_LETTER = {'8':'B','0':'D','6':'G','1':'K','9':'P',
                    '5':'S','7':'T','2':'Z','4':'A','3':'E'}
_VALID_SERIES    = set('ABCDEFGHKLMNPSTUVXYZ')


def _fix_char_at(ch: str, want_digit: bool) -> str:
    if want_digit:
        return _LETTER_TO_DIGIT.get(ch, ch) if not ch.isdigit() else ch
    else:
        c = _DIGIT_TO_LETTER.get(ch, ch) if ch.isdigit() else ch
        return c if c in _VALID_SERIES else ch


def _normalize_plate_for_output(raw: str) -> str:
    if not raw:
        return ""
    return re.sub(r'[^A-Z0-9]', '', raw.upper())


def _plate_pattern_score(text: str) -> float:
    """Score VN plate shape without adding separators.

    Accepted shapes are intentionally conservative:
    - 2 province digits
    - 1 or 2 series letters
    - 4 or 5 trailing digits
    """
    text = _normalize_plate_for_output(text)
    if len(text) < 7 or len(text) > 9:
        return 0.0
    if len(text) < 3 or not text[:2].isdigit():
        return 0.0

    best = 0.0
    for series_len in (1, 2):
        suffix_start = 2 + series_len
        if suffix_start >= len(text):
            continue
        series = text[2:suffix_start]
        suffix = text[suffix_start:]
        if len(suffix) not in (4, 5):
            continue
        if not series.isalpha() or any(ch not in _VALID_SERIES for ch in series):
            continue
        if not suffix.isdigit():
            continue

        score = 5.0
        score += 0.5 if series_len == 1 else 0.35
        score += 0.6 if len(suffix) == 5 else 0.25
        best = max(best, score)
    return best

def _plate_quality_score(text: str, conf: float, width: int, height: int, association_score: float) -> float:
    text = _correct_vn_plate(text)
    score = _plate_pattern_score(text)
    if score <= 0.0:
        return 0.0

    score += conf * 2.0
    size_factor = min(1.0, max(0, width) * max(0, height) / 10000.0)
    score += size_factor
    score += max(0.0, association_score)
    
    return max(0.0, score)

def _square_join_variants(top: str, bot: str) -> list:
    top = _normalize_plate_for_output(top)
    bot = _normalize_plate_for_output(bot)
    if not top and not bot:
        return []

    variants = []

    def _push(value: str):
        value = _correct_vn_plate(value)
        if value and value not in variants:
            variants.append(value)

    _push(top + bot)

    # LPRNet sometimes reads row-gap noise as an extra trailing character on
    # the top row (e.g. "30H7" instead of "30H", or "30HT" instead of "30H").
    # For 2-row square plates the top row should be exactly 3 chars
    # (2 province digits + 1 series letter).  Whenever top has ≥4 chars and
    # its first 3 chars form a valid VN province+series prefix, always add the
    # trimmed variant province_series[:3] + bot so the cleaner text can win.
    if len(top) >= 4 and len(bot) >= 4 and bot[0].isdigit():
        province_series = top[:3]
        if (
            len(province_series) == 3
            and province_series[:2].isdigit()
            and province_series[2].isalpha()
            and province_series[2] in _VALID_SERIES
        ):
            _push(province_series + bot)

    return variants

def _correct_vn_plate(raw: str) -> str:
    text = _normalize_plate_for_output(raw)
    if not text:
        return ""

    variants = [text]
    for series_len in (1, 2):
        suffix_start = 2 + series_len
        if suffix_start >= len(text):
            continue
        suffix_len = len(text) - suffix_start
        if suffix_len not in (4, 5):
            continue

        chars = list(text)
        for idx in range(min(2, len(chars))):
            chars[idx] = _fix_char_at(chars[idx], want_digit=True)
        for idx in range(2, min(suffix_start, len(chars))):
            chars[idx] = _fix_char_at(chars[idx], want_digit=False)
        for idx in range(suffix_start, len(chars)):
            chars[idx] = _fix_char_at(chars[idx], want_digit=True)
        variants.append("".join(chars))

    return max(variants, key=lambda v: (_plate_pattern_score(v), v == text))

def _text_similarity(a: str, b: str) -> float:
    a = _normalize_plate_for_output(a)
    b = _normalize_plate_for_output(b)
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / float(max_len)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for cb in b:
        curr = [prev[0] + 1]
        for i, ca in enumerate(a):
            curr.append(min(prev[i + 1] + 1, curr[i] + 1, prev[i] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _plate_similar_enough(a: str, b: str) -> bool:
    """True when a and b share the same province+series prefix and their
    numeric suffixes differ by at most 1 edit (substitution/insertion/deletion).
    Used for Levenshtein vote-clustering in _stable_plate so near-identical
    LPRNet readings (e.g. one digit wrong) pool their votes."""
    if a == b:
        return True
    for series_len in (1, 2):
        prefix = 2 + series_len
        if len(a) >= prefix + 3 and len(b) >= prefix + 3 and a[:prefix] == b[:prefix]:
            sa, sb = a[prefix:], b[prefix:]
            if abs(len(sa) - len(sb)) <= 1 and _levenshtein(sa, sb) <= 1:
                return True
    return False


def _major_plate_change(a: str, b: str) -> bool:
    a = _normalize_plate_for_output(a)
    b = _normalize_plate_for_output(b)
    if not a or not b:
        return False
    province_changed = len(a) >= 2 and len(b) >= 2 and a[:2] != b[:2]
    return province_changed or _text_similarity(a, b) < 0.55

def _should_replace_stable_text(current_text: str, current_score: float, current_votes: int,
                                new_text: str, new_score: float, new_votes: int) -> bool:
    if not current_text:
        return True
    if new_text == current_text:
        return new_score > current_score or new_votes > current_votes

    current_pattern = _plate_pattern_score(current_text)
    new_pattern = _plate_pattern_score(new_text)
    major_change = _major_plate_change(current_text, new_text)

    # Province changed + new candidate has worse plate shape = OCR drift / noise.
    # Block unconditionally; LPRNet confusion cannot win over a confirmed valid plate.
    if major_change and new_pattern < current_pattern:
        return False

    vote_margin = _PLATE_REPLACE_VOTE_MARGIN + (2 if major_change else 0)
    score_margin = _STABLE_REPLACE_SCORE_MARGIN + (1.0 if major_change else 0.0)

    enough_votes = new_votes >= current_votes + vote_margin
    clearly_better_shape = new_pattern > current_pattern and new_score >= current_score + 0.5
    clearly_better_score = new_score >= current_score + score_margin and new_votes >= max(_min_stable_votes, current_votes)
    return enough_votes or clearly_better_shape or clearly_better_score

def _is_valid_vn_plate_early(text: str) -> bool:
    return _plate_pattern_score(_correct_vn_plate(text)) > 0.0


def _stable_plate(track_key, raw_text: str, conf: float, width: int, height: int, assoc_score: float) -> str:
    if track_key not in _plate_history:
        _plate_history[track_key] = []
    
    hist = _plate_history[track_key]
    
    # Prune old candidates if history too large (keep last 30)
    if len(hist) > 30:
        hist = hist[-30:]
        _plate_history[track_key] = hist
        
    if raw_text:
        norm = _correct_vn_plate(raw_text)
        score = _plate_quality_score(norm, conf, width, height, assoc_score)
        if score > 0:
            hist.append({"text": norm, "score": score})
                
    if not hist:
        return ""
        
    counts = {}
    best_score = {}
    for item in hist:
        txt = item["text"]
        counts[txt] = counts.get(txt, 0) + 1
        best_score[txt] = max(best_score.get(txt, 0), item["score"])

    # Cluster near-identical plates (same province+series, suffix edit dist ≤ 1)
    # so that readings differing by one digit still pool their votes.
    all_texts = sorted(counts.keys(), key=lambda t: (best_score.get(t, 0.0), counts.get(t, 0)), reverse=True)
    cluster_for = {}
    for i, ti in enumerate(all_texts):
        if ti in cluster_for:
            continue
        cluster_for[ti] = ti
        for tj in all_texts[i + 1:]:
            if tj not in cluster_for and _plate_similar_enough(ti, tj):
                cluster_for[tj] = ti

    cluster_votes = {}
    cluster_score = {}
    for txt in all_texts:
        rep = cluster_for[txt]
        cluster_votes[rep] = cluster_votes.get(rep, 0) + counts[txt]
        cluster_score[rep] = max(cluster_score.get(rep, 0.0), best_score[txt])

    best_cand = ""
    best_val = -1.0
    for rep, c in cluster_votes.items():
        sc = cluster_score[rep]
        strong_single = (
            c >= 1
            and sc >= _SINGLE_VOTE_ACCEPT_SCORE
            and _plate_pattern_score(rep) >= 5.5
        )
        if c >= _min_stable_votes or strong_single:
            val = c + sc
            if val > best_val:
                best_val = val
                best_cand = rep

    return best_cand


def _plate_history_stats(track_key, text: str) -> tuple:
    text = _correct_vn_plate(text)
    hist = _plate_history.get(track_key, [])
    votes = 0
    best = 0.0
    for item in hist:
        if item.get("text") == text:
            votes += 1
            best = max(best, float(item.get("score", 0.0)))
    return votes, best


def _normalize_plate_text(raw_text: str) -> str:
    return _correct_vn_plate(raw_text)


def _bbox_tuple(obj_meta) -> tuple:
    r = obj_meta.rect_params
    return (int(r.left), int(r.top), int(r.width), int(r.height))



def _bbox_iou(a, b):
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0: return 0.0
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0

def _bbox_center_distance(a, b):
    cx1, cy1 = a[0] + a[2]/2.0, a[1] + a[3]/2.0
    cx2, cy2 = b[0] + b[2]/2.0, b[1] + b[3]/2.0
    return ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5

def _smooth_bbox(prev, new, alpha):
    return (
        int(alpha * new[0] + (1 - alpha) * prev[0]),
        int(alpha * new[1] + (1 - alpha) * prev[1]),
        int(alpha * new[2] + (1 - alpha) * prev[2]),
        int(alpha * new[3] + (1 - alpha) * prev[3]),
    )

def _pseudo_parent_lookup(sid: int, frame_num: int, class_id: int, left: int, top: int, height: int):
    key = (sid, frame_num, class_id, left, top, height)
    parent_id = _pseudo_parent_map.get(key)
    if parent_id is not None:
        return parent_id

    for (ksid, kframe, kclass, kleft, ktop, kheight), value in _pseudo_parent_map.items():
        if ksid != sid or kframe != frame_num or kclass != class_id:
            continue
        if abs(kleft - left) <= 2 and abs(ktop - top) <= 2 and abs(kheight - height) <= 2:
            return value
    return None

def _bbox_xyxy_from_tuple(bbox: tuple) -> tuple:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def _safe_parent(obj_meta):
    try:
        return obj_meta.parent
    except Exception:
        return None


def _is_vehicle_obj(obj_meta) -> bool:
    return (
        obj_meta is not None
        and obj_meta.unique_component_id == PGIE_UNIQUE_ID
        and obj_meta.class_id in VEHICLE_CLASS_IDS
    )


def _class_label(class_id: int) -> str:
    if 0 <= class_id < len(VEHICLE_LABELS):
        return VEHICLE_LABELS[class_id]
    return "cls{}".format(class_id)


def _get_vehicle_state(track_key: tuple) -> VehicleTrackState:
    state = _vehicle_states.get(track_key)
    if state is None:
        state = VehicleTrackState()
        _vehicle_states[track_key] = state
    return state


def _debug_event(event_type: str, payload: dict):
    if not _debug_jsonl_path:
        return
    try:
        row = dict(payload)
        row["event"] = event_type
        with open(_debug_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _geometry_associate_plate(plate_bbox: tuple, vehicles: dict):
    if not vehicles:
        return None
    px, py, pw, ph = plate_bbox
    pcx = px + pw / 2.0
    pcy = py + ph / 2.0
    best_key = None
    best_score = -1.0
    for track_key, data in vehicles.items():
        vx, vy, vw, vh = data["bbox"]
        if vw <= 0 or vh <= 0:
            continue
        contains_center = vx <= pcx <= vx + vw and vy <= pcy <= vy + vh
        ix1 = max(px, vx)
        iy1 = max(py, vy)
        ix2 = min(px + pw, vx + vw)
        iy2 = min(py + ph, vy + vh)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        plate_area = max(1, pw * ph)
        containment = inter / plate_area
        vertical_pos = max(0.0, min(1.0, (pcy - vy) / vh))
        score = containment * 2.0 + (1.0 if contains_center else 0.0) + vertical_pos * 0.25
        if score > best_score:
            best_score = score
            best_key = track_key
    return best_key if best_score >= 0.75 else None


def _resolve_vehicle_track_key(obj_meta, sid: int, vehicles: dict):
    parent = _safe_parent(obj_meta)
    if _is_vehicle_obj(parent):
        return (sid, parent.object_id)
    grandparent = _safe_parent(parent) if parent is not None else None
    if _is_vehicle_obj(grandparent):
        return (sid, grandparent.object_id)
    return _geometry_associate_plate(_bbox_tuple(obj_meta), vehicles)


def _crop_plate_from_frame(frame_image, bbox: tuple, pad_ratio: float = 0.15, min_width: int = 100):
    if frame_image is None:
        return None, bbox, 0.0, False
    fh, fw = frame_image.shape[:2]
    x, y, w, h = bbox
    pad_x = max(2, int(w * pad_ratio))
    pad_y = max(2, int(h * pad_ratio))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, 0, 0), 0.0, True

    raw = frame_image[y1:y2, x1:x2]
    crop = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR) if raw.shape[2] == 4 else raw.copy()
    ch, cw = crop.shape[:2]
    if cw < min_width and cw > 0:
        scale = max(2, int(math.ceil(min_width / float(cw))))
        crop = cv2.resize(crop, (cw * scale, ch * scale), interpolation=cv2.INTER_CUBIC)

    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        sharp = 0.0
    size_score = min(1.0, crop.shape[1] / 160.0) * 0.45 + min(1.0, crop.shape[0] / 48.0) * 0.20
    sharp_score = min(1.0, sharp / 120.0) * 0.25
    clipped = x1 == 0 or y1 == 0 or x2 == fw or y2 == fh
    clip_score = 0.10 if not clipped else 0.0
    quality = max(0.0, min(1.0, size_score + sharp_score + clip_score))
    return crop, (x1, y1, x2 - x1, y2 - y1), quality, clipped


# ── Visual Helpers ────────────────────────────────────────────────────────────

def _display_id(source_id: int, obj_id: int) -> int:
    global _next_short_id
    key = (source_id, obj_id)
    if key not in _short_id_map:
        _short_id_map[key] = _next_short_id
        _next_short_id += 1
    return _short_id_map[key]


def _hide_text(obj_meta):
    try:
        obj_meta.text_params.display_text = ""
        obj_meta.text_params.set_bg_clr = 0
        obj_meta.text_params.font_params.font_size = 1
        obj_meta.text_params.font_params.font_color.set(0.0, 0.0, 0.0, 0.0)
    except Exception:
        pass


def _place_label(obj_meta, y_pad: int = 18):
    try:
        r = obj_meta.rect_params
        obj_meta.text_params.x_offset = max(0, int(r.left))
        obj_meta.text_params.y_offset = max(0, int(r.top) - y_pad)
    except Exception:
        pass


def _update_continuous_fps(source_id: int):
    now = time.perf_counter()
    key = f"stream{source_id}"
    state = _fps_overlay_state.get(key)
    if state is None:
        _fps_overlay_state[key] = {"last_ts": now, "fps": 0.0}
        return

    dt = now - state["last_ts"]
    state["last_ts"] = now
    if dt <= 0.0:
        return

    instant_fps = 1.0 / dt
    prev_fps = state.get("fps", 0.0)
    if prev_fps <= 0.0:
        state["fps"] = instant_fps
    else:
        state["fps"] = (prev_fps * (1.0 - _fps_overlay_alpha)) + (instant_fps * _fps_overlay_alpha)


def _add_fps_overlay(batch_meta, frame_meta):
    if _fps_overlay_state:
        fps_text = " | ".join(
            f"{stream_id}: {data.get('fps', 0.0):.1f} FPS"
            for stream_id, data in sorted(_fps_overlay_state.items())
        )
    else:
        fps_text = "FPS: calculating..."

    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    display_meta.num_labels = 1
    text_params = display_meta.text_params[0]
    text_params.display_text = fps_text
    text_params.x_offset = 14
    text_params.y_offset = 14
    text_params.font_params.font_name = "Serif"
    text_params.font_params.font_size = 14
    text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    text_params.set_bg_clr = 1
    text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.65)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def _cleanup_history(active_ids: set):
    for key in active_ids:
        _object_last_seen[key] = _osd_probe_frame
    stale_cutoff = _osd_probe_frame - _STATE_STALE_AFTER_FRAMES
    stale_keys = {key for key, last_seen in _object_last_seen.items() if last_seen < stale_cutoff}
    for key in stale_keys:
        _object_last_seen.pop(key, None)
        _vehicle_states.pop(key, None)
        _short_id_map.pop(key, None)
        _plate_history.pop(key, None)
        _ocr_frame_cache.pop(key, None)
        _split_ocr.pop(key, None)
        _plate_text_seen.pop(key, None)


# ── Probes ────────────────────────────────────────────────────────────────────

def pgie_src_pad_buffer_probe(pad, info, u_data):
    """Keep only PGIE vehicle objects before nvtracker.

    The detector model can still output all 14 classes, but tracker input is
    constrained to vehicles so downstream state is keyed by vehicle track ID.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            try:
                next_obj = l_obj.next
            except StopIteration:
                next_obj = None

            if obj_meta.unique_component_id == PGIE_UNIQUE_ID and obj_meta.class_id not in VEHICLE_CLASS_IDS:
                try:
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                except Exception:
                    obj_meta.rect_params.border_width = 0
                    _hide_text(obj_meta)

            l_obj = next_obj

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def sgie3_sink_pad_buffer_probe(pad, info, u_data):
    """Before SGIE3: record plate bounding boxes in original frame coords.
    Also splits square plates into top/bottom pseudo-objects for LPRNet.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        sid = frame_meta.source_id
        fnum = frame_meta.frame_num

        if perf_data is not None:
            perf_data.update_fps("stream{}".format(sid))

        objs_to_add = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.class_id == LP_CLASS_ID and obj_meta.unique_component_id == PGIE_UNIQUE_ID:
                r = obj_meta.rect_params
                oid = obj_meta.object_id
                # Record plate rect for later use in OSD probe
                _plate_rects[(sid, oid)] = (r.left, r.top, r.width, r.height)

                ar = r.width / float(r.height) if r.height > 0 else 0
                if ar < _square_plate_ar_threshold:
                    parent_id = oid
                    frame_w = float(getattr(frame_meta, "source_frame_width", 0) or 0)
                    frame_h = float(getattr(frame_meta, "source_frame_height", 0) or 0)
                    
                    split_configs = [
                        (0.44 + _square_split_overlap/2.0, 0.56 + _square_split_overlap/2.0),
                        (0.5 + _square_split_overlap/2.0, 0.5 + _square_split_overlap/2.0),
                        (0.58 + _square_split_overlap/2.0, 0.58 + _square_split_overlap/2.0),
                    ]
                    
                    for s_top, s_bot in split_configs:
                        top_h = r.height * s_top
                        bot_h = r.height * s_bot
                        pad_x = r.width * _square_split_pad_x
                        pad_y = r.height * _square_split_pad_y
                        pseudo_left = max(0.0, r.left - pad_x)
                        pseudo_right = r.left + r.width + pad_x
                        if frame_w > 0:
                            pseudo_right = min(frame_w, pseudo_right)
                        pseudo_width = max(1.0, pseudo_right - pseudo_left)

                        top_y = max(0.0, r.top - pad_y)
                        top_bottom = r.top + top_h + pad_y
                        if frame_h > 0:
                            top_bottom = min(frame_h, top_bottom)
                        top_crop_h = max(1.0, top_bottom - top_y)

                        obj_top = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                        obj_top.class_id = LP_TOP_CLASS_ID
                        obj_top.unique_component_id = PGIE_UNIQUE_ID
                        obj_top.confidence = obj_meta.confidence
                        obj_top.rect_params.left   = pseudo_left
                        obj_top.rect_params.top    = top_y
                        obj_top.rect_params.width  = pseudo_width
                        obj_top.rect_params.height = top_crop_h
                        objs_to_add.append((obj_top, obj_meta))
                        _pseudo_parent_map[(sid, fnum, LP_TOP_CLASS_ID, int(pseudo_left), int(top_y), int(top_crop_h))] = parent_id

                        bot_y = max(0.0, r.top + r.height - bot_h - pad_y)
                        bot_bottom = r.top + r.height + pad_y
                        if frame_h > 0:
                            bot_bottom = min(frame_h, bot_bottom)
                        bot_crop_h = max(1.0, bot_bottom - bot_y)

                        obj_bot = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                        obj_bot.class_id = LP_BOT_CLASS_ID
                        obj_bot.unique_component_id = PGIE_UNIQUE_ID
                        obj_bot.confidence = obj_meta.confidence
                        obj_bot.rect_params.left   = pseudo_left
                        obj_bot.rect_params.top    = bot_y
                        obj_bot.rect_params.width  = pseudo_width
                        obj_bot.rect_params.height = bot_crop_h
                        objs_to_add.append((obj_bot, obj_meta))
                        _pseudo_parent_map[(sid, fnum, LP_BOT_CLASS_ID, int(pseudo_left), int(bot_y), int(bot_crop_h))] = parent_id

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        for new_obj, parent_obj in objs_to_add:
            pyds.nvds_add_obj_meta_to_frame(frame_meta, new_obj, parent_obj)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# YOLO-char decoding and custom parser functions removed.

import re as _re
# VN plates: 2-digit province + 1-2 letters (+optional digit) + dash + 4-5 alphanumeric
_VN_PLATE_RE = _re.compile(r'^\d{2}[A-Z]{1,2}\d?-[\dA-Z]{4,6}$')

def _is_valid_vn_plate(text: str) -> bool:
    """Rough sanity check: 6-10 chars including exactly one dash."""
    if text.count('-') != 1:
        return False
    prefix, suffix = text.split('-')
    if not (2 <= len(prefix) <= 5 and 3 <= len(suffix) <= 6):
        return False
    return True

# YOLO-char and multi-backend OCR runner functions removed.

def _associate_plate_to_vehicle(plate_meta, vehicles_list):
    parent = _safe_parent(plate_meta)
    if parent and _is_vehicle_obj(parent):
        return parent.object_id, "parent", 1.0

    px, py, pw, ph = _bbox_tuple(plate_meta)
    pcx = px + pw / 2
    pcy = py + ph / 2

    best_vid = -1
    best_score = 0.0
    
    for v in vehicles_list:
        vx, vy, vw, vh = _bbox_tuple(v)
        # Check if plate center is inside vehicle
        if vx <= pcx <= vx + vw and vy <= pcy <= vy + vh:
            # Containment score
            overlap_w = max(0, min(px + pw, vx + vw) - max(px, vx))
            overlap_h = max(0, min(py + ph, vy + vh) - max(py, vy))
            overlap_area = overlap_w * overlap_h
            plate_area = pw * ph
            containment = overlap_area / plate_area if plate_area > 0 else 0
            
            # Vertical pos (plates are usually in lower half)
            v_bottom_half = vy + vh / 2
            v_score = 1.0 if pcy >= v_bottom_half else 0.5
            
            score = containment * v_score
            if score > best_score:
                best_score = score
                best_vid = v.object_id
                
    if best_vid != -1:
        return best_vid, "geometry", best_score
        
    return -1, "none", 0.0

def _build_lpr_event(state: "VehicleTrackState", frame_num: int) -> dict:
    import datetime
    return {
        "event_id": f"{state.source_id}_{state.vehicle_tracker_id}_{frame_num}",
        "event_type": "license_plate_recognized",
        "schema_version": "1.0",
        "source_id": state.source_id,
        "source_uri": _source_uri_by_id.get(state.source_id, ""),
        "frame_num": frame_num,
        "pts": state.last_pts,
        "created_at": datetime.datetime.now().isoformat(),
        "vehicle": {
            "tracker_id": state.vehicle_tracker_id,
            "display_id": state.display_id,
            "class_id": state.vehicle_class,
            "class_name": state.vehicle_class_name,
            "confidence": round(state.vehicle_confidence, 4),
            "bbox": list(state.vehicle_bbox),
        },
        "plate": {
            "object_id": state.best_plate_object_id,
            "bbox": list(state.best_plate_bbox),
            "text_raw": state.best_plate_text_raw,
            "text_stable": state.best_plate_text_stable,
            "score": round(state.best_score, 4),
            "votes": int(state.best_votes),
            "ocr_confidence": round(state.ocr_confidence, 4),
            "ocr_backend": "lprnet",
            "stable": bool(state.best_plate_text_stable),
        },
        "association": {
            "method": state.association_method,
            "score": round(state.association_score, 4),
        },
        "media": {
            "plate_image_path": state.crop_plate_path,
            "vehicle_image_path": state.crop_vehicle_path,
            "frame_image_path": state.frame_path,
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


def _emit_event(state: "VehicleTrackState", frame_num: int):
    import json

    is_valid_vehicle = state.association_method != "none" and state.vehicle_class != -1

    if not is_valid_vehicle and _debug_jsonl_path:
        try:
            with open(_debug_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "event": "unassociated_plate",
                    "source_id": state.source_id,
                    "frame_num": frame_num,
                    "plate_object_id": state.best_plate_object_id,
                    "plate_bbox": list(state.best_plate_bbox),
                    "raw_text": state.best_plate_text_raw,
                }) + "\n")
        except Exception:
            pass

    if not is_valid_vehicle:
        return
    if not _event_jsonl_path and not _kafka_enabled:
        return

    event = _build_lpr_event(state, frame_num)
    event_json = json.dumps(event, ensure_ascii=False)

    if _event_jsonl_path:
        try:
            with open(_event_jsonl_path, "a", encoding="utf-8") as f:
                f.write(event_json + "\n")
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to write event JSONL: {e}\n")

    if _kafka_enabled and _kafka_producer is not None:
        try:
            kafka_key = f"{state.source_id}:{state.vehicle_tracker_id}".encode("utf-8")
            _kafka_producer.produce(
                _kafka_topic,
                key=kafka_key,
                value=event_json.encode("utf-8"),
                on_delivery=_kafka_delivery_cb,
            )
            _kafka_producer.poll(0)
        except Exception as e:
            sys.stderr.write(f"[WARN] Kafka produce failed (event still saved locally): {e}\n")

def metadata_src_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        sid = frame_meta.source_id
        frame_num = frame_meta.frame_num
        pts = getattr(frame_meta, "buf_pts", 0)
        if not pts:
            pts = getattr(gst_buffer, "pts", 0)
        
        frame_image = None
        if _event_output_dir or _save_event_frame:
            try:
                frame_image = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            except Exception:
                frame_image = None

        vehicles = []
        plates = []
        plate_parts = []

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
                
            if (
                obj_meta.unique_component_id == PGIE_UNIQUE_ID
                and obj_meta.class_id in VEHICLE_CLASS_IDS
                and obj_meta.object_id != UNTRACKED_OBJECT_ID
            ):
                vehicles.append(obj_meta)
            elif obj_meta.class_id == LP_CLASS_ID and obj_meta.unique_component_id == PGIE_UNIQUE_ID:
                plates.append(obj_meta)
            elif obj_meta.class_id in (LP_TOP_CLASS_ID, LP_BOT_CLASS_ID) and obj_meta.unique_component_id == PGIE_UNIQUE_ID:
                plate_parts.append(obj_meta)
                
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        for v in vehicles:
            vid = v.object_id
            track_key = (sid, vid)
            if track_key not in _vehicle_states:
                _vehicle_states[track_key] = VehicleTrackState(
                    display_id=_display_id(sid, vid),
                    source_id=sid,
                    vehicle_tracker_id=vid,
                    first_seen_frame=frame_num,
                    vehicle_class=v.class_id,
                    vehicle_class_name=_class_label(v.class_id).replace("_", " ").title(),
                )
                
            state = _vehicle_states[track_key]
            state.last_seen_frame = frame_num
            state.vehicle_confidence = float(v.confidence)
            
            raw_v_bbox = _bbox_tuple(v)
            state.vehicle_bbox_raw = raw_v_bbox
            
            if state.vehicle_bbox == (0, 0, 0, 0):
                state.vehicle_bbox = raw_v_bbox
            else:
                iou = _bbox_iou(state.vehicle_bbox, raw_v_bbox)
                cdist = _bbox_center_distance(state.vehicle_bbox, raw_v_bbox)
                max_jump = _bbox_max_center_jump_ratio * max(raw_v_bbox[2], raw_v_bbox[3])
                if iou < _bbox_reset_iou or cdist > max_jump:
                    state.vehicle_bbox = raw_v_bbox
                else:
                    state.vehicle_bbox = _smooth_bbox(state.vehicle_bbox, raw_v_bbox, _bbox_smooth_alpha)
            state.last_bbox_update_frame = frame_num

        plate_part_texts = {}
        for part in plate_parts:
            r = part.rect_params
            parent_id = _pseudo_parent_lookup(
                sid, frame_num, part.class_id, int(r.left), int(r.top), int(r.height)
            )
            if parent_id is not None:
                if parent_id not in plate_part_texts:
                    plate_part_texts[parent_id] = {"top": [], "bot": []}
                
                text, conf = _read_lpr_text(part, SGIE3_UNIQUE_ID)
                norm = _correct_vn_plate(text)
                if part.class_id == LP_TOP_CLASS_ID:
                    plate_part_texts[parent_id]["top"].append((norm, conf))
                else:
                    plate_part_texts[parent_id]["bot"].append((norm, conf))

        frame_best_plates = {}
        frame_plate_seen: dict = {}  # track_key → (p, area): best plate per vehicle this frame
        for p in plates:
            if p.confidence > 0.0 and p.confidence < _min_plate_conf:
                continue
            if p.rect_params.width < _min_plate_width or p.rect_params.height < _min_plate_height:
                continue
            _metrics["plate_objects"] += 1

            vid, assoc_method, assoc_score = _associate_plate_to_vehicle(p, vehicles)
            if vid == -1:
                vid = p.object_id
                if (sid, vid) not in _vehicle_states:
                    _vehicle_states[(sid, vid)] = VehicleTrackState(
                        display_id=_display_id(sid, vid), source_id=sid, vehicle_tracker_id=vid,
                        first_seen_frame=frame_num, vehicle_class=-1, vehicle_class_name="Unknown Vehicle",
                    )
                assoc_method = "none"
                assoc_score = 0.0
            
            track_key = (sid, vid)
            # Always track plate→vehicle so OSD can find the state even when OCR fails
            _p_area = p.rect_params.width * p.rect_params.height
            if track_key not in frame_plate_seen or _p_area > frame_plate_seen[track_key][1]:
                frame_plate_seen[track_key] = (p, _p_area)
            parent_id = p.object_id
            full_text, full_conf = _read_lpr_text(p, SGIE3_UNIQUE_ID)
            full_raw_text = _correct_vn_plate(full_text)
            
            candidates_to_eval = []
            seen_candidates = set()

            def _add_candidate(text: str, conf: float):
                text = _correct_vn_plate(text)
                if not text or text in seen_candidates:
                    return
                seen_candidates.add(text)
                candidates_to_eval.append((text, conf))

            _add_candidate(full_raw_text, full_conf)
            
            tops = plate_part_texts.get(parent_id, {}).get("top", [])
            bots = plate_part_texts.get(parent_id, {}).get("bot", [])
            
            for t_txt, t_conf in tops:
                for b_txt, b_conf in bots:
                    for joined in _square_join_variants(t_txt, b_txt):
                        _add_candidate(joined, (t_conf + b_conf) / 2.0)
                _add_candidate(t_txt, t_conf)
            for b_txt, b_conf in bots:
                _add_candidate(b_txt, b_conf)
                    
            best_cand_text = ""
            best_cand_score = -1.0
            best_cand_conf = 0.0
            
            for cand, conf in candidates_to_eval:
                score = _plate_quality_score(cand, conf, p.rect_params.width, p.rect_params.height, assoc_score)
                if score > best_cand_score:
                    best_cand_score = score
                    best_cand_text = cand
                    best_cand_conf = conf

            if _debug_jsonl_path:
                _p_ar = p.rect_params.width / float(p.rect_params.height) if p.rect_params.height > 0 else 99.0
                if _p_ar < _square_plate_ar_threshold:
                    _debug_event("square_plate_ocr", {
                        "frame_num": frame_num,
                        "source_id": sid,
                        "plate_object_id": parent_id,
                        "ar": round(_p_ar, 2),
                        "tops_found": len(tops),
                        "bots_found": len(bots),
                        "parent_in_parts": parent_id in plate_part_texts,
                        "full_text": full_raw_text,
                        "full_conf": round(full_conf, 3),
                        "top_cands": [(t, round(c, 3)) for t, c in tops[:3]],
                        "bot_cands": [(b, round(c, 3)) for b, c in bots[:3]],
                        "best_cand": best_cand_text,
                        "best_score": round(max(best_cand_score, 0.0), 3),
                        "display_text": _vehicle_states.get(track_key, VehicleTrackState()).display_plate_text,
                        "stable_text": _vehicle_states.get(track_key, VehicleTrackState()).best_plate_text_stable,
                    })

            if best_cand_score <= 0.0:
                continue
            _metrics["ocr_raw_events"] += 1

            if track_key not in frame_best_plates or best_cand_score > frame_best_plates[track_key]['score']:
                frame_best_plates[track_key] = {
                    'p': p, 'score': best_cand_score, 'text': best_cand_text, 'conf': best_cand_conf,
                    'assoc_method': assoc_method, 'assoc_score': assoc_score
                }

        for track_key, b in frame_best_plates.items():
            state = _vehicle_states[track_key]
            p = b['p']
            state.best_plate_object_id = p.object_id
            state.association_method = b['assoc_method']
            state.association_score = b['assoc_score']
            
            raw_p_bbox = _bbox_tuple(p)
            state.plate_bbox_raw = raw_p_bbox
            
            if state.best_plate_bbox == (0, 0, 0, 0):
                state.best_plate_bbox = raw_p_bbox
            else:
                iou = _bbox_iou(state.best_plate_bbox, raw_p_bbox)
                cdist = _bbox_center_distance(state.best_plate_bbox, raw_p_bbox)
                max_jump = _bbox_max_center_jump_ratio * max(raw_p_bbox[2], raw_p_bbox[3])
                if iou < _bbox_reset_iou * 1.5 or cdist > max_jump * 0.8:
                    state.best_plate_bbox = raw_p_bbox
                else:
                    state.best_plate_bbox = _smooth_bbox(state.best_plate_bbox, raw_p_bbox, _bbox_smooth_alpha)
            
            display_pattern_score = _plate_pattern_score(state.display_plate_text)
            candidate_pattern_score = _plate_pattern_score(b["text"])
            display_candidate_better = (
                not state.display_plate_text
                or b["score"] >= state.display_plate_score
                or candidate_pattern_score > display_pattern_score
            )
            if display_candidate_better:
                state.best_plate_text_raw = b['text']
                state.display_plate_text = b['text']
                state.display_plate_score = b['score']
                state.ocr_confidence = b['conf']

            stable_text = _stable_plate(track_key, b['text'], b['conf'], p.rect_params.width, p.rect_params.height, b['assoc_score'])
            stable_votes, stable_score = _plate_history_stats(track_key, stable_text) if stable_text else (0, 0.0)
            
            if b['score'] > state.best_score:
                state.best_plate_object_id = p.object_id
                state.association_method = b['assoc_method']
                state.association_score = b['assoc_score']
                state.best_plate_text_raw = b['text']
                state.ocr_confidence = b['conf']
                
            state.last_pts = pts
            
            if not stable_text or stable_score <= 0.0:
                continue
                
            current_text = state.best_plate_text_stable
            text_changed = stable_text != current_text
            candidate_better = _should_replace_stable_text(
                current_text, state.best_score, state.best_votes,
                stable_text, stable_score, stable_votes
            )
                
            if not candidate_better:
                continue
                
            if text_changed and current_text:
                state.plate_text_switches += 1

            state.best_plate_text_stable = stable_text
            state.display_plate_text = stable_text
            state.display_plate_score = max(state.display_plate_score, stable_score)
            state.best_votes = stable_votes
            state.best_score = stable_score if text_changed else max(state.best_score, stable_score)
            _plate_text_seen[track_key] = {
                "text": state.best_plate_text_stable, "stable": True,
                "score": state.best_score, "votes": state.best_votes,
            }
            
            # ── Anti-spam emit decision ───────────────────────────────────────
            if state.association_method == "none" or state.vehicle_class == -1:
                continue

            event_key = (state.source_id, state.vehicle_tracker_id, stable_text)
            frames_since_event = frame_num - state.last_event_frame

            if _emit_duplicates:
                emit_now = True
            elif event_key in _emitted_event_keys:
                # Same (source, vehicle, plate_text) already emitted this session.
                # Allow repeat only if caller explicitly set a cooldown.
                emit_now = (
                    _event_repeat_cooldown_frames > 0
                    and frames_since_event >= _event_repeat_cooldown_frames
                )
            else:
                # New key: first event for this vehicle, or a different plate text.
                if not state.last_emitted_plate_text:
                    emit_now = True
                else:
                    # Different plate text than last emitted — require it to be better.
                    stable_pattern_score = _plate_pattern_score(stable_text)
                    emitted_pattern_score = _plate_pattern_score(state.last_emitted_plate_text)
                    emit_now = (
                        stable_score >= state.last_emitted_score + 0.5
                        or stable_pattern_score > emitted_pattern_score
                    )

            if emit_now:
                if _event_output_dir and frame_image is not None:
                    p_bgr, _, _, _ = _crop_plate_from_frame(frame_image, state.best_plate_bbox, 0.0)
                    if p_bgr is not None:
                        fname = f"{sid}_{vid}_{frame_num}_plate.jpg"
                        state.crop_plate_path = os.path.abspath(os.path.join(_event_output_dir, fname))
                        cv2.imwrite(state.crop_plate_path, p_bgr)

                    if state.vehicle_bbox != (0, 0, 0, 0):
                        v_bgr, _, _, _ = _crop_plate_from_frame(frame_image, state.vehicle_bbox, 0.0)
                        if v_bgr is not None:
                            fname = f"{sid}_{vid}_{frame_num}_vehicle.jpg"
                            state.crop_vehicle_path = os.path.abspath(os.path.join(_event_output_dir, fname))
                            cv2.imwrite(state.crop_vehicle_path, v_bgr)

                    if _save_event_frame:
                        fname = f"{sid}_{vid}_{frame_num}_frame.jpg"
                        state.frame_path = os.path.abspath(os.path.join(_event_output_dir, fname))
                        full_bgr = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR) if frame_image.shape[2] == 4 else frame_image
                        cv2.imwrite(state.frame_path, full_bgr)

                _emitted_event_keys.add(event_key)
                state.last_event_frame = frame_num
                state.last_emitted_plate_text = state.best_plate_text_stable
                state.last_emitted_score = state.best_score
                _emit_event(state, frame_num)

        # Fallback: states that had a plate this frame but no valid OCR candidate still
        # need best_plate_object_id set so the OSD probe can find and label them.
        for track_key, (p_seen, _) in frame_plate_seen.items():
            if track_key not in frame_best_plates:
                state_s = _vehicle_states.get(track_key)
                if state_s is not None:
                    state_s.best_plate_object_id = p_seen.object_id
                    raw_p_bbox = _bbox_tuple(p_seen)
                    if state_s.best_plate_bbox == (0, 0, 0, 0):
                        state_s.best_plate_bbox = raw_p_bbox
                    else:
                        iou = _bbox_iou(state_s.best_plate_bbox, raw_p_bbox)
                        cdist = _bbox_center_distance(state_s.best_plate_bbox, raw_p_bbox)
                        max_jump = _bbox_max_center_jump_ratio * max(raw_p_bbox[2], raw_p_bbox[3])
                        if iou < _bbox_reset_iou * 1.5 or cdist > max_jump * 0.8:
                            state_s.best_plate_bbox = raw_p_bbox
                        else:
                            state_s.best_plate_bbox = _smooth_bbox(state_s.best_plate_bbox, raw_p_bbox, _bbox_smooth_alpha)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def osd_sink_pad_buffer_probe(pad, info, u_data):
    import pyds
    global _osd_probe_frame, _cleanup_counter
    _osd_probe_frame += 1
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    all_visible_keys = set()
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        sid = frame_meta.source_id
        frame_num = frame_meta.frame_num
        
        t_rows = max(1, _tiler_rows)
        t_cols = max(1, _tiler_cols)
        tile_w = TILER_WIDTH / t_cols
        tile_h = TILER_HEIGHT / t_rows
        
        visible_vehicle_keys = set()
        visible_plate_object_ids = set()

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
                
            if obj_meta.unique_component_id == SGIE3_UNIQUE_ID:
                obj_meta.rect_params.border_width = 0
                obj_meta.rect_params.has_bg_color = 0
                _hide_text(obj_meta)
            else:
                cid = obj_meta.class_id
                if obj_meta.unique_component_id == PGIE_UNIQUE_ID:
                    # Calculate original source ID from object coordinates on tiled frame
                    r = obj_meta.rect_params
                    cx = r.left + r.width / 2.0
                    cy = r.top + r.height / 2.0
                    col = max(0, min(t_cols - 1, int(cx / tile_w)))
                    row = max(0, min(t_rows - 1, int(cy / tile_h)))
                    obj_sid = row * t_cols + col

                    if cid in VEHICLE_CLASS_IDS and obj_meta.object_id != UNTRACKED_OBJECT_ID:
                        visible_vehicle_keys.add((obj_sid, obj_meta.object_id))
                    elif cid == LP_CLASS_ID:
                        visible_plate_object_ids.add(obj_meta.object_id)

                    if cid in (LP_TOP_CLASS_ID, LP_BOT_CLASS_ID):
                        obj_meta.rect_params.border_width = 0
                        obj_meta.rect_params.has_bg_color = 0
                        _hide_text(obj_meta)
                        try:
                            l_obj = l_obj.next
                        except StopIteration:
                            break
                        continue

                    color = _CLASS_COLORS[cid % len(_CLASS_COLORS)]
                    label = _class_label(cid).replace("_", " ").title()
                    
                    if cid == LP_CLASS_ID:
                        best_vid = obj_meta.object_id
                        shown_text = ""
                        for key, state in _vehicle_states.items():
                            if state.source_id == obj_sid and state.best_plate_object_id == obj_meta.object_id:
                                best_vid = state.vehicle_tracker_id
                                shown_text = state.best_plate_text_stable or state.display_plate_text
                                break

                        display_id = _display_id(obj_sid, best_vid)
                        if shown_text:
                            display_text = f"Plate: {shown_text} #{display_id}"
                        else:
                            display_text = f"Plate #{display_id}"
                    else:
                        display_id = _display_id(obj_sid, obj_meta.object_id)
                        display_text = f"{label} #{display_id}"
                        
                    r = obj_meta.rect_params
                    r.border_width = 2 if cid == LP_CLASS_ID else 3
                    r.border_color.set(color[0], color[1], color[2], color[3])
                    r.has_bg_color = 0
                    
                    if display_text:
                        obj_meta.text_params.display_text = display_text
                        _place_label(obj_meta, 18)
                        obj_meta.text_params.font_params.font_name = "Serif"
                        obj_meta.text_params.font_params.font_size = 11
                        obj_meta.text_params.font_params.font_color.set(color[0], color[1], color[2], 1.0)
                        obj_meta.text_params.set_bg_clr = 1
                        obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.65)
                    else:
                        _hide_text(obj_meta)
                
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        _add_fps_overlay(batch_meta, frame_meta)
        all_visible_keys.update(visible_vehicle_keys)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    _cleanup_counter += 1
    if _cleanup_counter % _CLEANUP_INTERVAL == 0:
        _cleanup_history(all_visible_keys)

    return Gst.PadProbeReturn.OK

def _make_el(factory: str, name: str):
    el = Gst.ElementFactory.make(factory, name)
    if not el:
        sys.stderr.write("[ERROR] Cannot create {} ({})\n".format(name, factory))
        sys.exit(1)
    return el


def cb_newpad(decodebin, decoder_src_pad, sinkpad):
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    features = caps.get_features(0)

    if gstname.find("video") != -1:
        if features.contains("memory:NVMM"):
            if decoder_src_pad.link(sinkpad) != Gst.PadLinkReturn.OK:
                sys.stderr.write("[ERROR] Failed to link decoder src pad to streammux sink pad\n")
        else:
            sys.stderr.write("[ERROR] Decodebin did not pick NVIDIA decoder plugin.\n")


def make_uri(s: str) -> str:
    if s.startswith("rtsp://") or s.startswith("file://") or s.startswith("http://") or s.startswith("https://"):
        return s
    return "file://" + os.path.abspath(s)

def _parse_args(args):
    no_display   = "--no-display" in args
    output_file  = None
    sources      = []
    ocr_backend  = "lprnet"
    save_crops   = None
    debug_jsonl  = None
    event_output_dir = None
    event_jsonl = None
    event_cooldown_frames = 60
    min_stable_votes = 2
    save_event_frame = False
    pgie_interval = 0
    min_plate_conf = 0.05
    min_plate_width = 20
    min_plate_height = 6
    bbox_smooth_alpha = 0.4
    bbox_reset_iou = 0.2
    bbox_max_center_jump_ratio = 0.5
    square_plate_ar_threshold = 1.7
    square_split_overlap = 0.12
    square_split_pad_x = 0.12
    square_split_pad_y = 0.08
    ocr_every_n  = 6
    ocr_min_conf = 0.0
    emit_duplicates = False
    event_repeat_cooldown_frames = 0
    kafka_enable = False
    kafka_bootstrap_server = "localhost:9092"
    kafka_topic = "lpr.events.v1"
    kafka_client_id = "ds-lpr-producer"
    skip_next    = False
    pgie_config  = None

    i = 1
    while i < len(args):
        a = args[i]
        if skip_next:
            skip_next = False
            i += 1
            continue

        def _nextval():
            nonlocal i
            if i + 1 < len(args):
                i += 1
                return args[i]
            sys.stderr.write(f"[ERROR] {a} requires a value\n")
            sys.exit(1)

        if a == "--no-display":
            pass
        elif a == "--pgie-config":
            pgie_config = _nextval()
        elif a.startswith("--pgie-config="):
            pgie_config = a.split("=", 1)[1]
        elif a in ("--output", "-o"):
            output_file = _nextval()
        elif a.startswith("--output="):
            output_file = a.split("=", 1)[1]
        elif a == "--save-crops":
            save_crops = _nextval()
        elif a.startswith("--save-crops="):
            save_crops = a.split("=", 1)[1]
        elif a == "--debug-jsonl":
            debug_jsonl = _nextval()
        elif a.startswith("--debug-jsonl="):
            debug_jsonl = a.split("=", 1)[1]
        elif a == "--event-output-dir":
            event_output_dir = _nextval()
        elif a.startswith("--event-output-dir="):
            event_output_dir = a.split("=", 1)[1]
        elif a == "--min-plate-conf":
            min_plate_conf = float(_nextval())
        elif a.startswith("--min-plate-conf="):
            min_plate_conf = float(a.split("=", 1)[1])
        elif a == "--pgie-interval":
            pgie_interval = int(_nextval())
        elif a.startswith("--pgie-interval="):
            pgie_interval = int(a.split("=", 1)[1])
        elif a == "--min-plate-width":
            min_plate_width = int(_nextval())
        elif a.startswith("--min-plate-width="):
            min_plate_width = int(a.split("=", 1)[1])
        elif a == "--min-plate-height":
            min_plate_height = int(_nextval())
        elif a.startswith("--min-plate-height="):
            min_plate_height = int(a.split("=", 1)[1])
        elif a == "--bbox-smooth-alpha":
            bbox_smooth_alpha = float(_nextval())
        elif a.startswith("--bbox-smooth-alpha="):
            bbox_smooth_alpha = float(a.split("=", 1)[1])
        elif a == "--bbox-reset-iou":
            bbox_reset_iou = float(_nextval())
        elif a.startswith("--bbox-reset-iou="):
            bbox_reset_iou = float(a.split("=", 1)[1])
        elif a == "--bbox-max-center-jump-ratio":
            bbox_max_center_jump_ratio = float(_nextval())
        elif a.startswith("--bbox-max-center-jump-ratio="):
            bbox_max_center_jump_ratio = float(a.split("=", 1)[1])
        elif a == "--square-plate-ar-threshold":
            square_plate_ar_threshold = float(_nextval())
        elif a.startswith("--square-plate-ar-threshold="):
            square_plate_ar_threshold = float(a.split("=", 1)[1])
        elif a == "--square-split-overlap":
            square_split_overlap = float(_nextval())
        elif a.startswith("--square-split-overlap="):
            square_split_overlap = float(a.split("=", 1)[1])
        elif a == "--square-split-pad-x":
            square_split_pad_x = float(_nextval())
        elif a.startswith("--square-split-pad-x="):
            square_split_pad_x = float(a.split("=", 1)[1])
        elif a == "--square-split-pad-y":
            square_split_pad_y = float(_nextval())
        elif a.startswith("--square-split-pad-y="):
            square_split_pad_y = float(a.split("=", 1)[1])
        elif a == "--save-event-frame":
            save_event_frame = True
        elif a == "--event-jsonl":
            event_jsonl = _nextval()
        elif a.startswith("--event-jsonl="):
            event_jsonl = a.split("=", 1)[1]
        elif a == "--event-cooldown-frames":
            event_cooldown_frames = int(_nextval())
        elif a.startswith("--event-cooldown-frames="):
            event_cooldown_frames = int(a.split("=", 1)[1])
        elif a == "--min-stable-votes":
            min_stable_votes = int(_nextval())
        elif a.startswith("--min-stable-votes="):
            min_stable_votes = int(a.split("=", 1)[1])
        elif a == "--ocr-every-n-frames":
            ocr_every_n = int(_nextval())
        elif a.startswith("--ocr-every-n-frames="):
            ocr_every_n = int(a.split("=", 1)[1])
        elif a == "--ocr-min-conf":
            ocr_min_conf = float(_nextval())
        elif a.startswith("--ocr-min-conf="):
            ocr_min_conf = float(a.split("=", 1)[1])
        elif a == "--emit-duplicates":
            emit_duplicates = True
        elif a == "--event-repeat-cooldown-frames":
            event_repeat_cooldown_frames = int(_nextval())
        elif a.startswith("--event-repeat-cooldown-frames="):
            event_repeat_cooldown_frames = int(a.split("=", 1)[1])
        elif a == "--kafka-enable":
            kafka_enable = True
        elif a == "--kafka-bootstrap-server":
            kafka_bootstrap_server = _nextval()
        elif a.startswith("--kafka-bootstrap-server="):
            kafka_bootstrap_server = a.split("=", 1)[1]
        elif a == "--kafka-topic":
            kafka_topic = _nextval()
        elif a.startswith("--kafka-topic="):
            kafka_topic = a.split("=", 1)[1]
        elif a == "--kafka-client-id":
            kafka_client_id = _nextval()
        elif a.startswith("--kafka-client-id="):
            kafka_client_id = a.split("=", 1)[1]
        elif not a.startswith("--"):
            sources.append(a)
        else:
            sys.stderr.write(f"[WARN] Unknown argument: {a}\n")
        i += 1

    return dict(
        no_display=no_display,
        output_file=output_file,
        sources=sources,
        ocr_backend=ocr_backend,
        save_crops=save_crops,
        debug_jsonl=debug_jsonl,
        event_output_dir=event_output_dir,
        event_jsonl=event_jsonl,
        event_cooldown_frames=event_cooldown_frames,
        min_stable_votes=min_stable_votes,
        save_event_frame=save_event_frame,
        pgie_interval=pgie_interval,
        min_plate_conf=min_plate_conf,
        min_plate_width=min_plate_width,
        min_plate_height=min_plate_height,
        bbox_smooth_alpha=bbox_smooth_alpha,
        bbox_reset_iou=bbox_reset_iou,
        bbox_max_center_jump_ratio=bbox_max_center_jump_ratio,
        square_plate_ar_threshold=square_plate_ar_threshold,
        square_split_overlap=square_split_overlap,
        square_split_pad_x=square_split_pad_x,
        square_split_pad_y=square_split_pad_y,
        ocr_every_n=ocr_every_n,
        ocr_min_conf=ocr_min_conf,
        emit_duplicates=emit_duplicates,
        event_repeat_cooldown_frames=event_repeat_cooldown_frames,
        kafka_enable=kafka_enable,
        kafka_bootstrap_server=kafka_bootstrap_server,
        kafka_topic=kafka_topic,
        kafka_client_id=kafka_client_id,
        pgie_config=pgie_config,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    global perf_data, _short_id_map, _plate_history, _next_short_id, _cleanup_counter, \
           _lpr_layer_warning_printed, _pseudo_parent_map, _split_ocr, \
           _ocr_frame_cache, \
           _ocr_backend, _OCR_EVERY_N, _OCR_MIN_CONF, \
           _save_crops_dir, _osd_probe_frame, _save_crop_seq, _vehicle_states, \
           _object_last_seen, _plate_text_seen, _debug_jsonl_path, _metrics, \
           _event_output_dir, _event_jsonl_path, _event_cooldown_frames, _min_stable_votes, \
           _source_uri_by_id, _save_event_frame, \
           _min_plate_conf, _min_plate_width, _min_plate_height, \
           _bbox_smooth_alpha, _bbox_reset_iou, _bbox_max_center_jump_ratio, \
           _square_plate_ar_threshold, _square_split_overlap, \
           _square_split_pad_x, _square_split_pad_y, \
           _emitted_event_keys, _emit_duplicates, _event_repeat_cooldown_frames, \
           _kafka_enabled, _kafka_producer, _kafka_topic

    _short_id_map      = {}
    _plate_history     = {}
    _vehicle_states    = {}
    _pseudo_parent_map = {}
    _split_ocr         = {}
    _ocr_frame_cache   = {}
    _osd_probe_frame   = 0
    _save_crop_seq     = {}
    _object_last_seen   = {}
    _plate_text_seen    = {}
    _emitted_event_keys = set()
    _debug_jsonl_path   = None
    _source_uri_by_id   = {}
    _save_event_frame   = False
    _emit_duplicates    = False
    _event_repeat_cooldown_frames = 0
    _kafka_enabled      = False
    _kafka_producer     = None
    _kafka_topic        = "lpr.events.v1"
    _min_plate_conf     = 0.05
    _min_plate_width    = 20
    _min_plate_height   = 6
    _bbox_smooth_alpha  = 0.4
    _bbox_reset_iou     = 0.2
    _bbox_max_center_jump_ratio = 0.5
    _square_plate_ar_threshold = 1.7
    _square_split_overlap = 0.12
    _square_split_pad_x = 0.12
    _square_split_pad_y = 0.08
    _metrics           = Counter()
    _next_short_id     = 1
    _cleanup_counter   = 0
    _lpr_layer_warning_printed = False

    cfg = _parse_args(args)
    if cfg.get("pgie_config"):
        global PGIE_CONFIG_PATH
        PGIE_CONFIG_PATH = os.path.abspath(cfg["pgie_config"])

    no_display  = cfg["no_display"]
    output_file = cfg["output_file"]
    sources_raw = cfg["sources"]

    _ocr_backend  = cfg["ocr_backend"]
    _OCR_EVERY_N  = cfg["ocr_every_n"]
    _OCR_MIN_CONF = cfg["ocr_min_conf"]
    _save_crops_dir = cfg["save_crops"]
    if _save_crops_dir:
        os.makedirs(_save_crops_dir, exist_ok=True)
    _debug_jsonl_path = cfg["debug_jsonl"]
    if _debug_jsonl_path is None and _save_crops_dir:
        _debug_jsonl_path = os.path.join(_save_crops_dir, "debug_events.jsonl")
    if _debug_jsonl_path:
        os.makedirs(os.path.dirname(os.path.abspath(_debug_jsonl_path)), exist_ok=True)

    _event_output_dir = os.path.abspath(cfg["event_output_dir"]) if cfg["event_output_dir"] else None
    _event_jsonl_path = os.path.abspath(cfg["event_jsonl"]) if cfg["event_jsonl"] else None
    _event_cooldown_frames = cfg["event_cooldown_frames"]
    _save_event_frame = cfg["save_event_frame"]
    _emit_duplicates = cfg["emit_duplicates"]
    _event_repeat_cooldown_frames = cfg["event_repeat_cooldown_frames"]
    _min_plate_conf = cfg["min_plate_conf"]
    _min_plate_width = cfg["min_plate_width"]
    _min_plate_height = cfg["min_plate_height"]
    _bbox_smooth_alpha = cfg["bbox_smooth_alpha"]
    _bbox_reset_iou = cfg["bbox_reset_iou"]
    _bbox_max_center_jump_ratio = cfg["bbox_max_center_jump_ratio"]
    _square_plate_ar_threshold = cfg["square_plate_ar_threshold"]
    _square_split_overlap = cfg["square_split_overlap"]
    _square_split_pad_x = cfg["square_split_pad_x"]
    _square_split_pad_y = cfg["square_split_pad_y"]
    if _event_output_dir:
        os.makedirs(_event_output_dir, exist_ok=True)
    if _event_jsonl_path:
        os.makedirs(os.path.dirname(_event_jsonl_path), exist_ok=True)

    # ── Kafka producer init ───────────────────────────────────────────────────
    _kafka_enabled = cfg["kafka_enable"]
    _kafka_topic   = cfg["kafka_topic"]
    if _kafka_enabled:
        try:
            from confluent_kafka import Producer as _KafkaProducer
            _kafka_producer = _KafkaProducer({
                "bootstrap.servers": cfg["kafka_bootstrap_server"],
                "client.id": cfg["kafka_client_id"],
            })
            print(f"[INFO] Kafka enabled: {cfg['kafka_bootstrap_server']} → topic={_kafka_topic}")
        except ImportError:
            sys.stderr.write(
                "[ERROR] --kafka-enable requires confluent-kafka package.\n"
                "  Install: pip install confluent-kafka\n"
            )
            sys.exit(1)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Kafka producer init failed: {e}\n")
            sys.exit(1)

    if not sources_raw:
        sys.stderr.write(
            "Usage: %s [options] <video1> [video2 ...]\n"
            "Options:\n"
            "  --no-display\n"
            "  --output <file.mp4>\n"
            "  --save-crops  <dir>   save plate crops to directory\n"
            "  --debug-jsonl <path>  write vehicle/plate/OCR debug events\n"
            "  --event-output-dir <dir>  write final event media to dir\n"
            "  --event-jsonl <path>  write server-ready events.jsonl\n"
            "  --save-event-frame  save original source frame for accepted events\n"
            "  --event-cooldown-frames <N>  min frames before emitting a later stable text unless clearly better (default: 60)\n"
            "  --min-stable-votes <N>  require N votes for stability (default: 2)\n"
            "  --pgie-interval <N>  PGIE frame skip override (default: 0)\n"
            "  --min-plate-conf <f>  skip low-confidence plates (default: 0.05)\n"
            "  --min-plate-width <N>  skip very small plates by width (default: 20)\n"
            "  --min-plate-height <N>  skip very small plates by height (default: 6)\n"
            "  --bbox-smooth-alpha <f>  bbox EMA smoothing alpha (default: 0.4)\n"
            "  --bbox-reset-iou <f>  reset smoothing below IoU (default: 0.2)\n"
            "  --bbox-max-center-jump-ratio <f>  reset smoothing on large center jumps (default: 0.5)\n"
            "  --square-plate-ar-threshold <f>  split plate below aspect ratio (default: 1.7)\n"
            "  --square-split-overlap <f>  overlap for square-plate split crops (default: 0.12)\n"
            "  --square-split-pad-x <f>  horizontal padding for split crops (default: 0.12)\n"
            "  --square-split-pad-y <f>  vertical padding for split crops (default: 0.08)\n"
            "  --ocr-every-n-frames <N>  throttle OCR bookkeeping (default: 6)\n"
            "  --ocr-min-conf <f>    min confidence to vote (default: 0.0)\n"
            "Event dedup:\n"
            "  --emit-duplicates     disable dedup, emit every stable update (debug)\n"
            "  --event-repeat-cooldown-frames <N>  re-emit same plate after N frames (default: 0=never)\n"
            "Kafka:\n"
            "  --kafka-enable\n"
            "  --kafka-bootstrap-server <host:port>  (default: localhost:9092)\n"
            "  --kafka-topic <topic>  (default: lpr.events.v1)\n"
            "  --kafka-client-id <id>  (default: ds-lpr-producer)\n"
            "OCR Backend: LPRNet (via config_sgie_lprnet.txt)\n"
            % args[0])
        sys.exit(1)

    uris        = [make_uri(s) for s in sources_raw]
    num_sources = len(uris)
    is_live     = any(u.startswith("rtsp://") for u in uris)
    pgie_interval = cfg["pgie_interval"]
    _min_stable_votes = cfg["min_stable_votes"]

    # ── Load OCR backend ──────────────────────────────────────────────────────
    _ocr_backend = "lprnet"
    print(f"[INFO] OCR backend  : {_ocr_backend}")
    print(f"[INFO] OCR throttle : every {_OCR_EVERY_N} frames")
    print(f"[INFO] OCR min conf : {_OCR_MIN_CONF}")
    print(f"[INFO] Stable votes : {_min_stable_votes}")
    if pgie_interval is not None:
        print(f"[INFO] PGIE interval: {pgie_interval}")
    
    platform_info = PlatformInfo()
    Gst.init(None)
    perf_data = PERF_DATA(num_streams=num_sources)

    # PGIE batch size selection:
    # If the model is vehicle_parking_detect.onnx (which only has batch=1), we force batch-size=1.
    # Otherwise, we use num_sources (dynamic batching).
    onnx_name = "vehicle_parking_detect.onnx"
    try:
        config = configparser.ConfigParser()
        config.read(PGIE_CONFIG_PATH)
        if config.has_option("property", "onnx-file"):
            onnx_path = config.get("property", "onnx-file")
            onnx_name = os.path.basename(onnx_path)
    except Exception:
        pass

    is_static_b1 = (onnx_name == "vehicle_parking_detect.onnx")
    pgie_batch_size = 1 if is_static_b1 else num_sources

    pgie_engine = _pgie_engine_path_for_batch(pgie_batch_size)
    pgie_overrides = {
        "batch-size": str(pgie_batch_size),
        "model-engine-file": pgie_engine,
    }
    if pgie_interval is not None:
        pgie_overrides["interval"] = str(max(0, pgie_interval))
    sgie3_overrides = {
        "interval": "0",
        "secondary-reinfer-interval": "0",
    }

    pgie_config    = _runtime_config_path(PGIE_CONFIG_PATH, pgie_overrides)
    tracker_config = _runtime_config_path(TRACKER_CONFIG_PATH)
    sgie3_config   = _runtime_config_path(SGIE3_CONFIG_PATH, sgie3_overrides)

    print("=" * 60)
    print(" ds_lpr_v2 | PGIE all classes → Tracker → Plate OCR → Display")
    print("=" * 60)
    print(" PGIE Config :", pgie_config)
    print(" PGIE Engine :", pgie_engine, f"[batch={pgie_batch_size} selected]")
    print(" Tracker Conf:", tracker_config)
    print(" SGIE OCR    :", sgie3_config)
    print(" Sources     :", uris)
    if output_file:
        print(" Output File :", output_file)
    print("=" * 60)

    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("[ERROR] Cannot create Pipeline\n")
        sys.exit(1)

    # ── Streammux ─────────────────────────────────────────────────────────────
    streammux = _make_el("nvstreammux", "streammux")
    pipeline.add(streammux)
    streammux.set_property("width",  MUXER_WIDTH)
    streammux.set_property("height", MUXER_HEIGHT)
    streammux.set_property("batch-size", num_sources)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("live-source", 1 if is_live else 0)

    # ── Sources ───────────────────────────────────────────────────────────────
    for i, uri in enumerate(uris):
        print("[INFO] Source {}: {}".format(i, uri))
        source  = _make_el("uridecodebin", "source-{}".format(i))
        sinkpad = streammux.request_pad_simple("sink_{}".format(i))
        if not sinkpad:
            sys.stderr.write("[ERROR] No sinkpad {} on streammux\n".format(i))
            sys.exit(1)
        source.set_property("uri", uri)
        _source_uri_by_id[i] = uri
        source.connect("pad-added", cb_newpad, sinkpad)
        pipeline.add(source)

    # ── Inference chain ───────────────────────────────────────────────────────
    pgie    = _make_el("nvinfer",   "pgie")
    tracker = _make_el("nvtracker", "tracker")
    sgie3   = _make_el("nvinfer",   "sgie-lprnet")

    pgie.set_property("config-file-path",  pgie_config)
    pgie.set_property("batch-size",        pgie_batch_size)  # dynamic batch size
    sgie3.set_property("config-file-path", sgie3_config)

    # Tracker properties from config file
    tcfg = configparser.ConfigParser()
    tcfg.read(tracker_config)
    if "tracker" in tcfg:
        sec = tcfg["tracker"]
        if "tracker-width"  in sec: tracker.set_property("tracker-width",  tcfg.getint("tracker","tracker-width"))
        if "tracker-height" in sec: tracker.set_property("tracker-height", tcfg.getint("tracker","tracker-height"))
        if "gpu-id"         in sec: tracker.set_property("gpu_id",         tcfg.getint("tracker","gpu-id"))
        if "ll-lib-file"    in sec: tracker.set_property("ll-lib-file",    tcfg.get("tracker","ll-lib-file"))
        if "ll-config-file" in sec: tracker.set_property("ll-config-file", tcfg.get("tracker","ll-config-file"))

    # ── Display/Tiler elements ────────────────────────────────────────────────
    tiler_rows = max(1, math.ceil(math.sqrt(num_sources)))
    tiler_cols = max(1, math.ceil(num_sources / tiler_rows))
    global _tiler_rows, _tiler_cols
    _tiler_rows = tiler_rows
    _tiler_cols = tiler_cols
    tiler = _make_el("nvmultistreamtiler", "tiler")
    tiler.set_property("rows",    tiler_rows)
    tiler.set_property("columns", tiler_cols)
    tiler.set_property("width",   TILER_WIDTH)
    tiler.set_property("height",  TILER_HEIGHT)

    nvvidconv = _make_el("nvvideoconvert", "convertor")
    nvvidconv.set_property("nvbuf-memory-type", 3)
    nvosd     = _make_el("nvdsosd",        "osd")
    nvosd.set_property("process-mode", 0)

    tee           = _make_el("tee",   "sink-tee")
    queue_display = _make_el("queue", "queue-display")

    # Display sink
    use_ximagesink = os.environ.get("USE_XIMAGESINK") == "1"
    nvvidconv_display = None
    caps_cpu_display = None
    videoconvert_display = None

    if no_display or not os.environ.get("DISPLAY"):
        print("[INFO] Display: fakesink")
        sink = _make_el("fakesink", "fakesink")
        sink.set_property("sync", False)
    elif use_ximagesink:
        print("[INFO] Display: ximagesink")
        sink = _make_el("ximagesink", "nvvideo-renderer")
        sink.set_property("sync", False)
        nvvidconv_display = _make_el("nvvideoconvert", "convertor-display")
        caps_cpu_display  = _make_el("capsfilter", "caps-cpu-display")
        caps_cpu_display.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))
        videoconvert_display = _make_el("videoconvert", "videoconvert-display")
    elif platform_info.is_integrated_gpu() or platform_info.is_platform_aarch64():
        print("[INFO] Display: nv3dsink")
        sink = _make_el("nv3dsink", "nv3d-sink")
        sink.set_property("sync", False)
    else:
        print("[INFO] Display: nveglglessink")
        sink = _make_el("nveglglessink", "nvvideo-renderer")
        sink.set_property("sync", False)

    # File sink (optional)
    save_to_file = output_file is not None
    if save_to_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        print("[INFO] Output file:", output_file)
        queue_file  = _make_el("queue",          "queue-file")
        nvvidconv2  = _make_el("nvvideoconvert",  "convertor-file")
        capsfilter  = _make_el("capsfilter",      "capsfilter-file")
        capsfilter.set_property("caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
        encoder     = _make_el("nvv4l2h264enc",  "h264-encoder")
        encoder.set_property("bitrate", 4000000)
        h264parse   = _make_el("h264parse",      "h264-parse")
        qtmux       = _make_el("qtmux",          "qt-mux")
        filesink    = _make_el("filesink",       "file-sink")
        filesink.set_property("location", output_file)
        filesink.set_property("sync", False)

    # ── Add to pipeline ───────────────────────────────────────────────────────
    caps_gpu = _make_el("capsfilter", "caps_gpu")
    caps_gpu.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    for el in [pgie, tracker, sgie3, tiler, nvvidconv, caps_gpu, nvosd, tee, queue_display, sink]:
        pipeline.add(el)
    if use_ximagesink:
        pipeline.add(nvvidconv_display)
        pipeline.add(caps_cpu_display)
        pipeline.add(videoconvert_display)
    if save_to_file:
        for el in [queue_file, nvvidconv2, capsfilter, encoder, h264parse, qtmux, filesink]:
            pipeline.add(el)

    # ── Link ──────────────────────────────────────────────────────────────────
    print("[INFO] Linking pipeline...")
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie3)
    
    # Format conversion BEFORE tiler so we can extract RGBA crops from original un-tiled frames
    sgie3.link(nvvidconv)
    nvvidconv.link(caps_gpu)
    caps_gpu.link(tiler)
    tiler.link(nvosd)
    nvosd.link(tee)

    tee_disp = tee.request_pad_simple("src_%u")
    tee_disp.link(queue_display.get_static_pad("sink"))
    if use_ximagesink:
        queue_display.link(nvvidconv_display)
        nvvidconv_display.link(caps_cpu_display)
        caps_cpu_display.link(videoconvert_display)
        videoconvert_display.link(sink)
    else:
        queue_display.link(sink)

    if save_to_file:
        tee_file = tee.request_pad_simple("src_%u")
        tee_file.link(queue_file.get_static_pad("sink"))
        queue_file.link(nvvidconv2)
        nvvidconv2.link(capsfilter)
        capsfilter.link(encoder)
        encoder.link(h264parse)
        h264parse.link(qtmux)
        qtmux.link(filesink)

    # ── Bus + Probes ──────────────────────────────────────────────────────────
    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    sgie3_sinkpad = sgie3.get_static_pad("sink")
    if not sgie3_sinkpad:
        sys.stderr.write("[ERROR] Cannot get sink pad of sgie3\n")
    else:
        sgie3_sinkpad.add_probe(Gst.PadProbeType.BUFFER, sgie3_sink_pad_buffer_probe, 0)
        print("[INFO] Attached sgie3 sink probe (pseudo objects for square plates)")

    caps_gpu_srcpad = caps_gpu.get_static_pad("src")
    if not caps_gpu_srcpad:
        sys.stderr.write("[ERROR] Cannot get src pad of caps_gpu\n")
    else:
        caps_gpu_srcpad.add_probe(Gst.PadProbeType.BUFFER, metadata_src_pad_buffer_probe, 0)
        print("[INFO] Attached caps_gpu src probe (event metadata & crop collection)")

    osd_sinkpad = nvosd.get_static_pad("sink")
    if not osd_sinkpad:
        sys.stderr.write("[ERROR] Cannot get sink pad of nvosd\n")
    else:
        osd_sinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    GLib.timeout_add(5000, perf_data.perf_print_callback)

    print("[INFO] Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.stderr.write("[ERROR] {}\n".format(e))

    pipeline.set_state(Gst.State.NULL)
    print("[INFO] Pipeline stopped.")

    if _kafka_enabled and _kafka_producer is not None:
        try:
            remaining = _kafka_producer.flush(timeout=5)
            if remaining:
                sys.stderr.write(f"[WARN] Kafka flush: {remaining} messages still in queue after timeout\n")
        except Exception as e:
            sys.stderr.write(f"[WARN] Kafka flush error: {e}\n")

    text_plate_tracks = [key for key, state in _vehicle_states.items() if state.best_plate_text_raw]
    stable_plate_tracks = [key for key, state in _vehicle_states.items() if state.best_plate_text_stable]
    print("[SUMMARY] tracked_objects={} plate_objects={} ocr_raw_events={} text_plate_tracks={} stable_plate_tracks={}".format(
        len(_vehicle_states), _metrics["plate_objects"], _metrics["ocr_raw_events"],
        len(text_plate_tracks), len(stable_plate_tracks)
    ))
    if _debug_jsonl_path:
        try:
            with open(_debug_jsonl_path, "a", encoding="utf-8") as f:
                for (sid, oid), item in _plate_text_seen.items():
                    f.write(json.dumps({
                        "event": "final_plate_track",
                        "sid": int(sid),
                        "object_id": int(oid),
                        "plate": item.get("text", ""),
                        "stable": bool(item.get("stable")),
                    }, ensure_ascii=True) + "\n")
        except Exception:
            pass
    if save_to_file and os.path.exists(output_file):
        print("[INFO] Output saved:", output_file)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
