import os
import re

# Cho phép ép các luồng RTSP thành chất lượng thấp (LQ) dùng cho mục đích Testing (do lỗi ffmpeg phát sai FPS). 
# Nếu cắm Camera thật, hãy đổi thành False.
FORCE_LQ_RTSP = True

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Config paths ──────────────────────────────────────────────────────────────
PGIE_CONFIG_PATH    = os.path.join(PROJECT_ROOT, "configs", "config_pgie_yolov11s.txt")
TRACKER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "ds_tracker_config.txt")
SGIE3_CONFIG_PATH   = os.path.join(PROJECT_ROOT, "configs", "config_sgie_lpr_ocr_2024.txt")
RUNTIME_CONFIG_DIR  = os.path.join("/tmp", "ds_lpr_v2_runtime_configs")

MUXER_BATCH_TIMEOUT_USEC = 16666
MUXER_WIDTH = 1920
MUXER_HEIGHT = 1080
TILER_WIDTH = 1280
TILER_HEIGHT = 720

# ── GIE unique IDs (must match gie-unique-id in config) ────────────────────
PGIE_UNIQUE_ID  = 1
SGIE1_UNIQUE_ID = 2
SGIE2_UNIQUE_ID = 3
SGIE3_UNIQUE_ID = 4
UNTRACKED_OBJECT_ID = (1 << 64) - 1

VEHICLE_CLASS_IDS = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11}
LP_CLASS_ID     = 13
LP_TOP_CLASS_ID = 14
LP_BOT_CLASS_ID = 15

# ── Class labels ──────────────────────────────────────────────────────────────
VEHICLE_LABELS = [
    "seater_12_16", "bus", "car", "club_cart", "human",
    "moto", "moto_rider", "shuttle_bus_5_7", "truck", "bike",
    "cyclist", "shuttle_bus_18", "head", "license_plate",
]

_CLASS_COLORS = [
    (0.2, 0.8, 1.0, 1.0),   (1.0, 0.5, 0.0, 1.0),   (0.2, 0.6, 1.0, 1.0),   (1.0, 1.0, 0.0, 1.0),
    (1.0, 0.2, 0.2, 1.0),   (0.8, 0.2, 0.8, 1.0),   (0.9, 0.5, 0.9, 1.0),   (0.0, 1.0, 0.8, 1.0),
    (1.0, 0.3, 0.3, 1.0),   (0.4, 1.0, 0.4, 1.0),   (0.3, 0.3, 1.0, 1.0),   (1.0, 0.6, 0.0, 1.0),
    (0.7, 0.7, 0.7, 1.0),   (0.0, 1.0, 0.0, 1.0),
]

_LPR_CHARS = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
_LPR_LAYER_NAME = "tf_op_layer_ArgMax"

_PLATE_HISTORY_LEN = 15
_PLATE_MIN_VOTES   = 3
_PLATE_REPLACE_VOTE_MARGIN = 2
_SINGLE_VOTE_ACCEPT_SCORE = 8.0
_STABLE_REPLACE_SCORE_MARGIN = 1.25
_STATE_STALE_AFTER_FRAMES = 450
_CLEANUP_INTERVAL = 150

_OLD_ROOTS = (
    "/workspace/ds_lpr_v2",
    "/workspace/new_ds_lpr",
    "/workspace/las_ds",
    "/workspace/last_ds",
    "/workspace/last_ds_cp",
)

_LETTER_TO_DIGIT = {'B':'8','D':'0','G':'6','I':'1','O':'0',
                    'P':'9','S':'5','T':'7','Z':'2','M':'1',
                    'A':'4','E':'3','H':'4'}
_DIGIT_TO_LETTER = {'8':'B','0':'D','6':'G','1':'K','9':'P',
                    '5':'S','7':'T','2':'Z','4':'A','3':'E'}
_VALID_SERIES    = set('ABCDEFGHKLMNPSTUVXYZ')

_VN_PLATE_RE = re.compile(r"^[1-9][0-9][A-Z][0-9A-Z]?\d{4,5}$")
