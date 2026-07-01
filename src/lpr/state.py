from collections import Counter
from lpr.models import VehicleTrackState

short_id_map = {}
plate_history = {}
pseudo_parent_map = {}
split_ocr = {}
ocr_frame_cache = {}
save_crop_seq = {}
object_last_seen = {}
plate_text_seen = {}
vehicle_states = {}
emitted_event_keys = set()
locked_plate_ids = set()
ocr_lock_min_votes = 3
ocr_lock_min_score = 4.0
fps_overlay_state = {}
source_uri_by_id = {}

next_short_id = 1
cleanup_counter = 0

lpr_layer_warning_printed = False
kafka_producer = None
perf_data = None
metrics = Counter()

# Runtime configurable parameters
ocr_backend = 'new_ocr_2024'
OCR_EVERY_N = 6
OCR_MIN_CONF = 0.0
save_crops_dir = None
osd_probe_frame = 0
tiler_rows = 1
tiler_cols = 1
debug_jsonl_path = None
square_plate_ar_threshold = 1.7
square_split_overlap = 0.12
square_split_pad_x = 0.12
square_split_pad_y = 0.08
min_plate_conf = 0.05
min_plate_width = 20
min_plate_height = 6
min_vehicle_width = 60
min_vehicle_height = 40
min_vehicle_width_ratio = 0.0   # fraction of muxer width; >0 overrides pixel threshold
min_vehicle_height_ratio = 0.0  # fraction of muxer height; >0 overrides pixel threshold
muxer_width = 1920
muxer_height = 1080
bbox_smooth_alpha = 0.4
bbox_reset_iou = 0.2
bbox_max_center_jump_ratio = 0.5
event_output_dir = None
event_jsonl_path = None
min_stable_votes = 2
save_event_frame = False
emit_duplicates = False
event_repeat_cooldown_frames = 0
kafka_enabled = False
kafka_topic = "lpr.events.v1"
fps_overlay_alpha = 0.2

def get_vehicle_state(track_key: tuple) -> VehicleTrackState:
    state_obj = vehicle_states.get(track_key)
    if state_obj is None:
        state_obj = VehicleTrackState()
        vehicle_states[track_key] = state_obj
    return state_obj

disable_laplacian = False
