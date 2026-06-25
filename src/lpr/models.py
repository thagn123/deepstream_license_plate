from collections import deque
from dataclasses import dataclass, field

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
