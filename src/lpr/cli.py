import sys


def _parse_args(args):
    no_display   = "--no-display" in args
    output_file  = None
    sources      = []
    ocr_backend  = "new_ocr_2024"
    save_crops   = None
    debug_jsonl  = None
    event_output_dir = None
    event_jsonl = None
    min_stable_votes = 2
    save_event_frame = False
    pgie_interval = 0
    min_plate_conf = 0.05
    min_plate_width = 20
    min_plate_height = 6
    min_vehicle_width = 50
    min_vehicle_height = 30
    min_vehicle_width_ratio = 0.0
    min_vehicle_height_ratio = 0.0
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
    disable_laplacian = False
    kafka_bootstrap_server = "localhost:9092"
    kafka_topic = "lpr.events.v1"
    kafka_client_id = "ds-lpr-producer"
    pgie_config  = None

    i = 1
    while i < len(args):
        a = args[i]

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
        elif a == "--ocr-backend":
            ocr_backend = _nextval()
        elif a.startswith("--ocr-backend="):
            ocr_backend = a.split("=", 1)[1]
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
        elif a == "--min-vehicle-width":
            min_vehicle_width = int(_nextval())
        elif a.startswith("--min-vehicle-width="):
            min_vehicle_width = int(a.split("=", 1)[1])
        elif a == "--min-vehicle-height":
            min_vehicle_height = int(_nextval())
        elif a.startswith("--min-vehicle-height="):
            min_vehicle_height = int(a.split("=", 1)[1])
        elif a == "--min-vehicle-width-ratio":
            min_vehicle_width_ratio = float(_nextval())
        elif a.startswith("--min-vehicle-width-ratio="):
            min_vehicle_width_ratio = float(a.split("=", 1)[1])
        elif a == "--min-vehicle-height-ratio":
            min_vehicle_height_ratio = float(_nextval())
        elif a.startswith("--min-vehicle-height-ratio="):
            min_vehicle_height_ratio = float(a.split("=", 1)[1])
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
        elif a == "--disable-laplacian":
            disable_laplacian = True
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
        min_stable_votes=min_stable_votes,
        save_event_frame=save_event_frame,
        pgie_interval=pgie_interval,
        min_plate_conf=min_plate_conf,
        min_plate_width=min_plate_width,
        min_plate_height=min_plate_height,
        min_vehicle_width=min_vehicle_width,
        min_vehicle_height=min_vehicle_height,
        min_vehicle_width_ratio=min_vehicle_width_ratio,
        min_vehicle_height_ratio=min_vehicle_height_ratio,
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
        disable_laplacian=disable_laplacian,
        kafka_bootstrap_server=kafka_bootstrap_server,
        kafka_topic=kafka_topic,
        kafka_client_id=kafka_client_id,
        pgie_config=pgie_config,
    )
