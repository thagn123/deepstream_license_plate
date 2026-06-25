import sys
import os
import json
import math
import configparser
from collections import Counter

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

import lpr_config as config
from lpr import state
from lpr.cli import _parse_args
from lpr.config_utils import _runtime_config_path, _pgie_engine_path_for_batch
from lpr.probes import (
    pgie_src_pad_buffer_probe,
    sgie3_sink_pad_buffer_probe,
    metadata_src_pad_buffer_probe,
    osd_sink_pad_buffer_probe,
)


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


def run(args, *, probe_overrides=None, ocr_backend=None):
    # ── Reset all module state ────────────────────────────────────────────────
    state.short_id_map      = {}
    state.plate_history     = {}
    state.vehicle_states    = {}
    state.pseudo_parent_map = {}
    state.split_ocr         = {}
    state.ocr_frame_cache   = {}
    state.osd_probe_frame   = 0
    state.save_crop_seq     = {}
    state.object_last_seen  = {}
    state.plate_text_seen   = {}
    state.emitted_event_keys = set()
    state.debug_jsonl_path  = None
    state.source_uri_by_id  = {}
    state.save_event_frame  = False
    state.emit_duplicates   = False
    state.event_repeat_cooldown_frames = 0
    state.kafka_enabled     = False
    state.kafka_producer    = None
    state.kafka_topic       = "lpr.events.v1"
    state.min_plate_conf    = 0.05
    state.min_plate_width   = 20
    state.min_plate_height  = 6
    state.min_vehicle_width  = 60
    state.min_vehicle_height = 40
    state.min_vehicle_width_ratio  = 0.0
    state.min_vehicle_height_ratio = 0.0
    state.muxer_width  = 1920
    state.muxer_height = 1080
    state.bbox_smooth_alpha = 0.4
    state.bbox_reset_iou    = 0.2
    state.bbox_max_center_jump_ratio = 0.5
    state.square_plate_ar_threshold = 1.7
    state.square_split_overlap = 0.12
    state.square_split_pad_x = 0.12
    state.square_split_pad_y = 0.08
    state.metrics           = Counter()
    state.next_short_id     = 1
    state.cleanup_counter   = 0
    state.lpr_layer_warning_printed = False

    cfg = _parse_args(args)
    if cfg.get("pgie_config"):
        config.PGIE_CONFIG_PATH = os.path.abspath(cfg["pgie_config"])

    no_display  = cfg["no_display"]
    output_file = cfg["output_file"]
    sources_raw = cfg["sources"]

    state.ocr_backend  = cfg["ocr_backend"]
    state.OCR_EVERY_N  = cfg["ocr_every_n"]
    state.OCR_MIN_CONF = cfg["ocr_min_conf"]
    state.save_crops_dir = cfg["save_crops"]
    if state.save_crops_dir:
        os.makedirs(state.save_crops_dir, exist_ok=True)
    state.debug_jsonl_path = cfg["debug_jsonl"]
    if state.debug_jsonl_path is None and state.save_crops_dir:
        state.debug_jsonl_path = os.path.join(state.save_crops_dir, "debug_events.jsonl")
    if state.debug_jsonl_path:
        os.makedirs(os.path.dirname(os.path.abspath(state.debug_jsonl_path)), exist_ok=True)

    state.event_output_dir = os.path.abspath(cfg["event_output_dir"]) if cfg["event_output_dir"] else None
    state.event_jsonl_path = os.path.abspath(cfg["event_jsonl"]) if cfg["event_jsonl"] else None
    state.event_cooldown_frames = cfg["event_cooldown_frames"]
    state.save_event_frame = cfg["save_event_frame"]
    state.emit_duplicates = cfg["emit_duplicates"]
    state.event_repeat_cooldown_frames = cfg["event_repeat_cooldown_frames"]
    state.min_plate_conf = cfg["min_plate_conf"]
    state.min_plate_width = cfg["min_plate_width"]
    state.min_plate_height = cfg["min_plate_height"]
    state.min_vehicle_width  = cfg["min_vehicle_width"]
    state.min_vehicle_height = cfg["min_vehicle_height"]
    state.min_vehicle_width_ratio  = cfg["min_vehicle_width_ratio"]
    state.min_vehicle_height_ratio = cfg["min_vehicle_height_ratio"]
    state.muxer_width  = config.MUXER_WIDTH
    state.muxer_height = config.MUXER_HEIGHT
    state.bbox_smooth_alpha = cfg["bbox_smooth_alpha"]
    state.bbox_reset_iou = cfg["bbox_reset_iou"]
    state.bbox_max_center_jump_ratio = cfg["bbox_max_center_jump_ratio"]
    state.square_plate_ar_threshold = cfg["square_plate_ar_threshold"]
    state.square_split_overlap = cfg["square_split_overlap"]
    state.square_split_pad_x = cfg["square_split_pad_x"]
    state.square_split_pad_y = cfg["square_split_pad_y"]
    if state.event_output_dir:
        os.makedirs(state.event_output_dir, exist_ok=True)
    if state.event_jsonl_path:
        os.makedirs(os.path.dirname(state.event_jsonl_path), exist_ok=True)

    # ── Kafka producer init ───────────────────────────────────────────────────
    state.disable_laplacian = cfg["disable_laplacian"]
    state.kafka_enabled = cfg["kafka_enable"]
    state.kafka_topic   = cfg["kafka_topic"]
    if state.kafka_enabled:
        try:
            from confluent_kafka import Producer as _KafkaProducer
            state.kafka_producer = _KafkaProducer({
                "bootstrap.servers": cfg["kafka_bootstrap_server"],
                "client.id": cfg["kafka_client_id"],
            })
            print(f"[INFO] Kafka enabled: {cfg['kafka_bootstrap_server']} → topic={state.kafka_topic}")
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
            "  --event-cooldown-frames <N>  min frames before emitting (default: 60)\n"
            "  --min-stable-votes <N>  require N votes for stability (default: 2)\n"
            "  --pgie-interval <N>  PGIE frame skip override (default: 0)\n"
            "  --min-plate-conf <f>  skip low-confidence plates (default: 0.05)\n"
            "  --min-plate-width <N>  skip small plates by width (default: 20)\n"
            "  --min-plate-height <N>  skip small plates by height (default: 6)\n"
            "  --min-vehicle-width <N>   drop vehicles narrower than N px before tracker (default: 60)\n"
            "  --min-vehicle-height <N>  drop vehicles shorter than N px before tracker (default: 40)\n"
            "  --min-vehicle-width-ratio <f>   ratio of muxer width (e.g. 0.031); overrides --min-vehicle-width when >0\n"
            "  --min-vehicle-height-ratio <f>  ratio of muxer height (e.g. 0.037); overrides --min-vehicle-height when >0\n"
            "  --bbox-smooth-alpha <f>  bbox EMA alpha (default: 0.4)\n"
            "  --bbox-reset-iou <f>  reset smoothing below IoU (default: 0.2)\n"
            "  --bbox-max-center-jump-ratio <f>  reset on large jumps (default: 0.5)\n"
            "  --square-plate-ar-threshold <f>  split below aspect ratio (default: 1.7)\n"
            "  --square-split-overlap <f>  overlap for split crops (default: 0.12)\n"
            "  --square-split-pad-x <f>  horizontal padding (default: 0.12)\n"
            "  --square-split-pad-y <f>  vertical padding (default: 0.08)\n"
            "  --ocr-every-n-frames <N>  throttle OCR bookkeeping (default: 6)\n"
            "  --ocr-min-conf <f>    min confidence to vote (default: 0.0)\n"
            "Event dedup:\n"
            "  --emit-duplicates     emit every stable update (debug)\n"
            "  --event-repeat-cooldown-frames <N>  re-emit same plate after N frames\n"
            "Kafka:\n"
            "  --kafka-enable\n"
            "  --kafka-bootstrap-server <host:port>  (default: localhost:9092)\n"
            "  --kafka-topic <topic>  (default: lpr.events.v1)\n"
            "  --kafka-client-id <id>  (default: ds-lpr-producer)\n"
            % args[0])
        sys.exit(1)

    uris        = [make_uri(s) for s in sources_raw]
    num_sources = len(uris)
    is_live     = any(u.startswith("rtsp://") for u in uris)
    pgie_interval = cfg["pgie_interval"]
    state.min_stable_votes = cfg["min_stable_votes"]

    state.ocr_backend = ocr_backend if ocr_backend is not None else "new_ocr_2024"
    print(f"[INFO] OCR backend  : {state.ocr_backend}")
    print(f"[INFO] OCR throttle : every {state.OCR_EVERY_N} frames")
    print(f"[INFO] OCR min conf : {state.OCR_MIN_CONF}")
    print(f"[INFO] Stable votes : {state.min_stable_votes}")
    if pgie_interval is not None:
        print(f"[INFO] PGIE interval: {pgie_interval}")

    from common.platform_info import PlatformInfo
    from common.FPS import PERF_DATA
    platform_info = PlatformInfo()
    Gst.init(None)
    state.perf_data = PERF_DATA(num_streams=num_sources)

    # ── PGIE batch size selection ─────────────────────────────────────────────
    onnx_name = "vehicle_parking_detect.onnx"
    try:
        ini = configparser.ConfigParser()
        ini.read(config.PGIE_CONFIG_PATH)
        if ini.has_option("property", "onnx-file"):
            onnx_path = ini.get("property", "onnx-file")
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

    pgie_cfg_path    = _runtime_config_path(config.PGIE_CONFIG_PATH, pgie_overrides)
    tracker_config   = _runtime_config_path(config.TRACKER_CONFIG_PATH)
    sgie3_config     = _runtime_config_path(config.SGIE3_CONFIG_PATH, sgie3_overrides)

    print("=" * 60)
    print(" ds_lpr_v2 | PGIE all classes → Tracker → Plate OCR → Display")
    print("=" * 60)
    print(" PGIE Config :", pgie_cfg_path)
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
    streammux.set_property("width",  config.MUXER_WIDTH)
    streammux.set_property("height", config.MUXER_HEIGHT)
    streammux.set_property("batch-size", num_sources)
    streammux.set_property("batched-push-timeout", config.MUXER_BATCH_TIMEOUT_USEC)
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
        state.source_uri_by_id[i] = uri
        source.connect("pad-added", cb_newpad, sinkpad)
        pipeline.add(source)

    # ── Inference chain ───────────────────────────────────────────────────────
    pgie      = _make_el("nvinfer",     "pgie")
    laplacian = _make_el("dslaplacian", "laplacian")
    tracker   = _make_el("nvtracker",   "tracker")
    sgie3     = _make_el("nvinfer",     "sgie-ocr")

    pgie.set_property("config-file-path",  pgie_cfg_path)
    pgie.set_property("batch-size",        pgie_batch_size)
    laplacian.set_property("class-id", 13)
    sgie3.set_property("config-file-path", sgie3_config)

    tcfg = configparser.ConfigParser()
    tcfg.read(tracker_config)
    if "tracker" in tcfg:
        sec = tcfg["tracker"]
        if "tracker-width"  in sec: tracker.set_property("tracker-width",  tcfg.getint("tracker", "tracker-width"))
        if "tracker-height" in sec: tracker.set_property("tracker-height", tcfg.getint("tracker", "tracker-height"))
        if "gpu-id"         in sec: tracker.set_property("gpu_id",         tcfg.getint("tracker", "gpu-id"))
        if "ll-lib-file"    in sec: tracker.set_property("ll-lib-file",    tcfg.get("tracker", "ll-lib-file"))
        if "ll-config-file" in sec: tracker.set_property("ll-config-file", tcfg.get("tracker", "ll-config-file"))

    # ── Display / Tiler elements ──────────────────────────────────────────────
    tiler_rows = max(1, math.ceil(math.sqrt(num_sources)))
    tiler_cols = max(1, math.ceil(num_sources / tiler_rows))
    state.tiler_rows = tiler_rows
    state.tiler_cols = tiler_cols
    tiler = _make_el("nvmultistreamtiler", "tiler")
    tiler.set_property("rows",    tiler_rows)
    tiler.set_property("columns", tiler_cols)
    tiler.set_property("width",   config.TILER_WIDTH)
    tiler.set_property("height",  config.TILER_HEIGHT)

    nvvidconv = _make_el("nvvideoconvert", "convertor")
    nvvidconv.set_property("nvbuf-memory-type", 3)
    nvosd     = _make_el("nvdsosd",        "osd")
    nvosd.set_property("process-mode", 0)

    tee           = _make_el("tee",   "sink-tee")
    queue_display = _make_el("queue", "queue-display")

    # Display sink
    use_ximagesink = os.environ.get("USE_XIMAGESINK") == "1" and not no_display
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

    if not state.disable_laplacian:
        for el in [pgie, tracker, laplacian, sgie3, tiler, nvvidconv, caps_gpu, nvosd, tee, queue_display, sink]:
            pipeline.add(el)
    else:
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
    if not state.disable_laplacian:
        tracker.link(laplacian)
        laplacian.link(sgie3)
    else:
        tracker.link(sgie3)
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
    from common.bus_call import bus_call
    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    _po = probe_overrides or {}
    _pgie_probe     = _po.get("pgie",     pgie_src_pad_buffer_probe)
    _sgie3_probe    = _po.get("sgie3",    sgie3_sink_pad_buffer_probe)
    _metadata_probe = _po.get("metadata", metadata_src_pad_buffer_probe)

    pgie_srcpad = pgie.get_static_pad("src")
    if not pgie_srcpad:
        sys.stderr.write("[ERROR] Cannot get src pad of pgie\n")
    else:
        pgie_srcpad.add_probe(Gst.PadProbeType.BUFFER, _pgie_probe, 0)
        print("[INFO] Attached pgie src probe (filter non-vehicle before tracker)")

    sgie3_sinkpad = sgie3.get_static_pad("sink")
    if not sgie3_sinkpad:
        sys.stderr.write("[ERROR] Cannot get sink pad of sgie3\n")
    else:
        sgie3_sinkpad.add_probe(Gst.PadProbeType.BUFFER, _sgie3_probe, 0)
        print("[INFO] Attached sgie3 sink probe")

    caps_gpu_srcpad = caps_gpu.get_static_pad("src")
    if not caps_gpu_srcpad:
        sys.stderr.write("[ERROR] Cannot get src pad of caps_gpu\n")
    else:
        caps_gpu_srcpad.add_probe(Gst.PadProbeType.BUFFER, _metadata_probe, 0)
        print("[INFO] Attached caps_gpu src probe (event metadata & crop collection)")

    osd_sinkpad = nvosd.get_static_pad("sink")
    if not osd_sinkpad:
        sys.stderr.write("[ERROR] Cannot get sink pad of nvosd\n")
    else:
        osd_sinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    GLib.timeout_add(5000, state.perf_data.perf_print_callback)

    print("[INFO] Starting pipeline...")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[INFO] Caught KeyboardInterrupt — sending EOS to flush output file...")
        pipeline.send_event(Gst.Event.new_eos())
        bus = pipeline.get_bus()
        msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        if msg and msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            sys.stderr.write("[ERROR] {} ({})\n".format(err, debug))
    except Exception as e:
        sys.stderr.write("[ERROR] {}\n".format(e))

    pipeline.set_state(Gst.State.NULL)
    print("[INFO] Pipeline stopped.")

    # ── Teardown ──────────────────────────────────────────────────────────────
    if state.kafka_enabled and state.kafka_producer is not None:
        try:
            remaining = state.kafka_producer.flush(timeout=5)
            if remaining:
                sys.stderr.write(f"[WARN] Kafka flush: {remaining} messages still in queue after timeout\n")
        except Exception as e:
            sys.stderr.write(f"[WARN] Kafka flush error: {e}\n")

    text_plate_tracks   = [k for k, vs in state.vehicle_states.items() if vs.best_plate_text_raw]
    stable_plate_tracks = [k for k, vs in state.vehicle_states.items() if vs.best_plate_text_stable]
    print("[SUMMARY] tracked_objects={} plate_objects={} ocr_raw_events={} text_plate_tracks={} stable_plate_tracks={}".format(
        len(state.vehicle_states), state.metrics["plate_objects"], state.metrics["ocr_raw_events"],
        len(text_plate_tracks), len(stable_plate_tracks)
    ))
    if state.debug_jsonl_path:
        try:
            with open(state.debug_jsonl_path, "a", encoding="utf-8") as f:
                for (sid, oid), item in state.plate_text_seen.items():
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
