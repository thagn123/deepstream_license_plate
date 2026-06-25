#!/usr/bin/env bash
# eval_run.sh — Chạy cả 2 OCR pipeline trên cùng tập video, lưu kết quả để so sánh.
#
# Chạy BÊN TRONG container:
#   docker exec -it ds90 bash /workspace/last_ds_cp/tools/eval_run.sh
#
# Hoặc với tham số tùy chỉnh:
#   docker exec -it ds90 bash /workspace/last_ds_cp/tools/eval_run.sh \
#       --videos "001 002 003" --min-votes 3

set -euo pipefail

WORKSPACE=/workspace/last_ds_cp
VIDEO_DIR="$WORKSPACE/videos/drive-download-20260616T102510Z-3-001"
OUT_BASE="$WORKSPACE/outputs/eval_$(date +%Y%m%d_%H%M%S)"
COMMON_ARGS="--no-display --pgie-interval 0 --min-stable-votes 2 --save-event-frame"

# Parse optional overrides
VIDEOS="001 002 003 004 005 006 007 008 009 010 011 012 013 014 015"
MIN_VOTES=2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --videos) VIDEOS="$2"; shift 2 ;;
        --min-votes) MIN_VOTES="$2"; shift 2 ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

COMMON_ARGS="$COMMON_ARGS --min-stable-votes $MIN_VOTES"

OUT_LPRNET="$OUT_BASE/lprnet"
OUT_NEW_OCR="$OUT_BASE/new_ocr"
mkdir -p "$OUT_LPRNET" "$OUT_NEW_OCR"

LOG="$OUT_BASE/run.log"

echo "================================================================" | tee "$LOG"
echo "  OCR EVALUATION RUN — $(date)" | tee -a "$LOG"
echo "  Videos : $VIDEOS" | tee -a "$LOG"
echo "  Out    : $OUT_BASE" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"

# ── Helper: run one pipeline on one video ──────────────────────────────────
run_pipeline() {
    local model="$1"      # lprnet | new_ocr
    local script="$2"     # python script path
    local video_id="$3"   # e.g. 001
    local out_dir="$4"    # output directory for this model

    local video="$VIDEO_DIR/lpr_230428_${video_id}.mp4"
    local evdir="$out_dir/video_${video_id}"
    mkdir -p "$evdir"

    if [[ ! -f "$video" ]]; then
        echo "[SKIP] $video not found" | tee -a "$LOG"
        return
    fi

    echo "  [$model] video_$video_id → $evdir" | tee -a "$LOG"

    local t0=$SECONDS
    GST_REGISTRY=/workspace/.gst_registry.bin \
    python3 "$script" "$video" \
        $COMMON_ARGS \
        --event-output-dir "$evdir" \
        --event-jsonl "$evdir/events.jsonl" \
        --debug-jsonl "$evdir/debug.jsonl" \
        >> "$evdir/stdout.log" 2>> "$evdir/stderr.log"
    local rc=$?
    local elapsed=$((SECONDS - t0))

    if [[ $rc -eq 0 ]]; then
        local n_events=$(wc -l < "$evdir/events.jsonl" 2>/dev/null || echo 0)
        # Extract FPS from stderr
        local fps=$(grep -oP 'Fps of stream\s+\K[\d.]+' "$evdir/stderr.log" 2>/dev/null | tail -1 || echo "?")
        echo "    → OK  events=$n_events  time=${elapsed}s  fps=$fps" | tee -a "$LOG"
        # Write timing metadata
        echo "{\"video\":\"$video_id\",\"model\":\"$model\",\"events\":$n_events,\"elapsed_s\":$elapsed,\"fps\":\"$fps\"}" \
            >> "$out_dir/timing.jsonl"
    else
        echo "    → FAIL rc=$rc (see $evdir/stderr.log)" | tee -a "$LOG"
    fi
}

# ── Run both models on all videos ─────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "── Phase 1: LPRNet (us_lprnet_baseline18) ──────────────────────" | tee -a "$LOG"
for vid in $VIDEOS; do
    run_pipeline "lprnet" "$WORKSPACE/src/app_lpr_v2.py" "$vid" "$OUT_LPRNET"
done

echo "" | tee -a "$LOG"
echo "── Phase 2: New OCR 2024 (lpr_ocr.20240305) ────────────────────" | tee -a "$LOG"
for vid in $VIDEOS; do
    run_pipeline "new_ocr" "$WORKSPACE/src/app_lpr_v2_new_ocr.py" "$vid" "$OUT_NEW_OCR"
done

# ── Merge all events into combined JSONL per model ────────────────────────
echo "" | tee -a "$LOG"
echo "── Merging events ───────────────────────────────────────────────" | tee -a "$LOG"
cat "$OUT_LPRNET"/video_*/events.jsonl 2>/dev/null > "$OUT_LPRNET/all_events.jsonl" || true
cat "$OUT_NEW_OCR"/video_*/events.jsonl 2>/dev/null > "$OUT_NEW_OCR/all_events.jsonl" || true

N_LPR=$(wc -l < "$OUT_LPRNET/all_events.jsonl" 2>/dev/null || echo 0)
N_NEW=$(wc -l < "$OUT_NEW_OCR/all_events.jsonl" 2>/dev/null || echo 0)
echo "  LPRNet  total events: $N_LPR" | tee -a "$LOG"
echo "  New OCR total events: $N_NEW" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "  DONE — $(date)" | tee -a "$LOG"
echo "  Run analysis:" | tee -a "$LOG"
echo "  python3 $WORKSPACE/tools/eval_compare.py \\" | tee -a "$LOG"
echo "      --lprnet  $OUT_LPRNET/all_events.jsonl \\" | tee -a "$LOG"
echo "      --new-ocr $OUT_NEW_OCR/all_events.jsonl \\" | tee -a "$LOG"
echo "      --lprnet-timing  $OUT_LPRNET/timing.jsonl \\" | tee -a "$LOG"
echo "      --new-ocr-timing $OUT_NEW_OCR/timing.jsonl" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
