#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${ROOT_DIR}/src/app_lpr_v2.py"
OUT_DIR="${ROOT_DIR}/outputs/validation_$(date +%Y%m%d_%H%M%S)"
SAVE_EVENT_FRAME="${SAVE_EVENT_FRAME:-0}"

VIDEOS=(
  "videos/test1.h264"
  "videos/test2.h264"
  "videos/test3.h264"
  "videos/test4.h264"
  "videos/test5.mp4"
)

mkdir -p "${OUT_DIR}"

echo "[INFO] Validation output: ${OUT_DIR}"

for video in "${VIDEOS[@]}"; do
  name="$(basename "${video}")"
  stem="${name%.*}"
  log_path="${OUT_DIR}/${stem}.log"
  jsonl_path="${OUT_DIR}/${stem}_events.jsonl"
  events_dir="${OUT_DIR}/${stem}_events"
  out_mp4="${OUT_DIR}/${stem}_annotated.mp4"

  echo "[INFO] Running ${video}"
  
  CMD=(python3 "${APP}" "${ROOT_DIR}/${video}"
    --no-display
    --output "${out_mp4}"
    --event-output-dir "${events_dir}"
    --event-jsonl "${jsonl_path}"
    --event-cooldown-frames 60
    --min-stable-votes 3
  )

  if [[ "${SAVE_EVENT_FRAME}" == "1" ]]; then
    CMD+=(--save-event-frame)
  fi

  "${CMD[@]}" >"${log_path}" 2>&1

  grep -E "^\[SUMMARY\]" "${log_path}" || true
done

echo "[INFO] Done. Inspect logs/events under ${OUT_DIR}"
