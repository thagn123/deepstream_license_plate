#!/usr/bin/env python3
"""
eval_compare.py — So sánh toàn diện 2 OCR model từ event JSONL outputs.

Cách dùng:
    python3 tools/eval_compare.py \
        --lprnet  outputs/eval_xxx/lprnet/all_events.jsonl \
        --new-ocr outputs/eval_xxx/new_ocr/all_events.jsonl \
        [--lprnet-timing  outputs/eval_xxx/lprnet/timing.jsonl] \
        [--new-ocr-timing outputs/eval_xxx/new_ocr/timing.jsonl] \
        [--report-md eval_report.md]
"""

import argparse
import json
import math
import re
import os
from collections import Counter
from typing import List, Dict, Optional


# ── Vietnamese plate regex ────────────────────────────────────────────────────
# Ngang: 51A-12345 / 30K-123456 / 29B1-23456
# Vuông (ghép): 30K1-23456 / 51A1-2345 → sau khi ghép top+bot
_VN_PLATE_RE = re.compile(
    r"^[1-9]\d[A-Z]\d{0,1}-?\d{4,5}$"
)
_VN_PLATE_2ROW_RE = re.compile(
    r"^[1-9]\d[A-Z]\d-\d{4,5}$"   # has separator → 2-row plate
)

def _is_valid_vn(text: str) -> bool:
    t = text.replace("-", "").replace("_", "")
    return bool(_VN_PLATE_RE.match(text.replace("_", "-"))) or bool(
        re.match(r"^[1-9]\d[A-Z]\d?\d{4,5}$", t)
    )

def _is_2row(text: str) -> bool:
    return "-" in text or "_" in text

def _plate_len(text: str) -> int:
    return len(text.replace("-", "").replace("_", ""))


# ── Stats helpers ─────────────────────────────────────────────────────────────
def _stats(vals: list) -> dict:
    if not vals:
        return {"n": 0, "mean": 0, "median": 0, "std": 0, "min": 0, "max": 0,
                "p25": 0, "p75": 0, "p90": 0, "p99": 0}
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    def pct(p): return s[min(int(n * p / 100), n - 1)]
    return {"n": n, "mean": mean, "median": pct(50), "std": math.sqrt(var),
            "min": s[0], "max": s[-1],
            "p25": pct(25), "p75": pct(75), "p90": pct(90), "p99": pct(99)}

def _fmt(v, decimals=3):
    return f"{v:.{decimals}f}"

def _pct(a, b):
    return f"{a/b*100:.1f}%" if b > 0 else "N/A"


# ── Load events ───────────────────────────────────────────────────────────────
def _normalize(e: dict) -> Optional[dict]:
    """Normalize both flat (old) and nested (new) event schemas to nested."""
    # Skip non-LPR events (debug rows)
    if e.get("event_type") == "license_plate_recognized":
        return e  # already nested
    if "plate_text_stable" in e:
        # Flat schema (older pipeline version)
        return {
            "source_uri": e.get("source_uri", ""),
            "frame_num":  e.get("frame_num", 0),
            "vehicle": {
                "tracker_id":   e.get("vehicle_tracker_id", 0),
                "display_id":   e.get("vehicle_display_id", 0),
                "class_id":     e.get("vehicle_class_id", -1),
                "class_name":   e.get("vehicle_class_name", "unknown"),
                "confidence":   e.get("vehicle_confidence", 0),
                "bbox":         e.get("vehicle_bbox", [0,0,0,0]),
            },
            "plate": {
                "object_id":      e.get("plate_object_id", 0),
                "bbox":           e.get("plate_bbox", [0,0,0,0]),
                "text_raw":       e.get("plate_text_raw", ""),
                "text_stable":    e.get("plate_text_stable", ""),
                "score":          e.get("plate_text_score", 0),
                "votes":          e.get("plate_text_votes", 0),
                "ocr_confidence": e.get("ocr_confidence", 0),
                "ocr_backend":    e.get("ocr_backend", "lprnet"),
                "stable":         e.get("stable", False),
            },
            "association": {
                "method": e.get("association_method", "none"),
                "score":  e.get("association_score", 0),
            },
        }
    return None  # skip unrecognized format (debug rows)


def load_events(path: str) -> List[Dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                norm = _normalize(raw)
                if norm is not None:
                    events.append(norm)
            except json.JSONDecodeError:
                pass
    return events

def load_timing(path: Optional[str]) -> List[Dict]:
    if not path or not os.path.exists(path):
        return []
    result = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    result.append(json.loads(line))
                except Exception:
                    pass
    return result


# ── Feature extraction ────────────────────────────────────────────────────────
class ModelStats:
    def __init__(self, name: str, events: List[Dict], timing: List[Dict]):
        self.name = name
        self.events = events
        self.timing = timing
        self._compute()

    def _compute(self):
        evts = self.events
        n = len(evts)
        self.total_events = n
        if n == 0:
            return

        # Per-video counts
        self.per_video: Counter = Counter()
        for e in evts:
            uri = e.get("source_uri", "")
            vid = os.path.basename(uri).replace(".mp4", "")
            self.per_video[vid] += 1

        # Plate texts
        texts_stable = [e["plate"]["text_stable"] for e in evts if e["plate"].get("text_stable")]
        texts_raw    = [e["plate"]["text_raw"]    for e in evts if e["plate"].get("text_raw")]

        self.n_has_stable  = len(texts_stable)
        self.n_has_raw     = len(texts_raw)
        self.stable_rate   = self.n_has_stable / n

        # Unique plates
        self.unique_stable = set(texts_stable)
        self.unique_raw    = set(texts_raw)

        # VN plate validity
        self.n_valid_stable = sum(1 for t in texts_stable if _is_valid_vn(t))
        self.valid_rate     = self.n_valid_stable / self.n_has_stable if self.n_has_stable else 0

        # 2-row vs 1-row
        self.n_2row = sum(1 for t in texts_stable if _is_2row(t))
        self.n_1row = self.n_has_stable - self.n_2row
        self.tworow_rate = self.n_2row / self.n_has_stable if self.n_has_stable else 0

        # Text length (excluding separator)
        self.text_lengths = [_plate_len(t) for t in texts_stable]
        self.len_stats     = _stats(self.text_lengths)
        self.len_dist      = Counter(self.text_lengths)

        # OCR confidence
        confs = [e["plate"]["ocr_confidence"] for e in evts if "ocr_confidence" in e["plate"]]
        self.conf_stats = _stats(confs)

        # Score
        scores = [e["plate"]["score"] for e in evts if "score" in e["plate"]]
        self.score_stats = _stats(scores)

        # Votes
        votes = [e["plate"]["votes"] for e in evts if "votes" in e["plate"]]
        self.vote_stats = _stats(votes)
        self.vote_dist  = Counter(votes)

        # Plate bbox size
        bboxes = [e["plate"]["bbox"] for e in evts if "bbox" in e["plate"]]
        widths  = [b[2] for b in bboxes if len(b) >= 4]
        heights = [b[3] for b in bboxes if len(b) >= 4]
        areas   = [w * h for w, h in zip(widths, heights)]
        self.pw_stats = _stats(widths)
        self.ph_stats = _stats(heights)
        self.pa_stats = _stats(areas)

        # Vehicle class distribution
        self.vclass_dist: Counter = Counter()
        for e in evts:
            cls = e.get("vehicle", {}).get("class_name", "unknown")
            self.vclass_dist[cls] += 1

        # Vehicle confidence
        vconfs = [e.get("vehicle", {}).get("confidence", 0) for e in evts]
        self.vconf_stats = _stats(vconfs)

        # Association method
        self.assoc_dist: Counter = Counter()
        for e in evts:
            m = e.get("association", {}).get("method", "none")
            self.assoc_dist[m] += 1
        assoc_scores = [e.get("association", {}).get("score", 0) for e in evts]
        self.assoc_score_stats = _stats(assoc_scores)

        # Confidence buckets
        self.conf_buckets = self._conf_buckets(confs)

        # Top frequent plates
        self.top_plates = Counter(texts_stable).most_common(10)

        # raw == stable match
        raw_stable_pairs = [(e["plate"]["text_raw"], e["plate"]["text_stable"])
                            for e in evts
                            if e["plate"].get("text_raw") and e["plate"].get("text_stable")]
        self.n_raw_eq_stable = sum(1 for r, s in raw_stable_pairs if r == s)
        self.raw_eq_rate     = self.n_raw_eq_stable / len(raw_stable_pairs) if raw_stable_pairs else 0

        # Timing
        if self.timing:
            elapsed = [t["elapsed_s"] for t in self.timing if "elapsed_s" in t]
            fps_vals = []
            for t in self.timing:
                try:
                    fps_vals.append(float(t.get("fps", 0) or 0))
                except (ValueError, TypeError):
                    pass
            self.total_time_s  = sum(elapsed)
            self.mean_fps      = sum(fps_vals) / len(fps_vals) if fps_vals else None
        else:
            self.total_time_s  = None
            self.mean_fps      = None

    @staticmethod
    def _conf_buckets(confs):
        buckets = {"<0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "≥0.9": 0}
        for c in confs:
            if c < 0.3:   buckets["<0.3"] += 1
            elif c < 0.5: buckets["0.3-0.5"] += 1
            elif c < 0.7: buckets["0.5-0.7"] += 1
            elif c < 0.9: buckets["0.7-0.9"] += 1
            else:          buckets["≥0.9"] += 1
        return buckets


# ── Report rendering ──────────────────────────────────────────────────────────
def _sep(char="─", w=72): return char * w
def _h(title, char="═", w=72): return f"\n{char*w}\n  {title}\n{char*w}"

def _bar2(v1, v2, max_v, w=24, c1="█", c2="▓"):
    def bar(v):
        n = int(round(v / max_v * w)) if max_v > 0 else 0
        return c1 * n + "░" * (w - n)
    return bar(v1), bar(v2)

def _row(label, v1, v2, w=30):
    return f"  {label:<28} {str(v1):<{w}} {str(v2):<{w}}"

def render_report(ms1: ModelStats, ms2: ModelStats, output_lines: list):
    def p(line=""):
        output_lines.append(line)

    n1, n2 = ms1.total_events, ms2.total_events
    if n1 == 0 and n2 == 0:
        p("[ERROR] Cả hai JSONL đều rỗng.")
        return

    p()
    p("╔" + "═"*70 + "╗")
    p("║{:^70}║".format("OCR MODEL EVALUATION REPORT"))
    p("║{:^70}║".format("DeepStream LPR Pipeline — Vietnam License Plate"))
    p("╚" + "═"*70 + "╝")

    def label(ms): return ms.name[:28]
    L1 = label(ms1)
    L2 = label(ms2)

    # ── 1. VOLUME ─────────────────────────────────────────────────────────────
    p(_h("1. KHỐI LƯỢNG KẾT QUẢ"))
    p(_row("Chỉ số", L1, L2))
    p("  " + _sep("-", 68))
    p(_row("Tổng events",     n1, n2))
    p(_row("Unique videos",   len(ms1.per_video), len(ms2.per_video)))
    if n1 > 0 and n2 > 0:
        p(_row("Events/video (avg)", f"{n1/max(len(ms1.per_video),1):.1f}", f"{n2/max(len(ms2.per_video),1):.1f}"))
    p(_row("Có text_stable",  f"{ms1.n_has_stable} ({_pct(ms1.n_has_stable,n1)})", f"{ms2.n_has_stable} ({_pct(ms2.n_has_stable,n2)})"))
    p(_row("Unique plates",   len(ms1.unique_stable), len(ms2.unique_stable)))

    # Per-video breakdown
    p()
    p(f"  {'Video':<20} {'LPRNet':>10} {'New OCR':>10}  {'Δ':>8}")
    p("  " + _sep("-", 52))
    all_vids = sorted(set(ms1.per_video) | set(ms2.per_video))
    for vid in all_vids:
        v1 = ms1.per_video.get(vid, 0)
        v2 = ms2.per_video.get(vid, 0)
        delta = v2 - v1
        sign = "+" if delta >= 0 else ""
        p(f"  {vid:<20} {v1:>10} {v2:>10}  {sign}{delta:>7}")

    # ── 2. ACCURACY ───────────────────────────────────────────────────────────
    p(_h("2. ĐỘ CHÍNH XÁC BIỂN SỐ"))
    p(_row("Chỉ số", L1, L2))
    p("  " + _sep("-", 68))
    p(_row("Valid VN plate (regex)",
           f"{ms1.n_valid_stable} ({_pct(ms1.n_valid_stable, ms1.n_has_stable)})",
           f"{ms2.n_valid_stable} ({_pct(ms2.n_valid_stable, ms2.n_has_stable)})"))
    p(_row("Raw == Stable (ổn định)",
           f"{ms1.n_raw_eq_stable} ({_pct(ms1.n_raw_eq_stable, ms1.n_has_stable)})",
           f"{ms2.n_raw_eq_stable} ({_pct(ms2.n_raw_eq_stable, ms2.n_has_stable)})"))
    p(_row("Biển 2 dòng (có '-')",
           f"{ms1.n_2row} ({_pct(ms1.n_2row, ms1.n_has_stable)})",
           f"{ms2.n_2row} ({_pct(ms2.n_2row, ms2.n_has_stable)})"))
    p(_row("Biển 1 dòng (ngang)",
           f"{ms1.n_1row} ({_pct(ms1.n_1row, ms1.n_has_stable)})",
           f"{ms2.n_1row} ({_pct(ms2.n_1row, ms2.n_has_stable)})"))

    # Text length distribution
    p()
    p(f"  {'Độ dài text':<12} {'LPRNet':>8} {'%':>6}  {'New OCR':>8} {'%':>6}")
    p("  " + _sep("-", 44))
    all_lens = sorted(set(ms1.len_dist) | set(ms2.len_dist))
    for l in all_lens:
        v1 = ms1.len_dist.get(l, 0)
        v2 = ms2.len_dist.get(l, 0)
        p(f"  {l:<12} {v1:>8} {_pct(v1,ms1.n_has_stable):>6}  {v2:>8} {_pct(v2,ms2.n_has_stable):>6}")

    # ── 3. OCR CONFIDENCE ─────────────────────────────────────────────────────
    p(_h("3. OCR CONFIDENCE"))
    p(_row("Chỉ số", L1, L2))
    p("  " + _sep("-", 68))
    for k in ("mean", "median", "std", "p25", "p75", "p90", "p99", "min", "max"):
        v1 = _fmt(ms1.conf_stats.get(k, 0))
        v2 = _fmt(ms2.conf_stats.get(k, 0))
        p(_row(k, v1, v2))

    p()
    p(f"  {'Bucket':<12} {'LPRNet':>8} {'%':>6}  {'New OCR':>8} {'%':>6}")
    p("  " + _sep("-", 44))
    for bucket in ("<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "≥0.9"):
        v1 = ms1.conf_buckets.get(bucket, 0)
        v2 = ms2.conf_buckets.get(bucket, 0)
        b1, b2 = _bar2(v1, v2, max(v1, v2, 1), w=10)
        p(f"  {bucket:<12} {v1:>8} {_pct(v1,ms1.conf_stats['n']):>6}  {v2:>8} {_pct(v2,ms2.conf_stats['n']):>6}")

    # ── 4. VOTE STABILITY ─────────────────────────────────────────────────────
    p(_h("4. VOTE STABILITY (số lần OCR đồng thuận)"))
    p(_row("Chỉ số", L1, L2))
    p("  " + _sep("-", 68))
    for k in ("mean", "median", "std", "p90", "max"):
        p(_row(k, _fmt(ms1.vote_stats.get(k,0), 2), _fmt(ms2.vote_stats.get(k,0), 2)))

    p()
    p(f"  {'Votes':<8} {'LPRNet':>8} {'%':>6}  {'New OCR':>8} {'%':>6}")
    p("  " + _sep("-", 44))
    all_vote_keys = sorted((set(ms1.vote_dist) | set(ms2.vote_dist)) & set(range(1, 31)))
    for v in all_vote_keys:
        c1 = ms1.vote_dist.get(v, 0)
        c2 = ms2.vote_dist.get(v, 0)
        if c1 == 0 and c2 == 0:
            continue
        p(f"  {v:<8} {c1:>8} {_pct(c1,n1):>6}  {c2:>8} {_pct(c2,n2):>6}")

    # ── 5. PLATE SCORE ────────────────────────────────────────────────────────
    p(_h("5. PLATE SCORE (tổng điểm bình chọn)"))
    p(_row("Chỉ số", L1, L2))
    p("  " + _sep("-", 68))
    for k in ("mean", "median", "std", "p25", "p75", "p90", "p99", "min", "max"):
        p(_row(k, _fmt(ms1.score_stats.get(k,0), 2), _fmt(ms2.score_stats.get(k,0), 2)))

    # ── 6. PLATE BBOX SIZE ────────────────────────────────────────────────────
    p(_h("6. KÍCH THƯỚC BIỂN SỐ (pixel trong frame 1920×1080)"))
    p(_row("Chỉ số", "LPRNet width/height/area", "New OCR width/height/area"))
    p("  " + _sep("-", 68))
    for k in ("mean", "median", "p90", "min", "max"):
        w1 = _fmt(ms1.pw_stats.get(k, 0), 1)
        h1 = _fmt(ms1.ph_stats.get(k, 0), 1)
        a1 = _fmt(ms1.pa_stats.get(k, 0), 0)
        w2 = _fmt(ms2.pw_stats.get(k, 0), 1)
        h2 = _fmt(ms2.ph_stats.get(k, 0), 1)
        a2 = _fmt(ms2.pa_stats.get(k, 0), 0)
        p(_row(k, f"{w1}×{h1} ({a1}px²)", f"{w2}×{h2} ({a2}px²)"))

    # ── 7. VEHICLE CLASS ──────────────────────────────────────────────────────
    p(_h("7. PHÂN LOẠI PHƯƠNG TIỆN"))
    all_classes = sorted(set(ms1.vclass_dist) | set(ms2.vclass_dist))
    p(f"  {'Loại xe':<20} {'LPRNet':>8} {'%':>6}  {'New OCR':>8} {'%':>6}")
    p("  " + _sep("-", 48))
    for cls in all_classes:
        c1 = ms1.vclass_dist.get(cls, 0)
        c2 = ms2.vclass_dist.get(cls, 0)
        p(f"  {cls:<20} {c1:>8} {_pct(c1,n1):>6}  {c2:>8} {_pct(c2,n2):>6}")

    p()
    p(_row("Vehicle confidence mean", _fmt(ms1.vconf_stats.get("mean",0)), _fmt(ms2.vconf_stats.get("mean",0))))

    # ── 8. ASSOCIATION ────────────────────────────────────────────────────────
    p(_h("8. VEHICLE-PLATE ASSOCIATION"))
    p(_row("Chỉ số", L1, L2))
    p("  " + _sep("-", 68))
    all_methods = sorted(set(ms1.assoc_dist) | set(ms2.assoc_dist))
    for m in all_methods:
        c1 = ms1.assoc_dist.get(m, 0)
        c2 = ms2.assoc_dist.get(m, 0)
        p(_row(f"method={m}", f"{c1} ({_pct(c1,n1)})", f"{c2} ({_pct(c2,n2)})"))
    p(_row("Assoc score mean", _fmt(ms1.assoc_score_stats.get("mean",0)), _fmt(ms2.assoc_score_stats.get("mean",0))))

    # ── 9. TOP PLATES ────────────────────────────────────────────────────────
    p(_h("9. TOP BIỂN SỐ XUẤT HIỆN NHIỀU NHẤT"))
    p(f"  {'#':<4} {'LPRNet':<28} {'N':>5}  {'New OCR':<28} {'N':>5}")
    p("  " + _sep("-", 70))
    tp1 = ms1.top_plates[:10]
    tp2 = ms2.top_plates[:10]
    for i in range(max(len(tp1), len(tp2))):
        t1, c1 = tp1[i] if i < len(tp1) else ("", "")
        t2, c2 = tp2[i] if i < len(tp2) else ("", "")
        p(f"  {i+1:<4} {t1:<28} {str(c1):>5}  {t2:<28} {str(c2):>5}")

    # ── 10. TIMING ───────────────────────────────────────────────────────────
    p(_h("10. THỜI GIAN XỬ LÝ"))
    p(_row("Chỉ số", L1, L2))
    p("  " + _sep("-", 68))
    t1 = f"{ms1.total_time_s:.0f}s" if ms1.total_time_s else "N/A"
    t2 = f"{ms2.total_time_s:.0f}s" if ms2.total_time_s else "N/A"
    p(_row("Tổng thời gian (wall)", t1, t2))
    f1 = f"{ms1.mean_fps:.1f}" if ms1.mean_fps else "N/A"
    f2 = f"{ms2.mean_fps:.1f}" if ms2.mean_fps else "N/A"
    p(_row("FPS trung bình", f1, f2))
    e1 = f"{n1/ms1.total_time_s:.1f} ev/s" if ms1.total_time_s else "N/A"
    e2 = f"{n2/ms2.total_time_s:.1f} ev/s" if ms2.total_time_s else "N/A"
    p(_row("Throughput (events/s)", e1, e2))

    # ── 11. SUMMARY ──────────────────────────────────────────────────────────
    p(_h("11. TỔNG KẾT & NHẬN XÉT"))

    def winner(v1, v2, higher_better=True):
        if v1 == v2: return "="
        return ("✓LPRNet" if (v1 > v2) == higher_better else "✓NewOCR")

    p()
    p(f"  {'Tiêu chí':<32} {'Kết quả'}")
    p("  " + _sep("-", 60))

    checks = []

    # Event volume
    diff_evt = n2 - n1
    sign = "+" if diff_evt >= 0 else ""
    checks.append(("Số events (coverage)",
                   f"New OCR: {sign}{diff_evt} ({_pct(abs(diff_evt), n1)} {'nhiều' if diff_evt>=0 else 'ít'} hơn)"))

    # Valid plate rate
    vr1, vr2 = ms1.valid_rate, ms2.valid_rate
    w = "New OCR" if vr2 > vr1 else "LPRNet" if vr1 > vr2 else "="
    checks.append(("Tỉ lệ biển VN hợp lệ",
                   f"{w}: LPRNet={_pct(ms1.n_valid_stable,ms1.n_has_stable)} vs New={_pct(ms2.n_valid_stable,ms2.n_has_stable)}"))

    # Confidence
    c1m, c2m = ms1.conf_stats.get("mean",0), ms2.conf_stats.get("mean",0)
    w = "New OCR" if c2m > c1m else "LPRNet" if c1m > c2m else "="
    checks.append(("OCR Confidence (mean)",
                   f"{w}: LPRNet={_fmt(c1m)} vs New={_fmt(c2m)}"))

    # Vote stability
    v1m, v2m = ms1.vote_stats.get("mean",0), ms2.vote_stats.get("mean",0)
    w = "New OCR" if v2m > v1m else "LPRNet" if v1m > v2m else "="
    checks.append(("Vote stability (mean)",
                   f"{w}: LPRNet={_fmt(v1m,2)} vs New={_fmt(v2m,2)}"))

    # 2-row detection
    checks.append(("Biển 2 dòng detected",
                   f"LPRNet={ms1.n_2row} ({_pct(ms1.n_2row,ms1.n_has_stable)}) "
                   f"vs New={ms2.n_2row} ({_pct(ms2.n_2row,ms2.n_has_stable)})"))

    # Raw=stable
    r1, r2 = ms1.raw_eq_rate, ms2.raw_eq_rate
    w = "New OCR" if r2 > r1 else "LPRNet" if r1 > r2 else "="
    checks.append(("Consistency raw→stable",
                   f"{w}: LPRNet={_pct(ms1.n_raw_eq_stable,ms1.n_has_stable)} vs New={_pct(ms2.n_raw_eq_stable,ms2.n_has_stable)}"))

    # Unique plates
    checks.append(("Unique plate texts",
                   f"LPRNet={len(ms1.unique_stable)} vs New={len(ms2.unique_stable)}"))

    for label, result in checks:
        p(f"  {label:<32} {result}")

    p()
    p("  ── Nhận xét chi tiết ──────────────────────────────────────────────")
    p()

    # Coverage
    if diff_evt > 0:
        p(f"  [+] New OCR phát hiện nhiều event hơn LPRNet ({diff_evt:+d}).")
        p(f"      Có thể do: bỏ pseudo-class 14/15 giảm noise, hoặc model nhạy hơn.")
    elif diff_evt < 0:
        p(f"  [-] New OCR ít event hơn LPRNet ({diff_evt:+d}).")
        p(f"      Có thể do model mới confidence threshold khác, hoặc bỏ sót 1 số xe.")

    # Confidence
    diff_conf = c2m - c1m
    if abs(diff_conf) > 0.02:
        better = "New OCR" if diff_conf > 0 else "LPRNet"
        p(f"  [{'+'if diff_conf>0 else '-'}] {better} có confidence cao hơn ({abs(diff_conf):.3f} delta).")
    else:
        p(f"  [=] OCR confidence tương đương nhau (Δ={diff_conf:+.3f}).")

    # Valid plate
    diff_vr = vr2 - vr1
    if abs(diff_vr) > 0.03:
        better = "New OCR" if diff_vr > 0 else "LPRNet"
        p(f"  [{'+'if diff_vr>0 else '-'}] {better} tạo ra nhiều biển đúng format VN hơn ({abs(diff_vr)*100:.1f}pp delta).")
    else:
        p(f"  [=] Tỉ lệ biển hợp lệ VN tương đương (Δ={diff_vr*100:+.1f}pp).")

    # 2-row
    p(f"  [i] Biển 2 dòng: LPRNet={_pct(ms1.n_2row,ms1.n_has_stable)}, "
      f"New OCR={_pct(ms2.n_2row,ms2.n_has_stable)}.")
    p(f"      LPRNet dùng TOP/BOT split → merge lại. New OCR nhận diện trực tiếp.")

    # Timing
    if ms1.total_time_s and ms2.total_time_s:
        dt = ms2.total_time_s - ms1.total_time_s
        p(f"  [i] Thời gian: LPRNet={ms1.total_time_s:.0f}s, New OCR={ms2.total_time_s:.0f}s (Δ={dt:+.0f}s).")
    else:
        p(f"  [i] Không có timing data. Chạy lại với eval_run.sh để đo FPS.")

    p()
    p("  ── Khuyến nghị ────────────────────────────────────────────────────")
    p()
    p("  Để đánh giá chính xác hơn, cần ground truth (biển số thật).")
    p("  So sánh text_stable với danh sách biển đã biết → tính Exact Match %.")
    p("  Tham khảo: tools/eval_run.sh để chạy chuẩn trên toàn bộ 15 video.")
    p()
    p(_sep("═", 72))
    p()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="So sánh 2 OCR model từ event JSONL")
    parser.add_argument("--lprnet",  required=True, help="Path JSONL của LPRNet")
    parser.add_argument("--new-ocr", required=True, help="Path JSONL của New OCR 2024")
    parser.add_argument("--lprnet-timing",  default=None, help="timing.jsonl của LPRNet")
    parser.add_argument("--new-ocr-timing", default=None, help="timing.jsonl của New OCR")
    parser.add_argument("--report-md", default=None, help="Lưu report ra file .md")
    args = parser.parse_args()

    print(f"[INFO] Loading LPRNet  : {args.lprnet}")
    evts1 = load_events(args.lprnet)
    print(f"[INFO] Loading New OCR : {args.new_ocr}")
    evts2 = load_events(args.new_ocr)
    timing1 = load_timing(args.lprnet_timing)
    timing2 = load_timing(args.new_ocr_timing)

    print(f"[INFO] LPRNet events   : {len(evts1)}")
    print(f"[INFO] New OCR events  : {len(evts2)}")
    print()

    ms1 = ModelStats("LPRNet (baseline18)", evts1, timing1)
    ms2 = ModelStats("New OCR 2024 (lpr_ocr.20240305)", evts2, timing2)

    lines = []
    render_report(ms1, ms2, lines)
    report_text = "\n".join(lines)

    print(report_text)

    if args.report_md:
        with open(args.report_md, "w", encoding="utf-8") as f:
            f.write("```\n")
            f.write(report_text)
            f.write("\n```\n")
        print(f"[INFO] Report saved to: {args.report_md}")


if __name__ == "__main__":
    main()
