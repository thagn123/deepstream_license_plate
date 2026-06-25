#!/usr/bin/env python3
"""
app_lpr_v2.py — DeepStream LPR pipeline using lpr_ocr.20240305.onnx.

  - SGIE3: lpr_ocr.20240305.onnx (37-class CTC, blank=0, _ = 2-line separator)
  - Class 13 only (no TOP/BOTTOM split — model handles square plates natively)

Usage:
  python3 src/app_lpr_v2.py <sources...> [--output out.mp4] [--no-display] ...
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from lpr.probes.sgie3 import sgie3_sink_pad_buffer_probe_new
from lpr.probes.metadata import metadata_src_pad_buffer_probe
from lpr.pipeline import run

if __name__ == "__main__":
    sys.exit(run(
        sys.argv,
        probe_overrides={
            "sgie3":    sgie3_sink_pad_buffer_probe_new,
            "metadata": metadata_src_pad_buffer_probe,
        },
        ocr_backend="new_ocr_2024",
    ))
