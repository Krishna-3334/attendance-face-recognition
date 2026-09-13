#!/usr/bin/env python3
"""Fetch the model weights the service needs.

* buffalo_l (SCRFD-10GF detector + ArcFace ResNet-50) is pulled by InsightFace
  itself into ~/.insightface/models on first use; this script triggers that
  download up front so the first API request is not a 300 MB surprise.
* The two MiniFASNet anti-spoofing ONNX files are pulled into backend/models.

Usage:  python scripts/download_models.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
MODELS = BACKEND / "models"

ANTISPOOF = {
    "AntiSpoofing_bin_1.5_128.onnx": "https://raw.githubusercontent.com/hairymax/Face-AntiSpoofing/main/saved_models/AntiSpoofing_bin_1.5_128.onnx",
    "AntiSpoofing_print-replay_1.5_128.onnx": "https://raw.githubusercontent.com/hairymax/Face-AntiSpoofing/main/saved_models/AntiSpoofing_print-replay_1.5_128.onnx",
}


def fetch_antispoof() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    for name, url in ANTISPOOF.items():
        dest = MODELS / name
        if dest.exists():
            print(f"  ✓ {name} (already present)")
            continue
        print(f"  ↓ {name}")
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ {name} ({dest.stat().st_size / 1e6:.1f} MB)")


def fetch_face_models() -> None:
    from insightface.app import FaceAnalysis

    print("  ↓ buffalo_l (SCRFD-10GF + ArcFace R50) — first run downloads ~300 MB")
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("  ✓ buffalo_l ready")


if __name__ == "__main__":
    print("Anti-spoofing models:")
    fetch_antispoof()
    print("Face models:")
    try:
        fetch_face_models()
    except Exception as exc:  # pragma: no cover
        print(f"  ! buffalo_l download failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print("\nAll models ready.")
