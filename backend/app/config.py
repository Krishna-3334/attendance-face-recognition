"""Runtime configuration.

Every threshold in this file is an *operating point*, not a magic number. The
recognition threshold is chosen from the FAR/FRR curves produced by
``eval/evaluate.py``; re-run that script on your own gallery and update
``MATCH_THRESHOLD`` (or set the environment variable) with the value it reports.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "attendance.db"))

# --- Face detection / recognition -------------------------------------------
# InsightFace 'buffalo_l' bundles SCRFD-10GF for detection and a ResNet-50
# ArcFace (w600k_r50) producing 512-D embeddings.
FACE_MODEL_PACK = os.getenv("FACE_MODEL_PACK", "buffalo_l")
DET_SIZE = (int(os.getenv("DET_WIDTH", 640)), int(os.getenv("DET_HEIGHT", 640)))
DET_THRESHOLD = float(os.getenv("DET_THRESHOLD", 0.5))

# Minimum face size (px, longest bbox side) accepted for enrolment/recognition.
MIN_FACE_PX = int(os.getenv("MIN_FACE_PX", 60))

# Cosine-similarity threshold for declaring a match. The default below is the
# value selected at ~0.1% FAR on the reference evaluation; treat it as a
# starting point and re-derive it for your deployment.
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", 0.42))

# Second-best margin: reject when the top two gallery identities are closer to
# each other than this, i.e. the decision is not confidently separable.
MATCH_MARGIN = float(os.getenv("MATCH_MARGIN", 0.05))

# --- Passive liveness (anti-spoofing) ---------------------------------------
LIVENESS_ENABLED = os.getenv("LIVENESS_ENABLED", "1") not in {"0", "false", "False"}
LIVENESS_BINARY_MODEL = MODELS_DIR / "AntiSpoofing_bin_1.5_128.onnx"
LIVENESS_ATTACK_MODEL = MODELS_DIR / "AntiSpoofing_print-replay_1.5_128.onnx"
# Probability of "live" required to pass. Raise it to trade convenience for
# security (lower APCER, higher BPCER).
LIVENESS_THRESHOLD = float(os.getenv("LIVENESS_THRESHOLD", 0.60))
# Face crop is expanded by this factor before liveness inference — the model was
# trained on 1.5x bbox crops, so context around the face carries the cue.
LIVENESS_BBOX_INC = 1.5
LIVENESS_INPUT_SIZE = 128

# --- Attendance rules --------------------------------------------------------
# Suppress duplicate check-ins for the same person inside this window.
ATTENDANCE_COOLDOWN_SECONDS = int(os.getenv("ATTENDANCE_COOLDOWN_SECONDS", 300))

# --- Server ------------------------------------------------------------------
# Vite dev server (3000 per vite.config.js), its default 5173, and `vite preview`.
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if o.strip()
]

DATA_DIR.mkdir(parents=True, exist_ok=True)
