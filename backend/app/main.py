"""FastAPI service: enrolment, liveness-gated recognition, attendance log.

Order of operations on /api/recognise is deliberate — detect, then liveness,
then match. A presentation attack is rejected before it is ever compared with
the gallery, so a spoofed frame can never produce an attendance record, and the
rejection is written to its own audit table.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config, db, liveness, recognition
from .schemas import (
    AttendanceOut,
    ConfigOut,
    EnrolOut,
    LivenessOut,
    RecogniseOut,
    UserOut,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bioaccess")

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load models once at startup rather than on the first request."""
    db.init_db()
    recognition.get_engine()
    liveness.get_detector()
    yield


app = FastAPI(title="BioAccess Face Attendance API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _decode(upload: UploadFile) -> np.ndarray:
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=f"could not decode image: {upload.filename}")
    return img


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/api/config", response_model=ConfigOut)
def get_config() -> ConfigOut:
    engine = recognition.get_engine()
    det = liveness.get_detector()
    return ConfigOut(
        detector="SCRFD-10GF (buffalo_l)",
        embedding_model="ArcFace ResNet-50 (w600k_r50)",
        embedding_dim=512,
        match_threshold=config.MATCH_THRESHOLD,
        match_margin=config.MATCH_MARGIN,
        liveness_enabled=det.enabled,
        liveness_threshold=config.LIVENESS_THRESHOLD,
        gallery_size=engine.gallery_size,
        cooldown_seconds=config.ATTENDANCE_COOLDOWN_SECONDS,
    )


@app.get("/api/stats")
def get_stats() -> dict:
    s = db.stats()
    s["gallery_size"] = recognition.get_engine().gallery_size
    return s


# --- enrolment ---------------------------------------------------------------


@app.post("/api/enrol", response_model=EnrolOut)
async def enrol(name: str = Form(...), files: list[UploadFile] = File(...)) -> EnrolOut:
    """Enrol an identity from one or more frames.

    Several frames are strongly preferred: the stored template is the
    renormalised mean of the per-frame embeddings, which suppresses the pose and
    lighting of any single capture. The response reports the mean pairwise
    similarity between the supplied frames, so a bad capture set (someone else in
    a frame, or a badly blurred one) is visible at enrolment time rather than as
    a mysterious false reject later.
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if db.user_exists(name):
        return EnrolOut(ok=False, detail=f"'{name}' is already enrolled")

    engine = recognition.get_engine()
    embeddings: list[np.ndarray] = []
    rejected: list[str] = []

    for upload in files:
        img = await _decode(upload)
        faces = engine.detect(img)
        if not faces:
            rejected.append(f"{upload.filename}: no face detected")
            continue
        if len(faces) > 1:
            rejected.append(f"{upload.filename}: {len(faces)} faces in frame")
            continue
        embeddings.append(faces[0].embedding)

    if not embeddings:
        return EnrolOut(ok=False, rejected_samples=rejected, detail="no usable frames")

    template = recognition.average_embeddings(embeddings)

    intra = None
    if len(embeddings) > 1:
        mat = np.stack(embeddings)
        sims = mat @ mat.T
        iu = np.triu_indices(len(embeddings), k=1)
        intra = float(round(float(sims[iu].mean()), 4))

    user_id = db.add_user(name, template, embeddings)
    engine.reload_gallery()

    users = {u["id"]: u for u in db.list_users()}
    return EnrolOut(
        ok=True,
        user=UserOut(**users[user_id]),
        accepted_samples=len(embeddings),
        rejected_samples=rejected,
        intra_class_similarity=intra,
    )


@app.get("/api/users", response_model=list[UserOut])
def get_users() -> list[UserOut]:
    return [UserOut(**u) for u in db.list_users()]


@app.delete("/api/users/{user_id}")
def remove_user(user_id: int) -> dict:
    if not db.delete_user(user_id):
        raise HTTPException(status_code=404, detail="user not found")
    recognition.get_engine().reload_gallery()
    return {"ok": True}


# --- recognition -------------------------------------------------------------


@app.post("/api/recognise", response_model=RecogniseOut)
async def recognise(file: UploadFile = File(...), log_attendance: bool = Form(True)) -> RecogniseOut:
    started = time.perf_counter()
    img = await _decode(file)
    engine = recognition.get_engine()
    detector = liveness.get_detector()

    face = engine.largest_face(img)
    if face is None:
        return RecogniseOut(
            ok=True,
            outcome="no_face",
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    live = detector.check(img, face.bbox)
    live_out = LivenessOut(
        enabled=live.enabled,
        is_live=live.is_live,
        live_score=round(live.live_score, 4),
        attack_type=live.attack_type,
        attack_scores={k: round(v, 4) for k, v in live.attack_scores.items()},
    )

    if not live.is_live:
        # Identify only for the audit trail — never log attendance for a spoof.
        probe = engine.match(face.embedding)
        db.log_spoof(live.live_score, live.attack_type, probe.name)
        return RecogniseOut(
            ok=True,
            outcome="spoof",
            bbox=list(face.bbox),
            liveness=live_out,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            detail=f"presentation attack rejected ({live.attack_type})",
        )

    result = engine.match(face.embedding)
    latency = round((time.perf_counter() - started) * 1000, 1)

    if not result.matched:
        return RecogniseOut(
            ok=True,
            outcome="ambiguous" if result.reason == "ambiguous" else "unknown",
            similarity=round(result.similarity, 4),
            runner_up=round(result.runner_up, 4),
            bbox=list(face.bbox),
            liveness=live_out,
            latency_ms=latency,
            detail=result.reason,
        )

    assert result.user_id is not None and result.name is not None

    if log_attendance:
        elapsed = db.seconds_since_last_checkin(result.user_id)
        if elapsed is not None and elapsed < config.ATTENDANCE_COOLDOWN_SECONDS:
            return RecogniseOut(
                ok=True,
                outcome="cooldown",
                name=result.name,
                user_id=result.user_id,
                similarity=round(result.similarity, 4),
                runner_up=round(result.runner_up, 4),
                bbox=list(face.bbox),
                liveness=live_out,
                latency_ms=latency,
                detail=f"already checked in {int(elapsed)}s ago",
            )
        record = db.log_attendance(result.user_id, result.name, result.similarity, live.live_score)
        return RecogniseOut(
            ok=True,
            outcome="match",
            name=result.name,
            user_id=result.user_id,
            similarity=round(result.similarity, 4),
            runner_up=round(result.runner_up, 4),
            bbox=list(face.bbox),
            liveness=live_out,
            record=AttendanceOut(**record),
            latency_ms=latency,
        )

    return RecogniseOut(
        ok=True,
        outcome="match",
        name=result.name,
        user_id=result.user_id,
        similarity=round(result.similarity, 4),
        runner_up=round(result.runner_up, 4),
        bbox=list(face.bbox),
        liveness=live_out,
        latency_ms=latency,
    )


# --- logs --------------------------------------------------------------------


@app.get("/api/attendance", response_model=list[AttendanceOut])
def get_attendance(limit: int = 200) -> list[AttendanceOut]:
    return [AttendanceOut(**r) for r in db.list_attendance(limit)]


@app.get("/api/spoof-events")
def get_spoof_events(limit: int = 100) -> list[dict]:
    return db.list_spoof_events(limit)
