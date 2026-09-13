"""Face detection, alignment and ArcFace embedding.

Pipeline per frame:
    SCRFD-10GF detection -> 5-point landmark similarity-transform alignment to
    112x112 -> ResNet-50 ArcFace (w600k_r50) -> 512-D embedding -> L2 normalise.

Matching is cosine similarity against the enrolled gallery. Because every
embedding is L2-normalised, cosine similarity is a single matrix-vector product,
so the cost of matching is O(N * 512) rather than a per-user Python loop.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from . import config, db

log = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    det_score: float
    embedding: np.ndarray  # L2-normalised, 512-D


@dataclass
class MatchResult:
    matched: bool
    user_id: int | None
    name: str | None
    similarity: float
    runner_up: float
    reason: str


class FaceEngine:
    """Thin wrapper around InsightFace with an in-memory gallery cache."""

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        self._lock = threading.Lock()
        self._app = FaceAnalysis(
            name=config.FACE_MODEL_PACK,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=config.DET_SIZE, det_thresh=config.DET_THRESHOLD)
        self._ids: list[int] = []
        self._names: list[str] = []
        self._matrix: np.ndarray = np.zeros((0, 512), dtype=np.float32)
        db.init_db()  # idempotent; lets the eval scripts run without the API
        self.reload_gallery()
        log.info("FaceEngine ready: %s, gallery=%d identities", config.FACE_MODEL_PACK, len(self._ids))

    # --- gallery -------------------------------------------------------------

    def reload_gallery(self) -> None:
        with self._lock:
            self._ids, self._names, self._matrix = db.load_gallery()

    @property
    def gallery_size(self) -> int:
        return len(self._ids)

    # --- inference -----------------------------------------------------------

    def detect(self, bgr: np.ndarray) -> list[DetectedFace]:
        faces = self._app.get(bgr)
        out: list[DetectedFace] = []
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            w, h = x2 - x1, y2 - y1
            if max(w, h) < config.MIN_FACE_PX:
                continue
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            out.append(
                DetectedFace(
                    bbox=(x1, y1, w, h),
                    det_score=float(f.det_score),
                    embedding=emb,
                )
            )
        out.sort(key=lambda d: d.bbox[2] * d.bbox[3], reverse=True)
        return out

    def largest_face(self, bgr: np.ndarray) -> DetectedFace | None:
        faces = self.detect(bgr)
        return faces[0] if faces else None

    def match(self, embedding: np.ndarray) -> MatchResult:
        with self._lock:
            matrix, ids, names = self._matrix, self._ids, self._names

        if matrix.shape[0] == 0:
            return MatchResult(False, None, None, 0.0, 0.0, "empty_gallery")

        sims = matrix @ embedding.astype(np.float32)
        order = np.argsort(-sims)
        best = int(order[0])
        best_sim = float(sims[best])
        runner_up = float(sims[order[1]]) if sims.size > 1 else 0.0

        if best_sim < config.MATCH_THRESHOLD:
            return MatchResult(False, None, None, best_sim, runner_up, "below_threshold")
        if sims.size > 1 and (best_sim - runner_up) < config.MATCH_MARGIN:
            return MatchResult(False, None, None, best_sim, runner_up, "ambiguous")

        return MatchResult(True, ids[best], names[best], best_sim, runner_up, "match")


_engine: FaceEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> FaceEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = FaceEngine()
    return _engine


def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """Mean of L2-normalised embeddings, renormalised — the standard way to build
    a multi-shot gallery template from several enrolment frames."""
    stacked = np.stack([np.asarray(e, dtype=np.float32) for e in embeddings])
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        raise ValueError("degenerate embedding average")
    return (mean / norm).astype(np.float32)
