"""Response models for the API."""

from __future__ import annotations

from pydantic import BaseModel


class UserOut(BaseModel):
    id: int
    name: str
    created_at: str
    n_samples: int


class EnrolOut(BaseModel):
    ok: bool
    user: UserOut | None = None
    accepted_samples: int = 0
    rejected_samples: list[str] = []
    intra_class_similarity: float | None = None
    detail: str | None = None


class LivenessOut(BaseModel):
    enabled: bool
    is_live: bool
    live_score: float
    attack_type: str
    attack_scores: dict[str, float]


class AttendanceOut(BaseModel):
    id: int
    user_id: int
    name: str
    timestamp: str
    date: str
    similarity: float
    liveness: float | None = None
    status: str


class RecogniseOut(BaseModel):
    ok: bool
    outcome: str  # match | no_face | spoof | unknown | ambiguous | cooldown
    name: str | None = None
    user_id: int | None = None
    similarity: float = 0.0
    runner_up: float = 0.0
    bbox: list[int] | None = None
    liveness: LivenessOut | None = None
    record: AttendanceOut | None = None
    latency_ms: float = 0.0
    detail: str | None = None


class ConfigOut(BaseModel):
    detector: str
    embedding_model: str
    embedding_dim: int
    match_threshold: float
    match_margin: float
    liveness_enabled: bool
    liveness_threshold: float
    gallery_size: int
    cooldown_seconds: int
