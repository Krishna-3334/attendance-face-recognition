"""Passive anti-spoofing (presentation-attack detection).

Two MiniFASNet-family ONNX classifiers trained on CelebA-Spoof are run on a 1.5x
expanded face crop:

* ``AntiSpoofing_bin_1.5_128``          -> P(live) vs P(spoof)
* ``AntiSpoofing_print-replay_1.5_128`` -> P(live) / P(print) / P(replay)

Passive means no user action is required — no blink, no head turn. The cue the
network uses is the texture and moire/reflectance signature that a printed sheet
or a phone screen leaves behind, which is why the crop is expanded beyond the
face box: the paper edge and screen bezel carry signal.

Decision: a probe passes when P(live) from the binary model is at or above
``LIVENESS_THRESHOLD``. The attack-type head is advisory — it labels *what* the
attack looked like for the audit log, and does not gate the decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from . import config

log = logging.getLogger(__name__)

ATTACK_LABELS = ("live", "print", "replay")


@dataclass
class LivenessResult:
    is_live: bool
    live_score: float
    attack_type: str
    attack_scores: dict[str, float]
    enabled: bool = True


def increased_crop(rgb: np.ndarray, bbox: tuple[int, int, int, int], inc: float) -> np.ndarray:
    """Square crop around the face, expanded by ``inc``, zero-padded at borders.

    Mirrors the crop used to build the training set, so the model sees the same
    framing at inference as it did during training.
    """
    real_h, real_w = rgb.shape[:2]
    x, y, w, h = bbox
    side = max(w, h)
    xc, yc = x + w / 2, y + h / 2

    x0 = int(xc - side * inc / 2)
    y0 = int(yc - side * inc / 2)
    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(real_w, x0 + int(side * inc))
    y2 = min(real_h, y0 + int(side * inc))

    crop = rgb[y1:y2, x1:x2, :]
    if crop.size == 0:
        raise ValueError("empty crop")
    return cv2.copyMakeBorder(
        crop,
        y1 - y0,
        int(side * inc) - (y2 - y0),
        x1 - x0,
        int(side * inc) - (x2 - x0),
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


def _letterbox(img: np.ndarray, size: int) -> np.ndarray:
    old_h, old_w = img.shape[:2]
    ratio = float(size) / max(old_h, old_w)
    new_h, new_w = int(old_h * ratio), int(old_w * ratio)
    img = cv2.resize(img, (new_w, new_h))
    dh, dw = size - new_h, size - new_w
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return np.expand_dims(img.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class _OnnxClassifier:
    def __init__(self, path: Path, size: int) -> None:
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = size

    def __call__(self, crop_rgb: np.ndarray) -> np.ndarray:
        out = self.session.run([], {self.input_name: _letterbox(crop_rgb, self.size)})[0]
        return _softmax(np.asarray(out).ravel())


class LivenessDetector:
    def __init__(self) -> None:
        self.enabled = config.LIVENESS_ENABLED
        self.binary: _OnnxClassifier | None = None
        self.attack: _OnnxClassifier | None = None
        if not self.enabled:
            log.warning("Liveness disabled by configuration")
            return
        if not config.LIVENESS_BINARY_MODEL.exists():
            log.warning(
                "Liveness model missing at %s — run scripts/download_models.py. "
                "Continuing with liveness DISABLED.",
                config.LIVENESS_BINARY_MODEL,
            )
            self.enabled = False
            return
        self.binary = _OnnxClassifier(config.LIVENESS_BINARY_MODEL, config.LIVENESS_INPUT_SIZE)
        if config.LIVENESS_ATTACK_MODEL.exists():
            self.attack = _OnnxClassifier(config.LIVENESS_ATTACK_MODEL, config.LIVENESS_INPUT_SIZE)
        log.info("Liveness ready (threshold=%.2f)", config.LIVENESS_THRESHOLD)

    def check(self, bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> LivenessResult:
        if not self.enabled or self.binary is None:
            return LivenessResult(True, 1.0, "live", {"live": 1.0}, enabled=False)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop = increased_crop(rgb, bbox, config.LIVENESS_BBOX_INC)

        probs = self.binary(crop)
        live_score = float(probs[0])

        attack_scores = {"live": live_score, "spoof": float(1.0 - live_score)}
        attack_type = "live" if live_score >= config.LIVENESS_THRESHOLD else "spoof"

        if self.attack is not None:
            a = self.attack(crop)
            attack_scores = {ATTACK_LABELS[i]: float(a[i]) for i in range(min(3, a.size))}
            if live_score < config.LIVENESS_THRESHOLD:
                # Name the attack from the 3-class head, ignoring its 'live' logit.
                spoof_only = {k: v for k, v in attack_scores.items() if k != "live"}
                if spoof_only:
                    attack_type = max(spoof_only, key=spoof_only.get)

        return LivenessResult(
            is_live=live_score >= config.LIVENESS_THRESHOLD,
            live_score=live_score,
            attack_type=attack_type,
            attack_scores=attack_scores,
        )


_detector: LivenessDetector | None = None


def get_detector() -> LivenessDetector:
    global _detector
    if _detector is None:
        _detector = LivenessDetector()
    return _detector
