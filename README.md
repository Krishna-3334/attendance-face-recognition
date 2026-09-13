# BioAccess — Face-Recognition Attendance with Passive Liveness

Server-side face recognition for attendance, built on SCRFD detection and ArcFace
embeddings, with a MiniFASNet presentation-attack check that runs **before** any
gallery comparison, and an evaluation harness that derives the matching threshold
from measured FAR/FRR curves instead of inheriting a library default.

```
browser (React)                    FastAPI service (Python)
─────────────────                  ─────────────────────────────────────────
webcam frame  ──POST /api/recognise──▶  SCRFD-10GF detection
                                        └─▶ MiniFASNet liveness  ──reject──▶ spoof_events
                                             └─▶ ArcFace R50 → 512-D embedding
                                                  └─▶ cosine vs gallery → attendance
```

## What changed in v2, and why

Version 1 ran the whole pipeline in the browser through `face-api.js`: TinyFaceDetector
and a TF.js recognition net, descriptors kept in `localStorage`, and matching at
`face-api`'s default Euclidean distance of 0.6. It worked as a demo but it was not a
recognition system, for four reasons that v2 addresses directly.

**No liveness.** A photograph held up to the webcam marked attendance. v2 scores every
probe with a MiniFASNet classifier trained on CelebA-Spoof and rejects print and replay
presentations before the embedding is ever compared with the gallery, so a spoofed frame
cannot produce an attendance record. Rejections are written to a separate audit table
together with the identity they would otherwise have matched.

**An inherited threshold.** 0.6 was a library default, not a decision. v2 ships
`eval/evaluate.py`, which builds genuine and impostor score distributions over a stated
gallery, reports ROC AUC and EER, and selects the cosine threshold that meets a target
false-accept rate — then tells you the false-reject rate you are paying for it. The
number in `app/config.py` is that operating point.

**A weaker model.** `face-api.js` is unmaintained and its recognition net is a small
TF.js port. v2 uses InsightFace `buffalo_l`: SCRFD-10GF for detection with five-point
landmark alignment, and a ResNet-50 ArcFace (`w600k_r50`) producing 512-D embeddings,
run through ONNX Runtime.

**`localStorage` is not a database.** Enrolments, per-frame samples, attendance and
rejected attempts now live in SQLite. Gallery embeddings are loaded once into a single
matrix so matching against N identities is one dot product rather than a Python loop.

## Quick start

```bash
# 1. backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py      # buffalo_l (~300 MB) + 2 anti-spoofing models
./run.sh                               # http://127.0.0.1:8000/docs

# 2. frontend (second terminal)
cp .env.example .env.local
npm install
npm run dev                            # http://localhost:3000
```

Enrol yourself under **Enrolment** (three frames are captured and averaged into one
template), then switch to **Mark Attendance**. Hold a photo of yourself up to the camera
to see the liveness stage reject it.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/enrol` | Enrol an identity from one or more frames; returns intra-class similarity |
| `POST` | `/api/recognise` | Liveness-gated 1:N identification; logs attendance on a match |
| `GET` | `/api/users` · `DELETE /api/users/{id}` | Manage the gallery |
| `GET` | `/api/attendance` | Check-in log with cosine and liveness scores |
| `GET` | `/api/spoof-events` | Rejected presentation attempts |
| `GET` | `/api/config` · `/api/stats` | Operating point, model card, live counters |

## Evaluation

The harness answers three questions: how separable are genuine and impostor pairs, what
threshold should be deployed, and how often does the deployed system get it wrong.

```bash
cd backend
python eval/fetch_lfw.py --min-faces 10 --out data/lfw     # ~158 identities, ~4.3k images
python eval/evaluate.py --data data/lfw --target-far 0.001
```

It prints and writes to `eval/reports/`:

* **Verification (1:1)** — ROC AUC, EER, and the cosine threshold meeting the FAR budget,
  with the FRR incurred. Plots: score distributions, FAR/FRR curves, ROC.
* **Identification (1:N, leave-one-out)** — for each image, the remaining images of that
  identity form the gallery template exactly as `/api/enrol` would; reports rank-1
  accuracy and the TAR/FAR/FRR actually experienced at the chosen threshold.
* **Liveness** — pass `--live-dir` and `--spoof-dir` to get APCER, BPCER and ACER
  (ISO/IEC 30107-3), i.e. attacks wrongly accepted and real faces wrongly rejected.

A FAR of *p* needs roughly `10/p` impostor pairs to be measurable at all; the script
refuses to recommend a threshold when the dataset cannot resolve the target, rather than
quoting a number the data does not support.

Record your results here:

| Metric | Value | Dataset |
| --- | --- | --- |
| ROC AUC | | |
| EER | | |
| Operating threshold @ 0.1% FAR | | |
| FRR at that threshold | | |
| Rank-1 accuracy (1:N) | | |
| Liveness APCER / BPCER / ACER | | |
| End-to-end latency (CPU) | | |

## Configuration

Everything is an environment variable, read in `app/config.py`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATCH_THRESHOLD` | `0.42` | Cosine similarity required for a match |
| `MATCH_MARGIN` | `0.05` | Minimum gap to the runner-up; below it the probe is ambiguous and rejected |
| `LIVENESS_THRESHOLD` | `0.60` | P(live) required to pass; raise for lower APCER at higher BPCER |
| `LIVENESS_ENABLED` | `1` | Set `0` to bypass liveness (measurement only, never deployment) |
| `ATTENDANCE_COOLDOWN_SECONDS` | `300` | Duplicate check-in suppression window |
| `DET_THRESHOLD`, `MIN_FACE_PX` | `0.5`, `60` | Detector confidence and minimum usable face size |

## Tests

```bash
cd backend
# drop alice_a / alice_b (same person) and bob_a (someone else) into tests/fixtures/
python tests/test_api.py
```

Covers enrolment, duplicate refusal, a genuine match, cooldown suppression, rejection of
an unenrolled person, and gallery deletion.

## Known limitations

* Liveness is a single-frame passive check. It stops printed photos and phone screens; it
  is not evaluated against 3D masks, and no depth or IR sensor is used.
* The gallery is a flat matrix scanned in full on every probe — fine to a few thousand
  identities, beyond which an ANN index (FAISS, HNSW) belongs here.
* Recognition accuracy on a demographic that differs from the ArcFace training
  distribution should be measured before deployment, not assumed from the LFW number.
* The service has no authentication; anything facing a network needs an auth layer in
  front of the enrolment and deletion endpoints.

## Credits

* [InsightFace](https://github.com/deepinsight/insightface) — SCRFD detector and ArcFace `buffalo_l` model pack.
* [hairymax/Face-AntiSpoofing](https://github.com/hairymax/Face-AntiSpoofing) — MiniFASNet ONNX weights trained on CelebA-Spoof, following [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing).
