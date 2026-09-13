# Backend — recognition service

```
app/
  config.py        operating points and paths (all overridable by env vars)
  db.py            SQLite: users, per-frame samples, attendance, spoof_events
  recognition.py   SCRFD detection + ArcFace embedding + cosine matching
  liveness.py      MiniFASNet passive presentation-attack detection
  schemas.py       response models
  main.py          FastAPI routes
eval/
  fetch_lfw.py     build a folder-per-identity dataset from LFW
  evaluate.py      ROC / EER / FAR / FRR / rank-1 / APCER / BPCER, and threshold choice
scripts/
  download_models.py
tests/
  test_api.py      enrol -> match -> reject, end to end
```

Run:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
./run.sh
```

Interactive API docs at <http://127.0.0.1:8000/docs>.

Model weights are not committed. `buffalo_l` lands in `~/.insightface/models`; the two
anti-spoofing ONNX files land in `backend/models/` and are gitignored.
