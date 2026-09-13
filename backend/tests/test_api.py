"""End-to-end test of the enrol -> recognise -> reject paths.

Run with:  python -m pytest tests/ -v      (or simply: python tests/test_api.py)

The test needs a handful of face images. Point FACE_TEST_DIR at a directory
containing ``<name>_1.jpg`` style files, or drop images into tests/fixtures/:

    tests/fixtures/
        alice_a.jpg   alice_b.jpg     # same person, two captures
        bob_a.jpg                     # a different person
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

# Use a scratch database so a test run never touches real enrolments.
_tmpdir = tempfile.mkdtemp(prefix="bioaccess-test-")
os.environ.setdefault("DATA_DIR", _tmpdir)
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test.db"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

FIXTURES = Path(os.getenv("FACE_TEST_DIR", BACKEND / "tests" / "fixtures"))


def _fixture(stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = FIXTURES / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    same_a, same_b, other = _fixture("alice_a"), _fixture("alice_b"), _fixture("bob_a")
    if not (same_a and same_b and other):
        print(f"! no fixtures in {FIXTURES} — add alice_a, alice_b, bob_a images to run this test")
        return 0

    failures = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"  [{'PASS' if condition else 'FAIL'}] {label} {detail}")
        if not condition:
            failures += 1

    with TestClient(app) as client:
        cfg = client.get("/api/config").json()
        print(f"config: {cfg}")

        with same_a.open("rb") as fh:
            r = client.post("/api/enrol", data={"name": "Alice"}, files={"files": fh})
        body = r.json()
        check("enrol Alice", r.status_code == 200 and body["ok"], str(body.get("detail", "")))

        with same_a.open("rb") as fh:
            r = client.post("/api/enrol", data={"name": "Alice"}, files={"files": fh})
        check("duplicate enrolment refused", not r.json()["ok"])

        with same_b.open("rb") as fh:
            r = client.post("/api/recognise", files={"file": fh})
        body = r.json()
        check(
            "second capture of Alice matches",
            body["outcome"] == "match" and body["name"] == "Alice",
            f"(outcome={body['outcome']}, sim={body['similarity']})",
        )

        with same_b.open("rb") as fh:
            r = client.post("/api/recognise", files={"file": fh})
        check("cooldown suppresses duplicate log", r.json()["outcome"] == "cooldown")

        with other.open("rb") as fh:
            r = client.post("/api/recognise", files={"file": fh})
        body = r.json()
        check(
            "different person is not matched",
            body["outcome"] == "unknown",
            f"(outcome={body['outcome']}, sim={body['similarity']})",
        )

        log = client.get("/api/attendance").json()
        check("exactly one attendance record", len(log) == 1, f"(got {len(log)})")

        users = client.get("/api/users").json()
        r = client.delete(f"/api/users/{users[0]['id']}")
        check("user deleted", r.status_code == 200)
        check("gallery now empty", client.get("/api/users").json() == [])

    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
