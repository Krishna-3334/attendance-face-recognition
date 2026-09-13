#!/usr/bin/env python3
"""Build a folder-per-identity dataset from LFW for the evaluation harness.

LFW (Labeled Faces in the Wild) is the standard public benchmark for face
verification and is what turns the evaluation from "two friends in my room" into
a number worth quoting. This pulls it via scikit-learn and writes it in the
layout ``evaluate.py`` expects.

    python eval/fetch_lfw.py --min-faces 10 --out data/lfw
    python eval/evaluate.py --data data/lfw --target-far 0.001

``--min-faces 10`` keeps identities with at least ten images: roughly 158
people and 4,300 images, which yields ~9 million impostor pairs — enough to
resolve a FAR of 0.001 comfortably. Lower it for a bigger, harder gallery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-faces", type=int, default=10, help="minimum images per identity")
    ap.add_argument("--out", type=Path, default=Path("data/lfw"))
    args = ap.parse_args()

    try:
        from sklearn.datasets import fetch_lfw_people
    except ImportError:
        raise SystemExit("pip install scikit-learn first")

    print(f"Fetching LFW (min_faces_per_person={args.min_faces}) — first run downloads ~200 MB")
    people = fetch_lfw_people(
        min_faces_per_person=args.min_faces, color=True, resize=1.0, funneled=True
    )

    images = (people.images * 255).astype(np.uint8)
    names = [people.target_names[t] for t in people.target]

    args.out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for img, name in zip(images, names):
        slug = name.replace(" ", "_")
        person_dir = args.out / slug
        person_dir.mkdir(exist_ok=True)
        idx = counts.get(slug, 0)
        counts[slug] = idx + 1
        cv2.imwrite(str(person_dir / f"{idx:03d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    total = sum(counts.values())
    print(f"Wrote {total} images across {len(counts)} identities to {args.out}")
    print(f"Approx. impostor pairs available: {total * (total - 1) // 2 - sum(c * (c - 1) // 2 for c in counts.values()):,}")


if __name__ == "__main__":
    main()
