#!/usr/bin/env python3
"""Evaluate the recognition and liveness stages, and choose operating points.

This script is the reason the project is more than a demo: it replaces the
inherited "0.6 threshold" of the old face-api.js build with a threshold derived
from measured genuine/impostor score distributions on a stated gallery.

Two evaluations are produced.

1. VERIFICATION (1:1).  Every image is embedded; all same-identity pairs form
   the genuine distribution and all different-identity pairs the impostor
   distribution.  From those we report AUC, EER, and the cosine threshold that
   meets a target FAR (false accept rate), together with the FRR paid for it.

2. IDENTIFICATION (1:N), leave-one-out.  For each identity the remaining images
   build the gallery template exactly as /api/enrol would; each held-out image is
   then matched against the full gallery.  We report rank-1 accuracy, and — at
   the chosen threshold — the true accept, false accept and false reject rates,
   which is what the deployed system actually experiences.

Optionally, LIVENESS is scored on bona-fide and attack folders using the
ISO/IEC 30107-3 metrics APCER, BPCER and ACER.

Dataset layout (folder per identity)::

    dataset/
        alice/img01.jpg img02.jpg ...
        bob/img01.jpg ...

Usage::

    python eval/evaluate.py --data dataset --target-far 0.001
    python eval/evaluate.py --data dataset --live-dir pa/live --spoof-dir pa/attack
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, liveness, recognition  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --- data loading ------------------------------------------------------------


def load_dataset(root: Path) -> dict[str, list[Path]]:
    identities: dict[str, list[Path]] = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if images:
            identities[person_dir.name] = images
    if not identities:
        raise SystemExit(f"no identity folders with images found under {root}")
    return identities


def embed_dataset(
    engine: recognition.FaceEngine, identities: dict[str, list[Path]]
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Return {identity: matrix[n,512]} plus a list of skipped-file messages."""
    embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    skipped: list[str] = []
    for name, paths in identities.items():
        for path in paths:
            img = cv2.imread(str(path))
            if img is None:
                skipped.append(f"{path}: unreadable")
                continue
            face = engine.largest_face(img)
            if face is None:
                skipped.append(f"{path}: no face detected")
                continue
            embeddings[name].append(face.embedding)
    return {k: np.stack(v) for k, v in embeddings.items() if v}, skipped


# --- verification (1:1) ------------------------------------------------------


def verification_scores(embeddings: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    names = sorted(embeddings)
    genuine: list[float] = []
    impostor: list[float] = []

    for name in names:
        mat = embeddings[name]
        if mat.shape[0] > 1:
            sims = mat @ mat.T
            iu = np.triu_indices(mat.shape[0], k=1)
            genuine.extend(sims[iu].tolist())

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            impostor.extend((embeddings[a] @ embeddings[b].T).ravel().tolist())

    return np.asarray(genuine, dtype=np.float64), np.asarray(impostor, dtype=np.float64)


def far_frr_curve(
    genuine: np.ndarray, impostor: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    far = np.array([(impostor >= t).mean() for t in thresholds])
    frr = np.array([(genuine < t).mean() for t in thresholds])
    return far, frr


MIN_SENSIBLE_THRESHOLD = 0.25  # ArcFace cosine scores below this are not a decision


def pick_threshold(genuine: np.ndarray, impostor: np.ndarray, target_far: float) -> dict:
    """Operating point = the (1 - target_far) quantile of the impostor scores.

    Taking the quantile of the impostor distribution rather than the lowest grid
    point that happens to satisfy the budget matters on small datasets: with few
    impostor pairs, *every* low threshold shows FAR = 0 purely because no
    impostor pair was sampled there, and picking the lowest one would ship an
    absurdly permissive system. The quantile always sits at the top of the
    observed impostor mass instead.

    A FAR of p can only be *measured* with roughly 10/p impostor pairs, so the
    result carries a ``resolvable`` flag; when it is false the reported FAR is an
    upper bound imposed by sample size, not a measurement.
    """
    n_imp = int(impostor.size)
    resolvable = n_imp * target_far >= 10

    threshold = float(np.quantile(impostor, 1.0 - target_far)) if n_imp else 0.0
    threshold = float(np.nextafter(threshold, np.inf))  # strictly above the quantile

    far = float((impostor >= threshold).mean()) if n_imp else 0.0
    frr = float((genuine < threshold).mean()) if genuine.size else 0.0

    return {
        "target_far": target_far,
        "threshold": threshold,
        "far": far,
        "frr": frr,
        "resolvable": resolvable,
        "impostor_pairs_needed_for_target": int(np.ceil(10 / target_far)),
        "sensible": threshold >= MIN_SENSIBLE_THRESHOLD,
    }


def equal_error_rate(
    genuine: np.ndarray, impostor: np.ndarray, thresholds: np.ndarray
) -> dict:
    far, frr = far_frr_curve(genuine, impostor, thresholds)
    idx = int(np.argmin(np.abs(far - frr)))
    return {"eer": float((far[idx] + frr[idx]) / 2), "threshold": float(thresholds[idx])}


def roc_auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Mann-Whitney U statistic — the probability that a random genuine pair
    scores above a random impostor pair. No sklearn dependency needed."""
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    rank_of = np.empty_like(ranks)
    rank_of[order] = ranks
    n_pos, n_neg = len(genuine), len(impostor)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = rank_of[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --- identification (1:N), leave-one-out -------------------------------------


def identification_eval(embeddings: dict[str, np.ndarray], threshold: float) -> dict:
    names = sorted(embeddings)
    usable = [n for n in names if embeddings[n].shape[0] >= 2]
    if len(usable) < 2:
        return {"note": "need >=2 identities with >=2 images each for 1:N evaluation"}

    rank1_hits = 0
    true_accept = 0
    false_accept = 0
    false_reject = 0
    trials = 0

    for probe_name in usable:
        mat = embeddings[probe_name]
        for k in range(mat.shape[0]):
            probe = mat[k]
            gallery_names: list[str] = []
            gallery_vecs: list[np.ndarray] = []
            for name in usable:
                rows = embeddings[name]
                if name == probe_name:
                    rows = np.delete(rows, k, axis=0)
                    if rows.shape[0] == 0:
                        continue
                template = rows.mean(axis=0)
                template = template / np.linalg.norm(template)
                gallery_names.append(name)
                gallery_vecs.append(template)

            if len(gallery_vecs) < 2:
                continue

            sims = np.stack(gallery_vecs) @ probe
            best = int(np.argmax(sims))
            best_sim = float(sims[best])
            predicted = gallery_names[best]

            trials += 1
            if predicted == probe_name:
                rank1_hits += 1
                if best_sim >= threshold:
                    true_accept += 1
                else:
                    false_reject += 1
            else:
                if best_sim >= threshold:
                    false_accept += 1
                else:
                    false_reject += 1

    return {
        "trials": trials,
        "rank1_accuracy": rank1_hits / trials if trials else float("nan"),
        "true_accept_rate": true_accept / trials if trials else float("nan"),
        "false_accept_rate": false_accept / trials if trials else float("nan"),
        "false_reject_rate": false_reject / trials if trials else float("nan"),
        "threshold": threshold,
    }


# --- liveness ----------------------------------------------------------------


def liveness_eval(
    engine: recognition.FaceEngine, live_dir: Path | None, spoof_dir: Path | None
) -> dict | None:
    if live_dir is None and spoof_dir is None:
        return None

    detector = liveness.get_detector()
    if not detector.enabled:
        return {"note": "liveness disabled or model missing"}

    def score_folder(folder: Path) -> list[float]:
        out: list[float] = []
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            face = engine.largest_face(img)
            if face is None:
                continue
            out.append(detector.check(img, face.bbox).live_score)
        return out

    bona_fide = score_folder(live_dir) if live_dir else []
    attacks = score_folder(spoof_dir) if spoof_dir else []
    t = config.LIVENESS_THRESHOLD

    result: dict = {"threshold": t, "n_bona_fide": len(bona_fide), "n_attack": len(attacks)}
    if attacks:
        # APCER: attacks wrongly accepted as bona fide.
        result["apcer"] = float(np.mean(np.asarray(attacks) >= t))
    if bona_fide:
        # BPCER: bona-fide presentations wrongly rejected.
        result["bpcer"] = float(np.mean(np.asarray(bona_fide) < t))
    if "apcer" in result and "bpcer" in result:
        result["acer"] = (result["apcer"] + result["bpcer"]) / 2
        result["auc"] = roc_auc(np.asarray(bona_fide), np.asarray(attacks))
    return result


# --- plots -------------------------------------------------------------------


def make_plots(
    genuine: np.ndarray,
    impostor: np.ndarray,
    thresholds: np.ndarray,
    chosen: dict,
    out_dir: Path,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    far, frr = far_frr_curve(genuine, impostor, thresholds)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(impostor, bins=60, alpha=0.65, label=f"impostor (n={impostor.size})", density=True)
    ax.hist(genuine, bins=60, alpha=0.65, label=f"genuine (n={genuine.size})", density=True)
    ax.axvline(chosen["threshold"], ls="--", color="k", label=f"threshold {chosen['threshold']:.3f}")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("Genuine vs impostor score distributions")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "score_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, far, label="FAR (false accept)")
    ax.plot(thresholds, frr, label="FRR (false reject)")
    ax.axvline(chosen["threshold"], ls="--", color="k", label=f"operating point {chosen['threshold']:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("cosine threshold")
    ax.set_ylabel("rate (log scale)")
    ax.set_title("FAR / FRR trade-off")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "far_frr.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(far, 1 - frr)
    ax.set_xscale("log")
    ax.set_xlabel("FAR (log scale)")
    ax.set_ylabel("TAR = 1 - FRR")
    ax.set_title("ROC")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "roc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    return written


# --- main --------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="dataset root, one folder per identity")
    ap.add_argument("--target-far", type=float, default=0.001, help="FAR budget (default 0.001 = 0.1%%)")
    ap.add_argument("--live-dir", type=Path, default=None, help="bona-fide images for liveness eval")
    ap.add_argument("--spoof-dir", type=Path, default=None, help="attack images for liveness eval")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "reports")
    args = ap.parse_args()

    engine = recognition.get_engine()

    identities = load_dataset(args.data)
    print(f"Loaded {len(identities)} identities, {sum(len(v) for v in identities.values())} images")

    embeddings, skipped = embed_dataset(engine, identities)
    n_images = sum(m.shape[0] for m in embeddings.values())
    print(f"Embedded {n_images} faces across {len(embeddings)} identities ({len(skipped)} skipped)")

    genuine, impostor = verification_scores(embeddings)
    if genuine.size == 0 or impostor.size == 0:
        raise SystemExit("need at least 2 identities and 2 images per identity")

    thresholds = np.linspace(-0.2, 1.0, 1201)
    chosen = pick_threshold(genuine, impostor, args.target_far)
    eer = equal_error_rate(genuine, impostor, thresholds)
    auc = roc_auc(genuine, impostor)
    ident = identification_eval(embeddings, chosen["threshold"])
    live = liveness_eval(engine, args.live_dir, args.spoof_dir)

    report = {
        "dataset": {
            "root": str(args.data),
            "identities": len(embeddings),
            "images_embedded": n_images,
            "images_skipped": len(skipped),
            "genuine_pairs": int(genuine.size),
            "impostor_pairs": int(impostor.size),
        },
        "models": {
            "detector": "SCRFD-10GF (buffalo_l)",
            "embedding": "ArcFace ResNet-50 w600k_r50, 512-D",
            "liveness": "MiniFASNet (CelebA-Spoof), 128x128, 1.5x crop",
        },
        "verification": {
            "auc": auc,
            "eer": eer,
            "genuine_mean": float(genuine.mean()),
            "impostor_mean": float(impostor.mean()),
            "operating_point": chosen,
        },
        "identification_leave_one_out": ident,
        "liveness": live,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    plots = make_plots(genuine, impostor, thresholds, chosen, args.out)

    print("\n" + "=" * 68)
    print("VERIFICATION (1:1)")
    print(f"  ROC AUC                 {auc:.4f}")
    print(f"  EER                     {eer['eer'] * 100:.2f}%  @ threshold {eer['threshold']:.3f}")
    print(f"  Operating point         cosine >= {chosen['threshold']:.3f}")
    print(f"    FAR                   {chosen['far'] * 100:.3f}%  (budget {args.target_far * 100:.3f}%)")
    print(f"    FRR                   {chosen['frr'] * 100:.2f}%")
    if not chosen["resolvable"]:
        print(f"    ! only {impostor.size} impostor pairs — a FAR of {args.target_far * 100:.3f}% needs "
              f"~{chosen['impostor_pairs_needed_for_target']}. Quote a looser FAR, or use more identities.")
    if not chosen["sensible"]:
        print(f"    ! threshold {chosen['threshold']:.3f} is below {MIN_SENSIBLE_THRESHOLD} — the impostor set is "
              "too small or too easy to constrain the decision. Do not deploy this value.")
    if "rank1_accuracy" in ident:
        print("IDENTIFICATION (1:N, leave-one-out)")
        print(f"  Rank-1 accuracy         {ident['rank1_accuracy'] * 100:.2f}%  over {ident['trials']} probes")
        print(f"  TAR / FAR / FRR         {ident['true_accept_rate'] * 100:.2f}% / "
              f"{ident['false_accept_rate'] * 100:.2f}% / {ident['false_reject_rate'] * 100:.2f}%")
    if live and "apcer" in live:
        print("LIVENESS (ISO/IEC 30107-3)")
        print(f"  APCER                   {live['apcer'] * 100:.2f}%  (attacks accepted)")
        print(f"  BPCER                   {live['bpcer'] * 100:.2f}%  (real faces rejected)")
        print(f"  ACER                    {live['acer'] * 100:.2f}%")
    print("=" * 68)
    print(f"\nWrote {args.out / 'metrics.json'}")
    for p in plots:
        print(f"Wrote {p}")
    if chosen["sensible"] and chosen["resolvable"]:
        print(f"\nSet MATCH_THRESHOLD={chosen['threshold']:.3f} (or edit app/config.py) to deploy this operating point.")
    else:
        print("\nNot recommending a threshold from this run — the dataset is too small. "
              "Re-run on more identities (LFW, or 20+ people from your own captures) before deploying.")


if __name__ == "__main__":
    main()
