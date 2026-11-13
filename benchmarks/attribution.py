"""Health x accuracy ATTRIBUTION (pre-registered before any
ladder data was seen).

The unique insight the GT ladder adds: per-window, the GT error joins
the health stream, so "the system fell apart" becomes "MECHANISM X
fired in window W and cost Y m of ATE there". Conventions, fixed up
front:

- ONE full-run Umeyama alignment (scale + rotation + translation)
  per sequence; windows measure the RESIDUAL error under that single
  gauge (per-window re-alignment would hide exactly the drift the
  attribution is supposed to expose).
- windowed local scale = Umeyama scale fit INSIDE the window (the
  scale-drift trajectory; deviation from the full-run scale = local
  scale error).
- rotation error = mean angle of R_est^T R_gt per window (deg).
- health columns come from the shared schema rows (quality, obs, map
  size, low-q fraction, event counts within the window).
- window length 100 frames (pre-registered; the TUM/EuRoC cadences
  are 20-30 Hz, so a window is 3-5 s of content).

Usage:
    uv run python -m benchmarks.attribution --sequences tum_fr1_desk ...
    (writes output/benchmarks/ladder/<sequence>_attribution.json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.datasets import GT_SEQUENCES, prepare
from benchmarks.harness import load_trajectory, run_slam
from benchmarks.metrics import umeyama_alignment

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "benchmarks" / "ladder"
WINDOW = 100   # frames (pre-registered)


def _rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip(
        (np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def _rotation_alignment(est, gt, n):
    """The ROTATION gauge: orthogonal Procrustes over the per-frame
    orientation pairs — R_rot minimizing sum ||R_rot @ R_est_i -
    R_gt_i||^2. Independent of the center alignment on purpose:
    Umeyama's R comes from CENTER clouds, which are degenerate on
    near-point trajectories (fr1_xyz) and then mis-gauges rotation
    by the arbitrary world-frame offset (measured: ~141 deg of
    phantom error with ATE 0.03)."""
    W = np.zeros((3, 3))
    for i in range(n):
        W += est[i, :3, :3] @ gt[i, :3, :3].T
    U, _, Vt = np.linalg.svd(W)
    R_rot = Vt.T @ U.T
    if np.linalg.det(R_rot) < 0:
        R_rot = Vt.T @ np.diag([1.0, 1.0, -1.0]) @ U.T
    return R_rot


def windowed_attribution(result, gt_poses, window: int = WINDOW):
    """The per-window attribution rows. Returns (full_alignment, rows)."""
    est = result.poses
    n = min(len(est), len(gt_poses))
    s, R, t = umeyama_alignment(est[:n, :3, 3], gt_poses[:n, :3, 3])
    aligned = (s * (R @ est[:n, :3, 3].T)).T + t
    # rotation errors are gauged by the rotation-only alignment (see
    # _rotation_alignment); the Umeyama R gauges the CENTERS only.
    R_rot = _rotation_alignment(est[:n], gt_poses[:n], n)

    rows = []
    for start in range(0, n - window // 2, window):
        stop = min(start + window, n)
        sl = slice(start, stop)
        err = np.linalg.norm(aligned[sl] - gt_poses[sl, :3, 3], axis=1)
        rot_errs = [_rot_angle_deg(
                        (R_rot @ est[i, :3, :3]).T @ gt_poses[i, :3, :3])
                    for i in range(start, stop)]
        # windowed LOCAL scale (the scale-drift trajectory)
        try:
            s_local, _, _ = umeyama_alignment(
                est[sl, :3, 3], gt_poses[sl, :3, 3])
        except Exception:                       # degenerate window
            s_local = None

        hrows = [r for r in result.health_rows
                 if start <= r["i"] < stop]
        q = np.asarray([r["q"] for r in hrows], dtype=float)
        row = {
            "frame_start": start, "frame_stop": stop - 1,
            "ate_rmse": float(np.sqrt(np.mean(err ** 2))),
            "ate_max": float(err.max()),
            "rot_err_deg_mean": float(np.mean(rot_errs)),
            "scale_local": s_local,
            "scale_dev": (float(s_local - s)
                          if s_local is not None else None),
            "q_p50": float(np.median(q)) if len(q) else None,
            "obs_p50": float(np.median(
                [r["obs"] for r in hrows])) if hrows else None,
            "pts_median": float(np.median(
                [r["pts"] for r in hrows])) if hrows else None,
            "lowq_frac": (float(np.mean(
                [r["low_q"] > 0 for r in hrows]))
                if hrows else None),
            "resets": sum(1 for i in
                          getattr(result, "health_reset_indices", [])
                          if start <= i < stop),
            "relocs": sum(r["ev"].get("reloc_success", 0)
                          + r["ev"].get("reloc_aggressive_success", 0)
                          for r in hrows),
            "kf_inserts": sum(1 for r in hrows if r["kf"]),
        }
        rows.append(row)
    return {"full_scale": s}, rows


def attribution_table(rows):
    cols = [("win", 12), ("ate rmse", 8), ("ate max", 8),
            ("rot deg", 8), ("scale loc", 9), ("q p50", 6),
            ("obs p50", 7), ("pts med", 8), ("lowq%", 6),
            ("resets", 6), ("relocs", 6), ("kf", 4)]
    header = " ".join(c.ljust(w) for c, w in cols)
    lines = ["\n" + header, "-" * len(header)]
    for r in rows:
        row = [
            f"{r['frame_start']:5d}-{r['frame_stop']:5d}",
            f"{r['ate_rmse']:.4f}", f"{r['ate_max']:.3f}",
            f"{r['rot_err_deg_mean']:7.2f}",
            f"{r['scale_local']:.4f}" if r["scale_local"] else "-",
            f"{r['q_p50']:.2f}" if r["q_p50"] is not None else "-",
            f"{r['obs_p50']:.0f}" if r["obs_p50"] is not None else "-",
            f"{r['pts_median']:.0f}" if r["pts_median"] is not None else "-",
            f"{100 * r['lowq_frac']:.0f}" if r["lowq_frac"] is not None else "-",
            str(r["resets"]), str(r["relocs"]), str(r["kf_inserts"]),
        ]
        lines.append(" ".join(str(v).ljust(w)
                              for v, (_, w) in zip(row, cols)))
    return "\n".join(lines)


def analyze_sequence(name: str, seed: int = 42, window: int = WINDOW):
    seq_dir = prepare(name)
    gt_poses = load_trajectory(seq_dir)
    result = run_slam(seq_dir, seed=seed)
    _, rows = windowed_attribution(result, gt_poses, window)
    return {
        "sequence": name, "seed": seed, "window": window,
        "frames": result.frames_processed,
        "summary": {
            "resets": result.resets,
            "loop_corrections": result.loop_corrections,
            "keyframes": result.num_keyframes,
            "map_points": len(result.map_points),
        },
        "windows": rows,
    }, result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    names = list(GT_SEQUENCES) if args.all else (args.sequences or [])
    if not names:
        ap.error("need --all or --sequences NAME ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in names:
        out = OUT_DIR / f"{name}_attribution.json"
        if out.exists() and not args.overwrite:
            print(f"[attr] {name}: exists, skipping (resume)", flush=True)
            continue
        print(f"[attr] running {name} ...", flush=True)
        data, _ = analyze_sequence(name, args.seed, args.window)
        if args.overwrite and out.exists():
            out.unlink()
        out.write_text(json.dumps(data, indent=1) + "\n")
        print(f"[attr] {name}: {len(data['windows'])} windows -> {out}")
        print(attribution_table(data["windows"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
