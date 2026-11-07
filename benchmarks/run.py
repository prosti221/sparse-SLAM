"""Benchmark CLI: run sequences headlessly, evaluate against ground truth,
print a report, and check results against checked-in baselines.

Usage:
    uv run python -m benchmarks.run                       # all sequences
    uv run python -m benchmarks.run --sequences translation orbit
    uv run python -m benchmarks.run --regen              # regenerate data
    uv run python -m benchmarks.run --update-baselines    # re-record thresholds
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.synthetic.generator import SEQUENCES, generate_sequence
from benchmarks.tum import prepare_tum_sequence
from benchmarks.harness import run_slam, load_trajectory, load_scene
from benchmarks import metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_FILE = Path(__file__).resolve().parent / "baselines.json"
REPORT_DIR = REPO_ROOT / "output" / "benchmarks"

# 'tum_xyz' is a real-world sequence (auto-downloaded on first use);
# the rest are generated synthetic worlds.
TUM_SEQUENCES = {"tum_xyz"}
DEFAULT_SEQUENCES = ["forward_rotation", "orbit", "translation", "tum_xyz"]

# Thresholds applied to each metric. Missing sequence entry = no baseline yet.
THRESHOLD_RULES = {
    "max": ["ate_rmse", "rpe_trans_rmse", "rpe_rot_rmse", "resets",
            "max_pose_jump"],
    "min": ["map_on_surface_pct", "map_near_scene_pct"],
}


def evaluate_sequence(name: str, seed: int) -> dict:
    if name in SEQUENCES:
        seq_dir = generate_sequence(name)
    elif name in TUM_SEQUENCES:
        seq_dir = prepare_tum_sequence()
    else:
        raise ValueError(f"Unknown sequence '{name}'. "
                         f"Available: {sorted(SEQUENCES) + sorted(TUM_SEQUENCES)}")

    gt_poses = load_trajectory(seq_dir)
    # Synthetic sequences ship analytic scene geometry; real-world ones don't.
    scene_points, planes = None, None
    if (seq_dir / "scene_points.txt").exists():
        scene_points, planes = load_scene(seq_dir)

    result = run_slam(seq_dir, seed=seed)
    return metrics.evaluate(result, gt_poses, scene_points, planes)


def check_baselines(results: dict, baselines: dict) -> dict:
    """Compare metrics against baselines. Returns {sequence: [failures]}."""
    failures = {}
    for seq, m in results.items():
        seq_failures = []
        thresholds = baselines.get("thresholds", {}).get(seq)
        if thresholds is None:
            continue
        for direction, keys in THRESHOLD_RULES.items():
            for key in keys:
                if key not in thresholds or key not in m:
                    continue
                limit = thresholds[key]
                value = m[key]
                if value is None:
                    continue  # metric not available for this dataset
                bad = (value > limit) if direction == "max" else (value < limit)
                if bad:
                    seq_failures.append(
                        f"{key}={value:.4g} {'>' if direction == 'max' else '<'} baseline {limit:.4g}")
        if seq_failures:
            failures[seq] = seq_failures
    return failures


def _cell(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def print_table(results: dict) -> None:
    columns = [
        ("sequence", 18), ("keyframes", 10), ("map_pts", 8),
        ("ATE rmse", 9), ("s", 7), ("scale dev", 10),
        ("RPE t", 8), ("RPE r deg", 10),
        ("surf %", 7), ("near %", 7), ("resets", 7), ("fps", 6),
    ]
    header = " ".join(name.ljust(width) for name, width in columns)
    print("\n" + header)
    print("-" * len(header))
    for seq, m in results.items():
        row = [
            seq, str(m["keyframes"]), str(m["map_points"]),
            _cell(m["ate_rmse"], ".4f"), _cell(m["scale"], ".4f"),
            _cell(m["scale_consistency"], ".3f"),
            _cell(m["rpe_trans_rmse"], ".4f"), _cell(m["rpe_rot_rmse"], ".2f"),
            _cell(m["map_on_surface_pct"], ".1f"), _cell(m["map_near_scene_pct"], ".1f"),
            str(m["resets"]), _cell(m["fps"], ".1f"),
        ]
        print(" ".join(str(v).ljust(w) for v, (_, w) in zip(row, columns)))


# Non-overlapping pipeline stages (see benchmarks/IMPROVEMENT.md phase 0);
# "other" = measured wall time not covered by any stage.
PROFILE_STAGES = ["extract", "match", "project", "proj_match", "pnp",
                   "kf_match", "map_optimize", "triangulate", "relocate",
                   "other"]


def print_profile_table(results: dict) -> None:
    """Where does the time go? Milliseconds per frame, per stage."""
    columns = [("sequence", 18)] + [(s, 11) for s in PROFILE_STAGES] + [("fps", 6)]
    header = " ".join(name.ljust(width) for name, width in columns)
    print("\n-- profile (ms/frame) --")
    print(header)
    print("-" * len(header))
    for seq, m in results.items():
        timing = m.get("timing_ms_per_frame", {})
        row = [seq] + [_cell(timing.get(s), ".1f") for s in PROFILE_STAGES] \
            + [_cell(m["fps"], ".1f")]
        print(" ".join(str(v).ljust(w) for v, (_, w) in zip(row, columns)))


def print_diagnostics_table(results: dict) -> None:
    """What happened along the way? Event counts + collapse tripwire."""
    event_keys = {
        "match_fb": "match_fallback",
        "pnp_ok": "pnp_frame_ok",
        "pnp_fail": "pnp_frame_fail",
        "ba": "ba_local_call",
        "ba_fail": "ba_failure",
        "tri_ok": "tri_accepted",
        "tri_rej": "tri_rejected",
    }
    columns = [("sequence", 18), ("match_fb", 9), ("pnp_ok", 7),
               ("pnp_fail", 8), ("ba", 5), ("ba_fail", 7),
               ("tri_ok", 8), ("tri_rej", 8), ("reloc", 6),
               ("resets", 6), ("max_jump", 9)]
    header = " ".join(name.ljust(width) for name, width in columns)
    print("\n-- diagnostics (counts; ba counts local+global calls) --")
    print(header)
    print("-" * len(header))
    for seq, m in results.items():
        ev = m.get("events", {})
        row = [seq]
        for label, _ in columns[1:]:
            if label == "ba":
                row.append(str(ev.get("ba_local_call", 0) + ev.get("ba_global_call", 0)))
            elif label == "reloc":
                row.append(str(ev.get("reloc_success", 0)
                               + ev.get("reloc_aggressive_success", 0)))
            elif label == "resets":
                row.append(str(m["resets"]))
            elif label == "max_jump":
                row.append(_cell(m.get("max_pose_jump"), ".2f"))
            else:
                row.append(str(ev.get(event_keys[label], 0)))
        print(" ".join(str(v).ljust(w) for v, (_, w) in zip(row, columns)))


def apply_overrides(pairs) -> None:
    """Temporarily override slam.utils.constants values for this run.

    Consumers of the overridable family read the constants through the
    module (see should_insert_keyframe / tracker / map), so threshold
    sweeps reach the pipeline live.
    """
    import slam.utils.constants as const

    for pair in pairs:
        name, sep, value = pair.partition("=")
        name = name.strip()
        if not sep or not hasattr(const, name):
            raise SystemExit(
                f"[bench] --override expects NAME=VALUE of a slam.utils.constants "
                f"constant; got {pair!r}")
        current = getattr(const, name)
        try:
            parsed = type(current)(value.strip())
        except ValueError:
            raise SystemExit(
                f"[bench] cannot parse {value!r} as {type(current).__name__} for {name}")
        setattr(const, name, parsed)
        print(f"[bench] override: {name} = {parsed} (default {current})", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES,
                        help=f"one of {DEFAULT_SEQUENCES}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regen", action="store_true",
                        help="force regeneration of synthetic sequences")
    parser.add_argument("--update-baselines", action="store_true",
                        help="write this run's metrics into baselines.json")
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip baseline comparison")
    parser.add_argument("--override", action="append", default=[],
                        metavar="NAME=VALUE",
                        help="override a slam.utils.constants value for this run "
                             "(repeatable), e.g. --override KEYFRAME_MINIMUM_ROTATION_DEG=1.5")
    args = parser.parse_args()

    apply_overrides(args.override)

    if args.regen:
        for name in args.sequences:
            generate_sequence(name, force=True)

    results = {}
    for name in args.sequences:
        print(f"[bench] running '{name}' ...", flush=True)
        t0 = time.perf_counter()
        results[name] = evaluate_sequence(name, args.seed)
        print(f"[bench] '{name}' done in {time.perf_counter() - t0:.1f}s", flush=True)

    print_table(results)
    print_profile_table(results)
    print_diagnostics_table(results)

    # Persist a machine-readable report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = REPORT_DIR / f"report_{stamp}.json"
    report = {"seed": args.seed, "results": results}
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[bench] report written to {report_file}")

    if args.update_baselines:
        with open(BASELINES_FILE, "w") as f:
            json.dump({"seed": args.seed, "thresholds": {
                seq: {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                      for k, v in m.items()}
                for seq, m in results.items()}}, f, indent=2)
        print(f"[bench] baselines updated: {BASELINES_FILE}")

    if args.no_baseline:
        return 0

    baselines = {}
    if BASELINES_FILE.exists():
        with open(BASELINES_FILE) as f:
            baselines = json.load(f)
    else:
        print("[bench] no baselines.json yet - skipping baseline check")
        return 0

    failures = check_baselines(results, baselines)
    if failures:
        print("\n[bench] BASELINE FAILURES:")
        for seq, fails in failures.items():
            for msg in fails:
                print(f"  {seq}: {msg}")
        return 1
    print("[bench] all sequences within baselines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
