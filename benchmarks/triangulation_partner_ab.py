"""Reproduce the fixed Phase 6d matrix, without changing runtime defaults.

uv run python -m benchmarks.triangulation_partner_ab --modes shadow --output PATH
Run full pytest BEFORE these jobs: determinism tests regenerate shared frames.
No baseline updates; instrumented timing is diagnostic, not adoption evidence.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import traceback
from unittest.mock import patch

import cv2 as cv
import numpy as np
import scipy
from scipy.spatial import cKDTree

from benchmarks import metrics as metric_mod, run as bench
from benchmarks.keyframe_audit import CONTROL, KeyframeAudit, SURVIVAL_HORIZONS, non_timing_differences
from benchmarks import triangulation_partners as partners
from slam.utils import logger


def manifest(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def geometry_coverage(result, directory):
    if not (directory / "scene_points.txt").exists() or not len(result.map_points):
        return None
    gt = bench.load_trajectory(directory)
    s, R, t = metric_mod.ate(result.poses, gt)["alignment"]
    points = s * (result.map_points @ R.T) + t
    scene, planes = bench.load_scene(directory)
    distance = np.min(np.abs(points @ planes[:, :3].T - planes[:, 3]), axis=1)
    covered = cKDTree(points).query(scene)[0] <= .5
    return {"on_surface_points_at_0_3m": int(sum(distance <= .3)),
            "scene_centers_covered_at_0_5m": int(sum(covered)),
            "scene_center_count": len(scene), "scene_center_coverage_pct": float(np.mean(covered) * 100),
            "warning": "GT-aligned fixed-tolerance proxies, not visibility-normalized completeness or proof of depth."}


def compare_raw(candidate, control_path, directory, first):
    with np.load(control_path) as control:
        common_indices = np.intersect1d(control["keyframe_gt_indices"], candidate.keyframe_gt_indices)
        result = {"common_final_kf_ate": {"count": len(common_indices)},
                  "exact_control_raw_arrays": all(np.array_equal(control[key], getattr(candidate, key))
                                                  for key in control.files)}
        if len(common_indices) >= 3:
            gt = bench.load_trajectory(directory)
            for label, poses, indices in (("control", control["keyframe_poses"], control["keyframe_gt_indices"]),
                                          ("candidate", candidate.keyframe_poses, candidate.keyframe_gt_indices)):
                by_index = dict(zip(indices, poses))
                chosen = np.array([by_index[i] for i in common_indices])
                result["common_final_kf_ate"][label + "_ate"] = metric_mod.ate(chosen, gt[common_indices])["ate_rmse"]
        if first is not None:
            result["exact_pose_prefix_before_first_partner_intervention"] = np.array_equal(
                control["poses"][:first], candidate.poses[:first])
            assert result["exact_pose_prefix_before_first_partner_intervention"]
        old, new = set(control["keyframe_indices"]), set(candidate.keyframe_indices)
        differences = sorted(old ^ new)
        result["first_insertion_schedule_difference"] = int(differences[0]) if differences else None
        result["control_only_insertions"] = len(old - new)
        result["candidate_only_insertions"] = len(new - old)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=partners.MODES, required=True)
    parser.add_argument("--sequences", nargs="+", choices=bench.DEFAULT_SEQUENCES, default=bench.DEFAULT_SEQUENCES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve old evidence: choose a new output path")
    logger.SEVERITY_FILTER = ["error"]
    control = json.loads(CONTROL.read_text())
    previous_audit = json.loads((bench.REPORT_DIR / "phase6a_keyframe_audit.json").read_text())
    baselines = json.loads(bench.BASELINES_FILE.read_text())
    sources = [Path(name) for name in (
        "benchmarks/triangulation_partners.py", "benchmarks/triangulation_partner_ab.py",
        "benchmarks/stages/test_triangulation_partners.py", "benchmarks/keyframe_audit.py",
        "benchmarks/baselines.json", "slam/core/tracker.py", "slam/core/map.py",
        "slam/utils/utils.py", "slam/utils/constants.py", "slam/features/matcher.py",
        "slam/ba/bundle_adjustment_g2o.py")]
    out = {"phase": "6d", "control_report": str(CONTROL), "seed": control["seed"],
           "configs": {"recent_older": partners.RECENT_OLDER, "additional_covisible": partners.COVISIBLE_OLDER,
                       "minimum_shared_for_covisible": partners.MIN_SHARED,
                       "minimum_new_accepted": partners.MIN_NEW_POINTS,
                       "modes": partners.MODES, "ratio": partners.const.LOWE_RATIO,
                       "creation_gates": {k: getattr(partners.const, k) for k in
                                          ("MIN_DEPTH", "MAX_DEPTH", "MIN_PARALLAX_DEG", "MAX_TRI_REPROJECTION_PX")}},
           "sources": {str(p): {**manifest(p), "source": p.read_text()} for p in sources},
           "versions": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "opencv": cv.__version__},
           "warning": "Shadow is fixed-control-map sizing, not future-map simulation. New-pair acceptance is not proof of correspondence/depth. Timing diagnostic only.",
           "runs": []}
    original_run = bench.run_slam
    for mode in args.modes:
        for sequence in args.sequences:
            captured = []

            def capture(directory, *a, **kw):
                result = original_run(directory, *a, **kw)
                captured.append((result, Path(directory)))
                return result

            row = {"mode": mode, "sequence": sequence}
            # The cohort hook must wrap the final triangulation output.
            with partners.TriangulationPartnerExperiment(mode) as experiment:
                with KeyframeAudit() as audit:
                    try:
                        with patch.object(bench, "run_slam", capture):
                            metrics = bench.evaluate_sequence(sequence, control["seed"])
                        assert len(captured) == 1
                        result, directory = captured[0]
                        first = next((r["frame_index"] for r in experiment.rows
                                      if (mode == "best_pair" and r["best_pair_is_older"])
                                      or (mode == "adjacent_plus_one" and r["supplement_available"])), None)
                        filename = bench.REPORT_DIR / f"phase6d_{mode}_{sequence}.npz"
                        if filename.exists():
                            raise FileExistsError(filename)
                        np.savez_compressed(filename, poses=result.poses, map_points=result.map_points,
                                            keyframe_poses=result.keyframe_poses,
                                            keyframe_gt_indices=np.asarray(result.keyframe_gt_indices, dtype=int),
                                            keyframe_indices=np.asarray(result.keyframe_indices, dtype=int))
                        changes = non_timing_differences(metrics, control["results"][sequence])
                        row.update(metrics=metrics, changed_control_metrics=changes,
                                   baseline_failures=bench.check_baselines({sequence: metrics}, baselines).get(sequence, []),
                                   raw_file=manifest(filename), geometry_coverage=geometry_coverage(result, directory),
                                   first_partner_intervention_frame=first,
                                   comparison=compare_raw(result, bench.REPORT_DIR / f"phase6c_shadow_{sequence}.npz",
                                                          directory, first))
                        if mode == "shadow":
                            assert not changes and row["comparison"]["exact_control_raw_arrays"]
                            assert audit.frames == previous_audit["runs"][sequence]["frames"]
                            # Horizon-not-reached sentinels are added below, as in 6a.
                    except Exception:
                        row["error"] = traceback.format_exc()
            for insertion in audit.insertions:
                for horizon in SURVIVAL_HORIZONS:
                    insertion["survival"].setdefault(str(horizon), {"status": "horizon_not_reached"})
            if mode == "shadow" and "error" not in row:
                assert audit.insertions == previous_audit["runs"][sequence]["insertions"]
                row["exact_control_frames_and_cohorts"] = True
            row.update(summary=audit.summary(), frames=audit.frames, insertions=audit.insertions,
                       partner_decisions=experiment.rows,
                       partner_summary={"evaluations": len(experiment.rows),
                                        "best_pair_older": sum(r["best_pair_is_older"] for r in experiment.rows),
                                        "supplement_possible": sum(r["supplement_available"] > 0 for r in experiment.rows),
                                        "supplement_available_points": sum(r["supplement_available"] for r in experiment.rows),
                                        "selected_partner_age_frames": dict(Counter(
                                            str(r["frame_index"] - r["best_pair_frame_index"])
                                            for r in experiment.rows if r["best_pair_is_older"]
                                            and r["best_pair_frame_index"] is not None))})
            out["runs"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
            m = row.get("metrics", {})
            print(json.dumps({"mode": mode, "sequence": sequence, "ate": m.get("ate_rmse"),
                              "kf_ate": m.get("ate_kf_rmse"), "rotation": m.get("rpe_rot_rmse"),
                              "surface": m.get("map_on_surface_pct"), "resets": m.get("resets"),
                              "keyframes": m.get("keyframes"), "insertions": len(audit.insertions),
                              "corrections": m.get("events", {}).get("loop_corrected", 0),
                              "partner_summary": row["partner_summary"],
                              "failures": row.get("baseline_failures"), "error": row.get("error")}), flush=True)
    print("Saved", args.output, flush=True)
    return int(any("error" in row for row in out["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
