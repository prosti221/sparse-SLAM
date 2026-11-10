"""6e fixed matrix: native/half-resolution/stride2 controls + held-out RPY.

Run tests BEFORE this CLI, never concurrently with canonical frame readers.
uv run python -m benchmarks.keyframe_sensitivity_ab --output OUTPUT --heldout
The default twelve rows are serial; held-out data must be prepared first.
No production changes, threshold retuning, baseline updates or speed claims.
"""
import argparse
import json
from pathlib import Path
import platform
import traceback

import cv2 as cv
import numpy as np
import scipy

from benchmarks import metrics as metric_mod, run as bench
from benchmarks.harness import load_trajectory, run_slam
from benchmarks.keyframe_audit import CONTROL, KeyframeAudit, SURVIVAL_HORIZONS, distribution, non_timing_differences
from benchmarks.keyframe_cost import KeyframeCostAudit
from benchmarks.keyframe_sensitivity import MODES, HELDOUT_NAME, matched_sample_comparison, prepare_view
from benchmarks.triangulation_partner_ab import geometry_coverage, manifest
from slam.utils import logger


def cadence_summary(audit, source_indices, timestamps):
    input_gaps, source_gaps, second_gaps = [], [], []
    previous = None
    for insertion in audit.insertions:
        index = insertion["frame_index"]
        if previous is not None and previous["epoch"] == insertion["epoch"]:
            old = previous["frame_index"]
            input_gaps.append(index - old)
            source_gaps.append(source_indices[index] - source_indices[old])
            if timestamps is not None:
                second_gaps.append(timestamps[index] - timestamps[old])
        previous = insertion
    return {"insertions_including_bootstrap_or_recovery": len(audit.insertions),
            "input_frame_gaps": distribution(input_gaps), "source_frame_gaps": distribution(source_gaps),
            "acquisition_seconds_gaps": distribution(second_gaps),
            "warning": "No gap crosses a map reset; counter age is separately retained in frame audit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--sequences", nargs="+", choices=bench.DEFAULT_SEQUENCES, default=bench.DEFAULT_SEQUENCES)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--data-root", type=Path, default=bench.REPO_ROOT / "data" / "phase6e" / "views")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve old evidence; choose a new output")
    logger.SEVERITY_FILTER = ["error"]
    control = json.loads(CONTROL.read_text())
    original_audit = json.loads((bench.REPORT_DIR / "phase6a_keyframe_audit.json").read_text())
    baselines = json.loads(bench.BASELINES_FILE.read_text())
    sources = [Path(p) for p in (
        "benchmarks/keyframe_cost.py", "benchmarks/keyframe_sensitivity.py",
        "benchmarks/keyframe_sensitivity_ab.py", "benchmarks/stages/test_keyframe_sensitivity.py",
        "benchmarks/keyframe_audit.py", "benchmarks/harness.py", "benchmarks/metrics.py",
        "benchmarks/baselines.json", "slam/core/tracker.py", "slam/core/map.py",
        "slam/utils/utils.py", "slam/utils/constants.py", "slam/features/feature_extractor.py",
        "slam/features/matcher.py", "slam/ba/bundle_adjustment_g2o.py")]
    report = {"phase": "6e", "control_report": str(CONTROL), "seed": control["seed"],
              "matrix": {"modes": list(args.modes), "sequences": args.sequences, "heldout": args.heldout,
                         "ordering": "serial, mode then sequence; held-out last", "retuned_constants": []},
              "sources": {str(p): {**manifest(p), "source": p.read_text()} for p in sources},
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "opencv": cv.__version__},
              "warning": "Characterization only. All timing includes observers; per-object/graph counts are not RSS. Synthetic acquisition seconds unknown. Held-out is not a new curated baseline.",
              "runs": []}
    jobs = [(sequence, mode) for mode in args.modes for sequence in args.sequences]
    if args.heldout:
        jobs.append((HELDOUT_NAME, "native"))
    for sequence, mode in jobs:
        row = {"sequence": sequence, "mode": mode}
        audit = KeyframeAudit()
        costs = None
        try:
            if sequence == HELDOUT_NAME:
                source = bench.REPO_ROOT / "data" / "phase6e" / "heldout" / HELDOUT_NAME
                if not (source / "source_manifest.json").exists():
                    raise FileNotFoundError("Prepare the predeclared held-out dataset first")
            else:
                source = bench.generate_sequence(sequence) if sequence in bench.SEQUENCES else bench.prepare_tum_sequence()
            directory = args.data_root / f"{sequence}_{mode}"
            inputs = prepare_view(source, directory, mode)
            row["input_manifest"] = inputs
            row["input_manifest_file"] = manifest(directory / "source_manifest.json")
            gt = load_trajectory(directory)
            scene, planes = bench.load_scene(directory) if (directory / "scene_points.txt").exists() else (None, None)
            costs = KeyframeCostAudit(inputs["source_indices"], inputs["timestamps_seconds"])
            with costs:
                with audit:
                    result = run_slam(directory, seed=control["seed"])
            metrics = metric_mod.evaluate(result, gt, scene, planes)
            metrics["final_intrinsics"] = result.final_intrinsics
            filename = bench.REPORT_DIR / f"phase6e_{sequence}_{mode}.npz"
            if filename.exists():
                raise FileExistsError(filename)
            np.savez_compressed(filename, poses=result.poses, map_points=result.map_points,
                                keyframe_poses=result.keyframe_poses,
                                keyframe_gt_indices=np.asarray(result.keyframe_gt_indices, dtype=int),
                                keyframe_indices=np.asarray(result.keyframe_indices, dtype=int),
                                source_indices=np.asarray(inputs["source_indices"], dtype=int))
            assert np.isfinite(result.poses).all() and np.isfinite(result.map_points).all()
            row.update(metrics=metrics, telemetry=result.telemetry, raw_file=manifest(filename),
                       costs=costs.summary(result.telemetry), geometry_coverage=geometry_coverage(result, directory),
                       cadence=cadence_summary(audit, inputs["source_indices"], inputs["timestamps_seconds"]),
                       culls=metrics["events"].get("kf_culled", 0),
                       reset_events=metrics["events"].get("system_reset", 0),
                       diagnostic_curated_gate_failures=bench.check_baselines({sequence: metrics}, baselines).get(sequence, []))
            if sequence != HELDOUT_NAME:
                row["changed_control_metrics"] = non_timing_differences(metrics, control["results"][sequence])
                with np.load(bench.REPORT_DIR / f"phase6c_shadow_{sequence}.npz") as native:
                    row["matched_source_sample_comparison"] = matched_sample_comparison(
                        result, native, inputs["source_indices"], load_trajectory(source), inputs["timestamps_seconds"])
                    if mode == "native":
                        row["exact_control_raw_arrays"] = all(np.array_equal(native[k], getattr(result, k)) for k in native.files)
                        assert row["exact_control_raw_arrays"] and not row["changed_control_metrics"]
                        assert audit.frames == original_audit["runs"][sequence]["frames"]
            else:
                row["heldout_provenance"] = json.loads((source / "source_manifest.json").read_text())
                row["heldout_has_curated_baseline"] = False
            row["selected_support_grid_cells"] = distribution(
                f["decision"]["signals"]["support_grid_at_3px"]["occupied_cells"] for f in audit.frames
                if f["decision"] is not None and f["decision"]["effective_selected"])
        except Exception:
            row["error"] = traceback.format_exc()
        for insertion in audit.insertions:
            for horizon in SURVIVAL_HORIZONS:
                insertion["survival"].setdefault(str(horizon), {"status": "horizon_not_reached"})
        if mode == "native" and sequence != HELDOUT_NAME and "error" not in row:
            assert audit.insertions == original_audit["runs"][sequence]["insertions"]
            row["exact_control_frames_and_cohorts"] = True
        row.update(audit_summary=audit.summary(), frames=audit.frames, insertions=audit.insertions,
                   map_samples=costs.map_samples if costs is not None else [],
                   graph_samples=costs.graphs if costs is not None else [],
                   cost_frames=costs.frames if costs is not None else [],
                   loop_windows=costs.loop_windows if costs is not None else [])
        report["runs"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        m = row.get("metrics", {})
        print(json.dumps({"sequence": sequence, "mode": mode, "ate": m.get("ate_rmse"),
                          "kf_ate": m.get("ate_kf_rmse"), "rotation": m.get("rpe_rot_rmse"),
                          "surface": m.get("map_on_surface_pct"), "insertions": len(audit.insertions),
                          "culls": row.get("culls"), "final_kfs": m.get("keyframes"),
                          "resets": row.get("reset_events"),
                          "map_peaks": row.get("costs", {}).get("map_peaks"),
                          "gates": row.get("diagnostic_curated_gate_failures"), "error": row.get("error")}), flush=True)
    print("Saved", args.output, flush=True)
    return int(any("error" in row for row in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
