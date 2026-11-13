"""Benchmark-only, observation-only keyframe decision audit.

Run: uv run python -m benchmarks.keyframe_audit

Hooks call the actual production functions exactly once and return their
original results. No runtime selector, thresholds, registry or quality
semantics change. The context patches process-global bindings, like other
benchmark probes: use it only for one sequential, single-threaded run.
Diagnostic residual cuts are NOT calibrated inlier probabilities or proof
of correct depth/matching. Instrumented FPS is not production timing.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
from unittest.mock import patch

import cv2 as cv
import numpy as np
import scipy

from benchmarks import run as bench
from slam.ba.bundle_adjustment_g2o import G2OBundleAdjustment
from slam.core.map import Map
import slam.core.tracker as tracker_mod
from slam.core.tracker import Tracker
from slam.utils import constants as const
from slam.utils import telemetry


CONTROL = bench.REPORT_DIR / "report_20261109_201653.json"
SUPPORT_CUTS_PX = (1.0, 3.0, 5.0)
REFERENCE_CUT_PX = 3.0
GRID_SHAPE = (4, 4)
SURVIVAL_HORIZONS = (2, 5)
NON_TIMING_EXCLUSIONS = {"fps", "timing_ms_per_frame"}
_active = False


def distribution(values):
    """JSON-safe summaries; distinguish missing/invalid values from zero."""
    a = np.asarray([v for v in values if v is not None], dtype=float)
    finite = a[np.isfinite(a)]
    result = {"finite_count": int(len(finite)),
              "nonfinite_count": int(len(a) - len(finite))}
    for name, percentile in (("min", 0), ("p10", 10), ("median", 50),
                             ("p90", 90), ("max", 100)):
        result[name] = float(np.percentile(finite, percentile)) if len(finite) else None
    return result


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _inside(frame, uv):
    # Same margin as the tracker projection search. Behind-camera rows
    # are NaN from Frame.project_points, and must not count as support.
    return (np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 10) & (uv[:, 0] < frame.W - 10)
            & (uv[:, 1] >= 10) & (uv[:, 1] < frame.H - 10))


def _authoritative_observations(frame, world):
    """Exclude dead identities, frame-only dedup and ALL pixel collisions.

    Reads keypoint arrays directly: accessing Observation cached_property
    fields would populate caches and violate the observer contract.
    """
    observations = []
    excluded = Counter()
    n_keypoints = len(frame.keypoints) if frame.keypoints is not None else 0
    for point_id, obs in frame.observed_points.items():
        if (point_id != obs.point.point_id
                or world.points_by_id.get(point_id) is not obs.point):
            excluded["dead_or_wrong_point_identity"] += 1
        elif (obs.frame is not frame
              or obs.point.observations.get(frame.frame_id) is not obs):
            excluded["frame_only_or_wrong_observation_identity"] += 1
        elif not 0 <= obs.kp_idx < n_keypoints:
            excluded["invalid_keypoint_index"] += 1
        else:
            observations.append(obs)
    assignments = Counter(obs.kp_idx for obs in observations)
    distinct = [obs for obs in observations if assignments[obs.kp_idx] == 1]
    excluded["colliding_keypoint_associations"] = len(observations) - len(distinct)
    return distinct, dict(excluded)


def _live_keyframe_count(point, world):
    return sum(obs.frame is world.keyframes_by_id.get(fid)
               for fid, obs in point.observations.items())


def grid_coverage(frame, observations):
    rows, cols = GRID_SHAPE
    grid = np.zeros((rows, cols), dtype=int)
    for obs in observations:
        u, v = frame.keypoints[obs.kp_idx].pt
        if np.isfinite([u, v]).all() and 0 <= u < frame.W and 0 <= v < frame.H:
            grid[min(int(v * rows / frame.H), rows - 1),
                 min(int(u * cols / frame.W), cols - 1)] += 1
    return {"counts": grid.tolist(), "occupied_cells": int(np.count_nonzero(grid)),
            "occupied_fraction": float(np.count_nonzero(grid) / grid.size)}


def bearing_parallax(frame, reference, pairs):
    """Angles between measured, undistorted bearings in a COMMON world frame.

    With exact poses pure rotation gives zero, despite raw image motion.
    Estimated rotations and correspondence errors remain assumptions.
    """
    if not pairs:
        return distribution([])
    current = np.array([frame.kp_pts_norm[a.kp_idx] for a, _ in pairs])
    previous = np.array([reference.kp_pts_norm[b.kp_idx] for _, b in pairs])
    rays = []
    for points, R in ((current, frame.pose[:3, :3]),
                      (previous, reference.pose[:3, :3])):
        directions = np.column_stack([points, np.ones(len(points))]) @ R.T
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        rays.append(directions)
    angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(*rays), axis=1),
                                  np.sum(rays[0] * rays[1], axis=1)))
    return distribution(angles)


def _reference_signals(frame, reference, world, support, maturity):
    observations, excluded = _authoritative_observations(reference, world)
    anchors = {obs.point_id: obs for obs in observations
               if maturity.setdefault(obs.point_id, _live_keyframe_count(obs.point, world)) >= 2}
    common = [(obs, anchors[obs.point_id]) for obs in support if obs.point_id in anchors]
    expected = set()
    median_depth = None
    if anchors:
        anchor_list = list(anchors.values())
        X = np.array([obs.point.pt_3d for obs in anchor_list])
        visible = _inside(frame, frame.project_points(X, normalized=False))
        expected = {obs.point_id for obs, valid in zip(anchor_list, visible) if valid}
        camera = (X - reference.pose[:3, 3]) @ reference.pose[:3, :3]
        depths = camera[:, 2]
        positive = depths[np.isfinite(depths) & (depths > 0)]
        if len(positive):
            median_depth = float(np.median(positive))
    baseline = float(np.linalg.norm(frame.pose[:3, 3] - reference.pose[:3, 3]))
    shared_expected = sum(a.point_id in expected for a, _ in common)
    pixel_flow = []
    for current, previous in common:
        uv1 = np.asarray(frame.keypoints[current.kp_idx].pt) / [frame.W, frame.H]
        uv2 = np.asarray(reference.keypoints[previous.kp_idx].pt) / [reference.W, reference.H]
        pixel_flow.append(float(np.linalg.norm(uv1 - uv2)))
    return {"mature_anchors": len(anchors), "shared_consistent": len(common),
            "overlap_fraction": _ratio(len(common), len(anchors)),
            "expected_visible_anchors": len(expected),
            "visible_match_fraction": _ratio(shared_expected, len(expected)),
            "median_reference_depth": median_depth,
            "baseline_over_depth": _ratio(baseline, median_depth),
            "parallax_deg": bearing_parallax(frame, reference, common),
            "image_fraction_flow": distribution(pixel_flow),
            "excluded_associations": excluded}


def capture_signals(frame, reference, world):
    """Snapshot AFTER the real policy consumes quality, BEFORE insertion.

    Does not read Frame.tracking_quality (a mutating lazy property).
    Missing match data is represented explicitly, even though the 6b
    helper supplies zero numeric evidence to the gate. This snapshot
    does not repair inputs or alter decisions.
    """
    n_features = len(frame.keypoints) if frame.keypoints is not None else 0
    matches = frame.matches if frame.matches is not None else []
    unique_mask = frame.kp_unique_mask if frame.kp_unique_mask is not None else []
    n_unmapped = int(sum(unique_mask))
    matched_indices = {m.queryIdx for m in matches if 0 <= m.queryIdx < len(unique_mask)}
    matched_unmapped = sum(bool(unique_mask[i]) for i in matched_indices)
    observations, excluded = _authoritative_observations(frame, world)
    errors = np.empty(0)
    inside = np.zeros(0, dtype=bool)
    if observations:
        X = np.array([obs.point.pt_3d for obs in observations])
        predicted = frame.project_points(X, normalized=False)
        measured = np.array([frame.keypoints[obs.kp_idx].pt for obs in observations])
        errors = np.linalg.norm(predicted - measured, axis=1)
        inside = _inside(frame, predicted)
    support = {str(cut): [obs for obs, valid in zip(observations, inside & (errors <= cut))
                         if valid] for cut in SUPPORT_CUTS_PX}
    consistent = support[str(REFERENCE_CUT_PX)]
    maturity = {obs.point_id: _live_keyframe_count(obs.point, world) for obs in consistent}
    local_refs = world.keyframes[-max(const.LOCAL_MAP_WINDOW_SIZE, const.RELOCALIZATION_WINDOW):]
    # Latest wins ties. Score by shared point-side anchors; the detailed
    # reference report additionally excludes that reference's collisions.
    best = reference
    best_count = -1
    for candidate in reversed(local_refs):
        count = sum(maturity[obs.point_id] >= 2
                    and (candidate_obs := obs.point.observations.get(candidate.frame_id)) is not None
                    and candidate_obs.frame is candidate for obs in consistent)
        if count > best_count:
            best, best_count = candidate, count
    last = _reference_signals(frame, reference, world, consistent, maturity)
    best_signals = (dict(last) if best is reference else
                    _reference_signals(frame, best, world, consistent, maturity))
    best_signals["same_as_last"] = best is reference
    best_signals["insertion_order_index"] = next(
        (i for i, kf in enumerate(world.keyframes) if kf is best), None)
    relative_R = frame.pose[:3, :3].T @ reference.pose[:3, :3]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(relative_R) - 1) / 2, -1, 1))))
    image = frame.image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    sharpness = float(cv.Laplacian(gray, cv.CV_64F).var())
    return {"features": n_features, "previous_frame_matches": len(matches),
            "match_data_missing": frame.matches is None,
            "unmapped_features": n_unmapped,
            "legacy_new_points_ratio": _ratio(n_unmapped, len(matches)),
            "unmapped_feature_fraction": _ratio(n_unmapped, n_features),
            "matched_unmapped_fraction": _ratio(matched_unmapped, len(matched_indices)),
            "legacy_quality": float(frame._tracking_quality),
            "baseline": float(np.linalg.norm(frame.pose[:3, 3] - reference.pose[:3, 3])),
            "rotation_deg": angle, "laplacian_variance": sharpness,
            "authoritative_distinct_associations": len(observations),
            "excluded_associations": excluded,
            "fresh_pixel_residual": distribution(errors),
            "invalid_or_outside_predictions": int(sum(~inside)),
            "consistent_support": {cut: len(obs) for cut, obs in support.items()},
            "support_grid_at_3px": grid_coverage(frame, consistent),
            "last_reference": last, "best_local_reference": best_signals}


def policy_attribution(signals, counter_gap, policy_selected):
    """Mirror reasons, not decisions: the real policy's result is authoritative."""
    flags = {"translation": signals["baseline"] >= const.MINIMUM_BASELINE_THRESHOLD,
             "rotation": (signals["rotation_deg"] >= const.KEYFRAME_MINIMUM_ROTATION_DEG
                          and counter_gap >= const.KEYFRAME_ROTATION_MIN_FRAME_GAP),
             # Keep descriptive missingness separate from the helper's
             # zero-evidence sentinel, including under threshold overrides.
             "new_ratio": ((signals["legacy_new_points_ratio"]
                            if signals["legacy_new_points_ratio"] is not None else 0.0)
                           >= const.MINIMUM_NEW_POINTS_RATIO),
             "quality": signals["legacy_quality"] >= const.MINIMUM_KEYFRAME_QUALITY_THRESHOLD,
             "maximum_gap": counter_gap >= const.MAX_NUMBER_OF_FRAMES_BETWEEN_KEYFRAMES}
    if policy_selected:
        primary = "translation" if flags["translation"] else "rotation"
    else:
        primary = "maximum_gap_only" if flags["maximum_gap"] else "not_selected"
    return {"policy_selected": bool(policy_selected),
            "effective_selected": bool(policy_selected or flags["maximum_gap"]),
            "primary_reason": primary, "flags": flags}


class KeyframeAudit:
    """Temporary observers; restore every patched binding even on failure."""

    def __init__(self):
        self.frames = []
        self.insertions = []
        self._pending = {}
        self._kf_frames = {}
        self._current = None
        self._tracker = None
        self._epoch = 0
        self.collection_seconds = 0.0

    def __enter__(self):
        global _active
        if _active:
            raise RuntimeError("KeyframeAudit contexts cannot overlap")
        _active = True
        self._stack = ExitStack()
        original_update = Tracker.update
        original_policy = tracker_mod.should_insert_keyframe
        original_pnp = G2OBundleAdjustment.refine_pose_pnp
        original_add = Map.add_keyframe
        original_tri = Tracker._triangulate
        original_reset = Map.reset

        def update(tracker, frame):
            self._tracker = tracker
            self._current = {"frame_index": len(self.frames), "epoch": self._epoch,
                             "decision": None, "pnp_calls": []}
            recorder = telemetry._recorder
            before = dict(recorder.events) if recorder is not None else {}
            try:
                result = original_update(tracker, frame)
                start = perf_counter()
                self._finish_frame(tracker.map)
                after = recorder.events if recorder is not None else {}
                self._current["events"] = {key: value - before.get(key, 0)
                                           for key, value in after.items()
                                           if value != before.get(key, 0)}
                self.frames.append(self._current)
                self.collection_seconds += perf_counter() - start
                return result
            finally:
                self._current = None
                self._tracker = None

        def policy(frame, reference, frames_since_last_keyframe=0, **kwargs):
            # the tracker passes the run's video_fps through
            # here (time-normalized cadence constants); the observer is
            # a pure passthrough - it never reads or alters it.
            result = original_policy(frame, reference, frames_since_last_keyframe,
                                     **kwargs)
            if self._current is not None:
                start = perf_counter()
                signals = capture_signals(frame, reference, self._tracker.map)
                reference_index = self._kf_frames.get(reference.frame_id)
                self._current["decision"] = {
                    **policy_attribution(signals, frames_since_last_keyframe, result[0]),
                    "counter_gap": frames_since_last_keyframe,
                    "actual_frame_gap": (self._current["frame_index"] - reference_index
                                         if reference_index is not None else None),
                    "signals": signals}
                self.collection_seconds += perf_counter() - start
            return result

        def pnp(optimizer, frame=None):
            target = frame if frame is not None else optimizer.map.cur_keyframe
            result = original_pnp(optimizer, frame)
            if self._current is not None:
                self._current["pnp_calls"].append({
                    "explicit_current_frame": frame is not None and frame is self._tracker.cur_frame,
                    "target_is_current_frame": target is self._tracker.cur_frame,
                    "success": bool(result[0]), "reported_input_count": int(result[1])})
            return result

        def add(world, frame):
            existed = frame.frame_id in world.keyframes_by_id
            result = original_add(world, frame)
            if self._current is not None and not existed and frame.frame_id in world.keyframes_by_id:
                index = self._current["frame_index"]
                self._kf_frames[frame.frame_id] = index
                decision = self._current["decision"]
                self.insertions.append({
                    "serial": len(self.insertions), "frame_index": index, "epoch": self._epoch,
                    "reason": (decision["primary_reason"] if decision else "bootstrap_or_recovery"),
                    "triangulated_candidates_accepted": 0,
                    "retained_new_points_at_frame_end": 0, "survival": {}})
            return result

        def triangulate(tracker):
            result = original_tri(tracker)
            if self._current is not None and self.insertions:
                insertion = self.insertions[-1]
                insertion["triangulated_candidates_accepted"] += len(result)
                self._pending[insertion["serial"]] = {
                    "ids": {point.point_id for point in result}, "initialized": False,
                    "epoch": self._epoch}
            return result

        def reset(world):
            result = original_reset(world)
            for serial in self._pending:
                for horizon in SURVIVAL_HORIZONS:
                    self.insertions[serial]["survival"].setdefault(str(horizon), {"status": "map_reset"})
            self._pending.clear()
            self._epoch += 1
            return result

        for owner, name, wrapper in ((Tracker, "update", update),
                                     (tracker_mod, "should_insert_keyframe", policy),
                                     (G2OBundleAdjustment, "refine_pose_pnp", pnp),
                                     (Map, "add_keyframe", add),
                                     (Tracker, "_triangulate", triangulate),
                                     (Map, "reset", reset)):
            self._stack.enter_context(patch.object(owner, name, wrapper))
        return self

    def _finish_frame(self, world):
        current_serial = len(self.insertions) - 1
        for serial, cohort in list(self._pending.items()):
            insertion = self.insertions[serial]
            if not cohort["initialized"]:
                cohort["ids"].intersection_update(world.points_by_id)
                insertion["retained_new_points_at_frame_end"] = len(cohort["ids"])
                cohort["initialized"] = True
            for horizon in SURVIVAL_HORIZONS:
                if current_serial - serial < horizon or str(horizon) in insertion["survival"]:
                    continue
                live = [world.points_by_id[pid] for pid in cohort["ids"] if pid in world.points_by_id]
                insertion["survival"][str(horizon)] = {
                    "status": "observed", "remaining": len(live),
                    "fraction": _ratio(len(live), insertion["retained_new_points_at_frame_end"]),
                    "with_at_least_3_live_keyframes": sum(_live_keyframe_count(p, world) >= 3 for p in live)}
            if all(str(h) in insertion["survival"] for h in SURVIVAL_HORIZONS):
                del self._pending[serial]

    def __exit__(self, *exc):
        global _active
        try:
            return self._stack.__exit__(*exc)
        finally:
            _active = False

    def summary(self):
        decisions = [frame["decision"] for frame in self.frames if frame["decision"] is not None]
        calls = [call for frame in self.frames for call in frame["pnp_calls"]
                 if call["explicit_current_frame"]]
        groups = {}
        for selected in (False, True):
            rows = [d for d in decisions if d["effective_selected"] == selected]
            groups["selected" if selected else "not_selected"] = {
                "frames": len(rows),
                **{key: distribution(d["signals"][key] for d in rows)
                   for key in ("legacy_new_points_ratio", "legacy_quality", "baseline", "rotation_deg")},
                "consistent_support_at_3px": distribution(d["signals"]["consistent_support"]["3.0"] for d in rows),
                "last_reference_parallax_deg": distribution(
                    d["signals"]["last_reference"]["parallax_deg"]["median"] for d in rows),
                "last_reference_baseline_over_depth": distribution(
                    d["signals"]["last_reference"]["baseline_over_depth"] for d in rows)}
        return {"frames": len(self.frames), "policy_evaluations": len(decisions),
                "insertions_including_bootstrap": len(self.insertions),
                "primary_reasons": dict(Counter(d["primary_reason"] for d in decisions)),
                "overlapping_flags": dict(Counter(key for d in decisions for key, value in d["flags"].items() if value)),
                "legacy_ratio_above_one": sum((d["signals"]["legacy_new_points_ratio"] or 0) > 1 for d in decisions),
                "selected_with_zero_consistent_support_at_3px": sum(
                    d["effective_selected"] and d["signals"]["consistent_support"]["3.0"] == 0 for d in decisions),
                "actual_per_frame_pnp": {"calls": len(calls), "success": sum(c["success"] for c in calls),
                                         "failure": sum(not c["success"] for c in calls)},
                "counter_minus_actual_gap": distribution(
                    d["counter_gap"] - d["actual_frame_gap"] for d in decisions if d["actual_frame_gap"] is not None),
                "groups": groups, "diagnostic_collection_seconds": self.collection_seconds}


def non_timing_differences(actual, control):
    return [key for key in sorted(actual.keys() | control.keys())
            if key not in NON_TIMING_EXCLUSIONS and actual.get(key) != control.get(key)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", nargs="+", choices=bench.DEFAULT_SEQUENCES,
                        default=bench.DEFAULT_SEQUENCES)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--output", type=Path, default=bench.REPORT_DIR / "phase6a_keyframe_audit.json")
    args = parser.parse_args()
    control = json.loads(args.control.read_text())
    baselines = json.loads(bench.BASELINES_FILE.read_text())
    source = Path(__file__)
    report = {"phase": "6a", "control_report": str(args.control), "seed": control["seed"],
              "diagnostic_assumptions": {
                  "support_cuts_px": SUPPORT_CUTS_PX, "reference_cut_px": REFERENCE_CUT_PX,
                  "grid_shape": GRID_SHAPE, "minimum_live_kf_anchors": 2,
                  "survival_horizons_insertions": SURVIVAL_HORIZONS,
                  "warning": "Fixed-map pixel consistency is not match correctness or calibrated uncertainty. No counterfactual trajectory is simulated. Timing includes diagnostic work."},
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "opencv": cv.__version__, "scipy": scipy.__version__},
              "probe_source": source.read_text(), "probe_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "runs": {}}
    failed = False
    for sequence in args.sequences:
        with KeyframeAudit() as audit:
            metrics = bench.evaluate_sequence(sequence, control["seed"])
        differences = non_timing_differences(metrics, control["results"][sequence])
        failures = bench.check_baselines({sequence: metrics}, baselines).get(sequence, [])
        for insertion in audit.insertions:
            for horizon in SURVIVAL_HORIZONS:
                insertion["survival"].setdefault(str(horizon), {"status": "horizon_not_reached"})
        report["runs"][sequence] = {"metrics": metrics, "changed_control_metrics": differences,
                                      "baseline_failures": failures, "summary": audit.summary(),
                                      "frames": audit.frames, "insertions": audit.insertions}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        failed |= bool(differences or failures)
        print(json.dumps({"sequence": sequence, "changed_control_metrics": differences,
                          "baseline_failures": failures, "summary": audit.summary()}), flush=True)
    print(f"Saved {args.output}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
