"""Diagnostic definitions and strict observer transparency."""
from types import SimpleNamespace

import cv2 as cv
import numpy as np
import pytest

from benchmarks.harness import run_slam
from benchmarks.keyframe_audit import (
    KeyframeAudit, capture_signals, grid_coverage, non_timing_differences,
    policy_attribution,
)
from benchmarks.stages.helpers import make_frame, project_gt, set_keypoints, yaw_pose
from benchmarks.synthetic.generator import ensure_sequence
from slam.ba.bundle_adjustment_g2o import G2OBundleAdjustment
from slam.core.map import Map
from slam.core.observation import Observation
from slam.core.point import Point
from slam.core.tracker import Tracker
import slam.core.tracker as tracker_mod
from slam.utils import telemetry


def _geometry_pair(scale=1.0, current_pose=None):
    world = Map()
    reference = make_frame(yaw_pose(0.0))
    older = make_frame(yaw_pose(0.0, (-0.4 * scale, 0.0, 0.0)))
    current = make_frame(current_pose if current_pose is not None
                         else yaw_pose(0.0, (0.3 * scale, 0.0, 0.0)))
    truth = scale * np.array([[-0.6, -0.4, 5.0], [0.6, -0.4, 5.0],
                              [-0.6, 0.4, 6.0], [0.6, 0.4, 6.0],
                              [0.0, -0.2, 7.0], [0.0, 0.2, 7.0]])
    for frame in (reference, older, current):
        set_keypoints(frame, [project_gt(frame.pose, X, frame.K)[0] for X in truth])
    world.add_keyframe(older)
    world.add_keyframe(reference)
    points = []
    for i, X in enumerate(truth):
        point = Point(X.copy())
        for frame in (reference, older, current):
            point.add_observation(Observation(frame, point, i))
        points.append(point)
    world.add_points(points)
    current.set_match_data([cv.DMatch(i, i, 0.0) for i in range(len(points))])
    current.kp_unique_mask = [False] * len(points)
    return world, current, reference, points


@pytest.mark.parametrize("scale", [0.1, 10.0])
def test_audit_geometry_is_monocular_scale_invariant(scale):
    world, current, reference, _ = _geometry_pair()
    base = capture_signals(current, reference, world)
    world, current, reference, _ = _geometry_pair(scale)
    scaled = capture_signals(current, reference, world)

    assert scaled["baseline"] == pytest.approx(base["baseline"] * scale)
    for key in ("baseline_over_depth", "overlap_fraction", "visible_match_fraction"):
        assert scaled["last_reference"][key] == pytest.approx(base["last_reference"][key])
    assert scaled["last_reference"]["parallax_deg"]["median"] == pytest.approx(
        base["last_reference"]["parallax_deg"]["median"], abs=1e-6)
    assert scaled["consistent_support"] == base["consistent_support"]


def test_audit_pure_rotation_has_image_motion_but_no_depth_parallax():
    world, current, reference, _ = _geometry_pair(current_pose=yaw_pose(0.12))
    signals = capture_signals(current, reference, world)

    assert signals["rotation_deg"] > 6.0
    assert signals["last_reference"]["image_fraction_flow"]["median"] > 0.05
    assert signals["last_reference"]["parallax_deg"]["median"] < 2e-6
    assert signals["last_reference"]["baseline_over_depth"] == 0.0


def test_grid_coverage_distinguishes_clustered_from_distributed_support():
    frame = make_frame()
    observations = [SimpleNamespace(kp_idx=i) for i in range(16)]
    set_keypoints(frame, [(30 + i, 30) for i in range(16)])
    cluster = grid_coverage(frame, observations)
    set_keypoints(frame, [(80 + 160 * x, 60 + 120 * y) for y in range(4) for x in range(4)])
    distributed = grid_coverage(frame, observations)

    assert cluster["occupied_cells"] == 1
    assert distributed["occupied_cells"] == 16
    assert sum(map(sum, cluster["counts"])) == sum(map(sum, distributed["counts"])) == 16


@pytest.mark.parametrize("contamination,expected_support", [
    ("frame_only", 5), ("dead_point", 5), ("wrong_point_identity", 5), ("collision", 4),
])
def test_support_uses_live_authoritative_distinct_measurements(contamination, expected_support):
    world, current, reference, points = _geometry_pair()
    if contamination == "frame_only":
        current.add_point_observation(Observation(current, points[0], 0))
    elif contamination == "dead_point":
        del world.points_by_id[points[0].point_id]
    elif contamination == "wrong_point_identity":
        impostor = Point(points[0].pt_3d.copy())
        impostor.point_id = points[0].point_id
        world.points_by_id[impostor.point_id] = impostor
    else:
        points[1].observations[current.frame_id].kp_idx = 0

    signals = capture_signals(current, reference, world)

    assert signals["consistent_support"]["3.0"] == expected_support
    assert signals["authoritative_distinct_associations"] == expected_support
    assert sum(signals["excluded_associations"].values()) == 6 - expected_support


def test_tracking_frames_cannot_supply_reference_maturity():
    world, current, reference, points = _geometry_pair()
    older = world.keyframes[0]
    points[0].remove_observation(older.frame_id)
    tracker_frame = set_keypoints(make_frame(), [current.keypoints[0].pt])
    points[0].add_observation(Observation(tracker_frame, points[0], 0))
    assert points[0].num_observations == 3

    signals = capture_signals(current, reference, world)

    assert signals["consistent_support"]["3.0"] == 6
    assert signals["last_reference"]["mature_anchors"] == 5
    assert signals["last_reference"]["shared_consistent"] == 5


def test_audit_exposes_ratio_population_mismatch_without_repairing_it():
    world, current, reference, _ = _geometry_pair()
    set_keypoints(current, [current.keypoints[i].pt for i in range(6)] + [(20, 20)] * 994)
    current.kp_unique_mask = [False] * 100 + [True] * 900
    current.set_match_data([cv.DMatch(i, i, 0.0) for i in range(100)])

    signals = capture_signals(current, reference, world)

    assert signals["legacy_new_points_ratio"] == 9.0
    assert signals["unmapped_feature_fraction"] == 0.9
    assert signals["matched_unmapped_fraction"] == 0.0


def test_missing_data_is_unavailable_not_perfect_geometry():
    world = Map()
    reference = make_frame()
    world.add_keyframe(reference)
    current = make_frame()

    signals = capture_signals(current, reference, world)

    assert signals["match_data_missing"]
    assert signals["legacy_new_points_ratio"] is None
    assert signals["last_reference"]["overlap_fraction"] is None
    assert signals["last_reference"]["baseline_over_depth"] is None
    assert signals["last_reference"]["parallax_deg"]["median"] is None
    assert signals["consistent_support"]["3.0"] == 0


def test_invalid_projections_are_not_support():
    world, current, reference, points = _geometry_pair()
    points[0].pt_3d[2] = -2.0

    signals = capture_signals(current, reference, world)

    assert signals["consistent_support"]["3.0"] == 5
    assert signals["invalid_or_outside_predictions"] == 1
    assert signals["fresh_pixel_residual"]["nonfinite_count"] == 1


def test_snapshot_does_not_mutate_quality_poses_registries_caches_or_cv_rng():
    world, current, reference, points = _geometry_pair()
    current._tracking_quality = 0.432
    current.tracking_quality_needs_update = True
    frames = [current, *world.keyframes]
    frame_keys = [set(frame.__dict__) for frame in frames]
    poses = [frame.pose for frame in frames]
    qualities = [(frame._tracking_quality, frame.tracking_quality_needs_update) for frame in frames]
    obs = [obs for point in points for obs in point.observations.values()]
    obs_keys = [set(observation.__dict__) for observation in obs]
    registries = [dict(point.observations) for point in points]
    cv.setRNGSeed(1847)
    expected = np.empty((12, 2))
    cv.randu(expected, 0.0, 1.0)
    cv.setRNGSeed(1847)

    signals = capture_signals(current, reference, world)

    actual = np.empty_like(expected)
    cv.randu(actual, 0.0, 1.0)
    np.testing.assert_array_equal(actual, expected)
    assert signals["legacy_quality"] == 0.432
    assert [set(frame.__dict__) for frame in frames] == frame_keys
    assert [(frame._tracking_quality, frame.tracking_quality_needs_update) for frame in frames] == qualities
    assert [set(observation.__dict__) for observation in obs] == obs_keys
    assert [dict(point.observations) for point in points] == registries
    for frame, pose in zip(frames, poses):
        np.testing.assert_array_equal(frame.pose, pose)


def test_reason_attribution_preserves_overlap_and_forced_gap_override():
    signals = {"baseline": 2.0, "rotation_deg": 1.5,
               "legacy_new_points_ratio": 0.4, "legacy_quality": 0.01}
    selected = policy_attribution(signals, 3, True)
    assert selected["primary_reason"] == "translation"
    assert all(selected["flags"][key] for key in ("translation", "rotation", "new_ratio", "quality"))
    signals.update(baseline=0.0, rotation_deg=0.0, legacy_new_points_ratio=None, legacy_quality=0.0)
    forced = policy_attribution(signals, 10, False)
    assert forced["primary_reason"] == "maximum_gap_only"
    assert forced["effective_selected"] and not forced["policy_selected"]
    assert not policy_attribution(signals, 9, False)["effective_selected"]


def test_audit_restores_actual_bindings_on_failure_and_rejects_overlap():
    bindings = [(Tracker, "update"), (tracker_mod, "should_insert_keyframe"),
                (G2OBundleAdjustment, "refine_pose_pnp"), (Map, "add_keyframe"),
                (Tracker, "_triangulate"), (Map, "reset")]
    originals = [getattr(owner, name) for owner, name in bindings]
    with pytest.raises(ValueError, match="injected"):
        with KeyframeAudit():
            assert all(getattr(owner, name) is not original
                       for (owner, name), original in zip(bindings, originals))
            with pytest.raises(RuntimeError, match="overlap"):
                with KeyframeAudit():
                    pass
            raise ValueError("injected")
    assert all(getattr(owner, name) is original for (owner, name), original in zip(bindings, originals))
    with KeyframeAudit():
        pass


def test_audit_records_real_pnp_return_without_fixing_legacy_tuple_truthiness(monkeypatch):
    calls = []

    def pnp(optimizer, frame=None):
        calls.append(frame)
        return False, 7

    def update(tracker, frame):
        tracker.cur_frame = frame
        tracker.map.add_keyframe(frame)
        result = tracker.map.g2o_optimizer.refine_pose_pnp(frame)
        telemetry.event("pnp_frame_ok" if result else "pnp_frame_fail")
        return frame.is_keyframe

    monkeypatch.setattr(G2OBundleAdjustment, "refine_pose_pnp", pnp)
    monkeypatch.setattr(Tracker, "update", update)
    frame = set_keypoints(make_frame(), [(20, 20)])
    recorder = telemetry.TelemetryRecorder()
    telemetry.set_recorder(recorder)
    try:
        with KeyframeAudit() as audit:
            assert Tracker(Map()).update(frame)
    finally:
        telemetry.set_recorder(None)

    assert calls == [frame]
    assert audit.summary()["actual_per_frame_pnp"] == {"calls": 1, "success": 0, "failure": 1}
    assert recorder.events == {"pnp_frame_ok": 1}  # Deliberately NOT repaired in 6a.


def test_survival_uses_insertion_serial_not_retained_keyframe_count():
    world, _, _, points = _geometry_pair()
    audit = KeyframeAudit()

    def insertion(serial):
        return {"serial": serial, "retained_new_points_at_frame_end": 0, "survival": {}}

    audit.insertions = [insertion(0)]
    audit._pending[0] = {"ids": {p.point_id for p in points}, "initialized": False, "epoch": 0}
    audit._finish_frame(world)
    del world.points_by_id[points[0].point_id]
    audit.insertions.extend([insertion(1), insertion(2)])
    assert len(world.keyframes) == 2
    audit._finish_frame(world)

    survived = audit.insertions[0]["survival"]
    assert survived["2"]["remaining"] == 5
    assert survived["2"]["fraction"] == pytest.approx(5 / 6)
    assert "5" not in survived
    audit.insertions.extend(insertion(i) for i in range(3, 6))
    audit._finish_frame(world)
    assert survived["5"]["remaining"] == 5
    assert not audit._pending


def test_real_tracker_audit_preserves_exact_poses_map_events_and_keyframe_indices(tmp_path):
    sequence = ensure_sequence("translation")
    (tmp_path / "calibration.txt").symlink_to(sequence / "calibration.txt")
    (tmp_path / "frames").mkdir()
    for path in sorted((sequence / "frames").glob("*.png"))[:16]:
        (tmp_path / "frames" / path.name).symlink_to(path)
    control = run_slam(tmp_path)
    with KeyframeAudit() as audit:
        observed = run_slam(tmp_path)

    np.testing.assert_array_equal(observed.poses, control.poses)
    np.testing.assert_array_equal(observed.map_points, control.map_points)
    assert observed.keyframe_indices == control.keyframe_indices
    assert observed.telemetry["events"] == control.telemetry["events"]
    assert observed.telemetry["stage_calls"] == control.telemetry["stage_calls"]
    assert len(audit.frames) == 16
    assert audit.summary()["policy_evaluations"] == 15
    assert len(audit.insertions) == len(control.keyframe_indices)
    assert all(frame["decision"]["effective_selected"] == (i in observed.keyframe_indices)
               for i, frame in enumerate(audit.frames) if frame["decision"] is not None)


def test_exact_control_comparison_ignores_only_timing():
    reference = {"ate_rmse": 0.1, "events": {"pnp_ok": 1}, "fps": 12, "timing_ms_per_frame": {}}
    observed = dict(reference, fps=1, timing_ms_per_frame={"audit": 1})
    assert non_timing_differences(observed, reference) == []
    observed["events"] = {"pnp_ok": 2}
    assert non_timing_differences(observed, reference) == ["events"]
