"""6d: explicit pair ownership, unchanged gates and RNG-transparent shadow."""
from copy import deepcopy

import cv2 as cv
import numpy as np
import pytest

from benchmarks.harness import run_slam
from benchmarks.keyframe_audit import KeyframeAudit
import benchmarks.triangulation_partners as partners
import slam.features.matcher as matcher_mod
from benchmarks.stages.helpers import make_frame, project_gt, set_keypoints, yaw_pose
from benchmarks.synthetic.generator import ensure_sequence
from slam.core.map import Map
from slam.core.observation import Observation
from slam.core.point import Point
from slam.core.tracker import Tracker
from slam.utils import constants as const, telemetry


def _fixture(n=16):
    world = Map()
    older, adjacent, current = [make_frame(yaw_pose(0.0, (x, 0, 0))) for x in (0, 0.99, 1)]
    truth = np.random.default_rng(47).uniform([-1, -.7, 4], [1, .7, 8], (n, 3))
    descriptors = np.random.default_rng(46).integers(0, 256, (n, 32), dtype=np.uint8)
    for index, frame in enumerate((older, adjacent, current)):
        order = np.arange(n)[::-1] if frame is older else np.arange(n)
        set_keypoints(frame, [project_gt(frame.pose, truth[i], frame.K)[0] for i in order], descriptors[order])
        world.add_keyframe(frame)
        frame._bench_frame_index = index
    current.matches = [cv.DMatch(i, i, 0.) for i in range(n)]
    tracker = Tracker(world)
    tracker.cur_frame = current
    return tracker, older, adjacent, current, truth


def _proposal(n, reference=None):
    return partners.PairProposal(reference, [cv.DMatch(i, i, 0.) for i in range(n)],
                                 {i: np.array([0, 0, 5, 1.]) for i in range(n)},
                                 {i: 0. for i in range(n)}, {}, 0)


def test_old_pair_uses_query_train_ownership_metric_geometry_and_both_registries():
    tracker, older, adjacent, current, truth = _fixture()
    matches_before = current.matches
    matches = partners.match_new_pair(current, older, "ORB")
    assert [(m.queryIdx, m.trainIdx) for m in matches] == [(i, 15 - i) for i in range(16)]
    proposal = partners.evaluate_pair(current, older, matches)
    points = partners.materialize(current, proposal)
    assert len(points) == 16
    for i, point in enumerate(points):
        np.testing.assert_allclose(point.pt_3d, truth[i], atol=2e-5)
        assert set(point.observations) == {current.frame_id, older.frame_id}
        for frame, kp in ((current, i), (older, 15 - i)):
            obs = point.observations[frame.frame_id]
            assert obs.frame is frame and obs.kp_idx == kp
            assert frame.observed_points[point.point_id] is obs
            assert frame.get_point_by_keypoint(frame.keypoints[kp]) is point
        np.testing.assert_array_equal(point.descriptor, current.descriptors[i])
    assert not adjacent.observed_points
    assert current.matches is matches_before
    assert tracker.map.prev_keyframe is adjacent


@pytest.mark.parametrize("source", ["mask", "keypoint_registry", "frame_only", "dead_registry"])
@pytest.mark.parametrize("endpoint", ["current", "reference"])
def test_no_mapped_stale_or_frame_only_endpoint_can_be_reused(source, endpoint):
    _, older, _, current, _ = _fixture()
    frame, index = (current, 0) if endpoint == "current" else (older, 15)
    point = Point(np.array([0., 0., 5.]))
    if source == "mask":
        frame.kp_unique_mask[index] = False
    elif source in ("keypoint_registry", "dead_registry"):
        frame.keypoint_to_point_map[frame.keypoints[index]] = point
    else:
        frame.add_point_observation(Observation(frame, point, index))
    matches = partners.match_new_pair(current, older, "ORB")
    assert len(matches) == 15
    assert all(m.queryIdx != 0 for m in matches)
    assert not point.observations  # no silent promotion


@pytest.mark.parametrize("missing", ["current_desc", "reference_desc", "one_train", "empty_query", "no_features"])
def test_missing_match_inputs_produce_an_explicit_empty_pair(missing):
    _, older, _, current, _ = _fixture()
    if missing == "current_desc":
        current.descriptors = None
    elif missing == "reference_desc":
        older.descriptors = None
    elif missing == "one_train":
        older.descriptors = older.descriptors[:1]
    elif missing == "empty_query":
        current.descriptors = current.descriptors[:0]
    else:
        current.keypoints = None
    assert partners.match_new_pair(current, older, "ORB") == []


def test_bf_lowe_preserves_competitors_and_one_to_one_order():
    _, older, _, current, _ = _fixture()
    current.descriptors[1] = current.descriptors[0]
    older.kp_unique_mask[15] = False
    matches = partners.match_new_pair(current, older, "ORB")
    assert not any(m.queryIdx in (0, 1) for m in matches)
    assert len({m.queryIdx for m in matches}) == len(matches)
    assert len({m.trainIdx for m in matches}) == len(matches)


@pytest.mark.parametrize("name,value,code", [
    ("MIN_DEPTH", 100., "MIN_DEPTH_VIOLATION"),
    ("MAX_DEPTH", .1, "MAX_DEPTH_VIOLATION"),
    ("MIN_PARALLAX_DEG", 90., "MIN_PARALLAX_VIOLATION"),
    ("MAX_TRI_REPROJECTION_PX", -1., "MAX_REPROJECTION_ERROR_VIOLATION"),
])
def test_proposals_use_the_actual_unchanged_creation_gates(monkeypatch, name, value, code):
    _, older, _, current, _ = _fixture()
    monkeypatch.setattr(const, name, value)
    proposal = partners.evaluate_pair(current, older, partners.match_new_pair(current, older, "ORB"))
    assert not proposal.points
    assert len(proposal.failures) == 16
    assert all(const.TRIANGULATION_VALIDATION_CODE[code] in codes for codes in proposal.failures.values())


def test_pure_rotation_does_not_create_depth_but_an_older_baseline_can():
    _, older, adjacent, current, _ = _fixture()
    base = partners.evaluate_pair(current, adjacent, current.matches)
    old = partners.evaluate_pair(current, older, partners.match_new_pair(current, older, "ORB"))
    assert not base.points
    assert len(old.points) == 16
    assert partners.choose_pair(base, [old]) is old
    # With identical centers even a rotated, perfectly matched view fails.
    truth = np.array([[.2, .1, 5.]])
    adjacent.pose = yaw_pose(.15, tuple(current.pose[:3, 3]))
    for frame in (adjacent, current):
        set_keypoints(frame, [project_gt(frame.pose, X, frame.K)[0] for X in truth])
    current.matches = [cv.DMatch(0, 0, 0.)]
    assert not partners.evaluate_pair(current, adjacent, current.matches).points


def test_selection_has_minimum_overlap_adjacent_tie_and_temporal_older_ties():
    base, short, tied, better = [_proposal(n) for n in (12, 9, 12, 13)]
    assert partners.choose_pair(base, [short]) is base
    assert partners.choose_pair(base, [tied]) is base
    assert partners.choose_pair(base, [better, _proposal(13)]) is better
    assert partners.choose_pair(base, [better], blocked=range(4), supplement=True) is None


def test_supplement_never_reuses_an_adjacent_query_or_registers_on_wrong_frame():
    tracker, older, adjacent, current, _ = _fixture()
    # Let adjacent retain four points with useful baseline; remaining queries
    # can only be born with the older partner, not duplicated across pairs.
    adjacent.pose = yaw_pose(0., (.5, 0, 0))
    truth = _fixture()[4]
    set_keypoints(adjacent, [project_gt(adjacent.pose, X, adjacent.K)[0] for X in truth], adjacent.descriptors)
    current.matches = current.matches[:4]
    with partners.TriangulationPartnerExperiment("adjacent_plus_one") as experiment:
        points = tracker._triangulate()
    assert len(points) == 16
    queries = [p.observations[current.frame_id].kp_idx for p in points]
    assert len(set(queries)) == 16
    assert all(adjacent.frame_id in p.observations for p in points[:4])
    assert all(older.frame_id in p.observations and adjacent.frame_id not in p.observations for p in points[4:])
    assert experiment.rows[0]["supplement_available"] == 12


def test_shortlist_is_bounded_live_and_latest_tied_without_cache_writes(monkeypatch):
    world = Map()
    frames = [make_frame() for _ in range(12)]
    for frame in frames:
        world.add_keyframe(frame)
    monkeypatch.setattr(partners, "shared_counts", lambda *a: {f.frame_id: 20 for f in frames})
    shortlist, _ = partners.shortlist(frames[-1], frames[-2], world)
    assert shortlist == frames[6:10][::-1] + [frames[5], frames[4]]
    del world.keyframes_by_id[frames[9].frame_id]
    shortlist, _ = partners.shortlist(frames[-1], frames[-2], world)
    assert frames[9] not in shortlist and len(shortlist) == 6
    assert not world._covisibility_score_cache


def test_shared_counts_exclude_frame_only_dead_and_colliding_measurements():
    tracker, older, _, current, truth = _fixture()
    points = []
    for i, X in enumerate(truth):
        point = Point(X)
        point.add_observation(Observation(current, point, i))
        point.add_observation(Observation(older, point, 15 - i))
        points.append(point)
    tracker.map.add_points(points)
    assert partners.shared_counts(current, tracker.map)[older.frame_id] == 16
    current.add_point_observation(Observation(current, points[0], 0))
    del tracker.map.points_by_id[points[1].point_id]
    points[3].observations[current.frame_id].kp_idx = 2
    assert partners.shared_counts(current, tracker.map)[older.frame_id] == 12


def test_pair_evaluation_is_read_only_and_cv_rng_transparent():
    tracker, older, adjacent, current, _ = _fixture()
    frames = (older, adjacent, current)
    keys = [set(f.__dict__) for f in frames]
    qualities = [(f._tracking_quality, f.tracking_quality_needs_update) for f in frames]
    poses = [f.pose for f in frames]
    masks = [f.kp_unique_mask.copy() for f in frames]
    matches = current.matches
    graph = deepcopy(tracker.map.covisibility_graph)
    cv.setRNGSeed(1847)
    expected = np.empty((12, 2))
    cv.randu(expected, 0., 1.)
    cv.setRNGSeed(1847)
    partners.shortlist(current, adjacent, tracker.map)
    partners.evaluate_pair(current, older, partners.match_new_pair(current, older, "ORB"))
    actual = np.empty_like(expected)
    cv.randu(actual, 0., 1.)
    np.testing.assert_array_equal(actual, expected)
    assert [set(f.__dict__) for f in frames] == keys
    assert [(f._tracking_quality, f.tracking_quality_needs_update) for f in frames] == qualities
    assert [f.kp_unique_mask for f in frames] == masks
    for f, pose in zip(frames, poses):
        np.testing.assert_array_equal(f.pose, pose)
        assert not f.observed_points and not f.keypoint_to_point_map
    assert current.matches is matches
    assert tracker.map.covisibility_graph == graph


def test_zero_w_and_gate_failures_are_not_registered(monkeypatch):
    _, older, _, current, _ = _fixture()
    monkeypatch.setattr(partners, "compute_triangulation", lambda *a, **kw: np.array([[1., 1, 1, 0]]))
    proposal = partners.evaluate_pair(current, older, [cv.DMatch(0, 15, 0.)])
    assert proposal.zero_w == 1 and not proposal.points
    assert partners.materialize(current, proposal) == []
    assert not current.observed_points


def test_materialization_refuses_late_registry_overwrite():
    _, older, _, current, _ = _fixture()
    proposal = partners.evaluate_pair(current, older, partners.match_new_pair(current, older, "ORB"))
    current.keypoint_to_point_map[current.keypoints[0]] = Point(np.zeros(3))
    with pytest.raises(RuntimeError, match="overwrite"):
        partners.materialize(current, proposal)


def test_context_restores_both_bindings_on_exception_and_rejects_overlap():
    original = Tracker._triangulate, Tracker.update
    with pytest.raises(ValueError, match="injected"):
        with partners.TriangulationPartnerExperiment("shadow"):
            with pytest.raises(RuntimeError, match="overlap"):
                with partners.TriangulationPartnerExperiment("shadow"):
                    pass
            raise ValueError("injected")
    assert (Tracker._triangulate, Tracker.update) == original


@pytest.mark.parametrize("scale", [.1, 10.])
def test_partner_geometry_is_scale_invariant_inside_unchanged_depth_guards(scale):
    tracker, older, adjacent, current, _ = _fixture()
    matches = partners.match_new_pair(current, older, "ORB")
    base = partners.evaluate_pair(current, older, matches)
    for frame in tracker.map.keyframes:
        pose = frame.pose
        pose[:3, 3] *= scale
        frame.pose = pose
    scaled = partners.evaluate_pair(current, older, matches)
    assert list(scaled.points) == list(base.points)
    np.testing.assert_allclose([p[:3] for p in scaled.points.values()],
                               np.array([p[:3] for p in base.points.values()]) * scale,
                               atol=1e-5, rtol=1e-5)
    assert scaled.summary(current)["accepted_bearing_parallax_deg"]["median"] == pytest.approx(
        base.summary(current)["accepted_bearing_parallax_deg"]["median"], abs=1e-6)


def test_float_descriptor_matching_is_bf_rng_transparent_too(monkeypatch):
    """The float branch is reached THROUGH the declared
    seam now - descriptor_norm raises on unknown names, so the pin
    registers a float-descriptor extractor kind for the call (the
    seam's dict is read at call time; the pre-23 name-string list
    silently defaulted every non-binary name to L2). The RNG
    transparency contract itself is unchanged: float descriptors
    match on BFMatcher(cv.NORM_L2) - no FLANN, no cv-RNG consumption."""
    monkeypatch.setitem(matcher_mod.EXTRACTOR_DESCRIPTOR_KINDS, "SIFT", "float")
    _, older, _, current, _ = _fixture()
    current.descriptors = current.descriptors.astype(np.float32)
    older.descriptors = older.descriptors.astype(np.float32)
    cv.setRNGSeed(1741)
    expected = np.empty((4, 2))
    cv.randu(expected, 0., 1.)
    cv.setRNGSeed(1741)
    matches = partners.match_new_pair(current, older, "SIFT")
    actual = np.empty_like(expected)
    cv.randu(actual, 0., 1.)
    assert len(matches) == 16
    np.testing.assert_array_equal(actual, expected)


def test_best_pair_context_preserves_adjacent_map_order_and_match_binding():
    tracker, older, adjacent, current, _ = _fixture()
    match_binding = current.matches
    order = tracker.map.keyframes.copy()
    with partners.TriangulationPartnerExperiment("best_pair") as experiment:
        points = tracker._triangulate()
    assert len(points) == 16 and experiment.rows[0]["best_pair_is_older"]
    assert tracker.map.keyframes == order and tracker.map.prev_keyframe is adjacent
    assert current.matches is match_binding and not adjacent.observed_points
    assert all(older.frame_id in p.observations for p in points)


def test_materialization_telemetry_counts_only_the_selected_pair_and_unblocked_failures():
    _, older, _, current, _ = _fixture()
    proposal = partners.evaluate_pair(current, older, partners.match_new_pair(current, older, "ORB"))
    proposal.failures[15] = [const.TRIANGULATION_VALIDATION_CODE["MIN_PARALLAX_VIOLATION"]]
    del proposal.points[15]
    recorder = telemetry.TelemetryRecorder()
    telemetry.set_recorder(recorder)
    try:
        points = partners.materialize(current, proposal, blocked={0, 15})
    finally:
        telemetry.set_recorder(None)
    assert len(points) == 14
    assert recorder.events == {"tri_accepted": 14, "tri_rejected": 0}


def test_real_shadow_preserves_exact_arrays_events_calls_and_cohorts(tmp_path):
    sequence = ensure_sequence("translation")
    (tmp_path / "calibration.txt").symlink_to(sequence / "calibration.txt")
    (tmp_path / "frames").mkdir()
    for path in sorted((sequence / "frames").glob("*.png"))[:24]:
        (tmp_path / "frames" / path.name).symlink_to(path)
    with KeyframeAudit() as control_audit:
        control = run_slam(tmp_path)
    with partners.TriangulationPartnerExperiment("shadow") as experiment:
        with KeyframeAudit() as audit:
            actual = run_slam(tmp_path)
    for key in ("poses", "map_points", "keyframe_poses"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(control, key))
    assert actual.telemetry["events"] == control.telemetry["events"]
    assert actual.telemetry["stage_calls"] == control.telemetry["stage_calls"]
    assert audit.frames == control_audit.frames
    assert audit.insertions == control_audit.insertions
    assert experiment.rows and all(len(r["older"]) <= 6 for r in experiment.rows)
