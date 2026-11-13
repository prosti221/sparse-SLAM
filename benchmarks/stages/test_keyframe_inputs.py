"""Absent evidence and match-pair ownership, not a new policy."""
import cv2 as cv
import numpy as np
import pytest

from benchmarks.stages.helpers import make_frame, set_keypoints, yaw_pose
from benchmarks.keyframe_audit import capture_signals, policy_attribution
from slam.core.map import Map
from slam.core.tracker import Tracker
import slam.features.matcher as matcher_mod
from slam.utils import constants as const
from slam.utils.utils import compute_new_points_ratio, should_insert_keyframe


@pytest.mark.parametrize("keypoint_count", [None, 0, 10])
@pytest.mark.parametrize("matches", [None, []])
def test_missing_matches_or_features_supply_no_ratio_evidence(keypoint_count, matches):
    frame = make_frame()
    if keypoint_count is not None:
        set_keypoints(frame, [(30, 30)] * keypoint_count)
    frame.matches = matches

    assert compute_new_points_ratio(frame) == 0.0
    assert frame.matches is matches


@pytest.mark.parametrize("matches", [None, []])
@pytest.mark.parametrize("quality,expected", [(0.0, False), (0.5, True)])
def test_no_match_evidence_does_not_remove_the_independent_quality_branch(matches, quality, expected):
    reference = set_keypoints(make_frame(), [(30, 30)] * 10)
    current = set_keypoints(make_frame(yaw_pose(0.0, (3.0, 0.0, 0.0))), [(30, 30)] * 10)
    current.matches = matches
    current._tracking_quality = quality

    selected, _ = should_insert_keyframe(current, reference)

    assert selected is expected


@pytest.mark.parametrize("previous_pair", [False, True])
def test_zero_match_result_replaces_match_data_from_a_previous_pair(monkeypatch, previous_pair):
    current = set_keypoints(make_frame(), [(30, 30)])
    reference = set_keypoints(make_frame(), [(40, 30)])
    current.matches = [cv.DMatch(0, 99, 0.0)] if previous_pair else None
    mask = list(current.kp_unique_mask)
    pose = current.pose
    calls = []

    def match(frame1, frame2, method, motion_prior=None):
        calls.append((frame1, frame2))
        return [], None

    monkeypatch.setattr(matcher_mod, "match_features", match)

    assert matcher_mod.match_features_between_frames(current, reference, "ORB") is None

    assert calls == [(current, reference)]
    assert current.matches == []
    assert current.kp_unique_mask == mask
    np.testing.assert_array_equal(current.pose, pose)


@pytest.mark.parametrize("pose_available", [False, True])
def test_nonempty_match_data_and_pose_return_are_preserved(monkeypatch, pose_available):
    current = set_keypoints(make_frame(), [(30, 30), (40, 40)])
    reference = set_keypoints(make_frame(), [(35, 30), (45, 40)])
    matches = [cv.DMatch(1, 0, 0.0), cv.DMatch(0, 1, 0.0)]
    pose = yaw_pose(0.1) if pose_available else None
    monkeypatch.setattr(matcher_mod, "match_features", lambda *a, **kw: (matches, pose))

    result = matcher_mod.match_features_between_frames(current, reference, "ORB")

    assert result is pose
    assert current.matches is matches  # Preserve the exact list and query/train order.


@pytest.mark.parametrize("initial_matches", [None, [], "stale_other_pair"])
def test_forced_gap_still_inserts_without_reusing_another_pairs_matches(monkeypatch, initial_matches):
    """The existing backstop survives; no-match triangulation stays empty.

    trainIdx=99 belongs to an earlier pair, NOT this one-keypoint KF.
    Leaving it attached would index the wrong reference or crash dedup.
    """
    world = Map()
    reference = set_keypoints(make_frame(), [(30, 30)])
    world.add_keyframe(reference)
    current = set_keypoints(make_frame(), [(30, 30)])
    current.matches = [cv.DMatch(0, 99, 0.0)] if initial_matches == "stale_other_pair" else initial_matches
    tracker = Tracker(world)
    tracker.cur_frame = current
    tracker.prev_frame = reference
    tracker.iterations_since_last_keyframe = 10
    monkeypatch.setattr(matcher_mod, "match_features", lambda *a, **kw: ([], None))
    triangulations = []
    monkeypatch.setattr(tracker, "_on_keyframe_inserted", lambda: triangulations.append(tracker._triangulate()))

    tracker._handle_keyframe_insertion()

    assert current.is_keyframe
    assert world.cur_keyframe is current
    assert tracker.iterations_since_last_keyframe == 0
    assert current.matches == []
    assert triangulations == [[]]
    assert not world.points


def test_valid_legacy_ratio_is_not_clamped_or_redefined():
    frame = set_keypoints(make_frame(), [(30, 30)] * 1000)
    frame.kp_unique_mask = [False] * 100 + [True] * 900
    frame.set_match_data([cv.DMatch(i, i, 0.0) for i in range(100)])

    assert compute_new_points_ratio(frame) == 9.0


def test_audit_preserves_missingness_but_mirrors_the_numeric_gate_under_override(monkeypatch):
    world = Map()
    reference = set_keypoints(make_frame(), [(30, 30)])
    world.add_keyframe(reference)
    current = set_keypoints(make_frame(yaw_pose(0.0, (3.0, 0.0, 0.0))), [(30, 30)])
    monkeypatch.setattr(const, "MINIMUM_NEW_POINTS_RATIO", 0.0)
    selected, _ = should_insert_keyframe(current, reference)
    signals = capture_signals(current, reference, world)

    assert signals["legacy_new_points_ratio"] is None
    assert compute_new_points_ratio(current) == 0.0
    assert selected  # Explicitly overriding the threshold to zero admits the sentinel.
    assert policy_attribution(signals, 0, selected)["flags"]["new_ratio"]
