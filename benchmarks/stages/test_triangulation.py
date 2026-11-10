"""Stage test: triangulation against known 3D geometry.

With perfect correspondences and correct poses, triangulation must recover
the exact metric points; validation must accept good configurations and
reject degenerate ones with the right reason codes.
"""
import numpy as np
import cv2 as cv
import pytest

import slam.utils.constants as const
import slam.core.point as point_mod
import slam.utils.utils as utils
from slam.core.observation import Observation
from slam.core.point import Point
from benchmarks.stages.helpers import make_K, make_frame, yaw_pose, project_gt, set_keypoints
from slam.utils.utils import triangulate, is_valid_triangulated_point
from slam.utils.constants import TRIANGULATION_VALIDATION_CODE

K = make_K()


def _normalized(uv):
    return (np.linalg.inv(K) @ np.append(uv, 1.0))[:2]


def test_triangulate_recovers_known_metric_points():
    """Perfect input -> exact output: scale included, since projection
    matrices come from metric poses and points are normalized by K."""
    T1 = yaw_pose(0.0, (0.0, 0.0, 0.0))
    T2 = yaw_pose(0.0, (1.0, 0.0, 0.0))
    P1 = np.linalg.inv(T1)[:3, :]
    P2 = np.linalg.inv(T2)[:3, :]

    X_true = np.array([
        [0.3, 0.1, 5.0], [-0.4, -0.2, 3.0], [0.1, 0.4, 8.0], [-0.2, 0.0, 4.0],
    ])
    pts1, pts2 = [], []
    for X in X_true:
        uv1, _ = project_gt(T1, X, K)
        uv2, _ = project_gt(T2, X, K)
        pts1.append(_normalized(uv1))
        pts2.append(_normalized(uv2))

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    points_4d = triangulate(P1, P2, pts1, pts2)
    recovered = points_4d[:, :3] / points_4d[:, 3:4]

    assert np.allclose(recovered, X_true, atol=1e-6)


def _validated_pair(T1, T2, X):
    """Build (cur, prev) frames + matches for one world point."""
    prev_f = make_frame(T1, K)
    cur_f = make_frame(T2, K)
    uv1, _ = project_gt(T1, X, K)
    uv2, _ = project_gt(T2, X, K)
    set_keypoints(prev_f, [uv1])
    set_keypoints(cur_f, [uv2])
    cur_f.set_match_data([cv.DMatch(0, 0, 0.0)])
    return cur_f, prev_f


def test_validation_accepts_good_point():
    T1 = yaw_pose(0.0, (0.0, 0.0, 0.0))
    T2 = yaw_pose(0.0, (1.0, 0.0, 0.0))
    X = np.array([0.2, 0.1, 5.0])
    cur_f, prev_f = _validated_pair(T1, T2, X)

    result = is_valid_triangulated_point(0, cur_f, prev_f, np.append(X, 1.0))
    assert result["validation_code"] == TRIANGULATION_VALIDATION_CODE["VALID"]
    assert result["failed_codes"] == []
    assert result["reprojection_error"] < 1e-6


def test_validation_rejects_low_parallax():
    """Identical camera poses: no parallax, triangulation is undefined."""
    T = yaw_pose(0.0, (0.0, 0.0, 0.0))
    X = np.array([0.2, 0.1, 5.0])
    cur_f, prev_f = _validated_pair(T, T, X)

    result = is_valid_triangulated_point(0, cur_f, prev_f, np.append(X, 1.0))
    assert result["validation_code"] == \
        TRIANGULATION_VALIDATION_CODE["MIN_PARALLAX_VIOLATION"]
    # A single clean failure: the only failed gate, first and whole.
    assert result["failed_codes"] == \
        [TRIANGULATION_VALIDATION_CODE["MIN_PARALLAX_VIOLATION"]]


def test_validation_rejects_point_behind_camera():
    T1 = yaw_pose(0.0, (0.0, 0.0, 5.0))
    T2 = yaw_pose(0.0, (1.0, 0.0, 5.0))
    # Cameras look toward +z; a point at z=4 is behind both. Keypoints claim
    # to observe it anyway - the validator must catch the impossibility.
    X = np.array([0.1, 0.1, 4.0])
    prev_f = make_frame(T1, K)
    cur_f = make_frame(T2, K)
    set_keypoints(prev_f, [(329.0, 249.0)])
    set_keypoints(cur_f, [(329.0, 249.0)])
    cur_f.set_match_data([cv.DMatch(0, 0, 0.0)])

    result = is_valid_triangulated_point(0, cur_f, prev_f, np.append(X, 1.0))
    # Phase 5c: the cascade is gone. A behind-camera point fails BOTH
    # cheirality (z <= 0) and reprojection (behind-camera projection is
    # None -> inf error); parallax passes (~48 deg here). The FIRST
    # failure is the root-cause code, and failed_codes carries all of
    # them. (The old cascade reported only MAX_REPROJECTION_ERROR_
    # VIOLATION - last failing check wins - the precedence quirk this
    # pin replaces; the 5a probe showed this population is sizable:
    # 9.6% of orbit's candidates.)
    assert result["validation_code"] == \
        TRIANGULATION_VALIDATION_CODE["MIN_DEPTH_VIOLATION"]
    assert result["failed_codes"] == [
        TRIANGULATION_VALIDATION_CODE["MIN_DEPTH_VIOLATION"],
        TRIANGULATION_VALIDATION_CODE["MAX_REPROJECTION_ERROR_VIOLATION"],
    ]


def test_gate_thresholds_are_override_sweepable(monkeypatch):
    """5c (the 2i property, pinned): the creation gate reads its
    thresholds through the module alias - an --override-style
    monkeypatch reaches the live gate - and a single-failure reject
    reports exactly one code."""
    T1 = yaw_pose(0.0, (0.0, 0.0, 0.0))
    T2 = yaw_pose(0.0, (1.0, 0.0, 0.0))
    X = np.array([0.2, 0.1, 5.0])
    cur_f, prev_f = _validated_pair(T1, T2, X)

    monkeypatch.setattr(const, "MIN_PARALLAX_DEG", 90.0)
    result = is_valid_triangulated_point(0, cur_f, prev_f, np.append(X, 1.0))
    assert result["validation_code"] == \
        TRIANGULATION_VALIDATION_CODE["MIN_PARALLAX_VIOLATION"]
    assert result["failed_codes"] == \
        [TRIANGULATION_VALIDATION_CODE["MIN_PARALLAX_VIOLATION"]]
    assert result["validation_code"] == result["failed_codes"][0]


def test_point_health_threshold_is_split_from_tri_gate(monkeypatch):
    """5c: the pruning path (Point.is_good) reads its OWN constant,
    name-bound at import in point.py. The creation gate's --override
    name (MAX_TRI_REPROJECTION_PX) cannot leak into point health, and
    point.py's own binding is what drives it - the 2i isolation
    property, now by construction (separate names)."""
    kf = make_frame(yaw_pose(0.0, (0.0, 0.0, 0.0)), K)
    set_keypoints(kf, [(0.0, 0.0)])  # Observation.keypoint resolves lazily
    pt = Point(np.zeros(3))
    pt.add_observation(Observation(kf, pt, 0))
    pt.update_reprojection_error(2.0)  # pixel-scale: above 1.0 -> bad
    assert pt.average_reprojection_error == 2.0
    assert not pt.is_good

    # sweeping the creation gate's threshold must not touch point health
    monkeypatch.setattr(const, "MAX_TRI_REPROJECTION_PX", 999.0)
    assert not pt.is_good

    # point.py's own module binding drives the decision
    monkeypatch.setattr(point_mod, "MAX_POINT_AVG_REPROJECTION_ERROR", 999.0)
    assert pt.is_good


def test_validation_codes_dict_is_consistent():
    """5c: the retired codes (MIN_BASELINE - never implemented;
    WELL_CONDITIONED - unreachable) are gone; the dict stays
    bidirectional for the tracker's telemetry reverse lookup."""
    codes = TRIANGULATION_VALIDATION_CODE
    for name, code in [("VALID", 0), ("MIN_DEPTH_VIOLATION", 1),
                       ("MAX_DEPTH_VIOLATION", 2),
                       ("MAX_REPROJECTION_ERROR_VIOLATION", 4),
                       ("MIN_PARALLAX_VIOLATION", 6)]:
        assert codes[name] == code  # forward mapping, historical numbers kept
        assert codes[code] == name  # the reverse mapping (bidirectional)
    assert "MIN_BASELINE_VIOLATION" not in codes
    assert "WELL_CONDITIONED_VIOLATION" not in codes


def test_zero_reprojection_error_does_not_prove_depth_accuracy():
    """5d: epipolar-consistent image error can change depth, not residual.

    A point reconstructed from the perturbed pair passes every current
    gate but is >25% wrong in depth. Observed reprojection residual
    cannot serve as the noise standard deviation for depth covariance.
    """
    prev = make_frame(yaw_pose(0.0, (0.0, 0.0, 0.0)), K)
    cur = make_frame(yaw_pose(0.0, (0.1, 0.0, 0.0)), K)
    truth = np.array([0.2, 0.1, 5.0])
    uv_prev, _ = project_gt(prev.pose, truth, K)
    uv_cur, _ = project_gt(cur.pose, truth, K)
    uv_cur = uv_cur + np.array([2.0, 0.0])
    set_keypoints(prev, [uv_prev])
    set_keypoints(cur, [uv_cur])
    cur.set_match_data([cv.DMatch(0, 0, 0.0)])

    homogeneous = triangulate(
        np.linalg.inv(cur.pose)[:3], np.linalg.inv(prev.pose)[:3],
        np.array([_normalized(uv_cur)]), np.array([_normalized(uv_prev)]))[0]
    recovered = homogeneous / homogeneous[3]
    result = is_valid_triangulated_point(0, cur, prev, recovered)

    assert result["validation_code"] == TRIANGULATION_VALIDATION_CODE["VALID"]
    assert result["reprojection_error"] < 1e-8
    assert abs(recovered[2] / truth[2] - 1.0) > 0.25


@pytest.mark.parametrize("use_optimization", [False, True])
def test_compute_triangulation_preserves_metric_geometry_and_match_order(use_optimization):
    """5f: the optional refinement uses inverse Twc and query/train order.

    Perfect fixed-pose correspondences retain metric scale. Permuted
    matches must not be confused with the keypoint array order.
    """
    prev = make_frame(yaw_pose(-0.1, (-0.2, 0.0, 0.1)), K)
    cur = make_frame(yaw_pose(0.2, (0.8, 0.1, 0.0)), K)
    truth = np.array([[0.3, 0.1, 5.0], [-0.4, -0.2, 3.0], [0.1, 0.4, 8.0]])
    set_keypoints(prev, [project_gt(prev.pose, X, K)[0] for X in truth])
    set_keypoints(cur, [project_gt(cur.pose, X, K)[0] for X in truth[::-1]])
    cur.set_match_data([cv.DMatch(0, 2, 0.0), cv.DMatch(2, 0, 0.0),
                        cv.DMatch(1, 1, 0.0)])
    poses_before = [prev.pose.copy(), cur.pose.copy()]

    homogeneous = utils.compute_triangulation(cur, prev, use_optimization)
    recovered = homogeneous[:, :3] / homogeneous[:, 3:4]

    np.testing.assert_allclose(recovered, truth[[2, 0, 1]], atol=1e-5, rtol=0)
    np.testing.assert_array_equal(prev.pose, poses_before[0])
    np.testing.assert_array_equal(cur.pose, poses_before[1])


def test_optional_refinement_reduces_its_normalized_geometric_objective():
    """5f: LM refines geometric, not DLT algebraic, error at fixed poses.

    This is a local objective pin, not a claim of better true depth or
    lower pixel error under arbitrary intrinsics/distortion.
    """
    P1 = np.linalg.inv(yaw_pose(0.0))[:3]
    P2 = np.linalg.inv(yaw_pose(0.25, (0.7, 0.1, 0.2)))[:3]
    truth = np.array([0.2, 0.1, 5.0])

    def project(P, X):
        v = P @ np.append(X, 1.0)
        return v[:2] / v[2]

    pt1 = project(P1, truth) + [0.002, -0.003]
    pt2 = project(P2, truth) + [-0.001, 0.004]
    homogeneous = triangulate(P1, P2, pt1[None], pt2[None])[0]
    linear = homogeneous[:3] / homogeneous[3]
    refined = utils.optimize_triangulation(pt1, pt2, P1, P2, linear)

    def objective(X):
        return np.sum((project(P1, X) - pt1) ** 2) + np.sum((project(P2, X) - pt2) ** 2)

    assert refined is not None
    assert objective(refined) < objective(linear) * 0.995


def test_refinement_cannot_repair_epipolar_consistent_wrong_depth():
    """5f: minimizing an already-zero residual cannot identify image noise."""
    truth = np.array([0.2, 0.1, 5.0])
    cur, prev = _validated_pair(yaw_pose(0.0), yaw_pose(0.0, (0.1, 0.0, 0.0)), truth)
    observed = np.array(cur.keypoints[0].pt) + [2.0, 0.0]
    set_keypoints(cur, [observed])

    homogeneous = utils.compute_triangulation(cur, prev, use_optimization=True)[0]
    recovered = homogeneous / homogeneous[3]
    result = is_valid_triangulated_point(0, cur, prev, recovered)

    assert result["failed_codes"] == []
    assert result["reprojection_error"] < 1e-5
    assert abs(recovered[2] / truth[2] - 1.0) > 0.25


def test_refinement_failure_falls_back_to_linear_candidate(monkeypatch):
    """5f: solver failure must not drop/reorder a match or modify its DLT row."""
    cur, prev = _validated_pair(yaw_pose(0.0), yaw_pose(0.2, (1.0, 0.0, 0.0)),
                                np.array([0.2, 0.1, 5.0]))
    linear = utils.compute_triangulation(cur, prev, use_optimization=False)
    calls = []

    def fail(*args):
        calls.append(args)
        return None

    monkeypatch.setattr(utils, "optimize_triangulation", fail)
    refined = utils.compute_triangulation(cur, prev, use_optimization=True)

    assert len(calls) == 1
    np.testing.assert_array_equal(refined, linear)
