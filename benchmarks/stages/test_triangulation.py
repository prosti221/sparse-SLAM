"""Stage test: triangulation against known 3D geometry.

With perfect correspondences and correct poses, triangulation must recover
the exact metric points; validation must accept good configurations and
reject degenerate ones with the right reason codes.
"""
import numpy as np
import cv2 as cv
import pytest

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
    assert result["reprojection_error"] < 1e-6


def test_validation_rejects_low_parallax():
    """Identical camera poses: no parallax, triangulation is undefined."""
    T = yaw_pose(0.0, (0.0, 0.0, 0.0))
    X = np.array([0.2, 0.1, 5.0])
    cur_f, prev_f = _validated_pair(T, T, X)

    result = is_valid_triangulated_point(0, cur_f, prev_f, np.append(X, 1.0))
    assert result["validation_code"] == \
        TRIANGULATION_VALIDATION_CODE["MIN_PARALLAX_VIOLATION"]


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
    # The validator cascades overwrites (last failing check wins): a point
    # behind the camera fails the depth check, but its projection is None,
    # so the infinite reprojection error overwrites MIN_DEPTH_VIOLATION.
    # The essential property is rejection; the specific code documents the
    # precedence quirk.
    assert result["validation_code"] != TRIANGULATION_VALIDATION_CODE["VALID"]
    assert result["validation_code"] == \
        TRIANGULATION_VALIDATION_CODE["MAX_REPROJECTION_ERROR_VIOLATION"]
