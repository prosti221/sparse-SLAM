"""Stage test: Frame geometry - projection, pose conventions, visibility.

These are the cheapest, highest-value tests: every downstream stage trusts
Frame's math. Codifies the Twc (camera-to-world) convention everywhere.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.stages.helpers import make_K, make_frame, se3, yaw_pose, project_gt
from slam.utils.utils import compute_baseline_distance

K = make_K()


def test_project_point_matches_manual_computation():
    T = yaw_pose(0.3, (1.0, -2.0, 0.5))
    frame = make_frame(T, K)
    X = np.array([0.4, -0.2, 3.0])

    uv_manual, _ = project_gt(T, X, K)
    uv = frame.project_point(X, normalized=False)

    assert np.allclose(uv, uv_manual, atol=1e-9)


def test_normalized_projection_is_pixel_normalized_by_k():
    T = yaw_pose(-0.4, (0.2, 0.1, 1.0))
    frame = make_frame(T, K)
    X = np.array([-0.3, 0.5, 2.5])

    uv = frame.project_point(X, normalized=False)
    xy = frame.project_point(X, normalized=True)

    assert np.allclose(xy, frame.normalize_keypoint(uv), atol=1e-9)
    assert np.allclose((frame.K @ np.append(xy, 1.0))[:2], uv, atol=1e-9)


def test_world_to_camera_is_inverse_pose():
    T = yaw_pose(1.1, (2.0, 0.3, -1.0))
    frame = make_frame(T, K)
    X = np.array([0.5, 0.5, 0.5])

    Xc_expected = (np.linalg.inv(T) @ np.append(X, 1.0))[:3]
    assert np.allclose(frame.world_to_camera(X), Xc_expected, atol=1e-9)


def test_batch_projection_matches_single():
    T = yaw_pose(0.2, (0.0, 0.0, 1.0))
    frame = make_frame(T, K)
    pts = np.array([[0.5, 0.2, 4.0], [-0.4, -0.1, 3.0], [0.1, 0.3, 9.0]])

    batch = frame.project_points(pts, normalized=False)
    for i, X in enumerate(pts):
        single = frame.project_point(X, normalized=False)
        assert np.allclose(batch[i], single, atol=1e-9)


def test_point_behind_camera_is_not_visible():
    T = yaw_pose(0.0, (0.0, 0.0, 5.0))  # camera at z=5 looking +z
    frame = make_frame(T, K)

    assert frame.project_point(np.array([0.0, 0.0, 4.0])) is None  # behind
    assert not frame.is_point_visible(np.array([0.0, 0.0, 4.0]))
    assert frame.is_point_visible(np.array([0.0, 0.0, 9.0]))       # ahead


def test_visibility_respects_image_margin():
    frame = make_frame(np.eye(4), K)
    # Projects to (320+45, 240): inside
    assert frame.is_point_visible(np.array([0.1, 0.0, 1.0]))
    # Projects to (320+450*4, 240): far outside
    assert not frame.is_point_visible(np.array([4.0, 0.0, 1.0]))


def test_camera_center_is_pose_translation():
    """Regression: camera_center returned -R^T t (the world-to-camera formula)
    although poses are camera-to-world, misplacing rotated cameras."""
    T = yaw_pose(np.pi / 2, (1.0, -2.0, 3.0))
    frame = make_frame(T, K)
    assert np.allclose(frame.camera_center, [1.0, -2.0, 3.0])


def test_baseline_distance_ignores_rotation():
    """Baseline is pure camera-position distance; orientation must not
    corrupt it (this fed keyframe-insertion decisions before the fix)."""
    f1 = make_frame(yaw_pose(np.pi / 2, (2.0, 0.0, 0.0)), K)
    f2 = make_frame(yaw_pose(0.0, (2.0, 3.0, 0.0)), K)

    # True positions differ by exactly 3 units in y.
    assert compute_baseline_distance(f1, f2) == pytest.approx(3.0)


def test_pose_6dof_round_trip():
    T = yaw_pose(0.7, (1.0, 2.0, 3.0))
    frame = make_frame(T, K)

    frame.pose_6dof = frame.pose_6dof  # getter -> setter round trip
    assert np.allclose(frame.pose, T, atol=1e-9)


def test_reprojection_error_zero_for_consistent_observation():
    T = yaw_pose(0.25, (0.5, 0.0, 0.0))
    frame = make_frame(T, K)
    X = np.array([0.2, -0.1, 3.0])
    uv, _ = project_gt(T, X, K)

    assert frame.compute_reprojection_error(X, uv) == pytest.approx(0.0, abs=1e-6)
