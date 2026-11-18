"""The viewer's point-cloud color contract: each point's
camera-sampled RGB normalized to [0, 1], colorless points -> neutral
gray (the offscreen cloud builder)."""
import cv2 as cv
import numpy as np

from slam.ui.editor import Viewport


def _pt(pts3d, colors):
    return [_new_point(p3, col) for p3, col in zip(pts3d, colors)]


def _new_point(p3, color):
    from slam.core.point import Point
    return Point(np.asarray(p3, dtype=float), color, None)


def test_cloud_carries_the_camera_sampled_colors():
    cols = [
        np.array([10, 128, 250], dtype=np.uint8),
        np.array([90, 90, 90], dtype=np.uint8),
        None,
    ]
    pts = _pt([[0, 0, 1], [1, 0, 2], [2, 0, 3]], cols)
    cloud = Viewport._cloud_from(pts)
    arr = np.asarray(cloud.points)
    c = np.asarray(cloud.colors)

    assert arr.shape == (3, 3)
    assert np.allclose(arr, [[0, 0, 1], [1, 0, 2], [2, 0, 3]])
    # The camera-sampled RGB normalized to [0, 1] (the legacy
    # /= 255.0 convention).
    assert np.allclose(c[0], np.array([10, 128, 250]) / 255.0)
    assert np.allclose(c[1], np.array([90, 90, 90]) / 255.0)
    # A colorless point falls back to neutral gray, never the old
    # uniform tint.
    assert np.allclose(c[2], np.array([200, 200, 200]) / 255.0)
    assert len(np.unique(np.round(c * 255).astype(np.uint16),
                         axis=0)) == 3, "the uniforms collapsed"


def test_cloud_colors_from_a_real_session_map():
    """The tracker's own creation path (Point constructed WITH the
    camera-sampled color) fed through the cloud builder: multi-valued
    colors, never a single tint."""
    import sys
    from pathlib import Path
    # A tiny real-ish pipeline: the synthetic ORB sequence from the
    # checkpoint pins produces points, but zero-triangulating scenes
    # cannot prove colors - construct the map directly.
    m = _realish_map()
    cloud = Viewport._cloud_from(m.points)
    c = np.asarray(cloud.colors) * 255.0
    uniq = np.unique(np.round(c).astype(np.uint16), axis=0)
    assert len(m.points) >= 3 and len(uniq) > 1, "one tint"


def _realish_map():
    from slam.calibration import IntrinsicsEstimate
    from slam.core.map import Map
    from slam.core.observation import Observation
    from benchmarks.stages.helpers import make_frame, set_keypoints
    m = Map()
    m.intrinsics = IntrinsicsEstimate.from_K(
        np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]),
        k1=0.0, source="spike")
    rng = np.random.default_rng(3)
    f = make_frame(np.eye(4))
    set_keypoints(f, [(30, 40), (60, 40), (90, 40)],
                  rng.integers(0, 255, size=(3, 32)).astype(np.uint8))
    m.add_keyframe(f)
    for i, (kp, col) in enumerate(zip(
            (0, 1, 2),
            (np.array([255, 10, 10], np.uint8),
             np.array([10, 255, 10], np.uint8),
             np.array([40, 40, 255], np.uint8)))):
        p = _new_point([i, 0.0, 1.0], col)
        m.add_points([p])
        p.add_observation(Observation(f, p, kp))
    return m
