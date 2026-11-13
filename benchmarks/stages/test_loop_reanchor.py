"""Stage test: the loop-correction point re-anchor.

Pins the C3 fix (the verdict: the global BA after a correction
started from moved cameras + STALE structure and teleported
keyframes 21-160 map units per solve, both hosts; the re-anchor
gives it a consistent initialization):

- the math: every point moves by its EARLIEST observing keyframe's
  rigid correction, exactly — multi-observer points follow the
  earliest observer, not the latest; points with no live keyframe
  observer stay put
- the wrapper: optimize_with_loop_constraints re-anchors on graph
  success (the tracker's correction path inherits it)
- the kill-switch: LOOP_POINT_REANCHOR_ENABLED=0 is the pre-17
  passthrough (graph success, points untouched)
- the flag default (a measured adoption fact: 1, the A/B
  may flip it only with a recorded decision)
"""
import cv2 as cv
import numpy as np

import slam.loop_closure as lc
from benchmarks.stages.helpers import make_frame, yaw_pose
from slam.core.map import Map
from slam.core.observation import Observation
from slam.core.point import Point
from slam.loop_closure import PoseGraphConstraint

K = np.array([[450.0, 0.0, 320.0],
              [0.0, 450.0, 240.0],
              [0.0, 0.0, 1.0]])


def _reanchor_fixture():
    """Three insertion-ordered keyframes with DISTINCT known
    corrections, and points covering every observation pattern:
    observed by the middle KF only, by middle+last (earliest =
    middle), by the last only, by the first only, and by NO
    keyframe (must stay put)."""
    m = Map()
    kfs = [make_frame(yaw_pose(0.0 * i, [i, 0.2 * i, 0.0]), K)
           for i in range(3)]
    # observation attachment reads frame.keypoints[kp_idx] eagerly
    # (add_point_observation) — dummy features suffice for the
    # re-anchor math (nothing reads the pixels)
    dummy_kps = [cv.KeyPoint(100.0, 100.0, 20), cv.KeyPoint(200.0, 200.0, 20)]
    dummy_descs = np.zeros((2, 32), dtype=np.uint8)
    for kf in kfs:
        kf.set_features(dummy_kps, dummy_descs)
        m.add_keyframe(kf)

    X = {"mid_only": np.array([0.3, -0.2, 4.0]),
         "mid_and_last": np.array([-0.5, 0.4, 3.0]),
         "last_only": np.array([0.1, 0.9, 2.5]),
         "first_only": np.array([-0.7, -0.3, 5.0]),
         "no_kf": np.array([9.9, 9.9, 9.9])}
    points = {k: Point(v.copy(), None, None) for k, v in X.items()}
    m.add_points(list(points.values()))

    def _obs(kf_key, pt_key, kp_idx):
        pt = points[pt_key]
        pt.add_observation(Observation(kfs[kf_key], pt, kp_idx))

    _obs(1, "mid_only", 0)
    _obs(1, "mid_and_last", 0)
    _obs(2, "mid_and_last", 1)
    _obs(2, "last_only", 0)
    _obs(0, "first_only", 0)

    # DISTINCT known corrections per keyframe (rotation + translation
    # — genuinely SE3, not a bare shift), applied WORLD-frame
    # (kf.pose' = D @ kf.pose) so the re-anchored point expectation
    # is exactly point' = D @ point.
    deltas = {i: yaw_pose(0.11 * (i + 1), [0.3 * i, -0.25, 0.4 * i])
              for i in range(3)}
    pre = {kf.frame_id: kf.pose.copy() for kf in kfs}
    return m, kfs, points, deltas, pre


def test_reanchor_moves_points_with_earliest_observer():
    """THE math pin: point' = (post @ inv(pre)) @ point of the point's
    EARLIEST observing keyframe — multi-observer points follow the
    earliest, not the latest; unobserved-by-KF points stay put."""
    m, kfs, points, deltas, pre = _reanchor_fixture()
    X_orig = {n: p.pt_3d.copy() for n, p in points.items()}

    # the graph's write-back: each KF moves by its own correction
    for i, kf in enumerate(kfs):
        kf.pose = deltas[i] @ kf.pose

    m.pose_graph_optimizer._reanchor_points_with_correction(pre)

    expected = {"mid_only": deltas[1],       # only observer: kf1
                "mid_and_last": deltas[1],   # earliest of {1,2} = kf1
                "last_only": deltas[2],       # only observer: kf2
                "first_only": deltas[0],      # only observer: kf0
                "no_kf": None}                # no live observer
    for name, d in expected.items():
        got = points[name].pt_3d
        if d is None:
            assert np.allclose(got, X_orig[name]), \
                f"{name}: no live observer must leave the point unmoved"
        else:
            want = d[:3, :3] @ X_orig[name] + d[:3, 3]
            assert np.allclose(got, want, atol=1e-12), \
                f"{name}: {got} != {want}"


def test_reanchor_wrapper_applies_on_graph_success(monkeypatch):
    """The tracker's correction path inherits the re-anchor:
    optimize_with_loop_constraints re-anchors the points after the
    g2o pose graph succeeds. (The flag is pinned ON for this pin —
    the shipped default is 0 after the A/B rejection; the
    machinery is probed here exactly as the A/B probes pin it.)"""
    m, kfs, points, deltas, _ = _reanchor_fixture()

    def fake_pg(constraints):
        for i, kf in enumerate(kfs):
            kf.pose = deltas[i] @ kf.pose
        return True

    monkeypatch.setattr(lc, "LOOP_POINT_REANCHOR_ENABLED", 1)
    monkeypatch.setattr(m.g2o_optimizer, "pose_graph_optimization",
                        fake_pg)

    X_mid = points["mid_only"].pt_3d.copy()
    ok = m.pose_graph_optimizer.optimize_with_loop_constraints([
        PoseGraphConstraint(kfs[2].frame_id, kfs[0].frame_id,
                           np.eye(4))])
    assert ok
    want = (deltas[1][:3, :3] @ X_mid + deltas[1][:3, 3])
    assert np.allclose(points["mid_only"].pt_3d, want, atol=1e-12), \
        "the wrapper must re-anchor after a successful graph"


def test_reanchor_flag_off_is_passthrough(monkeypatch):
    """Kill-switch transparency: LOOP_POINT_REANCHOR_ENABLED=0
    reproduces the pre-17 behavior EXACTLY — the graph still moves
    the keyframes, the points are untouched."""
    m, kfs, points, deltas, _ = _reanchor_fixture()
    pts_before = {n: p.pt_3d.copy() for n, p in points.items()}
    kf_before = [kf.pose.copy() for kf in kfs]

    def fake_pg(constraints):
        for i, kf in enumerate(kfs):
            kf.pose = deltas[i] @ kf.pose
        return True

    monkeypatch.setattr(m.g2o_optimizer, "pose_graph_optimization",
                        fake_pg)
    monkeypatch.setattr(lc, "LOOP_POINT_REANCHOR_ENABLED", 0)

    ok = m.pose_graph_optimizer.optimize_with_loop_constraints([
        PoseGraphConstraint(kfs[2].frame_id, kfs[0].frame_id,
                           np.eye(4))])
    assert ok
    # the graph moved the keyframes...
    assert any(not np.allclose(kf.pose, kf_before[i])
               for i, kf in enumerate(kfs)), \
        "the fake graph must move the keyframes for this pin to mean"
    # ...and the points did NOT move (the pre-17 passthrough)
    for n, p in points.items():
        assert np.allclose(p.pt_3d, pts_before[n]), \
            f"flag OFF must leave points untouched: {n}"


def test_reanchor_flag_default_pinned():
    """The measured adoption fact: REJECTED
    standalone — all 8 contract legs failed; the stale-gauge
    decoupling between the moved keyframes and the unmoved points
    is LOAD-BEARING (it shields the live stream from harmful
    corrections). The machinery stays OFF pending the combination
    (a corrected graph + the BA acceptance gate, then the
    re-anchor re-tried as one unit)."""
    assert lc.LOOP_POINT_REANCHOR_ENABLED == 0
