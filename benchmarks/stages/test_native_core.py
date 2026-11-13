"""[C1] the native map window pins.

The MapCore stores the BA input window with REPLACE semantics; handles
ARE the Python u64 ids (migration locked decision 3); the error
write-back must reproduce Point.update_reprojection_error's running
average bit-exactly; and Map.get_observations must leave a mirror that
equals the Python graph (the C2 fold's input is only as good as that
consistency).
"""
import numpy as np
import pytest

slam_native = pytest.importorskip("slam_native")
from slam_native import MapCore  # noqa: E402

from benchmarks.stages.helpers import (  # noqa: E402
    make_K, make_frame, make_world_points, set_keypoints, yaw_pose)
from slam.core.frame import Frame  # noqa: E402
from slam.core.map import Map  # noqa: E402
from slam.core.observation import Observation  # noqa: E402
from slam.core.point import Point  # noqa: E402


def _synthetic_window(frame_ids, point_ids, obs_fids, obs_pids,
                      obs_kidx=None, obs_norm=None, n_kp=2, desc_w=32,
                      desc_f32=False):
    F = len(frame_ids)
    N = F * n_kp
    O = len(obs_fids)
    rng = np.random.default_rng(3)
    descs = (rng.uniform(-1, 1, (N, desc_w // 4)).astype(np.float32)
             if desc_f32 else
             rng.integers(0, 256, (N, desc_w), dtype=np.uint8))
    if desc_f32:
        frame_desc = np.ascontiguousarray(descs).view(np.uint8).reshape(N, -1)
        dtype_code = 2
    else:
        frame_desc = descs
        dtype_code = 1
    obs_kidx = np.arange(O, dtype=np.int64) if obs_kidx is None else obs_kidx
    obs_norm = (np.arange(O * 2, dtype=np.float64).reshape(O, 2)
                if obs_norm is None else obs_norm)
    return dict(
        frame_ids=frame_ids,
        poses=np.stack([np.eye(4) + 0.01 * i for i in range(F)]),
        Ks=np.stack([make_K()] * F),
        k1s=np.linspace(0.0, 0.05, F),
        kp_offsets=np.arange(F + 1, dtype=np.int64) * n_kp,
        kp_pts=np.arange(N * 2, dtype=np.float64).reshape(N, 2),
        kp_pts_norm=np.arange(N * 2, dtype=np.float64).reshape(N, 2) * 0.1,
        frame_desc_bytes=frame_desc,
        frame_desc_dtype=dtype_code,
        point_ids=point_ids,
        positions=rng.uniform(-5, 5, (len(point_ids), 3)),
        point_desc_bytes=rng.integers(
            0, 256, (len(point_ids), desc_w), dtype=np.uint8),
        point_desc_dtype=1,
        point_num_obs=np.full(len(point_ids), O // max(len(point_ids), 1),
                              dtype=np.int64),
        obs_frame_ids=obs_fids,
        obs_point_ids=obs_pids,
        obs_kp_idx=obs_kidx,
        obs_pt2d_norm=obs_norm)


def test_window_round_trip_bit_exact():
    core = MapCore()
    w = _synthetic_window([1, 2, 3], [10, 11], [1, 1, 2, 3], [10, 11, 10, 11])
    core.push_window(**w)
    assert core.num_frames() == 3 and core.num_points() == 2
    assert core.num_obs() == 4
    for i, fid in enumerate([1, 2, 3]):
        np.testing.assert_array_equal(
            np.asarray(core.frame_pose(fid)).reshape(4, 4), w["poses"][i])
        np.testing.assert_array_equal(
            np.asarray(core.frame_K(fid)).reshape(3, 3), w["Ks"][i])
        assert core.frame_k1(fid) == w["k1s"][i]
    for i, pid in enumerate([10, 11]):
        np.testing.assert_array_equal(
            np.asarray(core.point_position(pid)), w["positions"][i])
    np.testing.assert_array_equal(
        np.asarray(core.window_pt2d_norm()), w["obs_pt2d_norm"])


def test_obs_order_preserved():
    core = MapCore()
    w = _synthetic_window([5, 6], [20, 21],
                          [6, 5, 6, 5, 6], [21, 20, 20, 21, 21],
                          obs_kidx=np.array([4, 0, 1, 9, 3], dtype=np.int64))
    core.push_window(**w)
    assert core.window_frame_ids() == [6, 5, 6, 5, 6]
    assert core.window_point_ids() == [21, 20, 20, 21, 21]
    assert core.window_kp_idx() == [4, 0, 1, 9, 3]


def test_replace_semantics():
    core = MapCore()
    core.push_window(**_synthetic_window([1, 2], [10], [1, 2], [10, 10]))
    core.push_window(**_synthetic_window([7], [30, 31], [7, 7], [30, 31]))
    assert core.num_frames() == 1 and core.num_points() == 2
    assert not core.has_frame(1) and not core.has_point(10)
    assert core.has_frame(7) and core.has_point(31)
    assert core.window_frame_ids() == [7, 7]


def test_rejected_push_leaves_window_intact():
    core = MapCore()
    good = _synthetic_window([1, 2], [10], [1, 2], [10, 10])
    core.push_window(**good)
    bad = _synthetic_window([1, 1], [40], [1, 1], [40, 40])  # duplicate id
    with pytest.raises(ValueError):
        core.push_window(**bad)
    assert core.num_frames() == 2 and core.num_obs() == 2
    np.testing.assert_array_equal(
        np.asarray(core.window_pt2d_norm()), good["obs_pt2d_norm"])


def test_update_point_errors_matches_point_recurrence():
    """The core's write-back must reproduce the Python running average
    bit-exactly: avg = (avg*(n-1) + err) / n with n = the point's
    GLOBAL observation count (the fold feeds per-local-obs errors)."""
    K = make_K()
    m = Map()
    frames = [make_frame(yaw_pose(0.0, (0.5 * i, 0, 0)), K) for i in range(3)]
    pts3d = make_world_points(1, z_range=(7.0, 9.0), rng_seed=11)
    pt = Point(pts3d[0].copy(), None, np.zeros(32, dtype=np.uint8))
    for i, kf in enumerate(frames):
        set_keypoints(kf, [(10.0 + i, 10.0)])
        m.add_keyframe(kf)
        pt.add_observation(Observation(kf, pt, 0))
    m.get_observations(m.keyframes)
    core = m._native_core
    assert core.has_point(pt.point_id)

    rng = np.random.default_rng(17)
    errors = rng.uniform(0.0, 3.0, 7)
    for e in errors:
        pt.update_reprojection_error(e, update_quality=False)
    core.update_point_errors([pt.point_id] * len(errors), list(errors))
    assert core.point_avg_err(pt.point_id) == pt.average_reprojection_error


def test_map_pushes_ba_window():
    """The mirror-consistency pin: get_observations leaves a native
    window that EQUALS the Python graph (structure + geometry), in the
    exact observation-list order the fold consumes."""
    K = make_K()
    m = Map()
    frames = [make_frame(yaw_pose(0.0, (0.5 * i, 0, 0)), K) for i in range(3)]
    pts3d = make_world_points(4, z_range=(7.0, 9.0), rng_seed=5)
    points = [Point(x.copy(), None, np.zeros(32, dtype=np.uint8))
              for x in pts3d]
    for i, kf in enumerate(frames):
        set_keypoints(kf, [(10.0 + i, 10.0 + j) for j in range(4)])
        m.add_keyframe(kf)
    for pi, pt in enumerate(points):
        for ki, kf in enumerate(frames):
            if (pi + ki) % 3 == 0:
                continue  # leave some observation gaps
            pt.add_observation(Observation(kf, pt, pi))
    obs = m.get_observations(m.keyframes)
    assert len(obs) >= 8

    core = m._native_core
    assert core.num_frames() == 3
    assert core.window_frame_ids() == [ob.frame_id for ob in obs]
    assert core.window_point_ids() == [ob.point_id for ob in obs]
    assert core.window_kp_idx() == [ob.kp_idx for ob in obs]
    np.testing.assert_array_equal(
        np.asarray(core.window_pt2d_norm()),
        np.array([ob.pt_2d_norm for ob in obs]))
    for kf in frames:
        assert core.has_frame(kf.frame_id)
        np.testing.assert_array_equal(
            np.asarray(core.frame_pose(kf.frame_id)).reshape(4, 4), kf.pose)
    for pt in points:
        assert core.has_point(pt.point_id)
        np.testing.assert_array_equal(
            np.asarray(core.point_position(pt.point_id)), pt.pt_3d)
        assert core.point_avg_err(pt.point_id) == \
            pt.average_reprojection_error


def test_descriptor_round_trip():
    K = make_K()
    m = Map()
    kf = make_frame(yaw_pose(0.0, (0.0, 0.0, 0.0)), K)
    set_keypoints(kf, [(10.0, 10.0), (20.0, 20.0)])
    kf.descriptors = np.random.default_rng(9).uniform(
        -1, 1, (2, 128)).astype(np.float32)
    kf.kp_pts_norm = np.zeros((2, 2))
    pt = Point(np.array([1.0, 2.0, 8.0]), None,
               np.random.default_rng(10).integers(0, 256, 32, dtype=np.uint8))
    m.add_keyframe(kf)
    pt.add_observation(Observation(kf, pt, 0))
    m.get_observations(m.keyframes)
    core = m._native_core

    dtype_code, width = core.frame_desc_meta(kf.frame_id)
    assert (dtype_code, width) == (2, 128 * 4)  # float32 raw bytes
    got = np.asarray(core.frame_descriptors(kf.frame_id)).reshape(2, -1)
    np.testing.assert_array_equal(
        got, np.ascontiguousarray(kf.descriptors).view(np.uint8))
    dtype_code, width = core.point_desc_meta(pt.point_id)
    assert (dtype_code, width) == (1, 32)
    np.testing.assert_array_equal(
        np.asarray(core.point_descriptor(pt.point_id)), pt.descriptor)


def test_identity_handles_are_u64():
    """Locked decision 3: point/frame ids are plain monotonic ints -
    the native core's handles ARE these ids, no mapping layer."""
    K = make_K()
    f1 = make_frame(yaw_pose(0.0, (0.0, 0.0, 0.0)), K)
    f2 = make_frame(yaw_pose(0.0, (0.5, 0.0, 0.0)), K)
    p1 = Point(np.array([1.0, 2.0, 8.0]))
    p2 = Point(np.array([2.0, 2.0, 8.0]))
    for x in (f1.frame_id, f2.frame_id, p1.point_id, p2.point_id):
        assert isinstance(x, int) and x > 0
    assert f2.frame_id > f1.frame_id
    assert p2.point_id > p1.point_id
    assert f1.observed_points == {}
