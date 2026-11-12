"""Stage tests: phase 20 (T3a) — the reset re-gauge continuation.

The shipped reset TELEPORTS the live trajectory to the world origin
(``cur_frame.pose = identity`` in ``Tracker._reset_system``) — the
single mechanism behind all three census SCALE_BLOWUP labels (road
|t| 382.3 -> 0 in one step, walking_woods_1 148.1 -> 0,
walking_woods_2_15 273.4 -> 0) and the reason nothing after a reset
is comparable to anything before it. These pins hold the flag and
both branches:

- the flag default 1 = the ADOPTED re-gauge continuation (a measured
  adoption fact: the phase-20 A/B flipped it 0 -> 1 with a recorded
  decision; the pre-adoption shipped default was the teleport)
- flag OFF = the EXACT shipped reset (identity pose; the archive,
  the clear, the streak resets — all unchanged)
- flag ON = RE-GAUGE CONTINUATION: the pre-reset pose is KEPT as
  the new map's anchor (the live trajectory stays in the pre-reset
  world frame; the pose-graph gauge anchors at KF0's CURRENT pose
  wherever it is; the E-chain/PnP compose relative motions; the
  harness's Umeyama is global — all verified gauge-agnostic in the
  phase-20 plan)
- the by-product: with the pose kept, the new map's frame IS the
  archive's frame, so the archive-reloc re-anchor
  (``_reanchor_map_to``) degenerates to a no-op instead of a
  second teleport (T5's mixed-gauge corruption resolved)
"""
import cv2 as cv
import numpy as np
import pytest

import slam.core.tracker as tracker_mod
from benchmarks.stages.helpers import make_frame, yaw_pose
from slam.core.map import Map
from slam.core.observation import Observation
from slam.core.point import Point
from slam.core.tracker import Tracker
from slam.utils import telemetry

K = np.array([[450.0, 0.0, 320.0],
              [0.0, 450.0, 0.0],
              [0.0, 0.0, 1.0]])


@pytest.fixture
def recorder():
    rec = telemetry.TelemetryRecorder()
    telemetry.set_recorder(rec)
    yield rec
    telemetry.set_recorder(None)


def _preworld(rng_seed=7, n_pts=40):
    """A live map with one keyframe at a NON-identity pose observing a
    described cloud, plus the frame about to anchor the new map
    carrying a NON-identity pre-reset estimate (the pose the teleport
    destroys). Returns (map, kf, points, origin, pre_origin_pose)."""
    rng = np.random.default_rng(rng_seed)
    m = Map()
    cloud = np.column_stack([
        rng.uniform(-1.0, 1.0, n_pts),
        rng.uniform(-0.7, 0.7, n_pts),
        rng.uniform(6.0, 9.0, n_pts)])
    descriptors = rng.integers(0, 256, size=(n_pts, 32), dtype=np.uint8)
    points = [Point(X.copy(), None, descriptors[i].copy())
              for i, X in enumerate(cloud)]
    m.add_points(points)

    kf_pose = yaw_pose(0.05, [0.3, -0.2, 0.1])
    kf = make_frame(kf_pose, K)
    uvs, keep = [], []
    for i, pt in enumerate(points):
        from benchmarks.stages.helpers import project_gt
        uv, _ = project_gt(kf_pose, pt.pt_3d, K)
        if not (0 <= uv[0] < 640 and 0 <= uv[1] < 480):
            continue
        uvs.append(uv)
        keep.append(i)
    kps = [cv.KeyPoint(float(u), float(v), size=20) for u, v in uvs]
    kf.set_features(kps, np.asarray([descriptors[i] for i in keep]))
    for kp_idx, i in enumerate(keep):
        points[i].add_observation(Observation(kf, points[i], kp_idx))
    m.add_keyframe(kf)

    origin = make_frame(yaw_pose(0.21, [2.5, -1.0, 0.6]), K)
    pre_pose = origin.pose.copy()
    m.add_keyframe(origin)  # the cadence: cur_frame CAN be a KF
    return m, kf, points, origin, pre_pose


def test_flag_default_pinned():
    """The ADOPTED default is the RE-GAUGE CONTINUATION (1) — the
    phase-20 A/B verdict (ADOPTED, 4/5 strict majority: woods_1,
    woods_2_15, kyoto_2_5, euroc_v1_01_easy; road the lone AGAINST on
    the new-label leg; greece bit-identical no-op guard; battery 0
    non-timing diffs; output/benchmarks/ladder/
    phase20_gauge_contract.json, 2026-11-12). The pre-adoption
    shipped default was the teleport (0)."""
    assert tracker_mod.RESET_GAUGE_CONTINUITY_ENABLED == 1


def test_flag_off_is_the_shipped_teleport(recorder):
    """Flag pinned 0: the exact shipped reset — identity pose, the
    archive frozen, the map cleared, the origin re-added as KF0, the
    streaks/velocity reset, system_reset fired."""
    m, kf, points, origin, pre_pose = _preworld()
    tracker = Tracker(m)
    tracker.cur_frame = origin
    tracker.consecutive_relocalization_failures = 2
    tracker.velocity = yaw_pose(0.1, [0.5, 0.5, 0.5])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tracker_mod, "RESET_GAUGE_CONTINUITY_ENABLED", 0)
        tracker._reset_system()

    np.testing.assert_array_equal(origin.pose, np.eye(4))
    assert len(m.archives) == 1
    assert m.archives[0].keyframes == [kf]
    assert [f for f in m.keyframes] == [origin]
    np.testing.assert_array_equal(tracker.velocity, np.eye(4))
    assert tracker.consecutive_relocalization_failures == 0
    assert recorder.events.get("system_reset") == 1


def test_flag_on_keeps_the_pregauge_pose(recorder):
    """Flag pinned 1: the pre-reset pose is KEPT — the post-reset KF0
    pose is bit-identical to the pre-reset estimate; the archive is
    frozen exactly as in the shipped path (its poses untouched, in
    the same world frame the live map now continues); the map's
    collections are cleared and re-seeded; the streaks/velocity
    reset; system_reset fired."""
    m, kf, points, origin, pre_pose = _preworld()
    tracker = Tracker(m)
    tracker.cur_frame = origin
    tracker.consecutive_relocalization_failures = 2
    tracker.velocity = yaw_pose(0.1, [0.5, 0.5, 0.5])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tracker_mod, "RESET_GAUGE_CONTINUITY_ENABLED", 1)
        tracker._reset_system()

    # the continuity: the pose is simply NOT overwritten
    np.testing.assert_array_equal(origin.pose, pre_pose)
    # the reset path's other writes are unchanged
    assert len(m.archives) == 1
    arch = m.archives[0]
    assert arch.keyframes == [kf]
    np.testing.assert_array_equal(arch.keyframes[0].pose, kf.pose)
    assert len(arch.points) == len(points)
    assert [f for f in m.keyframes] == [origin]
    np.testing.assert_array_equal(tracker.velocity, np.eye(4))
    assert tracker.consecutive_relocalization_failures == 0
    assert recorder.events.get("system_reset") == 1


def test_flag_on_live_map_and_archive_share_one_frame():
    """With the pose kept, the live map's frame and the archive's
    frame are ONE world frame: the archived keyframe's pose and the
    post-reset KF0 pose are both the pre-reset world's estimates
    (the mixed-gauge corruption T5 measured cannot arise)."""
    m, kf, points, origin, pre_pose = _preworld()
    tracker = Tracker(m)
    tracker.cur_frame = origin
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tracker_mod, "RESET_GAUGE_CONTINUITY_ENABLED", 1)
        tracker._reset_system()

    arch = m.archives[0]
    # both gauge anchors live in the pre-reset world frame W: the
    # archive's KF pose unchanged, the live KF0 pose the pre-reset
    # estimate — no transform relates them, they are the SAME frame
    np.testing.assert_array_equal(arch.keyframes[0].pose, kf.pose)
    np.testing.assert_array_equal(origin.pose, pre_pose)


def test_flag_on_archive_reanchor_degenerates_to_noop():
    """The by-product pin: with the pose kept, a verified archive
    reloc whose PnP pose IS the current estimate produces
    G = pose_pnp @ inv(cur_pose) ~= identity — the live map's poses
    and points are unchanged (the shipped path would rigidly move
    the identity-framed live map into the archive's frame)."""
    m, kf, points, origin, pre_pose = _preworld()
    tracker = Tracker(m)
    tracker.cur_frame = origin
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tracker_mod, "RESET_GAUGE_CONTINUITY_ENABLED", 1)
        tracker._reset_system()

    # a live point (the new map's first triangulation, in W)
    live_pt = Point(np.array([0.4, -0.3, 7.5]), None, None)
    m.add_points([live_pt])
    pre_kf_poses = [kf2.pose.copy() for kf2 in m.keyframes]
    pre_pt = live_pt.pt_3d.copy()

    pose_pnp = origin.pose.copy()  # the verified pose IS the estimate
    tracker._reanchor_map_to(pose_pnp)

    for kf2, want in zip(m.keyframes, pre_kf_poses):
        np.testing.assert_allclose(kf2.pose, want, atol=1e-9)
    np.testing.assert_allclose(live_pt.pt_3d, pre_pt, atol=1e-9)


def test_both_flags_fire_system_reset_telemetry(recorder):
    """The reset-path telemetry is identical in both branches (the
    event stream's shape does not depend on the gauge choice)."""
    for flag in (0, 1):
        m, kf, points, origin, pre_pose = _preworld()
        tracker = Tracker(m)
        tracker.cur_frame = origin
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(tracker_mod, "RESET_GAUGE_CONTINUITY_ENABLED", flag)
            tracker._reset_system()
    assert recorder.events.get("system_reset") == 2
