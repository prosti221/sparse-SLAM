"""Stage tests: archive-and-re-anchor (reset without amnesia).

The forced-reset probe measured what a reset destroys on this
codebase (a real mapped world - 4 KFs / 508 points in the controlled
case, 52-236 KFs on natural video) and measured the re-anchor
opportunity (600-1200 good descriptor matches per replay frame
against the destroyed world). These pins hold the machinery that
keeps that world:

- the ARCHIVE: _reset_system freezes the pre-reset map (poses, points,
  descriptors, observations) before the clear; the about-to-be-origin
  frame is excluded (its pose is overwritten to identity and the
  live map keeps mutating it - the archive must not alias it);
- the RECOVERY: archived keyframes are the last resort, verified by
  the loop-grade PnP stack (metric in the archive frame - the PnP
  pose IS the estimate; no unit-scale E-chain proposal exists);
- the RE-ANCHOR: a verified pose rigidly moves the ENTIRE live map
  into the archive's world frame (invalidating the persistent BA
  graph BEFORE the write - external pose/point mutations must never
  mix with the persistent graph's retained estimates), preserving
  every relative pose and the live map's internal scale;
- the SILENCE contract: no archive exists on any run that never
  reset (the whole curated battery), and the recovery returns before
  any event or work there.
"""
import cv2 as cv
import numpy as np
import pytest

import slam.loop_closure as loop_mod
from benchmarks.stages.helpers import make_frame, project_gt, yaw_pose
from slam.core.map import Map
from slam.core.observation import Observation
from slam.core.point import Point
from slam.core.tracker import Tracker
from slam.utils import telemetry

K = np.array([[450.0, 0.0, 320.0],
              [0.0, 450.0, 240.0],
              [0.0, 0.0, 1.0]])


def _rot_deg(T_a, T_b):
    dR = T_a[:3, :3].T @ T_b[:3, :3]
    return float(np.degrees(np.arccos(
        np.clip((np.trace(dR) - 1.) / 2., -1., 1.))))


@pytest.fixture
def recorder():
    rec = telemetry.TelemetryRecorder()
    telemetry.set_recorder(rec)
    yield rec
    telemetry.set_recorder(None)


def _archived_world(n_shared=60, seed=9):
    """A map whose single keyframe (at the origin) observes a described
    cloud (point-side registry) - the world a reset would destroy.
    Returns (map, kf, points, true_pose_of_the_revisit_view)."""
    rng = np.random.default_rng(seed)
    m = Map()
    cloud = np.column_stack([
        rng.uniform(-1.2, 1.2, n_shared),
        rng.uniform(-0.8, 0.8, n_shared),
        rng.uniform(7.0, 9.0, n_shared)])
    descriptors = rng.integers(0, 256, size=(n_shared, 32), dtype=np.uint8)

    points = [Point(X.copy(), None, descriptors[i].copy())
              for i, X in enumerate(cloud)]
    m.add_points(points)

    kf_pose = yaw_pose(0.0, [0.0, 0.0, 0.0])
    kf = make_frame(kf_pose, K)
    uvs, keep = [], []
    for i, pt in enumerate(points):
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
    return m, kf, points


def _viewer_of(points, pose, pose_attribute=None):
    """A frame viewing the given points from `pose` (keypoints at the
    TRUE projections carrying each point's own descriptor), whose
    POSE ATTRIBUTE is the LIVE-frame estimate - post-reset that is
    identity, NOT the revisit pose (which is what the archive's PnP
    is about to recover). Keeping the two apart is what makes the
    re-gauge transform non-trivial: G = pose_pnp @ inv(live_pose)."""
    frame = make_frame(pose, K)
    if pose_attribute is not None:
        frame.pose = np.asarray(pose_attribute, dtype=float)
    uvs, keep = [], []
    for i, pt in enumerate(points):
        uv, _ = project_gt(pose, pt.pt_3d, K)
        if not (0 <= uv[0] < 640 and 0 <= uv[1] < 480):
            continue
        uvs.append(uv)
        keep.append(i)
    kps = [cv.KeyPoint(float(u), float(v), size=20) for u, v in uvs]
    frame.set_features(kps, np.asarray([pt.descriptor for i, pt in
                                        enumerate(points) if i in keep]))
    return frame


REVISIT_POSE = yaw_pose(0.12, [0.4, 0.1, 0.2])


def _tracker_with_reset_world(monkeypatch):
    """A tracker whose live map is a fresh post-reset origin (the
    pre-reset world is frozen in the archive); map.optimize is
    neutralized (the 7a pattern) so on_relocalization_successful does
    not run BA over the fixture map."""
    m, kf, points = _archived_world()
    origin = make_frame()  # not a keyframe of the old map
    tracker = Tracker(m)
    tracker.cur_frame = origin
    monkeypatch.setattr(m, 'optimize', lambda **kw: None)
    tracker._reset_system()
    return tracker, m, kf, points, origin


# ===== the archive (10b) =====

def test_flags_default_on():
    assert loop_mod.RECOVERY_ARCHIVE_ENABLED == 1
    assert loop_mod.RECOVERY_ARCHIVE_SCAN_LIMIT == 40


def test_reset_archives_the_preworld():
    """A reset FREEZES the pre-reset map instead of discarding it: the
    archived keyframes and points are the SAME objects, with their
    poses, descriptors and the point-side observation registry intact
    (the cross-map PnP needs exactly this structure)."""
    m, kf, points = _archived_world()
    origin = make_frame()
    tracker = Tracker(m)
    tracker.cur_frame = origin

    tracker._reset_system()

    assert len(m.archives) == 1
    arch = m.archives[0]
    assert arch.keyframes == [kf]
    assert len(arch.points) == len(points)
    assert all(pt.is_observed_by(kf.frame_id) for pt in arch.points[:10])
    # the live map is the fresh origin only
    assert [f for f in m.keyframes] == [origin]
    np.testing.assert_array_equal(origin.pose, np.eye(4))


def test_reset_excludes_the_new_origin_from_the_archive():
    """The about-to-be-origin frame is EXCLUDED from the archive: its
    pose is overwritten to identity and the live map keeps mutating it
    - archiving it would alias an object the live pipeline owns."""
    m, kf, points = _archived_world()
    origin = make_frame()
    m.add_keyframe(origin)  # the recovery cadence: cur_frame CAN be a KF
    tracker = Tracker(m)
    tracker.cur_frame = origin

    tracker._reset_system()

    arch = m.archives[0]
    assert origin not in arch.keyframes
    assert arch.keyframes == [kf]
    assert [f for f in m.keyframes] == [origin]
    np.testing.assert_array_equal(origin.pose, np.eye(4))


def test_reset_on_a_minimal_map_archives_nothing():
    """A map whose only keyframe IS the about-to-be-origin has nothing
    to relocalize against - no archive, no crash (the reset-loop
    second cycle takes this path)."""
    m = Map()
    origin = make_frame()
    m.add_keyframe(origin)
    tracker = Tracker(m)
    tracker.cur_frame = origin
    tracker._reset_system()
    assert m.archives == []
    assert [f for f in m.keyframes] == [origin]


def test_flag_off_is_legacy_amnesia(monkeypatch):
    """Flag OFF: no archive is created at reset and the archive
    recovery returns before any work - legacy amnesia, bit-for-bit."""
    monkeypatch.setattr(loop_mod, 'RECOVERY_ARCHIVE_ENABLED', 0)
    m, kf, points = _archived_world()
    origin = make_frame()
    tracker = Tracker(m)
    tracker.cur_frame = origin
    tracker._reset_system()
    assert m.archives == []

    def forbidden(*args, **kwargs):
        pytest.fail('flag OFF must not query the archive')

    monkeypatch.setattr(m.loop_detector, 'verify_archive_recovery',
                        forbidden)
    assert not tracker._attempt_archive_relocalization()


# ===== the archive recovery (10c) =====

def test_archive_recovery_reanchors_the_pose(monkeypatch, recorder):
    """THE core pin: a post-reset frame that re-sees the archived world
    re-anchors - the committed pose is the METRIC map-anchored PnP
    pose IN THE ARCHIVE'S FRAME (the revisit's true pose), not a
    unit-scale E-chain proposal. The viewer's own pose attribute is the
    post-reset identity estimate - the PnP must recover the revisit
    pose FROM THE IMAGE ALONE."""
    tracker, m, kf, points, origin = _tracker_with_reset_world(monkeypatch)
    viewer = _viewer_of(points, REVISIT_POSE, pose_attribute=np.eye(4))
    tracker.cur_frame = viewer

    assert tracker._attempt_archive_relocalization()

    assert _rot_deg(tracker.cur_frame.pose, REVISIT_POSE) < 0.5
    assert np.linalg.norm(
        tracker.cur_frame.pose[:3, 3] - REVISIT_POSE[:3, 3]) < 5e-3
    assert tracker.consecutive_successful_relocalizations == 1
    np.testing.assert_array_equal(tracker.velocity, np.eye(4))
    # the quality-streak clear ran (on_relocalization_successful)
    assert m.consecutive_low_quality_frames == 0
    assert recorder.events.get('reloc_archive_attempt') == 1
    assert recorder.events.get('reloc_archive_success') == 1


def test_archive_recovery_preserves_rigid_relative_geometry(monkeypatch):
    """The re-anchor is a RIGID transform of the whole live map: every
    relative pose and every point's camera-frame position (against
    any live keyframe) is preserved, while absolute poses move into
    the archive's frame. The live map's internal scale is untouched
    (Sim3 stays the parked tier)."""
    tracker, m, kf, points, origin = _tracker_with_reset_world(monkeypatch)
    second = make_frame(yaw_pose(0.3, [0.5, 0.0, 0.2]))
    m.add_keyframe(second)
    live_pt = Point(np.array([1.0, 2.0, 8.0]), None, None)
    m.add_points([live_pt])

    viewer = _viewer_of(points, REVISIT_POSE, pose_attribute=np.eye(4))
    tracker.cur_frame = viewer
    rel_pose_before = np.linalg.inv(second.pose) @ viewer.pose
    X_cam_before = (np.linalg.inv(second.pose) @ np.append(live_pt.pt_3d, 1.0))[:3]

    assert tracker._attempt_archive_relocalization()

    # absolute poses moved (G is non-identity)...
    assert not np.allclose(second.pose, yaw_pose(0.3, [0.5, 0.0, 0.2]))
    # ...and every relative quantity is preserved to floating point
    rel_pose_after = np.linalg.inv(second.pose) @ viewer.pose
    np.testing.assert_allclose(rel_pose_after, rel_pose_before, atol=1e-9)
    X_cam_after = (np.linalg.inv(second.pose) @ np.append(live_pt.pt_3d, 1.0))[:3]
    np.testing.assert_allclose(X_cam_after, X_cam_before, atol=1e-9)
    # the dedup set followed the moved geometry
    assert tuple(np.round(live_pt.pt_3d, 3)) in m.point_coords


def test_archive_recovery_rejects_foreign_content(monkeypatch, recorder):
    """A frame that does not re-see any archived world finds no
    verified candidate - no re-anchor, no state change."""
    tracker, m, kf, points, origin = _tracker_with_reset_world(monkeypatch)
    # a DIFFERENT world: disjoint positions and descriptors
    other = _archived_world(seed=1234)[2]
    viewer = _viewer_of(other, REVISIT_POSE, pose_attribute=np.eye(4))
    tracker.cur_frame = viewer
    pose_before = viewer.pose.copy()

    assert not tracker._attempt_archive_relocalization()

    np.testing.assert_array_equal(viewer.pose, pose_before)
    assert tracker.consecutive_successful_relocalizations == 0
    assert recorder.events.get('reloc_archive_attempt') == 1
    assert 'reloc_archive_success' not in recorder.events


def test_rotation_cross_check_gates_the_archive_candidate(monkeypatch):
    """The archive path has no E-proposal for the disagreement
    check, so the PnP gets the loop detector's degeneracy guard
    instead: a mirrored/degenerate PnP fails the E-vs-PnP rotation
    cross-check and the candidate is skipped."""
    tracker, m, kf, points, origin = _tracker_with_reset_world(monkeypatch)
    viewer = _viewer_of(points, REVISIT_POSE, pose_attribute=np.eye(4))
    tracker.cur_frame = viewer

    monkeypatch.setattr(m.loop_detector, '_rotation_agrees',
                        lambda *a, **kw: False)
    assert not tracker._attempt_archive_relocalization()

    monkeypatch.setattr(m.loop_detector, '_rotation_agrees',
                        lambda *a, **kw: True)
    assert tracker._attempt_archive_relocalization()


def test_scan_limit_bounds_the_candidate_scan(monkeypatch, recorder):
    """A salem-scale pre-reset map holds 150+ keyframes; a recovery
    spiral must not pay for all of them every cycle. Newest-first,
    capped at RECOVERY_ARCHIVE_SCAN_LIMIT, early-out on the first
    verified match."""
    m = Map()
    kfs = []
    for i in range(50):
        kf = make_frame(yaw_pose(0.01 * i, [0.1 * i, 0.0, 0.0]))
        m.add_keyframe(kf)
        kfs.append(kf)
    m.archive_current_state()
    tracker = Tracker(m)
    tracker.cur_frame = make_frame()

    calls = []
    monkeypatch.setattr(m.loop_detector, 'verify_archive_recovery',
                        lambda frame, kf, pts: calls.append(kf) or None)

    assert not tracker._attempt_archive_relocalization()
    assert len(calls) == loop_mod.RECOVERY_ARCHIVE_SCAN_LIMIT
    # newest-first: the scan starts at the END of the archived list
    assert calls[0] is kfs[-1]
    assert recorder.events.get('reloc_archive_attempt') == 1


def test_no_archives_is_silent(recorder):
    """THE battery bit-transparency pin: on a run that never reset there
    are no archives; the recovery returns False BEFORE any event or
    work, so curated frozen-control comparisons stay exact."""
    m = Map()
    tracker = Tracker(m)
    tracker.cur_frame = make_frame()
    assert not tracker._attempt_archive_relocalization()
    assert dict(recorder.events) == {}


def test_persistent_graph_invalidated_before_the_reanchor_write(monkeypatch):
    """External-pose mutation discipline: the persistent local-BA graph's retained
    estimates must be invalidated BEFORE the external pose/point
    write - never mixed with a re-gauged map."""
    tracker, m, kf, points, origin = _tracker_with_reset_world(monkeypatch)
    second = make_frame(yaw_pose(0.3, [0.5, 0.0, 0.2]))
    m.add_keyframe(second)
    viewer = _viewer_of(points, REVISIT_POSE, pose_attribute=np.eye(4))
    tracker.cur_frame = viewer

    order = []

    real_reset = m.g2o_optimizer.reset_persistent_graph

    def recording_reset():
        order.append(('invalidate', m.keyframes[0].pose.copy()))

    monkeypatch.setattr(m.g2o_optimizer, 'reset_persistent_graph',
                        recording_reset)
    monkeypatch.setattr(m.loop_detector, 'verify_archive_recovery',
                        lambda frame, kf_, pts: order.append(('verify',)) or
                        REVISIT_POSE.copy())

    assert tracker._attempt_archive_relocalization()
    assert order[0][0] == 'verify'
    assert order[1][0] == 'invalidate'
    # the pose captured AT invalidation was still the pre-write pose
    np.testing.assert_array_equal(order[1][1], np.eye(4))
    # ...and the write happened after (the live origin moved)
    assert not np.allclose(m.keyframes[0].pose, np.eye(4))
