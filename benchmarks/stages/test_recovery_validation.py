"""Stage tests: phase 9 recovery validation (the map-anchored PnP gate).

Two layers of pins, following the established patterns:

- DETECTOR-level (the 3a loop-fixture pattern - real geometry, real
  descriptors, real deterministic PnP): a recovery proposal is
  validated against the candidate keyframe's point-side map
  observations; a genuine proposal passes, an implanted half-turn is
  rejected, and every inability to validate (sparse candidate,
  featureless frame, few Lowe matches, PnP failure, low-quality PnP)
  ACCEPTS exactly as the legacy path - the intake gate's no-basis
  principle.

- TRACKER-level (the 7a monkeypatch pattern): a rejected candidate
  returns exactly like a failed match - the recovery loop falls
  through to the next candidate, nothing is committed (rollback is
  factored out: the pose was never written, the streaks were never
  cleared, the aggressive path's remove_keyframes_after never ran).

The defect class is finding 29's: recovery accepted high-inlier E
proposals (including ~180-deg half-turns) with no geometric check and
no rollback; the per-frame PnP guard rejected the pose only AFTER
commit. The gate reuses the phase-3b reloc-grade deterministic PnP
machinery (seeded per call - cv-RNG-transparent).
"""
from types import SimpleNamespace

import cv2 as cv
import numpy as np
import pytest

import slam.loop_closure as loop_mod
from benchmarks.stages.helpers import make_frame, project_gt, set_keypoints, yaw_pose
from slam.core.map import Map
from slam.core.observation import Observation
from slam.core.point import Point
from slam.core.tracker import Tracker
import slam.core.tracker as tracker_mod
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


def _validation_fixture(n_shared=60):
    """A candidate keyframe at the origin observing a described cloud
    (point-side registry), and a current frame viewing the same cloud
    from a genuine pose - per-point DISTINCT descriptors (the 'perfect
    ORB' fixture: descriptor matching recovers exactly the right
    correspondences, so the PnP recovers the frame's TRUE pose and
    can arbitrate)."""
    rng = np.random.default_rng(9)
    m = Map()
    cloud = np.column_stack([
        rng.uniform(-1.2, 1.2, n_shared),
        rng.uniform(-0.8, 0.8, n_shared),
        rng.uniform(7.0, 9.0, n_shared)])
    descriptors = rng.integers(0, 256, size=(n_shared, 32), dtype=np.uint8)

    points = [Point(X.copy(), None, descriptors[i].copy())
              for i, X in enumerate(cloud)]
    m.add_points(points)

    def _see(frame, pose):
        """Keypoints at the TRUE projections, each point's own
        descriptor; registers point-side observations when asked."""
        uvs, keep = [], []
        for i, pt in enumerate(points):
            uv, _ = project_gt(pose, pt.pt_3d, K)
            if not (0 <= uv[0] < 640 and 0 <= uv[1] < 480):
                continue
            uvs.append(uv)
            keep.append(i)
        kps = [cv.KeyPoint(float(u), float(v), size=20) for u, v in uvs]
        frame.set_features(kps, np.asarray([descriptors[i] for i in keep]))
        return keep

    kf_pose = yaw_pose(0.0, [0.0, 0.0, 0.0])
    kf = make_frame(kf_pose, K)
    kept = _see(kf, kf_pose)
    for kp_idx, i in enumerate(kept):
        points[i].add_observation(Observation(kf, points[i], kp_idx))
    m.add_keyframe(kf)

    frame_pose = yaw_pose(0.12, [0.4, 0.1, 0.2])
    frame = make_frame(frame_pose, K)
    _see(frame, frame_pose)

    return m, kf, frame, frame_pose


# ===== detector-level pins (real geometry, real PnP) =====

def test_kill_switch_defaults_on_as_int():
    """The flag is an INT (the --override parser's bool("0") trap) and
    defaults ON; it lives on the loop_closure module until the user's
    constants sweep concludes."""
    assert loop_mod.RECOVERY_VALIDATION_ENABLED == 1


def test_genuine_proposal_passes():
    """THE core positive pin: the frame's true pose (what a genuine
    recovery proposes) agrees with the map-anchored PnP - accepted."""
    m, kf, frame, true_pose = _validation_fixture()
    assert m.loop_detector.validate_map_anchored_pose(frame, kf, true_pose)


def test_disagreement_within_tolerance_passes():
    """Proposals inside the measured agreement bound pass - genuine
    recoveries carry real pair error (RPY's post-7a worst accepted
    pair error is 3.07 deg; the bound is the loop-measured 20)."""
    m, kf, frame, true_pose = _validation_fixture()
    slightly_off = true_pose @ yaw_pose(np.radians(15.0))
    assert m.loop_detector.validate_map_anchored_pose(frame, kf, slightly_off)


def test_implanted_half_turn_is_rejected():
    """THE core negative pin (finding 29's defect class): a high-quality
    map-anchored PnP positively contradicting the proposal rejects it -
    the accepted 686/565-inlier half-turns die BEFORE commit."""
    m, kf, frame, true_pose = _validation_fixture()
    half_turn = true_pose @ yaw_pose(np.pi)
    assert not m.loop_detector.validate_map_anchored_pose(frame, kf, half_turn)


def test_sparse_candidate_cannot_validate_and_accepts():
    """Fewer described map points than LOOP_MIN_CORRESPONDENCES -> no
    basis to validate -> even the half-turn is ACCEPTED exactly as the
    legacy path (young / post-reset sparse maps keep their
    recovery)."""
    m, kf, frame, true_pose = _validation_fixture(n_shared=20)
    half_turn = true_pose @ yaw_pose(np.pi)
    assert m.loop_detector.validate_map_anchored_pose(frame, kf, half_turn)


def test_featureless_frame_cannot_validate_and_accepts():
    m, kf, frame, true_pose = _validation_fixture()
    bare = make_frame(true_pose, K)  # no features extracted
    half_turn = true_pose @ yaw_pose(np.pi)
    assert m.loop_detector.validate_map_anchored_pose(bare, kf, half_turn)


def test_pnp_failure_cannot_arbitrate_and_accepts(monkeypatch):
    m, kf, frame, true_pose = _validation_fixture()
    monkeypatch.setattr(m.loop_detector, '_pnp_ransac',
                        lambda obj, img, fx: (None, None, None))
    half_turn = true_pose @ yaw_pose(np.pi)
    assert m.loop_detector.validate_map_anchored_pose(frame, kf, half_turn)


def test_low_inlier_pnp_cannot_arbitrate_and_accepts(monkeypatch):
    m, kf, frame, true_pose = _validation_fixture()
    monkeypatch.setattr(
        m.loop_detector, '_pnp_ransac',
        lambda obj, img, fx: (np.zeros(3), np.zeros(3), np.array([], dtype=int)))
    half_turn = true_pose @ yaw_pose(np.pi)
    assert m.loop_detector.validate_map_anchored_pose(frame, kf, half_turn)


def test_flag_off_is_bit_transparent(monkeypatch):
    """Flag OFF: True without constructing a matcher or touching
    anything - the legacy path, bit-for-bit (probes patch the module
    attribute in memory for A/B)."""
    m, kf, frame, true_pose = _validation_fixture()
    monkeypatch.setattr(loop_mod, 'RECOVERY_VALIDATION_ENABLED', 0)

    def forbidden(*args, **kwargs):
        pytest.fail('flag OFF must not touch the matcher')

    monkeypatch.setattr(loop_mod, 'cv',
                        SimpleNamespace(BFMatcher=forbidden))
    half_turn = true_pose @ yaw_pose(np.pi)
    assert m.loop_detector.validate_map_anchored_pose(frame, kf, half_turn)


# ===== tracker-level pins (the 7a monkeypatch pattern) =====

def _recovery_tracker(candidate_poses):
    """A tracker whose cur_frame is a just-inserted query keyframe,
    with older candidate keyframes (insertion order = temporal order,
    newest-last). No map points: the candidate loop and the E-match
    are what matters here; validation is monkeypatched per test."""
    world = Map()
    kfs = []
    for pose in candidate_poses:
        kf = set_keypoints(make_frame(pose), [(30 + i, 30) for i in range(20)])
        world.add_keyframe(kf)
        kfs.append(kf)
    current = set_keypoints(make_frame(), [(30 + i, 30) for i in range(20)])
    world.add_keyframe(current)
    tracker = Tracker(world)
    tracker.cur_frame = current
    return tracker, kfs, current


def _accepting_match(monkeypatch, tracker, relative):
    """Accept every offered pair (inliers 20 >= MIN_RELOCALIZATION_
    INLIERS 12); records (query, reference) identity."""
    matches = [cv.DMatch(i, i, 0.) for i in range(20)]
    pairs = []

    def match(query, reference, method):
        pairs.append((query, reference))
        return matches, relative

    monkeypatch.setattr(tracker_mod, 'match_features', match)
    monkeypatch.setattr(tracker_mod, 'estimate_essential_matrix',
                        lambda a, b: (np.eye(3),
                                      np.ones((len(a), 1), dtype=np.uint8),
                                      len(a)))
    optimization_results = []
    monkeypatch.setattr(tracker.map, 'optimize',
                        lambda **kw: optimization_results.append(True))
    return pairs, optimization_results


def test_rejection_falls_through_to_the_next_candidate(monkeypatch, recorder):
    """A rejected newest candidate behaves exactly like a failed match:
    the loop continues to the next candidate, and only the VALIDATED
    acceptance commits - pose, success counter, optimization. The
    rejection event fires exactly once."""
    good = yaw_pose(0.05, (0.1, 0., 0.))
    bad = yaw_pose(0.2, (0.5, 0., 0.))
    tracker, (good_kf, bad_kf), current = _recovery_tracker([good, bad])
    relative = yaw_pose(np.radians(12.), (0.5, 0., 0.))
    pairs, optimization_results = _accepting_match(monkeypatch, tracker, relative)

    verdicts = []

    def validate(frame, kf, proposed):
        verdicts.append(kf.frame_id)
        return kf is good_kf

    monkeypatch.setattr(tracker.map.loop_detector,
                        'validate_map_anchored_pose', validate)

    assert tracker._attempt_relocalization()
    # newest candidate (bad_kf) matched first, rejected; good_kf then
    # matched and accepted
    assert pairs == [(current, bad_kf), (current, good_kf)]
    assert verdicts == [bad_kf.frame_id, good_kf.frame_id]
    np.testing.assert_allclose(current.pose, good_kf.pose @ relative,
                               atol=1e-15, rtol=0)
    assert tracker.consecutive_successful_relocalizations == 1
    assert optimization_results == [True]
    assert recorder.events.get('reloc_rejected_validation') == 1


def test_rejection_of_every_candidate_fails_recovery_without_mutation(
        monkeypatch, recorder):
    """All candidates rejected -> recovery FAILS with zero state
    change: the pose was never written, the success counter never
    incremented, the velocity model never reset. Rollback is factored
    out - there is nothing to roll back."""
    tracker, (kf,), current = _recovery_tracker(
        [yaw_pose(0.05, (0.1, 0., 0.))])
    pose_before = current.pose.copy()
    velocity_before = tracker.velocity.copy()
    pairs, optimization_results = _accepting_match(
        monkeypatch, tracker, yaw_pose(np.radians(12.), (0.5, 0., 0.)))

    monkeypatch.setattr(tracker.map.loop_detector,
                        'validate_map_anchored_pose',
                        lambda frame, kf, proposed: False)

    assert not tracker._attempt_relocalization()
    np.testing.assert_array_equal(current.pose, pose_before)
    assert tracker.consecutive_successful_relocalizations == 0
    np.testing.assert_array_equal(tracker.velocity, velocity_before)
    assert optimization_results == []
    assert recorder.events.get('reloc_rejected_validation') == 1


def test_validated_acceptance_preserves_the_legacy_commit(monkeypatch, recorder):
    """A passing validation changes NOTHING about the legacy acceptance
    flow: same candidate order, same committed pose, same counter and
    optimization. Acceptance is telemetry-silent (the 7a discipline)."""
    tracker, (older,), current = _recovery_tracker(
        [yaw_pose(0.1, (0.2, 0., 0.))])
    relative = yaw_pose(np.radians(12.), (0.5, 0., 0.))
    pairs, optimization_results = _accepting_match(monkeypatch, tracker, relative)

    calls = []
    monkeypatch.setattr(
        tracker.map.loop_detector, 'validate_map_anchored_pose',
        lambda frame, kf, proposed: calls.append((frame, kf, proposed)) or True)

    assert tracker._attempt_relocalization()
    assert pairs == [(current, older)]
    assert len(calls) == 1
    # the gate saw the exact proposal that gets committed
    assert calls[0][0] is current and calls[0][1] is older
    np.testing.assert_allclose(current.pose, older.pose @ relative,
                               atol=1e-15, rtol=0)
    assert tracker.consecutive_successful_relocalizations == 1
    np.testing.assert_array_equal(tracker.velocity, np.eye(4))
    assert optimization_results == [True]
    # acceptance adds NOTHING beyond the legacy recovery events
    assert dict(recorder.events) == {'reloc_attempt': 1, 'reloc_success': 1}


def test_aggressive_path_drops_the_branch_only_on_validated_acceptance(
        monkeypatch, recorder):
    """The aggressive path's destructive remove_keyframes_after runs
    only on the VALIDATED acceptance: the rejected newest candidate
    never defines the cut - the accepted older reference does (and the
    post-reference branch, including the just-inserted query itself,
    is dropped exactly as 7a pinned)."""
    good = yaw_pose(0.05, (0.1, 0., 0.))
    bad = yaw_pose(0.2, (0.5, 0., 0.))
    tracker, (good_kf, bad_kf), current = _recovery_tracker([good, bad])
    pairs, _ = _accepting_match(monkeypatch, tracker, yaw_pose(0.))

    verdicts = []

    def validate(frame, kf, proposed):
        verdicts.append(kf.frame_id)
        return kf is good_kf

    monkeypatch.setattr(tracker.map.loop_detector,
                        'validate_map_anchored_pose', validate)

    assert tracker._attempt_aggressive_relocalization()
    assert verdicts == [bad_kf.frame_id, good_kf.frame_id]
    # the branch was dropped at the ACCEPTED reference, not the rejected
    # candidate: everything after good_kf is gone (bad_kf + the query)
    assert [kf for kf in tracker.map.keyframes] == [good_kf]
    assert current.frame_id not in tracker.map.keyframes_by_id
    assert bad_kf.frame_id not in tracker.map.keyframes_by_id
    assert recorder.events.get('reloc_rejected_validation') == 1


def test_flag_off_reproduces_legacy_acceptance_at_the_tracker(monkeypatch,
                                                               recorder):
    """Flag OFF: the half-turn proposal that finding 29 measured being
    accepted (and only later PnP-rejected) is accepted again - the
    pinned demonstration of what the gate closes, and the A/B arm.
    The REAL validation method runs (not monkeypatched) with the
    matcher namespace poisoned: the flag must be consulted before any
    validation work - bit-transparent OFF."""
    tracker, (older,), current = _recovery_tracker(
        [yaw_pose(0.1, (0.2, 0., 0.))])
    relative = yaw_pose(np.pi)  # the half-turn
    pairs, _ = _accepting_match(monkeypatch, tracker, relative)
    monkeypatch.setattr(loop_mod, 'RECOVERY_VALIDATION_ENABLED', 0)

    def forbidden(*args, **kwargs):
        pytest.fail('flag OFF must not touch the matcher')

    monkeypatch.setattr(loop_mod, 'cv',
                        SimpleNamespace(BFMatcher=forbidden))

    assert tracker._attempt_relocalization()
    assert pairs == [(current, older)]
    np.testing.assert_allclose(current.pose, older.pose @ relative,
                               atol=1e-15, rtol=0)
    # the legacy event set only - no validation events on acceptance
    assert dict(recorder.events) == {'reloc_attempt': 1, 'reloc_success': 1}
