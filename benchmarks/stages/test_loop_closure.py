"""Stage test: phase 3 loop closure (3a: the machinery repair).

Homes the pins for the repaired constraint machinery:
- the reloc-grade PnP constraint (the metric fix for the milestone-3
  scale bug): recovers the TRUE relative pose from a ground-truth
  revisit fixture, INDEPENDENT of the query KF's (deliberately
  drifted) pose attribute
- the feature-db similarity separation (revisit > threshold >
  spurious) and the keyframe-exclusion window
- the spurious-candidate rejection (no shared map points -> the
  correspondence gate)

The pipeline wiring is 3b (telemetry-only first); the pose-graph
correction and its drifted-chain fixture are 3c. The DEFAULT
pipeline is untouched by this item (the tracker hook stays disabled).
"""
import cv2 as cv
import numpy as np
import pytest

import slam.utils.constants as const
from benchmarks.stages.helpers import make_frame, project_gt, yaw_pose
from slam.core.map import Map
from slam.core.observation import Observation
from slam.core.point import Point
from slam.core.tracker import Tracker
from slam.loop_closure import LoopClosureCandidate
from slam.utils import telemetry

K = np.array([[450.0, 0.0, 320.0],
              [0.0, 450.0, 240.0],
              [0.0, 0.0, 1.0]])


def _loop_fixture(n_shared=60):
    """A ground-truth REVISIT map (the 3a pin fixture):

    - a match keyframe and a query keyframe viewing the SAME cloud
      from a known relative pose (the revisit), the query KF's pose
      attribute deliberately DRIFTED (3.5+ units / 20 deg off truth -
      the constraint must be independent of it);
    - a spurious keyframe viewing a DISJOINT cloud behind it (the
      negative candidate - no shared map points, low similarity).

    Keypoints sit at the TRUE projections with per-point DISTINCT
    descriptors (identical across the keyframes that see the point -
    the 'perfect ORB' fixture), so descriptor matching recovers
    exactly the right correspondences. Only the match keyframe
    carries map observations (the constraint's map points are the
    match side by construction); the query is a bare keyframe.
    """
    rng = np.random.default_rng(5)
    m = Map()

    # Cloud A: in front of the origin cameras (the codebase convention
    # - the generator's level cameras at yaw 0 look along +z).
    cloud_a = np.column_stack([
        rng.uniform(-1.2, 1.2, n_shared),
        rng.uniform(-0.8, 0.8, n_shared),
        rng.uniform(7.0, 9.0, n_shared)])
    # Cloud B: behind, for the spurious keyframe (yaw pi).
    cloud_b = np.column_stack([
        rng.uniform(-1.2, 1.2, n_shared),
        rng.uniform(-0.8, 0.8, n_shared),
        rng.uniform(-9.0, -7.0, n_shared)])

    descriptors = rng.integers(0, 256, size=(2 * n_shared, 32),
                               dtype=np.uint8)

    points = {}
    pts = []
    for i, X in enumerate(cloud_a):
        p = Point(X.copy(), None, descriptors[i].copy())
        points[("a", i)] = p
        pts.append(p)
    for i, X in enumerate(cloud_b):
        p = Point(X.copy(), None, descriptors[n_shared + i].copy())
        points[("b", i)] = p
        pts.append(p)
    m.add_points(pts)

    def _see(frame, keys):
        """Attach keypoints at the TRUE projections of the given
        points, with each point's own descriptor."""
        uvs = []
        keep = []
        for key in keys:
            X = points[key].pt_3d
            uv, Xc = project_gt(frame.pose, X, K)
            if not (0 <= uv[0] < 640 and 0 <= uv[1] < 480):
                continue
            uvs.append(uv)
            keep.append(key)
        descs = np.asarray([points[k].descriptor for k in keep])
        kps = [cv.KeyPoint(float(u), float(v), size=20) for u, v in uvs]
        frame.set_features(kps, descs)
        return keep

    # The match keyframe at the origin: sees cloud A.
    match_kf = make_frame(yaw_pose(0.0, [0.0, 0.0, 0.0]), K)
    kept = _see(match_kf, [("a", i) for i in range(n_shared)])
    for kp_idx, key in enumerate(kept):
        pt = points[key]
        pt.add_observation(Observation(match_kf, pt, kp_idx))
    m.add_keyframe(match_kf)

    # The spurious keyframe: yaw pi, sees cloud B only.
    spurious_kf = make_frame(yaw_pose(np.pi, [0.0, 0.0, 0.5]), K)
    _see(spurious_kf, [("b", i) for i in range(n_shared)])
    m.add_keyframe(spurious_kf)

    # The query keyframe: the REVISIT of the match pose, at a known
    # relative pose. Its pose ATTRIBUTE is drifted far off truth (the
    # independence pin); the true pose is returned in info.
    true_pose = yaw_pose(0.17, [1.4, 0.25, 0.7])
    query_kf = make_frame(true_pose, K)
    _see(query_kf, [("a", i) for i in range(n_shared)])
    drift = yaw_pose(0.60, [3.4, -2.1, 2.6])
    query_kf.pose = drift
    m.add_keyframe(query_kf)

    info = {"query_true_pose": true_pose, "drift": drift}
    frames = {"match": match_kf, "spurious": spurious_kf, "query": query_kf}
    return m, frames, info


def _rot_deg(T_a, T_b):
    dR = T_a[:3, :3].T @ T_b[:3, :3]
    return np.degrees(np.arccos(
        np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))


def test_constraint_recovers_true_relative_pose():
    """3a, THE core pin: the PnP loop constraint recovers the TRUE
    relative pose of a ground-truth revisit - METRIC (the milestone-3
    E-matrix edge carried unit-scale translation; the scale bug this
    item exists to fix) and INDEPENDENT of the query KF's drifted pose
    attribute (the PnP is unseeded)."""
    m, f, info = _loop_fixture()
    T_true = info["query_true_pose"] @ np.linalg.inv(f["match"].pose)

    T_rel = m.loop_detector._estimate_loop_constraint(f["query"],
                                                       f["match"])

    assert T_rel is not None, "the true revisit must verify"
    assert _rot_deg(T_true, T_rel) < 0.5, \
        f"rotation off by {_rot_deg(T_true, T_rel):.3f} deg"
    assert np.linalg.norm(T_rel[:3, 3] - T_true[:3, 3]) < 5e-3, \
        f"translation off by {np.linalg.norm(T_rel[:3, 3] - T_true[:3, 3])}"

    # Independence: the drifted attribute is FAR from the estimate (a
    # seed-following solver would land there and fail above).
    drifted_rel = info["drift"] @ np.linalg.inv(f["match"].pose)
    assert np.linalg.norm(drifted_rel[:3, 3] - T_true[:3, 3]) > 3.0
    assert _rot_deg(drifted_rel, T_true) > 15.0


def test_verified_candidate_carries_pose_graph_measurement():
    """3a: the verified candidate's relative_pose IS the pose-graph
    edge measurement M(query, match) = pose_q_pnp @ inv(pose_match)
    (the milestone-3 code passed a CAMERA-FRAME Rt - a convention bug
    on top of the scale bug). With the fixture's drift-free match
    side, the measurement equals the true relative pose - and verify
    round-trips the constraint into the candidate."""
    m, f, info = _loop_fixture()
    cand = LoopClosureCandidate(
        query_keyframe_id=f["query"].frame_id,
        match_keyframe_id=f["match"].frame_id,
        similarity_score=0.9)

    verified = m.loop_detector.verify_loop_candidates([cand])

    assert len(verified) == 1
    T_true = info["query_true_pose"] @ np.linalg.inv(f["match"].pose)
    assert verified[0].geometric_verified
    assert _rot_deg(T_true, verified[0].relative_pose) < 0.5
    assert np.linalg.norm(
        verified[0].relative_pose[:3, 3] - T_true[:3, 3]) < 5e-3


def test_spurious_candidate_rejected():
    """3a: the negative candidate (a keyframe viewing a DISJOINT
    cloud) never verifies - the correspondence gate rejects it before
    any pose is estimated (no shared map points -> the Lowe ratio test
    leaves few random correspondences)."""
    m, f, _ = _loop_fixture()
    telemetry.set_recorder(telemetry.TelemetryRecorder())

    cand = LoopClosureCandidate(
        query_keyframe_id=f["query"].frame_id,
        match_keyframe_id=f["spurious"].frame_id,
        similarity_score=0.5)
    verified = m.loop_detector.verify_loop_candidates([cand])

    assert verified == []
    # The rejection reason is telemetry-distinguishable (3b's
    # calibration data).
    rec = telemetry._recorder
    telemetry.set_recorder(None)
    assert rec.events.get("loop_rejected_correspondences", 0) >= 1


def test_similarity_separates_revisit_from_spurious():
    """3a: the DB similarity score ORDERS the regimes: the revisit
    (identical descriptors) scores ~1.0, well above the threshold, and
    the spurious keyframe scores far below it.

    NOTE the fixture's honest limit: uniformly-random descriptors
    cross-check-match ~30% of features between UNRELATED sets (mutual
    nearest neighbors happen that often in iid data), so a fixture can
    prove the ORDERING and the revisit gate-clearing, not the exact
    threshold; real ORB descriptor structure owns the threshold
    calibration, which 3b measures on orbit/tum telemetry."""
    m, f, _ = _loop_fixture()
    db = m.feature_db
    q_desc = f["query"].descriptors

    s_match = db._compute_similarity_score(
        q_desc, db.get_keyframe_features(f["match"].frame_id))
    s_spur = db._compute_similarity_score(
        q_desc, db.get_keyframe_features(f["spurious"].frame_id))

    assert s_match > const.LOOP_SIMILARITY_THRESHOLD, \
        f"the revisit must clear the gate (got {s_match:.3f})"
    assert s_spur < 0.6 * s_match, \
        f"no revisit/spurious separation: revisit={s_match:.3f} " \
        f"spurious={s_spur:.3f}"


def test_exclusion_window_and_no_self_candidate(monkeypatch):
    """3a: the exclusion window is in KEYFRAMES (the milestone-3
    frames/keyframes inconsistency, resolved), the window collapsed
    to 0 excludes NOTHING (the [:-0] empty-slice latent bug, caught
    by this pin), and the query keyframe is never its own candidate
    (the pipeline inserts it into the DB before detection runs).
    Candidates the similarity gate passes are SUPPOSED to include the
    spurious keyframe here - killing it is VERIFICATION's job (pinned
    separately); detection only orders and gates by similarity."""
    m, f, _ = _loop_fixture()

    # Default window (20) excludes everything in a 3-KF map.
    assert m.loop_detector.detect_potential_loops(f["query"]) == []

    # Window = 2: only the match keyframe is eligible (the window
    # excludes the spurious AND the query by DB position).
    monkeypatch.setattr(const, "LOOP_EXCLUDE_RECENT_KF", 2)
    ids = {c.match_keyframe_id
           for c in m.loop_detector.detect_potential_loops(f["query"])}
    assert ids == {f["match"].frame_id}

    # Window = 1: the spurious keyframe becomes eligible too; the
    # QUERY (the most recent) is excluded - and even when it is NOT
    # (window 0), the detector skips the self-candidate.
    monkeypatch.setattr(const, "LOOP_EXCLUDE_RECENT_KF", 1)
    ids = {c.match_keyframe_id
           for c in m.loop_detector.detect_potential_loops(f["query"])}
    assert f["match"].frame_id in ids
    assert f["query"].frame_id not in ids

    monkeypatch.setattr(const, "LOOP_EXCLUDE_RECENT_KF", 0)
    ids = {c.match_keyframe_id
           for c in m.loop_detector.detect_potential_loops(f["query"])}
    assert f["query"].frame_id not in ids, "never a self-candidate"
    assert f["match"].frame_id in ids


def test_tracker_hook_is_telemetry_only(monkeypatch):
    """3b: the tracker's detection hook (keyframe cadence - the query
    is the freshly inserted cur_keyframe) runs detection+verification
    and STOPS: TELEMETRY-ONLY by construction (the correction path
    lands in 3c). With a verified loop in hand, nothing numeric moves
    - the query KF's (drifted) pose attribute and every map keyframe
    pose are untouched."""
    m, f, info = _loop_fixture()
    monkeypatch.setattr(const, "LOOP_EXCLUDE_RECENT_KF", 0)
    tracker = Tracker(m)
    assert m.cur_keyframe is f["query"], "the fixture's query is last"

    poses_before = [kf.pose.copy() for kf in m.keyframes]
    pts_before = [p.pt_3d.copy() for p in m.points]
    rec = telemetry.TelemetryRecorder()
    telemetry.set_recorder(rec)
    try:
        tracker._handle_loop_closure_detection()
    finally:
        telemetry.set_recorder(None)

    assert rec.events.get("loop_detected", 0) >= 1
    assert rec.events.get("loop_verified", 0) >= 1, \
        "the true revisit must verify through the hook"
    # Telemetry-only: nothing moved (the query pose stays DRIFTED -
    # the correction is 3c's to land).
    assert np.array_equal(f["query"].pose, info["drift"])
    for kf, pose in zip(m.keyframes, poses_before):
        assert np.array_equal(kf.pose, pose)
    for p, pt in zip(m.points, pts_before):
        assert np.array_equal(p.pt_3d, pt)


def test_tracker_hook_disabled_by_flag(monkeypatch):
    """3b: LOOP_CLOSURE_ENABLED=0 (an INT flag - the --override parser's
    bool("0") trap) short-circuits the hook BEFORE any detection - no
    telemetry, no matching, no map access."""
    m, f, _ = _loop_fixture()
    monkeypatch.setattr(const, "LOOP_EXCLUDE_RECENT_KF", 0)
    monkeypatch.setattr(const, "LOOP_CLOSURE_ENABLED", 0)
    tracker = Tracker(m)

    rec = telemetry.TelemetryRecorder()
    telemetry.set_recorder(rec)
    try:
        tracker._handle_loop_closure_detection()
    finally:
        telemetry.set_recorder(None)

    assert rec.events == {}
