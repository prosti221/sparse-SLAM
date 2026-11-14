"""Tracker call-site outcomes and input ownership contracts."""
import numpy as np
import pytest

from benchmarks.stages.helpers import make_K, make_frame, make_world_points, project_gt, set_keypoints, yaw_pose
from slam.core.map import Map
from slam.core.observation import Observation
from slam.core.point import Point
from slam.core.tracker import Tracker
import slam.core.tracker as tracker_mod
import slam.ba.bundle_adjustment_g2o as ba_mod
from slam.utils import telemetry


@pytest.mark.parametrize("success,reported_count", [(False, 0), (False, 10), (True, 10)])
def test_tracker_uses_pnp_success_element_without_changing_pose_or_insertion_flow(
        monkeypatch, success, reported_count):
    """A failed nonempty tuple is not success; PnP owns pose write-back.

    Exercise the REAL Tracker.update, isolating matching/mapping so the
    test cannot pass just because a replacement caller unpacks correctly.
    """
    world = Map()
    previous = set_keypoints(make_frame(yaw_pose(0.1, (0.2, 0.0, 0.0))), [(30, 30)] * 10)
    world.add_keyframe(previous)
    tracker = Tracker(world)
    tracker.cur_frame = previous
    current = set_keypoints(make_frame(), [(30, 30)] * 10)
    monkeypatch.setattr(tracker.feature_extractor, "extract",
                        lambda image: current.get_keypoints_descriptors())
    relative_pose = yaw_pose(0.01, (0.1, 0.0, 0.0))
    monkeypatch.setattr(tracker_mod, "match_features_between_frames", lambda *a, **kw: relative_pose)
    observations = [Observation(current, Point(np.array([i * 0.01, 0.0, 5.0])), i)
                    for i in range(10)]
    monkeypatch.setattr(tracker, "_project_visible_map_points",
                         lambda: (np.empty((0, 2)), np.empty(0, dtype=np.int64)))
    monkeypatch.setattr(tracker, "_match_projected_points",
                         lambda uv, rows: observations)
    seed = previous.pose @ relative_pose
    refined = seed @ yaw_pose(0.005, (0.01, 0.0, 0.0))
    pnp_calls = []

    def pnp(frame):
        assert frame is current
        pnp_calls.append(frame)
        np.testing.assert_allclose(frame.pose, seed, atol=1e-15, rtol=0)
        if success:
            frame.pose = refined
        return success, reported_count

    monkeypatch.setattr(world.g2o_optimizer, "refine_pose_pnp", pnp)
    insertion_poses = []
    monkeypatch.setattr(tracker, "_handle_keyframe_insertion", lambda: insertion_poses.append(current.pose))
    monkeypatch.setattr(tracker, "_handle_relocalization_check", lambda: None)
    logs = []
    monkeypatch.setattr(tracker_mod, "debug_log", lambda tag, message: logs.append(message))
    recorder = telemetry.TelemetryRecorder()
    previous_recorder = telemetry._recorder
    telemetry.set_recorder(recorder)
    previous_pose = previous.pose
    try:
        assert not tracker.update(current)
    finally:
        telemetry.set_recorder(previous_recorder)

    expected = refined if success else seed
    assert pnp_calls == [current]
    assert recorder.events.get("pnp_frame_ok", 0) == int(success)
    assert recorder.events.get("pnp_frame_fail", 0) == int(not success)
    assert len(insertion_poses) == 1
    np.testing.assert_allclose(insertion_poses[0], expected, atol=1e-15, rtol=0)
    np.testing.assert_allclose(current.pose, expected, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(previous.pose, previous_pose)
    np.testing.assert_allclose(tracker.velocity, np.linalg.inv(previous_pose) @ expected, atol=1e-15, rtol=0)
    assert any("Pose refinement failed" in message for message in logs) == (not success)


@pytest.mark.parametrize("observation_count", [0, 9])
def test_tracker_does_not_report_unattempted_pnp(monkeypatch, observation_count):
    world = Map()
    previous = set_keypoints(make_frame(), [(30, 30)] * 10)
    world.add_keyframe(previous)
    tracker = Tracker(world)
    tracker.cur_frame = previous
    current = set_keypoints(make_frame(), [(30, 30)] * 10)
    monkeypatch.setattr(tracker.feature_extractor, "extract",
                        lambda image: current.get_keypoints_descriptors())
    monkeypatch.setattr(tracker_mod, "match_features_between_frames", lambda *a, **kw: np.eye(4))
    observations = [Observation(current, Point(np.array([i * 0.01, 0.0, 5.0])), i)
                    for i in range(observation_count)]
    monkeypatch.setattr(tracker, "_project_visible_map_points",
                         lambda: (np.empty((0, 2)), np.empty(0, dtype=np.int64)))
    monkeypatch.setattr(tracker, "_match_projected_points",
                         lambda uv, rows: observations)
    monkeypatch.setattr(tracker, "_handle_keyframe_insertion", lambda: None)
    monkeypatch.setattr(tracker, "_handle_relocalization_check", lambda: None)

    def unexpected_pnp(*args):
        pytest.fail("PnP was called below the existing observation floor")

    monkeypatch.setattr(world.g2o_optimizer, "refine_pose_pnp", unexpected_pnp)
    recorder = telemetry.TelemetryRecorder()
    previous_recorder = telemetry._recorder
    telemetry.set_recorder(recorder)
    try:
        tracker.update(current)
    finally:
        telemetry.set_recorder(previous_recorder)
    assert "pnp_frame_ok" not in recorder.events
    assert "pnp_frame_fail" not in recorder.events


def test_pnp_count_and_log_describe_inputs_not_a_robust_inlier_set(monkeypatch):
    """ITERATIVE PnP uses all inputs, even with one corrupted observation."""
    world = Map()
    frame = make_frame()
    truth = make_world_points(40)
    set_keypoints(frame, [project_gt(frame.pose, X, frame.K)[0] for X in truth])
    for i, X in enumerate(truth):
        point = Point(X)
        point.add_observation(Observation(frame, point, i))
    frame.kp_pts_norm[0] += [0.3, 0.3]
    calls = []

    def solve(points_3d, points_2d, K, **kwargs):
        calls.append((points_3d.copy(), points_2d.copy(), kwargs))
        return True, np.zeros((3, 1)), np.zeros((3, 1))

    def no_ransac(*args, **kwargs):
        pytest.fail("The reporting repair must not replace the solver or consume RANSAC RNG")

    monkeypatch.setattr(ba_mod.cv, "solvePnP", solve)
    monkeypatch.setattr(ba_mod.cv, "solvePnPRansac", no_ransac)
    logs = []
    monkeypatch.setattr(ba_mod, "debug_log", lambda tag, message: logs.append(message))

    success, count = world.g2o_optimizer.refine_pose_pnp(frame)

    assert success and count == 40
    assert len(calls) == 1 and len(calls[0][0]) == len(calls[0][1]) == 40
    assert calls[0][2]["flags"] == ba_mod.cv.SOLVEPNP_ITERATIVE
    assert calls[0][2]["useExtrinsicGuess"] is True
    assert any("40 input correspondences" in message for message in logs)
    assert not any("inliers" in message for message in logs)


# --- focal-normalized projection-matching radius -------------

def _projection_tracker(fx):
    """A tracker whose cur_frame carries the given focal (K at the
    frame, exactly as the live pipeline reads it)."""
    world = Map()
    frame = make_frame(K=make_K(fx=fx, fy=fx))
    tracker = Tracker(world)
    tracker.cur_frame = frame
    return tracker, frame


def _binary_descriptor(fill):
    return np.full(32, fill, dtype=np.uint8)


def _projection_match_setup(fx=1449.6, method="ORB", point_desc=None):
    """A tracker whose map's registry holds ONE point (the C3c index
    flow requires registered points); returns (tracker, uv, rows,
    point)."""
    world = Map()
    frame = make_frame(K=make_K(fx=fx, fy=fx))
    tracker = Tracker(world, feature_extraction_method=method)
    tracker.cur_frame = frame
    desc = point_desc if point_desc is not None else _binary_descriptor(0)
    point = Point(np.array([0.0, 0.0, 5.0]), None, desc)
    world.add_points([point])
    uv = np.array([[30.0, 30.0]])
    rows = np.array([world._registry_rows[point.point_id]], dtype=np.int64)
    return tracker, uv, rows, point

@pytest.mark.parametrize("fx,offset_px,matched", [
    (450.0, 6.0, False),       # fr/orbit/translation focal: floor keeps 5 px
    (517.306409, 6.0, False),  # tum_xyz's true focal: still 5 px
    (450.0, 4.0, True),       # the floor itself still searches 5 px
    (1449.6, 6.0, True),       # walking_salem config focal: resolves ~14 px
    (1449.6, 12.0, True),      # well inside the resolved radius
])
def test_projection_match_radius_resolves_from_the_live_focal(fx, offset_px, matched):
    """The search radius is angular (0.55 deg), resolved per
    call from the live focal. The measured-provenance contract (finding
    32): every benchmark focal (450 / 517.3) resolves BELOW the
    historical 5 px floor and stays at exactly 5 px - the frozen battery
    is bit-identical by construction - while the walking_salem config
    focal (fx=1449.6, whose per-frame E-chain noise of ~0.56 deg is a
    14 px displacement, 3x the fixed radius that collapsed its map
    anchoring) resolves to ~14 px. A keypoint 6 px from the projection
    is therefore rejected at benchmark focals and accepted at the long
    focal; 4 px is accepted everywhere (the floor is intact)."""
    tracker, uv, rows, _point = _projection_match_setup(fx)
    set_keypoints(tracker.cur_frame, [(30.0 + offset_px, 30.0)],
                  np.zeros((1, 32), dtype=np.uint8))
    observations = tracker._match_projected_points(uv, rows)
    assert (len(observations) == 1) == matched


def test_projection_match_still_requires_descriptor_agreement_at_wide_radius():
    """The radius only widens the SEARCH. Acceptance still
    requires the <20 Hamming gate (random-pair distance ~128 for ORB),
    so the wide radius cannot admit accidental keypoint collisions."""
    tracker, uv, rows, _point = _projection_match_setup(1449.6)
    set_keypoints(tracker.cur_frame, [(30.0, 30.0)],
                  np.full((1, 32), 255, np.uint8))
    assert tracker._match_projected_points(uv, rows) == []


def test_projection_match_explicit_radius_override_still_honored():
    """An explicit dist_thresh (the probe/diagnostic path)
    overrides the focal resolution unchanged."""
    tracker, uv, rows, _point = _projection_match_setup(1449.6)
    set_keypoints(tracker.cur_frame, [(36.0, 30.0)],
                  np.zeros((1, 32), dtype=np.uint8))
    assert tracker._match_projected_points(uv, rows, dist_thresh=5) == []
    assert len(tracker._match_projected_points(uv, rows, dist_thresh=14)) == 1


def _projection_match_fixture(descriptors, kp_positions, uv=(30.0, 30.0)):
    """A tracker whose frame carries the given keypoints/descriptors and
    one map point at the fixed projection."""
    world = Map()
    frame = make_frame(K=make_K(fx=1449.6, fy=1449.6))
    tracker = Tracker(world)
    tracker.cur_frame = frame
    set_keypoints(frame, kp_positions, descriptors)
    point = Point(np.array([0.0, 0.0, 5.0]), None, _binary_descriptor(0))
    return tracker, [(point, np.array(uv))], point


def test_projection_match_hamming_threshold_is_strict():
    """[C3a] the descriptor accept is STRICTLY below the reloc
    threshold (20 for binary): distance 19 matches, 20 does not - the
    knife-edge pair regime the route-B port must preserve. The pair is
    crafted so the KIND DISPATCH is bound: a 19-bit Hamming distance is
    ACCEPTED under kind=1 but the same bytes' L2-on-reinterpreted-float
    distance is ~2.0 (far above the float threshold 0.8639) - so a
    kind=2 dispatch produces ZERO matches here, making this pin fail
    loudly on a kind-map regression (measured: the 'binary'/'HAMMING'
    literal mismatch dispatched ORB to L2 and the battery collapsed)."""
    # d_popcount(descA, descB) = popcount(A XOR B): craft A=0x00 and
    # B with exactly 19 / 20 differing bits.
    def desc_with_popcount(n):
        d = np.zeros(32, dtype=np.uint8)
        remaining = n
        for byte in range(32):
            if remaining >= 8:
                d[byte] = 0xFF
                remaining -= 8
            elif remaining > 0:
                d[byte] = (1 << remaining) - 1
                remaining = 0
        return d

    desc = np.stack([desc_with_popcount(19), desc_with_popcount(20)])
    positions = [(30.0, 30.0)] * 2  # both inside the ball, zero pixel dist
    tracker, uv, rows, _point = _projection_match_setup()
    set_keypoints(tracker.cur_frame, positions, desc)
    obs = tracker._match_projected_points(uv, rows, dist_thresh=5)
    assert len(obs) == 1
    # the accepted one is the 19-bit keypoint (index 0: strictly < 20)
    assert obs[0].kp_idx == 0


def test_projection_match_ties_break_to_the_first_scanned():
    """[C3a] two keypoints at the SAME descriptor distance: the match
    goes to the FIRST in the scan order (the pre-registered route-B
    tie-break, replacing scipy's implementation-defined order)."""
    # both keypoints share the SAME descriptor (distance 0 to the
    # point's zero descriptor), at different pixel positions.
    desc = np.zeros((2, 32), dtype=np.uint8)
    positions = [(30.0, 30.0), (32.0, 30.0)]
    tracker, uv, rows, _point = _projection_match_setup()
    set_keypoints(tracker.cur_frame, positions, desc)
    obs = tracker._match_projected_points(uv, rows, dist_thresh=5)
    assert len(obs) == 1
    assert obs[0].kp_idx == 0  # the first scanned wins the tie


def test_projection_match_l2_formulation_is_f64():
    """[C3a] the float-kind (ALIKED) descriptor distance is the
    pre-registered f64 accumulation: a knife-edge pair straddling the
    float reloc threshold (0.8639) behaves as f64 arithmetic dictates
    - distance just below matches, just above does not. (The f32 value
    0.8639 widened to f64 is 0.8639000058... > the threshold double
    0.8639 - so the straddle is crafted at 0.8638/0.8640.)"""
    below = np.zeros((1, 128), dtype=np.float32)
    below[0, 0] = np.float32(0.8638)   # widened: 0.86379998922... < 0.8639
    above = below.copy()
    above[0, 0] = np.float32(0.8640)   # widened: 0.86400005... > 0.8639
    # the POINT descriptor: the zero vector (width 128 float32 = 512 B)
    tracker, uv, rows, _point = _projection_match_setup(
        method="ALIKED", point_desc=np.zeros(512, dtype=np.uint8))
    set_keypoints(tracker.cur_frame, [(30.0, 30.0), (31.0, 30.0)],
                  np.concatenate([below, above]))
    obs = tracker._match_projected_points(uv, rows, dist_thresh=5)
    assert len(obs) == 1
    assert obs[0].kp_idx == 0  # the below-threshold keypoint


def test_update_skips_extraction_when_features_carried(monkeypatch):
    """[26c] the dedup: a frame whose features the caller attached
    (the intake gate's carry) takes NO extractor call, and the
    tracker's outcome is identical to the bare-frame extract path on
    the same image - the carry is a wall-clock win with zero numeric
    effect."""
    import cv2 as cv
    from benchmarks.stages.helpers import make_K
    from slam.core.frame import Frame
    from slam.core.map import Map
    from slam.core.tracker import Tracker

    img = (np.random.default_rng(9).integers(0, 256, (240, 320, 3))
           .astype(np.uint8))
    K = make_K()

    extract_calls = []
    real_extract = Tracker.__init__

    def counting_init(self, *a, **k):
        real_extract(self, *a, **k)
        orig = self.feature_extractor.extract

        def counting(img):
            extract_calls.append(1)
            return orig(img)

        self.feature_extractor.extract = counting

    monkeypatch.setattr(Tracker, "__init__", counting_init)

    # path A: bare frame (the extract path)
    extract_calls.clear()
    mA = Map()
    tA = Tracker(mA)
    fA = Frame(img, K)
    assert fA.keypoints is None
    tA.update(fA)
    assert len(extract_calls) == 1
    state_a = (fA.keypoints is not None, len(fA.descriptors),
               fA.pose.tobytes())

    # path B: the carried features (the gate-carry path)
    extract_calls.clear()
    mB = Map()
    tB = Tracker(mB)
    fB = Frame(img, K)
    from slam.core.intake import _frame_gray
    fB.set_features(*tB.feature_extractor.extract(_frame_gray(img)))
    extract_calls.clear()
    tB.update(fB)
    assert len(extract_calls) == 0, \
        "a carried frame must not re-extract"
    state_b = (fB.keypoints is not None, len(fB.descriptors),
               fB.pose.tobytes())
    assert state_a == state_b
    np.testing.assert_array_equal(fA.descriptors, fB.descriptors)
    assert [kp.pt for kp in fA.keypoints] == [kp.pt for kp in fB.keypoints]
