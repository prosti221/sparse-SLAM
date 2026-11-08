"""Stage tests: PnP pose refinement and local bundle adjustment.

Regressions for three fixed bugs:
- refine_pose_pnp must be seeded with the world-to-camera transform
  (the inverse of the stored Twc) and must write back camera-to-world.
- refine_pose_pnp must refine the frame it is GIVEN (the tracker passes
  the current frame) - it used to read/write the map's previous keyframe,
  so per-frame refinement never applied to the tracked trajectory
  (finding 9, phase 1b).
- local BA must be anchored: keyframes outside the window are fixed
  vertices, so the window cannot drift rigidly at zero reprojection cost.
- Divergent PnP solutions (unexplainable observations, teleporting
  refinements) must be rejected by the guards, keeping the seed pose
  (phase 1b).
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.stages.helpers import make_K, make_frame, se3, yaw_pose, project_gt, set_keypoints, make_world_points
from slam.core.map import Map
from slam.core.point import Point
from slam.core.observation import Observation


def _rot_error(R_a, R_b):
    return np.degrees(np.arccos(np.clip(
        (np.trace(R_a.T @ R_b) - 1.0) / 2.0, -1.0, 1.0)))


def _build_keyframe_map(pose, world_points, K):
    """Map with a single keyframe at `pose` observing all world points."""
    m = Map()
    kf = make_frame(pose, K)
    uvs = []
    points = []
    for X in world_points:
        uv, _ = project_gt(pose, X, K)
        uvs.append(uv)
        pt = Point(X.copy(), None, np.zeros(32, dtype=np.uint8))
        points.append(pt)
    set_keypoints(kf, uvs)
    for i, pt in enumerate(points):
        pt.add_observation(Observation(kf, pt, i))
    m.add_keyframe(kf)
    m.add_points(points)
    return m, kf


def test_refine_pose_pnp_recovers_perturbed_pose():
    """PnP must pull a slightly perturbed keyframe pose back to truth."""
    K = make_K()
    T_true = yaw_pose(0.2, (1.0, 2.0, 5.0))
    world_points = make_world_points(40, z_range=(8.0, 12.0))

    m, kf = _build_keyframe_map(T_true, world_points, K)

    # Perturb: 1.5 deg rotation + 0.05 units translation.
    perturb = se3(Rotation.from_euler("y", np.radians(1.5)).as_matrix(),
                  np.array([0.03, -0.02, 0.04]))
    kf.pose = T_true @ perturb

    ok, inliers = m.g2o_optimizer.refine_pose_pnp()

    assert ok
    assert inliers >= 10
    assert _rot_error(kf.pose[:3, :3], T_true[:3, :3]) < 0.2
    assert np.linalg.norm(kf.pose[:3, 3] - T_true[:3, 3]) < 0.02


def test_refine_pose_pnp_targets_requested_frame():
    """Finding 9 regression (phase 1b): refine_pose_pnp must refine the
    frame it is GIVEN - the tracker's current frame - and must not touch
    the map's previous keyframe. Pre-1b this call read the previous
    keyframe's observations and wrote the previous keyframe's pose, so
    per-frame refinement never applied to the tracked trajectory."""
    K = make_K()
    T_true = yaw_pose(0.2, (1.0, 2.0, 5.0))
    world_points = make_world_points(40, z_range=(8.0, 12.0))

    m, kf = _build_keyframe_map(T_true, world_points, K)
    kf_pose_before = kf.pose.copy()

    # A separate current frame: its keypoints are the TRUE projections of
    # the map points (real image measurements), but its stored pose is a
    # perturbed E-chain estimate - exactly the tracker's situation.
    cur = make_frame(T_true, K)
    uvs = [project_gt(T_true, X, K)[0] for X in world_points]
    set_keypoints(cur, uvs)
    for i, pt in enumerate(m.points):
        pt.add_observation(Observation(cur, pt, i))
    perturb = se3(Rotation.from_euler("y", np.radians(1.5)).as_matrix(),
                  np.array([0.03, -0.02, 0.04]))
    cur.pose = T_true @ perturb

    ok, inliers = m.g2o_optimizer.refine_pose_pnp(cur)

    assert ok
    assert _rot_error(cur.pose[:3, :3], T_true[:3, :3]) < 0.2
    assert np.linalg.norm(cur.pose[:3, 3] - T_true[:3, 3]) < 0.02
    # The keyframe is a DIFFERENT frame: it must be untouched.
    assert np.allclose(kf.pose, kf_pose_before, atol=1e-12), \
        "refine_pose_pnp(cur) must not move the previous keyframe"


def test_pnp_guards_reject_divergent_solution():
    """Phase 1b: observations no pose can explain (all corrupted by
    large random offsets) must yield a rejection, and the frame must
    keep its seed pose - a diverged PnP must never poison the map."""
    K = make_K()
    T_seed = yaw_pose(0.2, (1.0, 2.0, 5.0))
    world_points = make_world_points(40, z_range=(8.0, 12.0))

    m, kf = _build_keyframe_map(T_seed, world_points, K)

    cur = make_frame(T_seed, K)
    rng = np.random.default_rng(11)
    uvs = []
    for X in world_points:
        uv, _ = project_gt(T_seed, X, K)
        uvs.append(uv + rng.uniform(30.0, 60.0, size=2))
    set_keypoints(cur, uvs)
    for i, pt in enumerate(m.points):
        pt.add_observation(Observation(cur, pt, i))
    seed_pose = cur.pose.copy()

    ok, _ = m.g2o_optimizer.refine_pose_pnp(cur)

    assert not ok
    assert np.allclose(cur.pose, seed_pose, atol=1e-12), \
        "seed pose must be kept when guards reject"


def test_pnp_guards_reject_teleporting_solution():
    """Phase 1b: a 'refinement' that teleports the camera (center step
    far beyond the observed scene depth) is rejected even if the target
    is geometrically consistent - refinement must correct, not relocate
    (the pre-1b divergence mode reached |t| ~ 2e4)."""
    K = make_K()
    T_true = yaw_pose(0.2, (1.0, 2.0, 5.0))
    world_points = make_world_points(40, z_range=(8.0, 12.0))

    m, kf = _build_keyframe_map(T_true, world_points, K)

    # Observations come from T_true, but the stored seed is ~60 units away:
    # any consistent solution is a teleport relative to the seed.
    cur = make_frame(T_true, K)
    uvs = [project_gt(T_true, X, K)[0] for X in world_points]
    set_keypoints(cur, uvs)
    for i, pt in enumerate(m.points):
        pt.add_observation(Observation(cur, pt, i))
    cur.pose = yaw_pose(0.0, (60.0, 0.0, 0.0))
    seed_pose = cur.pose.copy()

    ok, _ = m.g2o_optimizer.refine_pose_pnp(cur)

    assert not ok
    assert np.allclose(cur.pose, seed_pose, atol=1e-12), \
        "seed pose must be kept when the refinement teleports"


def _build_trajectory_map(n_keyframes, world_points, K):
    """A map with keyframes marching forward along +z, all observing a
    shared cloud of world points (12 KFs x 40 pts -> strong covisibility)."""
    m = Map()
    kfs = []
    points = [Point(X.copy(), None, np.zeros(32, dtype=np.uint8))
              for X in world_points]
    for i in range(n_keyframes):
        T = yaw_pose(0.0, (0.0, 0.0, 0.5 * i))
        kf = make_frame(T, K)
        uvs = []
        for X in world_points:
            uv, _ = project_gt(T, X, K)
            uvs.append(uv)
        set_keypoints(kf, uvs)
        for j, pt in enumerate(points):
            pt.add_observation(Observation(kf, pt, j))
        m.add_keyframe(kf)
        kfs.append(kf)
    m.add_points(points)
    return m, kfs


def test_local_ba_is_anchored_and_recovers_perturbed_keyframe():
    """Regression for the gauge-freedom bug: with outer keyframes fixed,
    a moderately perturbed window keyframe must be pulled back to its true
    pose. Pre-fix, the window could absorb the perturbation as a rigid
    drift at zero reprojection cost instead."""
    K = make_K()
    world_points = make_world_points(40, z_range=(7.0, 10.0))
    m, kfs = _build_trajectory_map(12, world_points, K)

    T_true_last = kfs[-1].pose.copy()
    # Realistic perturbation: 1 deg rotation + 0.3 units along the optical
    # axis (the weakest LM direction - a strong test of anchoring).
    perturb = se3(Rotation.from_euler("y", np.radians(1.0)).as_matrix(),
                  np.array([0.0, 0.0, 0.3]))
    kfs[-1].pose = T_true_last @ perturb

    ok = m.g2o_optimizer.local_bundle_adjustment(
        kfs[-1].frame_id, window_size=5, fix_points=False)

    assert ok
    # The perturbed keyframe must be pulled back near its true pose.
    assert np.linalg.norm(kfs[-1].pose[:3, 3] - T_true_last[:3, 3]) < 0.1, \
        "local BA failed to recover the perturbed keyframe pose"
    assert _rot_error(kfs[-1].pose[:3, :3], T_true_last[:3, :3]) < 0.5
    # The fixed first keyframe must not have moved at all.
    assert np.allclose(kfs[0].pose, np.eye(4), atol=1e-9)
    # No other window keyframe may slide rigidly away from its pose.
    for idx, kf in enumerate(kfs):
        T_expected = yaw_pose(0.0, (0.0, 0.0, 0.5 * idx))
        assert np.linalg.norm(kf.pose[:3, 3] - T_expected[:3, 3]) < 0.1


def test_local_ba_keeps_window_keyframes_consistent():
    """Without any perturbation, local BA must be a no-op-ish refinement:
    poses were generated from exact geometry, so nothing may move much."""
    K = make_K()
    world_points = make_world_points(40, z_range=(7.0, 10.0))
    m, kfs = _build_trajectory_map(12, world_points, K)

    poses_before = [kf.pose.copy() for kf in kfs]

    ok = m.g2o_optimizer.local_bundle_adjustment(
        kfs[-1].frame_id, window_size=5, fix_points=False)

    assert ok
    for kf, before in zip(kfs, poses_before):
        moved = np.linalg.norm(kf.pose[:3, 3] - before[:3, 3])
        assert moved < 0.05, f"keyframe moved {moved:.3f} on clean input"


def test_timer_global_ba_disabled_by_default():
    """Phase 2a (finding 13): the timer-triggered global BA is disabled by
    default (interval 0). Measured A/B: it HURT three of four sequences
    (translation ATE 0.0845 -> 0.0181 without it) while dominating tum's
    BA cost; local BA keeps corrections local via fixed boundary
    keyframes, and the global re-solve traded well-constrained accuracy
    for global smoothing on a cadence with no evidence of need. Global
    BA stays available for loop closure (phase 3), triggered by loop
    events, not a timer."""
    from slam.utils import constants as const

    K = make_K()
    m = Map()
    for i in range(12):
        m.add_keyframe(make_frame(yaw_pose(0.0, (0.5 * i, 0.0, 0.0)), K))

    # The shipped constant disables the timer.
    assert const.GLOBAL_BUNDLE_ADJUSTMENT_KEYFRAME_INTERVAL == 0
    assert not m.should_perform_global_bundle_adjustment(
        const.GLOBAL_BUNDLE_ADJUSTMENT_KEYFRAME_INTERVAL)

    # Interval semantics still work for future re-enablement: > 0 fires
    # at keyframe-count multiples, 0 disables.
    assert m.should_perform_global_bundle_adjustment(4)      # 12 % 4 == 0
    assert not m.should_perform_global_bundle_adjustment(5)  # 12 % 5 != 0
    assert not m.should_perform_global_bundle_adjustment(0)


def test_point_remove_observation_updates_count():
    """Finding 16: Point.remove_observation used to assign to the
    setter-less num_observations property (AttributeError on every
    call that found the observation). Reachable via Map._remove_keyframe
    (the keyframe-culling path, unused by benchmarks so far) - the map
    management item cannot be enabled until this works."""
    import numpy as np

    from slam.core.point import Point
    from slam.core.observation import Observation

    K = make_K()
    kf = make_frame(yaw_pose(0.0, (0, 0, 0)), K)
    set_keypoints(kf, [(10.0, 10.0)])  # Observation reads frame.keypoints lazily
    pt = Point(np.zeros(3), None, np.zeros(32, dtype=np.uint8))

    obs = Observation(kf, pt, 0)
    pt.add_observation(obs)
    assert pt.num_observations == 1

    pt.remove_observation(kf.frame_id)  # must not raise
    assert pt.num_observations == 0


def test_filter_duplicate_registers_consistent_observations():
    """Finding 17 (phase 2g): _filter_duplicate_keypoints must register
    duplicate-keypoint matches CONSISTENTLY - point-side AND frame-side
    with the SAME object (they used to be frame-side only, invisible to
    BA), and must NOT overwrite an existing projection-matched
    observation for the same (point, frame) - the projection match is
    the better-anchored measurement."""
    from types import SimpleNamespace

    import cv2 as cv

    from slam.core.map import Map
    from slam.core.observation import Observation
    from slam.core.point import Point
    from slam.core.tracker import Tracker

    K = make_K()
    m = Map()
    prev_kf = make_frame(yaw_pose(0.0, (0, 0, 0)), K)
    cur = make_frame(yaw_pose(0.0, (0.5, 0, 0)), K)
    set_keypoints(prev_kf, [(10.0, 10.0), (50.0, 50.0)])
    set_keypoints(cur, [(12.0, 10.0), (52.0, 50.0)])

    # pt_dedup: observed by prev_kf only - the dedup match must become a
    # FULL observation (both registries, same object). Placed exactly on
    # cur's keypoint-0 ray so it passes the geometric validation gate.
    K_inv = np.linalg.inv(K)
    Xc = 5.0 * (K_inv @ np.array([12.0, 10.0, 1.0]))
    pt_dedup = Point((cur.pose @ np.append(Xc, 1.0))[:3], None,
                     np.zeros(32, dtype=np.uint8))
    pt_dedup.add_observation(Observation(prev_kf, pt_dedup, 0))

    # pt_both: prev_kf obs + an EXISTING projection-matched obs on cur -
    # the dedup match must keep the projection observation untouched.
    pt_both = Point(np.array([1.0, 0.0, 5.0]), None,
                    np.zeros(32, dtype=np.uint8))
    pt_both.add_observation(Observation(prev_kf, pt_both, 1))
    proj_obs = Observation(cur, pt_both, 1)
    pt_both.add_observation(proj_obs)

    m.add_keyframe(prev_kf)
    m.add_points([pt_dedup, pt_both])

    # cur matches: query 0 <-> prev train 0 (pt_dedup), query 1 <->
    # prev train 1 (pt_both)
    cur.set_match_data([cv.DMatch(0, 0, 0.0), cv.DMatch(1, 1, 0.0)])
    cur.kp_unique_mask = [True, True]

    tracker = SimpleNamespace(cur_frame=cur, map=m)
    Tracker._filter_duplicate_keypoints(tracker)

    # Dedup-only case: registers on the frame side ONLY (deliberate
    # status quo - these are load-bearing for insertion-time PnP and
    # tracking-quality counts, but were measured NOT to be reliable
    # enough to promote to full point-side observations; see the 2g
    # A/B record). The point side must NOT gain it.
    assert not pt_dedup.is_observed_by(cur.frame_id), \
        "dedup-only match leaked into the point-side registry"
    frame_obs = cur.observed_points[pt_dedup.point_id]
    assert frame_obs.kp_idx == 0

    # Both-case: the projection observation survived, same object both
    # sides, and no new observation replaced it.
    assert pt_both.observations[cur.frame_id] is proj_obs
    assert cur.observed_points[pt_both.point_id] is proj_obs

    # Both matched keypoints masked for triangulation.
    assert cur.kp_unique_mask == [False, False]
