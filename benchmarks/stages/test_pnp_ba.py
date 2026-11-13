"""Stage tests: PnP pose refinement and local bundle adjustment.

Regressions for pose-refinement, anchoring and graph-cache bugs:
- refine_pose_pnp must be seeded with the world-to-camera transform
  (the inverse of the stored Twc) and must write back camera-to-world.
- refine_pose_pnp must refine the frame it is GIVEN (the tracker passes
  the current frame) - it used to read/write the map's previous keyframe,
  so per-frame refinement never applied to the tracked trajectory.
- local BA must be anchored: keyframes outside the window are fixed
  vertices, so the window cannot drift rigidly at zero reprojection cost.
- Divergent PnP solutions (unexplainable observations, teleporting
  refinements) must be rejected by the guards, keeping the seed pose.
- External pose-graph/global-BA write-back must invalidate the local
  graph, without disabling ordinary local graph reuse.
"""
import numpy as np
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
    """Regression: refine_pose_pnp must refine the
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
    """Observations no pose can explain (all corrupted by
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
    """A 'refinement' that teleports the camera (center step
    far beyond the observed scene depth) is rejected even if the target
    is geometrically consistent - refinement must correct, not relocate
    (the pre-guard divergence mode reached |t| ~ 2e4)."""
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
    """The timer-triggered global BA is disabled by
    default (interval 0). Measured A/B: it HURT three of four sequences
    (translation ATE 0.0845 -> 0.0181 without it) while dominating tum's
    BA cost; local BA keeps corrections local via fixed boundary
    keyframes, and the global re-solve traded well-constrained accuracy
    for global smoothing on a cadence with no evidence of need. Global
    BA stays available for loop closure, triggered by loop
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


def _seed_persistent_graph():
    """A clean graph whose membership stays unchanged across a rewrite."""
    m, kfs = _build_trajectory_map(6, make_world_points(32), make_K())
    ba = m.g2o_optimizer
    assert ba.local_bundle_adjustment(kfs[-1].frame_id, window_size=3,
                                      fix_points=False)
    assert ba._local_opt is not None
    assert ba._kf_vertex and ba._pt_vertex
    return m, kfs, ba


def _assert_persistent_graph_invalidated(ba):
    assert ba._local_opt is None
    assert ba._kf_vertex == {}
    assert ba._pt_vertex == {}
    assert ba._graph_edges_by_pt == {}
    assert ba._graph_edges_by_kf == {}


def _assert_persistent_estimates_match_map(ba):
    assert ba._local_opt is not None
    for fid, vertex in ba._kf_vertex.items():
        kf = ba.map.keyframes_by_id.get(fid)
        if kf is not None:
            live = np.linalg.inv(kf.pose)
            est = vertex.estimate()
            assert np.allclose(est.rotation().matrix(), live[:3, :3], atol=1e-9, rtol=0)
            assert np.allclose(est.translation(), live[:3, 3], atol=1e-9, rtol=0)
    for pid, vertex in ba._pt_vertex.items():
        point = ba.map.points_by_id.get(pid)
        if point is not None:
            assert np.allclose(vertex.estimate(), point.pt_3d, atol=1e-9, rtol=0)


def test_global_ba_invalidates_local_cache_and_preserves_corrected_gauge():
    """5e prerequisite: global write-back must invalidate local estimates.

    Apply a rigid world-gauge change to poses AND points, leaving every
    image observation unchanged. Fresh global BA preserves that gauge.
    A retained pre-change local graph would restore its old fixed anchor
    on the next solve even though membership has not changed.
    """
    m, kfs, ba = _seed_persistent_graph()
    S = yaw_pose(0.17, (2.0, -1.0, 0.5))
    for kf in kfs:
        kf.pose = S @ kf.pose
    for point in m.points:
        point.pt_3d = S[:3, :3] @ point.pt_3d + S[:3, 3]

    assert ba.global_bundle_adjustment()
    corrected_anchor = kfs[0].pose.copy()
    corrected_points = np.array([point.pt_3d for point in m.points])
    _assert_persistent_graph_invalidated(ba)

    assert ba.local_bundle_adjustment(kfs[-1].frame_id, window_size=3,
                                      fix_points=False)
    assert np.allclose(kfs[0].pose, corrected_anchor, atol=1e-9)
    assert np.allclose(kfs[0].pose, S, atol=1e-9)
    assert np.allclose([point.pt_3d for point in m.points], corrected_points,
                       atol=1e-4)
    _assert_persistent_estimates_match_map(ba)


def test_pose_graph_invalidates_local_cache_without_requiring_global_ba():
    """The cache must already be invalid if later global BA fails/skips."""
    m, kfs, ba = _seed_persistent_graph()
    relative = kfs[-1].pose @ np.linalg.inv(kfs[0].pose)
    relative[0, 3] += 0.3  # force a real external pose change
    assert ba.pose_graph_optimization(
        [(kfs[-1].frame_id, kfs[0].frame_id, relative)])
    _assert_persistent_graph_invalidated(ba)


def test_no_write_paths_keep_persistent_graph(monkeypatch):
    """No loops / rejected global BA must not cause a needless rebuild."""
    from slam.utils import constants as const

    m, _, ba = _seed_persistent_graph()
    graph = ba._local_opt
    assert ba.pose_graph_optimization([])
    assert ba._local_opt is graph
    monkeypatch.setattr(const, "MINIMUM_GLOBAL_OBSERVATIONS_FOR_POINT", 10**9)
    assert not ba.global_bundle_adjustment()
    assert ba._local_opt is graph


def test_local_ba_still_reuses_its_graph():
    """Invalidation is for external write-back, not every local solve."""
    _, kfs, ba = _seed_persistent_graph()
    graph = ba._local_opt
    assert ba.local_bundle_adjustment(kfs[-1].frame_id, window_size=3,
                                      fix_points=False)
    assert ba._local_opt is graph


def test_partial_external_write_failure_cannot_reuse_stale_graph(monkeypatch):
    """Invalidate BEFORE write-back, so partial failures cannot retain it."""
    _, kfs, ba = _seed_persistent_graph()
    relative = kfs[-1].pose @ np.linalg.inv(kfs[0].pose)

    def partial_write_then_fail(vertices):
        moved = kfs[-1].pose
        moved[0, 3] += 1.0
        kfs[-1].pose = moved
        raise RuntimeError("injected partial external write failure")

    monkeypatch.setattr(ba, "_update_keyframe_poses_from_vertices",
                        partial_write_then_fail)
    assert not ba.pose_graph_optimization(
        [(kfs[-1].frame_id, kfs[0].frame_id, relative)])
    _assert_persistent_graph_invalidated(ba)


def test_point_remove_observation_updates_count():
    """Regression: Point.remove_observation used to assign to the
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
    """Regression: _filter_duplicate_keypoints must register
    duplicate-keypoint matches CONSISTENTLY and must NOT overwrite an
    existing projection-matched observation for the same (point,
    frame) - the projection match is the better-anchored measurement.
    Dedup-only matches stay FRAME-SIDE ONLY: promotion to full
    point-side observations was measured and rejected twice, across
    six gate configurations (see the rejection record)."""
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

    # pt_dedup: observed by prev_kf only - the match must stay
    # FRAME-SIDE ONLY (the deliberate status quo).
    pt_dedup = Point(np.zeros(3), None, np.zeros(32, dtype=np.uint8))
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
    # tracking-quality counts; promotion was measured and rejected
    # twice - see the 2g and 2i-Q1 records). The point side must NOT
    # gain it.
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


# --- BA-overhead cuts. All three must be BIT-IDENTICAL
# to the code they replaced: the vectorized per-group norms, the
# hoisted quality-flag pass in the error fold, and the incremental
# edge bookkeeping that replaced the per-call opt.edges() rebuild.

def _fold_map(n_kfs, world_points, K, include_behind_camera=True):
    """A trajectory map plus one point that is BEHIND every camera
    (exercises the finite-filter path: its fold never updates, so its
    observers must never be flagged)."""
    m = Map()
    kfs = []
    points = [Point(X.copy(), None, np.zeros(32, dtype=np.uint8))
              for X in world_points]
    if include_behind_camera:
        points.append(Point(np.array([0.0, 0.0, -1.0]), None,
                            np.zeros(32, dtype=np.uint8)))
    for i in range(n_kfs):
        T = yaw_pose(0.0, (0.0, 0.0, 0.5 * i))
        kf = make_frame(T, K)
        uvs = []
        for X in world_points:
            uv, _ = project_gt(T, X, K)
            uvs.append(uv)
        if include_behind_camera:
            uvs.append((320.0, 240.0))  # pixel for the behind-camera point
        set_keypoints(kf, uvs)
        for j, pt in enumerate(points):
            pt.add_observation(Observation(kf, pt, j))
        m.add_keyframe(kf)
        kfs.append(kf)
    m.add_points(points)
    return m, kfs, points


def _reference_fold(observations):
    """The reference fold: by-frame batched projection, the SAME
    vectorized row-norm the optimized fold computes (the [26d] step-2
    re-measurement: the per-observation 1-D np.linalg.norm rounded
    differently at the last ULP - BLAS FMA inside the dot, measured
    7.9% of random inputs - so the norm's formulation is pinned to
    the vectorized form on BOTH sides, and the pins keep verifying
    the fold ORDER, the recurrence, and the flag semantics), and
    per-update quality flags."""
    from collections import defaultdict
    by_frame = defaultdict(list)
    for obs in observations:
        by_frame[obs.frame].append(obs)

    error_by_obs = {}
    for frame, group in by_frame.items():
        pts_3d = np.empty((len(group), 3), dtype=np.float64)
        for i, obs in enumerate(group):
            pts_3d[i] = obs.pt_3d
        projected = frame.project_points(pts_3d, normalized=True)
        finite = np.all(np.isfinite(projected), axis=1)
        if finite.any():
            obs2d = np.array([obs.pt_2d_norm for obs in group],
                             dtype=np.float64)
            d = projected - obs2d
            norms = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
            for obs, ok, err in zip(group, finite, norms):
                if ok:
                    error_by_obs[obs] = err

    for obs in observations:
        error = error_by_obs.get(obs)
        if error is not None:
            obs.point.update_reprojection_error(error)  # flags on


def _flagged_frames(m):
    """Flagged keyframes by POSITION: twin maps have different uuid4s,
    so positional indices are the comparable identity."""
    return {i for i, kf in enumerate(m.keyframes)
            if kf.tracking_quality_needs_update}


def test_recompute_point_errors_bit_identical_point_grouped():
    """The 2i fold (vectorized norms + hoisted flags) must produce
    EXACTLY the old averages and the old final flag set, on the
    point-consecutive order local BA uses."""
    K = make_K()
    world_points = make_world_points(12, z_range=(7.0, 10.0))
    m_ref, kfs_ref, pts_ref = _fold_map(6, world_points, K)
    m_new, kfs_new, pts_new = _fold_map(6, world_points, K)

    obs_ref = [obs for p in pts_ref for obs in p.observations.values()]
    obs_new = [obs for p in pts_new for obs in p.observations.values()]
    assert len(obs_ref) == len(obs_new)

    _reference_fold(obs_ref)
    m_new.g2o_optimizer._recompute_point_errors(obs_new)

    for p_ref, p_new in zip(pts_ref, pts_new):
        assert p_new.average_reprojection_error == \
            p_ref.average_reprojection_error, \
            "running average diverged from the reference fold"
    assert _flagged_frames(m_new) == _flagged_frames(m_ref)
    # The behind-camera point never updates: its average stays 0 and it
    # contributes no flags (covered positionally by the set equality).
    assert pts_new[-1].average_reprojection_error == 0.0


def test_recompute_point_errors_bit_identical_frame_grouped():
    """Same equivalence on the frame-grouped order the global-BA path
    uses (observations grouped by keyframe, not by point): the hoist's
    change-detection must stay correct there too."""
    K = make_K()
    world_points = make_world_points(8, z_range=(7.0, 10.0))
    m_ref, kfs_ref, pts_ref = _fold_map(5, world_points, K)
    m_new, kfs_new, pts_new = _fold_map(5, world_points, K)

    obs_ref = [obs for kf in kfs_ref for obs in kf.observed_points.values()]
    obs_new = [obs for kf in kfs_new for obs in kf.observed_points.values()]

    _reference_fold(obs_ref)
    m_new.g2o_optimizer._recompute_point_errors(obs_new)

    for p_ref, p_new in zip(pts_ref, pts_new):
        assert p_new.average_reprojection_error == \
            p_ref.average_reprojection_error
    assert _flagged_frames(m_new) == _flagged_frames(m_ref)


def test_update_reprojection_error_flag_parameter():
    """update_quality=False must skip flagging entirely; the default
    must keep flagging (tracker and triangulation call sites)."""
    K = make_K()
    m, kfs, pts = _fold_map(3, make_world_points(4, z_range=(7.0, 10.0)), K)
    kf = kfs[0]
    pt = pts[0]
    kf.tracking_quality_needs_update = False

    pt.update_reprojection_error(0.5, update_quality=False)
    assert not kf.tracking_quality_needs_update
    # Running average: starts at 0, divides by the observation count.
    assert pt.average_reprojection_error == 0.5 / pt.num_observations

    pt.update_reprojection_error(0.5)
    assert kf.tracking_quality_needs_update


def test_persistent_graph_edge_bookkeeping_consistency():
    """The incremental pair bookkeeping must exactly
    mirror the graph across reconciles - including the stale-vertex
    cascade path exercised by a keyframe cull."""
    K = make_K()
    world_points = make_world_points(40, z_range=(7.0, 10.0))
    m, kfs = _build_trajectory_map(12, world_points, K)

    def bookkeeping_pairs(ba):
        return {(pid, fid)
                for pid, fids in ba._graph_edges_by_pt.items()
                for fid in fids}

    def graph_pairs(ba):
        vid2pt = {v: pid for pid, v in ba._pt_vertex.items()}
        vid2kf = {v: fid for fid, v in ba._kf_vertex.items()}
        return {(vid2pt[e.vertex(0)], vid2kf[e.vertex(1)])
                for e in ba._local_opt.edges()}

    ba = m.g2o_optimizer
    assert ba.local_bundle_adjustment(kfs[-1].frame_id,
                                       window_size=5, fix_points=False)
    assert bookkeeping_pairs(ba) == graph_pairs(ba)

    # Cull a middle keyframe (2h machinery), then reconcile again: the
    # cascade path must drop exactly the culled keyframe's pairs.
    assert m.remove_keyframe_by_id(kfs[2].frame_id)
    assert ba.local_bundle_adjustment(m.keyframes[-1].frame_id,
                                      window_size=5, fix_points=False)
    assert bookkeeping_pairs(ba) == graph_pairs(ba)
    assert kfs[2].frame_id not in ba._kf_vertex

    # A third reconcile with a sliding window (different stale set).
    assert ba.local_bundle_adjustment(m.keyframes[-1].frame_id,
                                      window_size=7, fix_points=False)
    assert bookkeeping_pairs(ba) == graph_pairs(ba)

    # Count consistency across both views.
    n_pairs = sum(len(s) for s in ba._graph_edges_by_pt.values())
    assert n_pairs == sum(len(s) for s in ba._graph_edges_by_kf.values())
    assert n_pairs == len(ba._local_opt.edges())
