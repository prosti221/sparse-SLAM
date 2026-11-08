"""Stage tests: keyframe insertion policy decision boundaries.

Pure logic tests for should_insert_keyframe - no SLAM run required.
Documents the actual policy semantics: baseline is a critical condition;
new-points-ratio and tracking quality are OR-ed quality conditions.
"""
import numpy as np
import pytest

from benchmarks.stages.helpers import make_K, make_frame, yaw_pose, set_keypoints
from slam.utils.utils import should_insert_keyframe

K = make_K()


def _policy_frame(pose, quality, n_matches=100, n_tracked=60):
    """A frame with controlled tracking-quality inputs.

    - tracking_quality is set directly (bypassing recompute)
    - new-points ratio = num_tracked_features / len(matches)
    """
    f = make_frame(pose, K)
    set_keypoints(f, [(10.0, 10.0)] * n_matches)
    f.matches = list(range(n_matches))
    f.kp_unique_mask = [True] * n_tracked + [False] * (n_matches - n_tracked)
    f._tracking_quality = quality
    f.tracking_quality_needs_update = False
    return f


def test_baseline_below_threshold_rejects():
    prev_kf = _policy_frame(yaw_pose(0.0, (0, 0, 0)), 0.5)
    # Baseline 1.0 < MINIMUM_BASELINE_THRESHOLD (2.0) -> no insertion,
    # even with perfect quality.
    cur = _policy_frame(yaw_pose(0.0, (1.0, 0, 0)), 0.5)
    should, _ = should_insert_keyframe(cur, prev_kf)
    assert not should


def test_baseline_and_ratio_pass_inserts():
    prev_kf = _policy_frame(yaw_pose(0.0, (0, 0, 0)), 0.0)
    # Baseline 3.0 >= 2.0, tracked ratio 60/100 = 0.6 >= 0.4
    cur = _policy_frame(yaw_pose(0.0, (3.0, 0, 0)), 0.0)
    should, _ = should_insert_keyframe(cur, prev_kf)
    assert should


def test_quality_conditions_are_ored():
    """Baseline is critical; ratio and quality are alternatives."""
    prev_kf = _policy_frame(yaw_pose(0.0, (0, 0, 0)), 0.0)

    # Ratio passes, quality terrible -> insert.
    cur = _policy_frame(yaw_pose(0.0, (3.0, 0, 0)), quality=0.0, n_tracked=60)
    should, _ = should_insert_keyframe(cur, prev_kf)
    assert should

    # Quality passes, ratio terrible -> insert.
    cur = _policy_frame(yaw_pose(0.0, (3.0, 0, 0)), quality=0.5, n_tracked=0)
    should, _ = should_insert_keyframe(cur, prev_kf)
    assert should

    # Both fail -> no insertion.
    cur = _policy_frame(yaw_pose(0.0, (3.0, 0, 0)), quality=0.0, n_tracked=0)
    should, _ = should_insert_keyframe(cur, prev_kf)
    assert not should


def test_reason_string_reports_failed_conditions():
    prev_kf = _policy_frame(yaw_pose(0.0, (0, 0, 0)), 0.5)
    cur = _policy_frame(yaw_pose(0.0, (0.5, 0, 0)), quality=0.0, n_tracked=0)
    should, reason = should_insert_keyframe(cur, prev_kf)
    assert not should
    assert "Baseline" in reason
    assert "new points" in reason
    assert "Tracking quality" in reason


# --- Phase 1c: rotation-aware viewpoint criterion ---

def test_rotation_triggers_insertion_without_baseline():
    """Phase 1c: accumulated rotation is viewpoint change - it may satisfy
    the critical condition where translation cannot. Rotation-dominant
    motion starved the map under the translation-only criterion
    (forward_rotation triangulation rejected ~100% of points)."""
    prev_kf = _policy_frame(yaw_pose(0.0, (0, 0, 0)), 0.0)
    # Baseline 0.5 fails, but 5 deg of accumulated rotation at a
    # sufficiently sparse cadence passes.
    cur = _policy_frame(yaw_pose(np.radians(5.0), (0.5, 0, 0)), 0.0, n_tracked=60)

    should, reason = should_insert_keyframe(cur, prev_kf,
                                            frames_since_last_keyframe=3)
    assert should, reason


def test_small_rotation_and_baseline_rejects():
    """Both translation and rotation below their thresholds: no keyframe,
    even with perfect quality conditions."""
    prev_kf = _policy_frame(yaw_pose(0.0, (0, 0, 0)), 0.5)
    cur = _policy_frame(yaw_pose(np.radians(0.5), (1.0, 0, 0)), 0.0, n_tracked=60)

    should, _ = should_insert_keyframe(cur, prev_kf,
                                       frames_since_last_keyframe=5)
    assert not should


def test_rotation_trigger_is_gap_capped():
    """Phase 1c self-limiting gate: the rotation trigger never fires
    frame-to-frame (gap below KEYFRAME_ROTATION_MIN_FRAME_GAP) - a
    degenerate 1-2 frame pair poisons the map with junk triangulations
    (measured on orbit with an ungated threshold: 44 -> 140 keyframes,
    surf 15.9 -> 0)."""
    prev_kf = _policy_frame(yaw_pose(0.0, (0, 0, 0)), 0.0)
    cur = _policy_frame(yaw_pose(np.radians(5.0), (0.5, 0, 0)), 0.0, n_tracked=60)

    # Dense cadence (gap < min): no trigger, however much rotation.
    should, _ = should_insert_keyframe(cur, prev_kf,
                                       frames_since_last_keyframe=2)
    assert not should, "rotation trigger fired on a dense frame-to-frame cadence"

    # Sparse cadence (gap >= min): trigger fires.
    should, _ = should_insert_keyframe(cur, prev_kf,
                                       frames_since_last_keyframe=3)
    assert should


def test_policy_reads_live_constants():
    """The overridability contract for benchmark sweeps (--override):
    the policy reads its constants through the module, so changing them
    changes behavior without code changes."""
    import slam.utils.constants as const
    saved = (const.MINIMUM_BASELINE_THRESHOLD,
              const.KEYFRAME_MINIMUM_ROTATION_DEG)
    try:
        const.MINIMUM_BASELINE_THRESHOLD = 10.0
        const.KEYFRAME_MINIMUM_ROTATION_DEG = 999.0
        prev_kf = _policy_frame(yaw_pose(0.0, (0, 0, 0)), 0.0)
        cur = _policy_frame(yaw_pose(0.0, (3.0, 0, 0)), 0.0, n_tracked=60)

        should, _ = should_insert_keyframe(cur, prev_kf)
        assert not should, "baseline 3.0 must reject under a 10.0 threshold"
    finally:
        (const.MINIMUM_BASELINE_THRESHOLD,
         const.KEYFRAME_MINIMUM_ROTATION_DEG) = saved


# --- Phase 2h: keyframe-removal machinery (Map._remove_keyframe) ---
# The culling hook lands with step 3; these pins hold the repaired
# semantics of the removal itself, whose only live caller today is the
# aggressive-relocalization reset path.

def _cull_map(K):
    """5 keyframes marching forward, three point families:
    - shared:  observed by all 5 KFs (survive any single cull)
    - weak:    observed by KFs 0-2 (orphaned when kf2 is culled: 2
               remaining KF observations < MINIMUM_OBSERVATIONS_FOR_POINT=3)
    - masked:  observed by KFs 1-2 plus two never-promoted TRACKER
               frames (orphaned when kf2 is culled even though 3
               observations remain total - only keyframes anchor a
               point: BA edges, covisibility, and refinement all come
               from keyframes; tracker poses are never optimized)."""
    from benchmarks.stages.helpers import project_gt, make_world_points
    from slam.core.map import Map
    from slam.core.point import Point
    from slam.core.observation import Observation

    m = Map()
    world = make_world_points(16, z_range=(7.0, 10.0))
    shared = [Point(world[i].copy(), None, np.zeros(32, dtype=np.uint8))
              for i in range(8)]
    weak = [Point(world[8 + i].copy(), None, np.zeros(32, dtype=np.uint8))
            for i in range(4)]
    masked = [Point(world[12 + i].copy(), None, np.zeros(32, dtype=np.uint8))
              for i in range(4)]

    kfs = []
    for i in range(5):
        kf = make_frame(yaw_pose(0.0, (0.0, 0.0, 0.5 * i)), K)
        observed = list(shared)
        if i in (0, 1, 2):
            observed += weak
        if i in (1, 2):
            observed += masked
        uvs = [project_gt(kf.pose, X.pt_3d, K)[0] for X in observed]
        set_keypoints(kf, uvs)
        for j, pt in enumerate(observed):
            pt.add_observation(Observation(kf, pt, j))
        m.add_keyframe(kf)
        kfs.append(kf)

    # Tracker frames: never promoted (is_keyframe False), pose never
    # optimized - their observations must not keep a point alive.
    trackers = []
    for i in range(2):
        tr = make_frame(yaw_pose(0.0, (0.0, 0.0, 1.0 + 0.25 * i)), K)
        uvs = [project_gt(tr.pose, X.pt_3d, K)[0] for X in masked]
        set_keypoints(tr, uvs)
        for j, pt in enumerate(masked):
            pt.add_observation(Observation(tr, pt, j))
        trackers.append(tr)

    m.add_points(shared + weak + masked)
    return m, kfs, shared, weak, masked


def test_remove_keyframe_cleans_all_registries():
    """Phase 2h machinery repair: removing a keyframe must clean every
    map registry (keyframes list/by-id, tracking-quality history,
    covisibility graph, loop-closure feature DB, orphaned points) and
    the persistent local-BA graph must self-heal on the next call."""
    K = make_K()
    m, kfs, shared, weak, masked = _cull_map(K)

    m.update_tracking_quality()
    assert m.g2o_optimizer.local_bundle_adjustment(
        kfs[-1].frame_id, window_size=5, fix_points=False)
    culled_id = kfs[2].frame_id
    assert culled_id in m.g2o_optimizer._kf_vertex

    assert m.remove_keyframe_by_id(culled_id)

    # Keyframe registries
    assert culled_id not in m.keyframes_by_id
    assert culled_id not in [kf.frame_id for kf in m.keyframes]
    assert culled_id not in m.tracking_quality_history
    # Loop-closure feature database
    assert culled_id not in m.feature_db.feature_storage
    assert culled_id not in m.feature_db.keyframe_order
    # Covisibility graph: neither row nor any survivor's column
    assert culled_id not in m.covisibility_graph
    for kf_id in m.covisibility_graph:
        assert culled_id not in m.covisibility_graph[kf_id]

    # Orphan semantics: shared survive (4 KF observers), weak and
    # masked die (2 resp. 1 KF observer left; masked still has 3 total
    # observations incl. the two tracker frames - only keyframes count)
    pt_ids = {pt.point_id for pt in m.points}
    for pt in shared:
        assert pt.point_id in pt_ids
        assert not pt.is_observed_by(culled_id)
    for pt in weak + masked:
        assert pt.point_id not in pt_ids
        assert pt.point_id not in m.points_by_id

    # Persistent local-BA graph self-heals: culled keyframe vertex and
    # orphaned point vertices drop out on the next reconcile.
    assert m.g2o_optimizer.local_bundle_adjustment(
        m.keyframes[-1].frame_id, window_size=5, fix_points=False)
    assert culled_id not in m.g2o_optimizer._kf_vertex
    for pt in weak + masked:
        assert pt.point_id not in m.g2o_optimizer._pt_vertex
    for pt in shared:
        assert pt.point_id in m.g2o_optimizer._pt_vertex


def test_remove_keyframes_after_is_positional():
    """Phase 2h repair: remove_keyframes_after must use keyframe LIST
    POSITION (insertion order), not frame_id ordering - frame ids are
    random uuid4 values, so the old `kf.frame_id <= reference` walk
    removed a RANDOM suffix of the keyframe list. The crafted ids below
    invert uuid ordering relative to insertion order, which made the
    old code keep keyframes that must go."""
    import uuid as _uuid
    from slam.core.map import Map

    K = make_K()
    m = Map()
    kfs = []
    for i in range(5):
        kf = make_frame(yaw_pose(0.0, (0.0, 0.0, 0.5 * i)), K)
        kf.frame_id = _uuid.UUID(int=5 - i)  # kf0=5 ... kf4=1
        set_keypoints(kf, [(10.0, 10.0)])
        m.add_keyframe(kf)
        kfs.append(kf)

    m.remove_keyframes_after(kfs[2].frame_id)

    assert [kf.frame_id for kf in m.keyframes] == \
        [kfs[0].frame_id, kfs[1].frame_id, kfs[2].frame_id], \
        "keyframes inserted after the reference must be positionally removed"


def _criterion_map(n_kfs, observer_lists, K):
    """n_kfs keyframes marching forward; observer_lists[j] = keyframe
    indices observing point j. Each point gets a dedicated keypoint on
    each observing frame (ground-truth projected, so observations are
    well-formed), registered point-side and frame-side."""
    from slam.core.map import Map
    from slam.core.point import Point
    from slam.core.observation import Observation
    from benchmarks.stages.helpers import make_world_points, project_gt

    m = Map()
    kfs = [make_frame(yaw_pose(0.0, (0.0, 0.0, 0.5 * i)), K)
           for i in range(n_kfs)]

    world = make_world_points(len(observer_lists), z_range=(7.0, 10.0))
    points = [Point(world[j].copy(), None, np.zeros(32, dtype=np.uint8))
              for j in range(len(observer_lists))]

    # Per-frame keypoint lists + per-point (kf_idx, kp_idx) entries.
    uvs = [[] for _ in range(n_kfs)]
    per_point_observers = []
    for j, observers in enumerate(observer_lists):
        entries = []
        for kf_idx in observers:
            uv, _ = project_gt(kfs[kf_idx].pose, world[j], K)
            uvs[kf_idx].append(uv)
            entries.append((kf_idx, len(uvs[kf_idx]) - 1))
        per_point_observers.append(entries)

    for i in range(n_kfs):
        set_keypoints(kfs[i], uvs[i])

    for j, pt in enumerate(points):
        for kf_idx, kp_idx in per_point_observers[j]:
            pt.add_observation(Observation(kfs[kf_idx], pt, kp_idx))

    for kf in kfs:
        m.add_keyframe(kf)
    m.add_points(points)
    return m, kfs, points


def test_cull_redundant_keyframe_criteria():
    """Phase 2h step 3: a keyframe is redundant when >=
    CULL_REDUNDANCY_RATIO of its point-side points are observed by >=
    CULL_MIN_OTHER_OBSERVERS other keyframes. At most ONE (the oldest
    redundant candidate) is culled per call; keyframes[0] and the
    BA/relocation windows are protected."""
    K = make_K()

    # 15 KFs all observing all 10 points -> every candidate redundant.
    all_obs = [[i for i in range(15)] for _ in range(10)]
    m, kfs, points = _criterion_map(15, all_obs, K)

    assert m.cull_redundant_keyframe() is True
    assert m.cull_redundant_keyframe() is True
    # Oldest-first, one per call; the gauge anchor (keyframes[0]) is
    # never a candidate even though fully redundant.
    assert [kf.frame_id for kf in m.keyframes] == \
        [kfs[0].frame_id] + [kfs[i].frame_id for i in range(3, 15)]
    # Fully-redundant points survive culling with no orphan cascade
    # (13 remaining keyframe observers >= MINIMUM_OBSERVATIONS_FOR_POINT).
    assert len(m.points) == 10
    assert all(pt.point_id in m.points_by_id for pt in points)


def test_cull_redundant_keyframe_windows_protected():
    """Protected windows: with fewer keyframes than the protection
    arithmetic allows, no candidate exists and nothing is culled - even
    if every keyframe is fully redundant."""
    K = make_K()
    # 11 KFs, protected tail = max(LOCAL_MAP_WINDOW_SIZE=7,
    # RELOCALIZATION_WINDOW=10) = 10 -> candidates = keyframes[1:1] = [].
    all_obs = [[i for i in range(11)] for _ in range(10)]
    m, kfs, points = _criterion_map(11, all_obs, K)

    assert m.cull_redundant_keyframe() is False
    assert len(m.keyframes) == 11


def test_cull_redundant_keyframe_ratio_boundary():
    """The ratio boundary and the orphan interaction: a candidate whose
    bucket is 9/10 redundantly-observed points is culled (0.9 >= 0.9),
    and its one NON-redundant point is orphan-removed with it (only 2
    keyframe observers remain). 8/10 is below the threshold: no cull."""
    K = make_K()

    def observers_of(redundant, weak):
        # 10 points in kf1's bucket: `redundant` observed by all 12 KFs,
        # `weak` observed by kf1 + only two others (0 and the protected
        # tail keyframe 11).
        return ([list(range(12))] * redundant +
                [[1, 0, 11]] * weak)

    # 8/10 -> no cull
    m, kfs, points = _criterion_map(12, observers_of(8, 2), K)
    assert m.cull_redundant_keyframe() is False
    assert len(m.keyframes) == 12

    # 9/10 -> cull; the weak point orphan-removes with kf1
    m, kfs, points = _criterion_map(12, observers_of(9, 1), K)
    assert m.cull_redundant_keyframe() is True
    assert kfs[1].frame_id not in m.keyframes_by_id
    assert len(m.points) == 9, \
        "the culled keyframe's non-redundant point must orphan-remove"
    # The 9 redundant points each had 11 keyframe observers: they
    # survive (10 remaining >= MINIMUM_OBSERVATIONS_FOR_POINT).
    assert len(m.points_by_id) == 9


def test_cull_redundant_keyframe_requires_min_other_observers():
    """Points observed by only 2 other keyframes are NOT redundant
    evidence (CULL_MIN_OTHER_OBSERVERS = 3): no cull even at ratio 1.0
    of the bucket."""
    K = make_K()
    # Each point observed by exactly 3 keyframes (n_others = 2).
    observers = [[j % 12, (j % 12 + 5) % 12, (j % 12 + 7) % 12]
                 for j in range(30)]
    m, kfs, points = _criterion_map(12, observers, K)

    assert m.cull_redundant_keyframe() is False
    assert len(m.keyframes) == 12
    assert len(m.points) == 30
