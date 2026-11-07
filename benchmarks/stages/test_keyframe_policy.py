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
