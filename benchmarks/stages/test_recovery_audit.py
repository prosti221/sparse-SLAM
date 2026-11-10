"""Diagnostic pins: the reviewed 7a self-reference exclusion and 8-3
recovery-streak hygiene, plus characterization of other retained
recovery risks.

The exclusion tests pin the REVIEWED REPAIR (finding 29): a frame is never
its own relocalization reference. The 8-3 tests pin the REVIEWED REPAIR
for the streak counters' lifetime (finding 30): genuinely healthy tracking
clears both streak counters, so the success cap counts consecutive
quality-failing recoveries only. The remaining tests characterize
retained behavior; any further recovery change needs separately
reviewed pins.
"""
import json
from types import SimpleNamespace

import cv2 as cv
import numpy as np
import pytest

from benchmarks.harness import run_slam
from benchmarks.keyframe_audit import KeyframeAudit
from benchmarks.recovery_audit import RecoveryAudit, analyze_frames, relative_rotation_error
import benchmarks.recovery_audit as recovery_mod
from benchmarks.stages.helpers import make_frame, set_keypoints, yaw_pose
from benchmarks.synthetic.generator import ensure_sequence
from slam.ba.bundle_adjustment_g2o import G2OBundleAdjustment
from slam.core.frame import Frame
from slam.core.map import Map
from slam.core.tracker import Tracker
import slam.core.tracker as tracker_mod
import slam.features.matcher as matcher_mod
from slam.utils import telemetry


def _tracker():
    world = Map()
    frame = set_keypoints(make_frame(), [(30 + i, 30) for i in range(20)])
    world.add_keyframe(frame)
    tracker = Tracker(world)
    tracker.cur_frame = frame
    return tracker, frame


def test_relative_rotation_error_is_world_gauge_and_scale_independent():
    reference, current = yaw_pose(.2, (1, 2, 3)), yaw_pose(.5, (2, 3, 4))
    gauge = yaw_pose(1.7, (100, -30, 2))
    estimate_reference, estimate = gauge @ reference, gauge @ current
    estimate_reference[:3, 3] *= 80.
    estimate[:3, 3] *= 80.
    assert relative_rotation_error(estimate, estimate_reference, current, reference) < 1e-5
    estimate = estimate @ yaw_pose(np.radians(30))
    assert relative_rotation_error(estimate, estimate_reference, current, reference) == pytest.approx(30.)


def test_stream_analysis_excludes_reset_crossings_not_other_large_errors():
    poses = [yaw_pose(0.), yaw_pose(.1), yaw_pose(1.), yaw_pose(1.3)]
    frames = [{'calls': [], 'epoch_after': int(i >= 2), 'final': {'pose': p.tolist()}}
              for i, p in enumerate(poses)]
    analysis = analyze_frames(frames, np.repeat(np.eye(4)[None], 4, axis=0))
    assert [r['input_index'] for r in analysis['stream_step_errors_above_diagnostic_5deg_excluding_resets']] == [1, 3]


def test_snapshot_does_not_consume_quality_mutate_registries_or_draw_cv_rng(monkeypatch):
    tracker, frame = _tracker()
    tracker.map.tracking_quality_history[frame.frame_id] = .17
    frame.tracking_quality_needs_update = True
    audit = RecoveryAudit()
    audit._tracker = tracker
    audit._indices[frame.frame_id] = 0
    before = set(frame.__dict__), frame.pose, dict(frame.observed_points), dict(frame.keypoint_to_point_map)

    def forbidden(_):
        pytest.fail('Observer consumed lazy tracking quality')

    monkeypatch.setattr(Frame, 'tracking_quality', property(forbidden))
    cv.setRNGSeed(4181)
    expected = np.empty((6, 2))
    cv.randu(expected, 0., 1.)
    cv.setRNGSeed(4181)
    snapshot = audit.snapshot()
    actual = np.empty_like(expected)
    cv.randu(actual, 0., 1.)
    np.testing.assert_array_equal(actual, expected)
    assert snapshot['map_quality_from_stored_history'] == .17
    assert frame.tracking_quality_needs_update
    assert set(frame.__dict__) == before[0]
    np.testing.assert_array_equal(frame.pose, before[1])
    assert frame.observed_points == before[2] and frame.keypoint_to_point_map == before[3]


def test_observer_restores_actual_aliases_on_exception_and_rejects_overlap():
    bindings = [(Tracker, 'update'), (tracker_mod, 'match_features'), (matcher_mod, 'match_features'),
                (tracker_mod, 'estimate_essential_matrix'), (matcher_mod, 'estimate_essential_matrix'),
                (G2OBundleAdjustment, 'refine_pose_pnp'), (Map, 'needs_recovery')]
    originals = [getattr(owner, name) for owner, name in bindings]
    with pytest.raises(ValueError, match='injected'):
        with RecoveryAudit():
            with pytest.raises(RuntimeError, match='overlap'):
                with RecoveryAudit():
                    pass
            raise ValueError('injected')
    assert [getattr(owner, name) for owner, name in bindings] == originals


@pytest.mark.parametrize('healthy_frames', [1, 100])
def test_healthy_tracking_clears_recovery_streak_counters(monkeypatch, healthy_frames):
    """Phase 8-3 (finding 30's lifetime fix): 'consecutive' means
    consecutive. Genuinely healthy tracking (map quality at or above
    the recovery threshold) clears BOTH streak counters - a later
    recovery need starts counting from zero, and no recovery attempt
    is made under healthy tracking."""
    for _ in range(healthy_frames):
        tracker, frame = _tracker()
        tracker.consecutive_successful_relocalizations = 4
        tracker.consecutive_relocalization_failures = 1
        tracker.map.tracking_quality_history[frame.frame_id] = 0.7  # healthy
        monkeypatch.setattr(tracker.map, 'needs_recovery', lambda: False)
        monkeypatch.setattr(tracker, '_attempt_relocalization',
                            lambda: pytest.fail('attempted under healthy tracking'))
        tracker._handle_relocalization_check()
        assert tracker.consecutive_successful_relocalizations == 0
        assert tracker.consecutive_relocalization_failures == 0


def test_low_quality_gaps_do_not_clear_the_success_streak(monkeypatch):
    """Phase 8-3: the clear is gated on HEALTH, not on
    needs_recovery() == False. on_relocalization_successful resets
    consecutive_low_quality_frames, so the frames between two recovery
    cycles are always 'no recovery needed'; clearing there would make
    the cap unreachable dead code. A sub-threshold gap preserves the
    streak, and the 5th consecutive quality-failing recovery still
    resets BEFORE another recovery attempt (retained cap ordering)."""
    tracker, frame = _tracker()
    tracker.consecutive_successful_relocalizations = 5
    tracker.map.tracking_quality_history[frame.frame_id] = 0.1  # below threshold
    monkeypatch.setattr(tracker.map, 'needs_recovery', lambda: False)
    resets = []
    monkeypatch.setattr(tracker, '_reset_system', lambda: resets.append(True))
    for _ in range(4):
        tracker._handle_relocalization_check()
    assert tracker.consecutive_successful_relocalizations == 5 and not resets

    monkeypatch.setattr(tracker.map, 'needs_recovery', lambda: True)

    def forbidden():
        pytest.fail('Success-cap reset must occur BEFORE another recovery attempt')

    monkeypatch.setattr(tracker, '_attempt_relocalization', forbidden)
    tracker._handle_relocalization_check()
    assert resets == [True]


def test_recovered_quality_stops_the_ineffective_streak_reset(monkeypatch):
    """Phase 8-3: five cumulative recoveries no longer reset when
    tracking recovered in between - the decisive behavioral change.
    Quality restoration clears the streak, so the next recovery need
    starts counting from zero and the cap cannot fire immediately."""
    tracker, frame = _tracker()
    tracker.consecutive_successful_relocalizations = 5
    tracker.map.tracking_quality_history[frame.frame_id] = 0.5  # recovered
    monkeypatch.setattr(tracker.map, 'needs_recovery', lambda: False)
    tracker._handle_relocalization_check()
    assert tracker.consecutive_successful_relocalizations == 0

    # A NEW recovery need (quality sub-threshold again, as needs_recovery
    # requires) starts the count from zero - the cap cannot fire on the
    # five pre-recovery successes.
    tracker.map.tracking_quality_history[frame.frame_id] = 0.1
    monkeypatch.setattr(tracker.map, 'needs_recovery', lambda: True)
    attempts = []
    monkeypatch.setattr(tracker, '_attempt_relocalization',
                        lambda: attempts.append(True) or True)
    resets = []
    monkeypatch.setattr(tracker, '_reset_system', lambda: resets.append(True))
    tracker._handle_relocalization_check()
    assert attempts == [True] and resets == []


def _inserted_query_tracker():
    """A just-inserted query keyframe plus one older keyframe (finding 29)."""
    world = Map()
    older = set_keypoints(make_frame(yaw_pose(0.1, (0.2, 0., 0.))), [(30 + i, 30) for i in range(20)])
    world.add_keyframe(older)
    current = set_keypoints(make_frame(), [(30 + i, 30) for i in range(20)])
    world.add_keyframe(current)
    tracker = Tracker(world)
    tracker.cur_frame = current
    return tracker, older, current


def _accepting_match(monkeypatch, tracker, relative):
    """Accept every offered pair; records (query, reference) identity."""
    matches = [cv.DMatch(i, i, 0.) for i in range(20)]
    pairs = []

    def match(query, reference, method):
        pairs.append((query, reference))
        return matches, relative

    monkeypatch.setattr(tracker_mod, 'match_features', match)
    monkeypatch.setattr(tracker_mod, 'estimate_essential_matrix',
                        lambda a, b: (np.eye(3), np.ones((len(a), 1), dtype=np.uint8), len(a)))
    optimization_results = []
    monkeypatch.setattr(tracker.map, 'optimize', lambda **kw: optimization_results.append(True))
    return pairs, optimization_results


def test_relocalization_never_matches_the_query_frame_against_itself(monkeypatch):
    """Finding 29 repair: zero-baseline self-pairs are excluded entirely.

    With the query as the ONLY keyframe, both recovery paths fail rather
    than accept a meaningless identical-image match; the matcher is never
    even called, and no pose/counter state changes.
    """
    tracker, frame = _tracker()
    pose = frame.pose
    pairs, _ = _accepting_match(monkeypatch, tracker, yaw_pose(np.pi, (1., 0., 0.)))

    assert not tracker._attempt_relocalization()
    assert not tracker._attempt_aggressive_relocalization()

    assert pairs == []  # the self candidate is skipped before any matching
    np.testing.assert_array_equal(tracker.cur_frame.pose, pose)
    assert tracker.consecutive_successful_relocalizations == 0
    assert tracker.consecutive_relocalization_failures == 0


def test_recovery_skips_the_query_and_uses_the_next_newest_keyframe(monkeypatch):
    """Exclusion must not disable recovery: the older reference is used."""
    tracker, older, current = _inserted_query_tracker()
    relative = yaw_pose(np.radians(12.), (0.5, 0., 0.))
    pairs, optimization_results = _accepting_match(monkeypatch, tracker, relative)

    assert tracker._attempt_relocalization()

    assert pairs == [(current, older)]  # self skipped, newest OTHER first
    np.testing.assert_allclose(current.pose, older.pose @ relative, atol=1e-15, rtol=0)
    assert optimization_results == [True]
    assert tracker.consecutive_successful_relocalizations == 1
    np.testing.assert_array_equal(tracker.velocity, np.eye(4))


def test_aggressive_relocalization_also_excludes_the_query_frame(monkeypatch):
    """The permissive path gets the same exclusion (finding 29).

    Existing branch-drop semantics are unchanged: accepting the older
    reference still removes keyframes inserted after it - including the
    just-inserted query itself.
    """
    tracker, older, current = _inserted_query_tracker()
    pairs, optimization_results = _accepting_match(monkeypatch, tracker, yaw_pose(0.))

    assert tracker._attempt_aggressive_relocalization()

    assert pairs == [(current, older)]
    assert [kf for kf in tracker.map.keyframes] == [older]
    assert current.frame_id not in tracker.map.keyframes_by_id
    assert optimization_results == [True]


def test_real_shadow_preserves_arrays_events_calls_cohorts_and_cv_rng(tmp_path, monkeypatch):
    sequence = ensure_sequence('translation')
    (tmp_path / 'calibration.txt').symlink_to(sequence / 'calibration.txt')
    (tmp_path / 'frames').mkdir()
    for path in sorted((sequence / 'frames').glob('*.png'))[:24]:
        (tmp_path / 'frames' / path.name).symlink_to(path)
    original_fit = cv.findEssentialMat
    fit_calls = []

    def fit(*args, **kwargs):
        fit_calls.append(True)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(cv, 'findEssentialMat', fit)
    with KeyframeAudit() as control_audit:
        control = run_slam(tmp_path)
    original_count = len(fit_calls)
    expected_rng = np.empty((8, 2))
    cv.randu(expected_rng, 0., 1.)
    fit_calls.clear()
    with RecoveryAudit() as trace:
        with KeyframeAudit() as audit:
            actual = run_slam(tmp_path)
    actual_rng = np.empty_like(expected_rng)
    cv.randu(actual_rng, 0., 1.)
    np.testing.assert_array_equal(actual_rng, expected_rng)
    assert len(fit_calls) == original_count
    for key in ('poses', 'map_points', 'keyframe_poses'):
        np.testing.assert_array_equal(getattr(actual, key), getattr(control, key))
    assert actual.keyframe_indices == control.keyframe_indices
    assert actual.keyframe_gt_indices == control.keyframe_gt_indices
    assert actual.telemetry['events'] == control.telemetry['events']
    assert actual.telemetry['stage_calls'] == control.telemetry['stage_calls']
    assert audit.frames == control_audit.frames and audit.insertions == control_audit.insertions
    assert len(trace.frames) == 24
    assert sum(call['name'] == 'essential_fit' for f in trace.frames for call in f['calls']) == original_count
    for index, frame in enumerate(trace.frames):
        np.testing.assert_array_equal(frame['final']['pose'], actual.poses[index])


@pytest.mark.parametrize('analysis_fails', [False, True])
def test_cli_calls_real_evaluator_contract_and_preserves_completed_run_on_analysis_error(
        tmp_path, monkeypatch, analysis_fails):
    """No ignored data dependency: exercise CLI/reporting with a small result."""
    directory = tmp_path / 'dataset'
    directory.mkdir()
    np.savetxt(directory / 'trajectory.txt', [[i, 0, 0, 0, 0, 0, 0, 1] for i in range(3)])
    result = SimpleNamespace(poses=np.repeat(np.eye(4)[None], 3, axis=0),
                             map_points=np.empty((0, 3)), keyframe_poses=np.eye(4)[None],
                             keyframe_indices=[0], keyframe_gt_indices=[0],
                             telemetry={'events': {}, 'stage_calls': {}}, final_intrinsics={})
    raw_control = tmp_path / 'control.npz'
    np.savez_compressed(raw_control, **{name: getattr(result, name) for name in
                        ('poses', 'map_points', 'keyframe_poses', 'keyframe_indices', 'keyframe_gt_indices')})
    matrix = tmp_path / 'matrix.json'
    matrix.write_text(json.dumps({'seed': 42, 'runs': [{
        'sequence': 'tum_rpy', 'mode': 'native',
        'input_manifest_file': {'path': str(directory / 'source_manifest.json')},
        'input_manifest': {'input_frame_tree_sha256': 'fixture', 'source_directory': str(directory)},
        'raw_file': {'path': str(raw_control)}, 'frames': [], 'insertions': [],
        'telemetry': result.telemetry, 'metrics': {'diagnostic_metric': 1, 'final_intrinsics': {}},
    }]}))
    monkeypatch.setattr(recovery_mod, 'frame_tree_sha256', lambda p: 'fixture')
    runs = []

    def run(path, seed):
        runs.append((path, seed))
        return result

    def evaluate(actual, gt, scene_points, planes):
        assert actual is result and len(gt) == 3
        assert scene_points is None and planes is None
        if analysis_fails:
            raise RuntimeError('injected postprocessing error')
        return {'diagnostic_metric': 1}

    monkeypatch.setattr(recovery_mod, 'run_slam', run)
    monkeypatch.setattr(recovery_mod.metrics, 'evaluate', evaluate)
    monkeypatch.setattr(recovery_mod.logger, 'SEVERITY_FILTER', list(recovery_mod.logger.SEVERITY_FILTER))
    output = tmp_path / 'trace.json'
    monkeypatch.setattr('sys.argv', ['recovery_audit', '--output', str(output), '--reference-matrix', str(matrix)])
    assert recovery_mod.main() == int(analysis_fails)
    assert runs == [(directory, 42)]
    saved = json.loads(output.read_text())
    assert saved['status'] == ('analysis_failed' if analysis_fails else 'complete')
    if analysis_fails:
        assert 'injected postprocessing error' in saved['error']
    else:
        assert all(saved['verification'].values())
    with np.load(output.with_suffix('.npz')) as raw:
        np.testing.assert_array_equal(raw['poses'], result.poses)
    with pytest.raises(FileExistsError):
        recovery_mod.main()
    assert len(runs) == 1
