"""Read-only rotation/recovery diagnosis; no candidate policy or extra solve.

uv run python -m benchmarks.recovery_audit --output output/benchmarks/rotation_recovery_trace.json
uv run python -m benchmarks.recovery_audit --sequence translation --output output/benchmarks/phase7a_baseline_trace_translation.json

Replays one frozen native input (--sequence names a 6e matrix row; curated
sequences reuse the exact 6e native symlink views), checking exact 6e arrays,
metrics/events/stage calls and frame/cohort traces. Run full pytest BEFORE
this CLI, never concurrently with sequence readers. RPY is now diagnostic
development evidence, not a fresh held-out validation set for future fixes.
Hooks are process-global, sequential-only and restore on exception. They
call original functions once, never read lazy tracking quality/observation
properties, and never introduce matching, RANSAC, optimization or RNG draws.
Nested call event deltas overlap; do NOT sum them as independent events.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import traceback
from unittest.mock import patch

import cv2 as cv
import numpy as np

from benchmarks import metrics
from benchmarks import run as bench
from benchmarks.harness import load_trajectory, run_slam
from benchmarks.keyframe_audit import KeyframeAudit, SURVIVAL_HORIZONS, non_timing_differences
from benchmarks.keyframe_sensitivity import file_sha256, frame_tree_sha256
from slam.ba.bundle_adjustment_g2o import G2OBundleAdjustment
from slam.core.map import Map
from slam.core.tracker import Tracker
import slam.core.tracker as tracker_mod
import slam.features.matcher as matcher_mod
from slam.utils import constants as const, logger, telemetry

_active = False


def rotation_distance(a, b):
    """Geodesic rotation error in degrees; no alignment or scale fitting."""
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1.) / 2., -1., 1.))))


def relative_rotation_error(pose, reference_pose, gt, reference_gt):
    return rotation_distance(reference_pose[:3, :3].T @ pose[:3, :3],
                             reference_gt[:3, :3].T @ gt[:3, :3])


def _events():
    return dict(telemetry._recorder.events) if telemetry._recorder is not None else {}


class RecoveryAudit:
    def __init__(self):
        self.frames = []
        self._indices = {}
        self._current = None
        self._tracker = None
        self._stack_names = []
        self._epoch = 0

    def snapshot(self):
        tracker = self._tracker
        if tracker is None:
            return None
        world, frame = tracker.map, tracker.cur_frame
        history = list(world.tracking_quality_history.values())[-const.LOCAL_MAP_WINDOW_SIZE:]
        return {
            'current_index': self._indices.get(frame.frame_id) if frame is not None else None,
            'pose': frame.pose.tolist() if frame is not None else None,
            'previous_pose': tracker.prev_frame.pose.tolist() if tracker.prev_frame is not None else None,
            'current_is_live_keyframe': frame is not None and world.keyframes_by_id.get(frame.frame_id) is frame,
            'features': len(frame.keypoints) if frame is not None and frame.keypoints is not None else 0,
            'frame_side_observations': len(frame.observed_points) if frame is not None else 0,
            'cached_frame_quality': float(frame._tracking_quality) if frame is not None else None,
            'frame_quality_dirty': bool(frame.tracking_quality_needs_update) if frame is not None else None,
            'map_quality_from_stored_history': float(sum(history) / len(history)) if history else 0.,
            'low_quality_frames': world.consecutive_low_quality_frames,
            'successful_relocalization_counter': tracker.consecutive_successful_relocalizations,
            'failed_relocalization_counter': tracker.consecutive_relocalization_failures,
            'live_keyframes': len(world.keyframes), 'live_points': len(world.points),
            'velocity': tracker.velocity.tolist(), 'epoch': self._epoch,
        }

    def _wrap(self, original, name, details=None, returned=None):
        def call(*args, **kwargs):
            if self._current is None:
                return original(*args, **kwargs)
            row = {'name': name, 'context': list(self._stack_names), 'before': self.snapshot()}
            if details is not None:
                row.update(details(*args, **kwargs))
            self._current['calls'].append(row)
            before_events = _events()
            self._stack_names.append(name)
            try:
                result = original(*args, **kwargs)
                if returned is not None:
                    row.update(returned(result, args, kwargs))
                row['after'] = self.snapshot()
                row['events'] = {k: v - before_events.get(k, 0) for k, v in _events().items()
                                 if v != before_events.get(k, 0)}
                return result
            finally:
                self._stack_names.pop()
        return call

    def __enter__(self):
        global _active
        if _active:
            raise RuntimeError('RecoveryAudit contexts cannot overlap')
        _active = True
        self._stack = ExitStack()
        original_update, original_reset = Tracker.update, Tracker._reset_system

        def update(tracker, frame):
            self._tracker = tracker
            self._indices[frame.frame_id] = len(self.frames)
            self._current = {'input_index': len(self.frames), 'epoch_before': self._epoch, 'calls': []}
            before_events = _events()
            try:
                result = original_update(tracker, frame)
                self._current.update(final=self.snapshot(), epoch_after=self._epoch,
                                     events={k: v - before_events.get(k, 0) for k, v in _events().items()
                                             if v != before_events.get(k, 0)})
                self.frames.append(self._current)
                return result
            finally:
                self._current = None
                self._tracker = None

        def reset(tracker):
            result = original_reset(tracker)
            self._epoch += 1
            return result

        def pair_details(frame, reference, *args, **kwargs):
            return {'query_index': self._indices.get(frame.frame_id),
                    'reference_index': self._indices.get(reference.frame_id),
                    'self_reference': frame is reference, 'reference_pose': reference.pose.tolist()}

        def pnp_details(optimizer, frame=None):
            target = frame if frame is not None else optimizer.map.cur_keyframe
            return {'explicit_frame': frame is not None,
                    'target_index': self._indices.get(target.frame_id) if target is not None else None,
                    'target_pose_before': target.pose.tolist() if target is not None else None}

        def pnp_return(result, args, kwargs):
            optimizer = args[0]
            frame = args[1] if len(args) > 1 else kwargs.get('frame')
            target = frame if frame is not None else optimizer.map.cur_keyframe
            return {'success': bool(result[0]), 'input_count': int(result[1]),
                    'target_pose_after': target.pose.tolist() if target is not None else None}

        def match_return(result, args, kwargs):
            matches, relative = result
            return {'matches': len(matches), 'relative_pose': relative.tolist() if relative is not None else None}

        def fit_return(result, args, kwargs):
            model, mask, count = result
            return {'model': model.tolist() if model is not None else None,
                    'reported_inliers': int(count), 'input_count': len(args[0]),
                    'mask_entries': int(np.count_nonzero(mask)) if mask is not None else 0}

        def observation_return(result, args, kwargs):
            return {'observations': len(result), 'distinct_keypoints': len({o.kp_idx for o in result})}

        bindings = [(Tracker, 'update', update)]
        for name in ('_handle_keyframe_insertion', '_handle_loop_closure_detection',
                     '_handle_relocalization_check', '_attempt_relocalization',
                     '_attempt_aggressive_relocalization'):
            returned = ((lambda r, a, k: {'success': bool(r)}) if name.startswith('_attempt_') else None)
            bindings.append((Tracker, name, self._wrap(getattr(Tracker, name), name, returned=returned)))
        bindings.extend([
            (Tracker, '_reset_system', self._wrap(reset, '_reset_system')),
            (Tracker, '_get_next_pose_estimate', self._wrap(
                Tracker._get_next_pose_estimate, 'prediction', returned=lambda r, a, k: {'predicted_pose': r.tolist()})),
            (Tracker, '_match_projected_points', self._wrap(
                Tracker._match_projected_points, 'projection_matches', returned=observation_return)),
            (Tracker, '_match_frame_to_keyframe', self._wrap(
                Tracker._match_frame_to_keyframe, 'relocalization_pair',
                details=lambda tracker, frame, reference: pair_details(frame, reference),
                returned=lambda r, a, k: {'returned_pose': r[0].tolist() if r[0] is not None else None,
                                         'reported_inliers': int(r[1])})),
            (G2OBundleAdjustment, 'refine_pose_pnp', self._wrap(
                G2OBundleAdjustment.refine_pose_pnp, 'pnp', pnp_details, pnp_return)),
            (Map, 'needs_recovery', self._wrap(Map.needs_recovery, 'needs_recovery',
                                             returned=lambda r, a, k: {'result': bool(r)})),
            (Map, 'optimize', self._wrap(Map.optimize, 'map_optimize')),
        ])
        for name in ('local_bundle_adjustment', 'global_bundle_adjustment', 'pose_graph_optimization'):
            bindings.append((G2OBundleAdjustment, name,
                             self._wrap(getattr(G2OBundleAdjustment, name), name,
                                        returned=lambda r, a, k: {'success': bool(r)})))
        for owner in (tracker_mod, matcher_mod):
            bindings.append((owner, 'match_features', self._wrap(
                owner.match_features, 'descriptor_match', pair_details, match_return)))
            bindings.append((owner, 'estimate_essential_matrix', self._wrap(
                owner.estimate_essential_matrix, 'essential_fit', returned=fit_return)))
        for owner, name, replacement in bindings:
            self._stack.enter_context(patch.object(owner, name, replacement))
        return self

    def __exit__(self, *exc):
        global _active
        try:
            return self._stack.__exit__(*exc)
        finally:
            _active = False


def analyze_frames(frames, gt):
    """Gauge-invariant direct-pair errors and streaming errors kept separate."""
    pairs, resets, large_steps = [], [], []
    for index, frame in enumerate(frames):
        for call in frame['calls']:
            if call['name'] == 'relocalization_pair' and call.get('returned_pose') is not None:
                ref = call['reference_index']
                pairs.append({'input_index': index, 'reference_index': ref,
                              'self_reference': call['self_reference'], 'reported_inliers': call['reported_inliers'],
                              'pair_rotation_error_deg': relative_rotation_error(
                                  np.asarray(call['returned_pose']), np.asarray(call['reference_pose']), gt[index], gt[ref]),
                              'pose_change_deg': rotation_distance(np.asarray(call['before']['pose'])[:3, :3],
                                                                   np.asarray(call['returned_pose'])[:3, :3])})
            if call['name'] == '_reset_system':
                resets.append({'input_index': index, 'before': call['before'], 'after': call['after']})
        if index and frame['epoch_after'] == frames[index - 1]['epoch_after']:
            previous = np.asarray(frames[index - 1]['final']['pose'])
            error = relative_rotation_error(np.asarray(frame['final']['pose']), previous, gt[index], gt[index - 1])
            if error > 5.:
                large_steps.append({'input_index': index, 'rotation_error_deg': error,
                                    'diagnostic_cut_only_deg': 5.})
    return {'accepted_relocalization_pairs': pairs, 'reset_calls': resets,
            'stream_step_errors_above_diagnostic_5deg_excluding_resets': large_steps,
            'warning': 'Streaming errors can include map-gauge changes between emitted samples. Direct pair errors use the matched reference BEFORE its subsequent optimization. Threshold is diagnostic, never a runtime gate.'}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence', default='tum_rpy',
                        help='native 6e matrix row to replay (curated rows reuse the exact frozen 6e views)')
    parser.add_argument('--reference-matrix', type=Path,
                        default=root / 'output/benchmarks/phase6e_matrix.json')
    args = parser.parse_args()
    raw_path = args.output.with_suffix('.npz')
    if args.output.exists() or raw_path.exists():
        raise FileExistsError('Preserve prior evidence; choose a new output')
    baseline = json.loads(args.reference_matrix.read_text())
    reference = next(r for r in baseline['runs']
                     if r['sequence'] == args.sequence and r['mode'] == 'native')
    directory = Path(reference['input_manifest_file']['path']).parent
    assert frame_tree_sha256(directory) == reference['input_manifest']['input_frame_tree_sha256']
    sources = ['benchmarks/recovery_audit.py', 'benchmarks/harness.py', 'benchmarks/keyframe_audit.py',
               'benchmarks/stages/test_recovery_audit.py', 'slam/core/tracker.py', 'slam/core/map.py',
               'slam/core/frame.py', 'slam/core/point.py', 'slam/features/matcher.py',
               'slam/features/feature_extractor.py', 'slam/utils/utils.py', 'slam/utils/constants.py',
               'slam/ba/bundle_adjustment_g2o.py', 'slam/config/config.yaml']
    provenance = {p: {'sha256': file_sha256(root / p), 'source': (root / p).read_text()} for p in sources}
    logger.SEVERITY_FILTER = ['error']
    with RecoveryAudit() as trace:
        with KeyframeAudit() as audit:
            result = run_slam(directory, seed=baseline['seed'])
    gt = load_trajectory(directory)
    for insertion in audit.insertions:
        for horizon in SURVIVAL_HORIZONS:
            insertion['survival'].setdefault(str(horizon), {'status': 'horizon_not_reached'})
    with np.load(reference['raw_file']['path']) as native:
        arrays_exact = all(np.array_equal(native[name], getattr(result, name)) for name in
                           ('poses', 'map_points', 'keyframe_poses', 'keyframe_indices', 'keyframe_gt_indices'))
    checks = {'raw_arrays_exact': arrays_exact,
              'stage_calls_exact': result.telemetry['stage_calls'] == reference['telemetry']['stage_calls'],
              'frames_exact': audit.frames == reference['frames'],
              'cohorts_exact': audit.insertions == reference['insertions']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, poses=result.poses, map_points=result.map_points,
                        keyframe_poses=result.keyframe_poses, keyframe_indices=result.keyframe_indices,
                        keyframe_gt_indices=result.keyframe_gt_indices)
    report = {'phase': 'rotation_recovery_diagnosis', 'seed': baseline['seed'], 'sources': provenance,
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv.__version__},
              'input_manifest': reference['input_manifest'], 'reference_matrix_sha256': file_sha256(args.reference_matrix),
              'raw_file': {'path': str(raw_path), 'bytes': raw_path.stat().st_size, 'sha256': file_sha256(raw_path)},
              'verification': checks, 'telemetry': result.telemetry, 'frames': trace.frames,
              'status': 'collected_pending_analysis',
              'scope': 'Observation only; no runtime policy/threshold change, extra RANSAC or optimization. RPY is diagnostic, not fresh held-out evidence for any future fix.'}
    # Preserve a completed run BEFORE metric/report analysis, which can
    # fail independently of tracking and should never require a new solve.
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    try:
        # Synthetic views carry scene geometry; the held-out row has none.
        # Match the 6e evaluation inputs exactly (scene/planes or None).
        scene, planes = (bench.load_scene(directory)
                         if (directory / 'scene_points.txt').exists() else (None, None))
        actual = metrics.evaluate(result, gt, scene, planes)
        actual['final_intrinsics'] = result.final_intrinsics
        changes = non_timing_differences(actual, reference['metrics'])
        checks['non_timing_metrics_and_events_exact'] = not changes
        report.update(metrics=actual, changed_metrics=changes, analysis=analyze_frames(trace.frames, gt),
                      status='complete' if all(checks.values()) else 'control_mismatch')
    except Exception:
        report.update(status='analysis_failed', error=traceback.format_exc())
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(report['error'])
        return 1
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'saved': str(args.output), 'verification': checks, 'analysis': report['analysis']}, indent=2))
    return int(not all(checks.values()))


if __name__ == '__main__':
    raise SystemExit(main())
