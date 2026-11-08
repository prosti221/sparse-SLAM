"""Observation-only telemetry for the SLAM pipeline.

The benchmark harness attaches a recorder; production runs without one
behave identically - every helper here is a no-op with negligible overhead
when no recorder is attached, so instrumentation cannot perturb results.

Stage timings are non-overlapping by construction: each `stage()` marks a
specific call site in the pipeline, so stage totals never double-count.
Time not covered by any stage shows up as "other" in the report.

Event keys (counted pipeline decisions - deterministic per run):

  match_fallback              frame-to-frame matching returned no pose;
                              constant-velocity estimate was used
  match_no_ransac             too few good matches for RANSAC estimation
  match_zero                  zero matches between two frames
  pnp_frame_ok / pnp_frame_fail   outcome of the per-frame PnP call site
  pnp_ok / pnp_few_points / pnp_no_solution / pnp_few_inliers
                              refine_pose_pnp outcomes (all call sites)
  ba_local_call / ba_global_call / ba_success / ba_failure
                              map.optimize BA branch outcomes
  ba_iterations              total LM iterations actually run by BA
                              (early-exit via the terminate action makes
                              this smaller than 30 x ba calls)
  ba_insufficient_observations / ba_global_insufficient_observations /
  ba_no_local_keyframes / ba_too_few_edges / ba_exception
                              BA rejection reasons
  tri_accepted / tri_rejected / tri_reject_<CODE>
                              triangulation validation totals and reasons
  dedup_skip_existing       duplicate-keypoint matches whose (point,
                              frame) pair was already registered by
                              projection matching - the projection
                              observation is kept and only the keypoint
                              is masked (phase 2g, finding 17)
  reloc_attempt / reloc_success / reloc_aggressive_attempt /
  reloc_aggressive_success / reloc_failed / system_reset
                              relocalization machinery
  points_pruned               outlier points removed by map pruning
  kf_culled / points_culled   keyframes removed by map culling, and the
                              map points orphan-removed with them
                              (below the keyframe-observation minimum;
                              phase 2h)
"""
import time
from contextlib import contextmanager

_recorder = None


def set_recorder(recorder):
    """Attach the telemetry recorder (or detach it with None)."""
    global _recorder
    _recorder = recorder


class TelemetryRecorder:
    """Accumulates stage timings and event counts for one run."""

    def __init__(self):
        self.stage_seconds = {}   # stage name -> total seconds
        self.stage_calls = {}     # stage name -> call count
        self.events = {}          # event name -> count

    def add_stage_time(self, name, seconds):
        self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + seconds
        self.stage_calls[name] = self.stage_calls.get(name, 0) + 1

    def add_event(self, name, count=1):
        self.events[name] = self.events.get(name, 0) + count

    def snapshot(self):
        return {
            "stage_seconds": dict(self.stage_seconds),
            "stage_calls": dict(self.stage_calls),
            "events": dict(self.events),
        }


@contextmanager
def stage(name):
    """Time a pipeline stage; no-op when no recorder is attached."""
    rec = _recorder
    if rec is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        rec.add_stage_time(name, time.perf_counter() - t0)


def event(name, count=1):
    """Count a pipeline event; no-op when no recorder is attached."""
    rec = _recorder
    if rec is not None:
        rec.add_event(name, count)
