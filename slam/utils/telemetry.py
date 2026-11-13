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
                               pnp_few_inliers is a legacy name: the
                               iterative solver counts INPUTS, not a
                               robust inlier set
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
                              is masked
  reloc_attempt / reloc_success / reloc_aggressive_attempt /
  reloc_aggressive_success / reloc_failed / system_reset
                              relocalization machinery
  points_pruned               outlier points removed by map pruning
  kf_culled / points_culled   keyframes removed by map culling, and the
                              map points orphan-removed with them
                              (below the keyframe-observation minimum)
  calib_gate_locked / calib_gate_freed / calib_gate_frozen
                              the 2j intrinsics refit's observability
                              gate: stays locked on critical-motion
                              sequences (single-axis translation, less
                              than 2 accumulated rotation axes - orbit,
                              translation, forward_rotation by design),
                              frees on diverse motion; the estimate
                              freezes after the one-shot stream-end
                              basin search
  calib_hook_frozen / calib_hook_free / calib_final_gate_locked
                              post-freeze hooks are no-ops; hooks after
                              the gate freed (the hook never searches
                              mid-stream - v3.1); the stream-end search
                              skipped because the gate never freed
  calib_search_done / calib_propagated / calib_search_failed
                              a basin search completed (the estimate
                              landed in the winning basin), a K/k1
                              propagation ran (the two normalized-
                              coordinate caches recomputed), or no
                              candidate produced a usable solve
  loop_detected / loop_verified
                              loop closure: the detector found
                              similarity candidates for a new keyframe;
                              a candidate PASSED the reloc-grade PnP
                              verification (the metric constraint)
  loop_rejected_correspondences / loop_rejected_pnp /
  loop_rejected_gates / loop_rejected_rotation
                              the verification rejection reasons: too
                              few map-point correspondences, PnP
                              failure, inlier-ratio / reprojection
                              gates, or the two-view rotation
                              disagreement (the degeneracy guard)
  loop_correction_skipped / loop_corrected / loop_global_ba /
  loop_correction_failed
                              the correction: a verified
                              constraint that AGREES with the odometry
                              chain (drift below
                              LOOP_MIN_DRIFT_M - tum's near-revisits);
                              a correction applied (pose graph), the
                              follow-up global BA re-aligning points,
                              or a failed pose graph
  intake_first_frame / intake_feed_displacement / intake_forced_max_gap /
  intake_few_features / intake_few_matches / intake_skip
                              displacement-gated frame
                              intake (video path only - the harness
                              feeds frame files and never constructs
                              a gate): why each DECODED frame was fed
                              (run start; displacement cleared the
                              threshold vs the last fed frame; the
                              max-gap time valve fired; measurement
                              impossible - too few features or too
                              few matches) or skipped
"""
import threading
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
        self.values = {}          # name -> [recorded values, in order]
        # Worker threads emit into the same dicts (the loop-scan pool):
        # `+=` on a dict value is not atomic under threads, so every
        # mutation takes this lock. Uncontended on the single-thread
        # path (one acquire per event, ~microseconds).
        self._lock = threading.Lock()

    def add_stage_time(self, name, seconds):
        with self._lock:
            self.stage_seconds[name] = (self.stage_seconds.get(name, 0.0)
                                        + seconds)
            self.stage_calls[name] = self.stage_calls.get(name, 0) + 1

    def add_event(self, name, count=1):
        with self._lock:
            self.events[name] = self.events.get(name, 0) + count

    def add_value(self, name, value):
        """Append to the keyed value store. SEPARATE from the events
        dict: nothing reads it unless the consumer asks, so runs that
        never call value() are byte-identical in their event/stage
        output (diagnostic-only instrumentation hangs off this)."""
        with self._lock:
            self.values.setdefault(name, []).append(value)

    def snapshot(self):
        return {
            "stage_seconds": dict(self.stage_seconds),
            "stage_calls": dict(self.stage_calls),
            "events": dict(self.events),
            "values": {k: list(v) for k, v in self.values.items()},
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


def value(name, v):
    """Record a diagnostic value (drift, a stage's max move, ...);
    no-op when no recorder is attached. Never read by the pipeline —
    the keyed store exists for the attribution tooling only."""
    rec = _recorder
    if rec is not None:
        rec.add_value(name, v)
