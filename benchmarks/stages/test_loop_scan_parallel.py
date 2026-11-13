"""Stage pins: the loop-scan worker pool ([26b]).

The pool parallelizes two independent, order-independent loops - the
DB candidate scan (one similarity score per stored keyframe) and the
candidate verification (one reloc-grade PnP per candidate). The
[26a] audit proved every worker call cv-RNG-clean and GIL-releasing;
telemetry counting is lock-guarded. The contract: with the pool
engaged, candidate lists, similarity scores, verified constraints,
and event counts are IDENTICAL to the sequential path, invariant to
the worker count. Flag 0/1 = the sequential shipped path.
"""
import threading

import numpy as np
import pytest

from benchmarks.stages.test_loop_closure import _loop_fixture
from slam.core import tracker as tracker_mod
from slam.loop_closure import LOOP_SCAN_THREADS
from slam.utils import telemetry


def test_scan_threads_shipped_default_is_sequential():
    assert LOOP_SCAN_THREADS in (0, 1)


def _run_paths(monkeypatch, threads):
    """Run the scan + verification on ONE fixture per flag value; the
    fixture's keyframe ids must match across runs, so the fixture is
    built once and both paths run against it."""
    monkeypatch.setattr("slam.loop_closure.LOOP_SCAN_THREADS", threads)
    import slam.utils.constants as const
    monkeypatch.setattr(const, "LOOP_EXCLUDE_RECENT_KF", 0)
    # push BOTH fixture keyframes past the gate so the verification
    # pool engages (the spurious keyframe sits below the 0.3 gate)
    monkeypatch.setattr(const, "LOOP_SIMILARITY_THRESHOLD", 0.0)
    m, f, info = _loop_fixture()
    detector = m.loop_detector
    query = f["query"]
    out = []
    for threads_value in threads:
        monkeypatch.setattr("slam.loop_closure.LOOP_SCAN_THREADS",
                            threads_value)
        rec = telemetry.TelemetryRecorder()
        telemetry.set_recorder(rec)
        try:
            candidates = detector.detect_potential_loops(query)
            verified = detector.verify_loop_candidates(candidates)
        finally:
            telemetry.set_recorder(None)
        out.append((candidates, verified, dict(rec.events)))
    return out


def _verified_key(pairs):
    """match id + exact pose bytes: the per-call-seeded PnP makes
    verified poses deterministic, so byte equality is the contract."""
    return [(v.match_keyframe_id, v.relative_pose.tobytes())
            for v in pairs]


def test_pool_results_identical_to_sequential(monkeypatch):
    (seq_c, seq_v, seq_ev), (par_c, par_v, par_ev) = \
        _run_paths(monkeypatch, [0, 4])
    assert len(par_c) > 1, "the pin needs several candidates for the pool"
    # candidate order and scores bit-exact
    assert [(c.match_keyframe_id, c.similarity_score)
            for c in seq_c] == [(c.match_keyframe_id, c.similarity_score)
                                for c in par_c]
    # verified set bit-exact (poses are per-call-seeded deterministic)
    assert _verified_key(seq_v) == _verified_key(par_v)
    # event counts exact (lock-guarded merging)
    assert seq_ev == par_ev


def test_pool_thread_count_invariance(monkeypatch):
    (two_c, two_v, two_ev), (four_c, four_v, four_ev) = \
        _run_paths(monkeypatch, [2, 4])
    assert [(c.match_keyframe_id, c.similarity_score)
            for c in two_c] == [(c.match_keyframe_id, c.similarity_score)
                                for c in four_c]
    assert _verified_key(two_v) == _verified_key(four_v)
    assert two_ev == four_ev


def test_telemetry_lock_survives_concurrent_events():
    rec = telemetry.TelemetryRecorder()
    N_THREADS, N_EVENTS = 8, 2000

    def emit():
        for _ in range(N_EVENTS):
            telemetry.set_recorder(rec)
            telemetry.event("concurrent_probe")
            telemetry.set_recorder(None)

    threads = [threading.Thread(target=emit)
               for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert rec.events["concurrent_probe"] == N_THREADS * N_EVENTS
