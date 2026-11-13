"""Telemetry contract: observation-only, comprehensive, deterministic.

Phase 0 of the improvement program (benchmarks/IMPROVEMENT.md):
instrumentation must be a pure observer - it cannot perturb pipeline
numerics, it must actually cover the run, and its event counts (pipeline
decisions) must be as deterministic as the poses themselves.
"""
from slam.utils import telemetry
from benchmarks.synthetic.generator import ensure_sequence
from benchmarks.harness import run_slam


def test_stages_cover_run_without_double_counting():
    seq = ensure_sequence("translation")
    r = run_slam(seq, seed=42)

    assert r.telemetry["stage_seconds"], "no stages recorded"
    stage_total = sum(r.telemetry["stage_seconds"].values())
    # Non-overlapping by construction: stage totals can never exceed the
    # measured wall time.
    assert stage_total <= r.seconds + 1e-6, "stages double-count"
    # And instrumentation must cover the bulk of the run (image IO and
    # frame construction are the expected "other").
    assert stage_total >= 0.3 * r.seconds, "stage coverage too low"

    # The per-frame stages always run.
    expected = {"extract", "match", "project", "proj_match"}
    assert expected <= set(r.telemetry["stage_seconds"])
    # Keyframe machinery runs on every sequence that builds a map.
    assert r.num_keyframes >= 1
    assert {"map_optimize", "triangulate", "kf_match"} \
        <= set(r.telemetry["stage_seconds"])


def test_event_counts_are_deterministic():
    """Events count pipeline decisions, so they must be identical across
    identical runs - a stronger determinism signal than poses alone
    (an event flip without a pose change would still be a bug)."""
    seq = ensure_sequence("translation")
    r1 = run_slam(seq, seed=42)
    r2 = run_slam(seq, seed=42)

    assert r1.telemetry["events"] == r2.telemetry["events"], \
        "telemetry events differ between identical runs"
    assert r1.telemetry["stage_calls"] == r2.telemetry["stage_calls"]


def test_recorder_detached_and_noop_without_one():
    # run_slam detaches in a finally block; nothing may leak between runs.
    assert telemetry._recorder is None

    # With no recorder attached, stage()/event() must be no-ops.
    with telemetry.stage("nothing"):
        pass
    telemetry.event("nothing")
    assert telemetry._recorder is None
