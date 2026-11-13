"""Stage pins: the loop-correction attribution instrumentation.

The attribution flag is observation-only: at either value the
pipeline's events and numerics are unchanged, and at the shipped
default (0) the recorder's value store stays EMPTY - the battery's
event/stage output is byte-identical by construction. The pins hold:

- the shipped default (0) and its silence (a correction that FIRES
  still records no values);
- the flag-on emission contract: a correction records one drift per
  constraint plus the stage magnitudes (graph max move, BA outcome
  + max move, constraint count, post map size);
- event-silence: the flag changes no event counts (the value store
  is separate from the events dict);
- the recorder's value() contract itself (append order, events
  untouched, snapshot copies).
"""
import numpy as np
import pytest

from benchmarks.stages.test_loop_closure import _loop_fixture
from slam.core import tracker as tracker_mod
from slam.utils import telemetry


def test_attrib_shipped_default_is_off():
    assert tracker_mod.LOOP_CORRECTION_ATTRIB_ENABLED == 0


def _run_hook(monkeypatch, flag):
    m, f, info = _loop_fixture()
    monkeypatch.setattr(tracker_mod, "LOOP_CORRECTION_ATTRIB_ENABLED", flag)
    import slam.utils.constants as const
    monkeypatch.setattr(const, "LOOP_EXCLUDE_RECENT_KF", 0)
    tracker = tracker_mod.Tracker(m)
    rec = telemetry.TelemetryRecorder()
    telemetry.set_recorder(rec)
    try:
        tracker._handle_loop_closure_detection()
    finally:
        telemetry.set_recorder(None)
    assert rec.events.get("loop_corrected", 0) >= 1, \
        "the fixture's drifted revisit must correct for this pin"
    return rec


def test_attrib_flag_off_records_no_values(monkeypatch):
    """Shipped default: the correction FIRES and the value store stays
    empty - the battery never sees attribution data."""
    rec = _run_hook(monkeypatch, 0)
    assert rec.values == {}


def test_attrib_flag_on_records_magnitudes(monkeypatch):
    """Flag ON: one drift value per constraint, plus the stage
    magnitudes - all observation, no behavior change."""
    rec = _run_hook(monkeypatch, 1)
    n_constraints = rec.values["loop_attr_constraints"][0]
    assert n_constraints >= 1
    assert len(rec.values["loop_attr_drift"]) == n_constraints
    for d in rec.values["loop_attr_drift"]:
        assert d > 0.0
    assert rec.values["loop_attr_graph_maxmove"][0] >= 0.0
    assert rec.values["loop_attr_ba_ok"][0] in (0.0, 1.0)
    assert rec.values["loop_attr_ba_maxmove"][0] >= 0.0
    assert rec.values["loop_attr_map_points"][0] > 0


def test_attrib_is_event_silent(monkeypatch):
    """The flag changes NO event counts: a flag-off run and a flag-on
    run of the same correction produce identical events dicts."""
    off = _run_hook(monkeypatch, 0)
    on = _run_hook(monkeypatch, 1)
    assert off.events == on.events


def test_value_store_contract():
    """value() appends in order into its own store; the events dict is
    untouched; snapshot() deep-copies so later writes cannot mutate a
    taken snapshot."""
    rec = telemetry.TelemetryRecorder()
    telemetry.set_recorder(rec)
    try:
        telemetry.value("probe_a", 1.5)
        telemetry.value("probe_a", 2.5)
        telemetry.value("probe_b", 7)
        telemetry.event("probe_event", 2)
    finally:
        telemetry.set_recorder(None)
    assert rec.values == {"probe_a": [1.5, 2.5], "probe_b": [7]}
    assert rec.events == {"probe_event": 2}
    snap = rec.snapshot()
    rec.add_value("probe_a", 9.9)
    assert snap["values"]["probe_a"] == [1.5, 2.5]
    assert rec.values["probe_a"] == [1.5, 2.5, 9.9]
