"""6c experimental boundaries; these selectors are not runtime defaults."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.harness import run_slam
from benchmarks.keyframe_audit import KeyframeAudit
from benchmarks.keyframe_policy import (
    KeyframePolicyExperiment, SELECTORS, SelectorState, evaluate_selector,
)
from benchmarks.synthetic.generator import ensure_sequence
import slam.core.tracker as tracker_mod


def _signals():
    reference = {"parallax_deg": {"median": 0.9, "finite_count": 20},
                 "baseline_over_depth": 0.03, "shared_consistent": 20,
                 "visible_match_fraction": 0.7}
    return {"features": 100, "previous_frame_matches": 80,
            "baseline": 3.0, "rotation_deg": 0.0, "legacy_quality": 0.5,
            "legacy_new_points_ratio": 9.0, "matched_unmapped_fraction": 0.8,
            "support_grid_at_3px": {"occupied_cells": 12, "occupied_fraction": 0.75},
            "last_reference": deepcopy(reference), "best_local_reference": deepcopy(reference)}


def _decide(name, signals=None, state=None, gap=3, reference="kf", legacy=False, pose=True):
    return evaluate_selector(SELECTORS[name], state if state is not None else SelectorState(),
                             signals if signals is not None else _signals(), gap, reference, legacy, pose)


@pytest.mark.parametrize("name,selected", [
    ("parallax_080", True), ("parallax_120", False),
    ("depth_025", True), ("depth_040", False),
])
def test_geometry_thresholds_replace_raw_distance_only_with_evidence(name, selected):
    decision = _decide(name, legacy=not selected)
    assert decision.selected is selected
    assert decision.evidence["geometry_available"]


@pytest.mark.parametrize("missing", ["shared", "parallax", "grid", "pose_failed", "pose_unattempted"])
@pytest.mark.parametrize("legacy", [False, True])
def test_weak_mapped_context_preserves_legacy_not_a_new_rescue(missing, legacy):
    signals = _signals()
    pose = True
    if missing == "shared":
        signals["last_reference"]["parallax_deg"]["finite_count"] = 9
    elif missing == "parallax":
        signals["last_reference"]["parallax_deg"]["median"] = None
    elif missing == "grid":
        signals["support_grid_at_3px"]["occupied_cells"] = 3
    elif missing == "pose_failed":
        pose = False
    else:
        pose = None
    decision = _decide("hybrid_last", signals, legacy=legacy, pose=pose)
    assert decision.selected is legacy
    assert decision.reason == ("legacy_map_renewal" if legacy else "legacy_wait")


@pytest.mark.parametrize("name", ["parallax_120", "depth_040", "hybrid_last"])
def test_tracking_rotation_does_not_require_depth_parallax(name):
    signals = _signals()
    signals["rotation_deg"] = 5.0
    signals["last_reference"]["parallax_deg"]["median"] = 0.0
    signals["last_reference"]["baseline_over_depth"] = 0.0
    assert _decide(name, signals).reason == "view_rotation"
    assert _decide(name, signals).selected
    assert not _decide(name, signals, gap=2).selected


def test_reference_renewal_requires_persistence_and_resets_on_missing_evidence():
    signals = _signals()
    signals["last_reference"]["parallax_deg"]["median"] = 0.7
    signals["last_reference"]["visible_match_fraction"] = 0.1
    state = SelectorState()
    assert not _decide("hybrid_last", signals, state, gap=3).selected
    assert _decide("hybrid_last", signals, state, gap=4).reason == "reference_renewal"
    assert not _decide("hybrid_last", signals, state, gap=3, reference="next_kf").selected
    assert not _decide("hybrid_last", signals, state, gap=4, reference="next_kf", pose=False).selected
    assert state.renewal_streak == 0
    assert not _decide("hybrid_last", signals, state, gap=5, reference="next_kf").selected


def test_coverage_renewal_is_persistent_and_needs_geometric_change():
    signals = _signals()
    signals["support_grid_at_3px"] = {"occupied_cells": 6, "occupied_fraction": 6 / 16}
    signals["last_reference"]["parallax_deg"]["median"] = 0.7
    state = SelectorState()
    assert not _decide("hybrid_last", signals, state, gap=3).selected
    assert _decide("hybrid_last", signals, state, gap=4).reason == "coverage_renewal"
    signals["last_reference"]["parallax_deg"]["median"] = 0.2
    assert not _decide("hybrid_last", signals, state, gap=5).selected


def test_local_tracking_reference_does_not_change_the_triangulation_baseline_signal():
    signals = _signals()
    signals["last_reference"]["visible_match_fraction"] = 0.1
    signals["last_reference"]["parallax_deg"]["median"] = 0.7
    signals["best_local_reference"]["parallax_deg"]["median"] = 20.0
    last_state, local_state = SelectorState(), SelectorState()
    for gap in (3, 4):
        last = _decide("hybrid_last", signals, last_state, gap=gap)
        local = _decide("hybrid_local", signals, local_state, gap=gap)
    assert last.selected
    assert not local.selected
    assert local.evidence["parallax_deg"] == 0.7


@pytest.mark.parametrize("field,value", [
    ("features", 0), ("previous_frame_matches", 0),
    ("baseline", float("nan")), ("rotation_deg", float("nan")),
])
def test_unusable_image_evidence_is_not_a_policy_admission(field, value):
    signals = _signals()
    signals[field] = value
    decision = _decide("hybrid_last", signals, gap=10, legacy=True)
    assert not decision.selected
    assert decision.reason == "unusable_image_evidence"
    # The production caller STILL has its independent forced-gap rule.


@pytest.mark.parametrize("name", ["parallax_080", "parallax_120", "depth_025", "depth_040"])
def test_mapped_geometry_branch_is_independent_of_map_scale(name):
    small, large = _signals(), _signals()
    small["baseline"] *= 0.001
    large["baseline"] *= 1000
    assert _decide(name, small, legacy=False).selected == _decide(name, large, legacy=True).selected


def test_hysteresis_resets_when_the_same_reference_counter_rolls_back():
    signals = _signals()
    signals["last_reference"]["parallax_deg"]["median"] = 0.7
    signals["last_reference"]["visible_match_fraction"] = 0.1
    state = SelectorState()
    _decide("hybrid_last", signals, state, gap=4)
    _decide("hybrid_last", signals, state, gap=5)
    assert not _decide("hybrid_last", signals, state, gap=3).selected
    assert state.renewal_streak == 1


def test_fraction_only_candidate_uses_the_correct_population_but_keeps_quality_or():
    signals = _signals()
    signals["matched_unmapped_fraction"] = 0.2
    signals["legacy_quality"] = 0.0
    assert not _decide("matched_fraction", signals, legacy=True).selected
    signals["matched_unmapped_fraction"] = 0.4
    assert _decide("matched_fraction", signals).selected
    signals["matched_unmapped_fraction"] = None
    signals["legacy_quality"] = 0.5
    assert _decide("matched_fraction", signals).selected


def test_candidate_reads_do_not_mutate_the_signal_snapshot():
    signals = _signals()
    original = deepcopy(signals)
    for name in SELECTORS:
        _decide(name, signals)
    assert signals == original


def test_experiment_restores_bindings_and_respects_callers_forced_override(monkeypatch):
    original = lambda *a, **kw: (False, "legacy")
    monkeypatch.setattr(tracker_mod, "should_insert_keyframe", original)
    signals = _signals()
    signals["features"] = 0
    row = {"signals": signals, "policy_selected": False, "primary_reason": "maximum_gap_only",
           "flags": {"maximum_gap": True}}
    audit = SimpleNamespace(_current={"decision": row, "pnp_calls": []})
    with pytest.raises(ValueError, match="injected"):
        with KeyframePolicyExperiment(audit, "hybrid_last"):
            selected, _ = tracker_mod.should_insert_keyframe(None, SimpleNamespace(frame_id="kf"), 10)
            assert not selected
            assert row["effective_selected"]  # Caller override is not hidden by the experiment.
            assert row["primary_reason"] == "maximum_gap_only"
            raise ValueError("injected")
    assert tracker_mod.should_insert_keyframe is original


def test_actual_shadow_is_exact_and_records_every_declared_candidate(tmp_path):
    sequence = ensure_sequence("translation")
    (tmp_path / "calibration.txt").symlink_to(sequence / "calibration.txt")
    (tmp_path / "frames").mkdir()
    for path in sorted((sequence / "frames").glob("*.png"))[:16]:
        (tmp_path / "frames" / path.name).symlink_to(path)
    control = run_slam(tmp_path)
    with KeyframeAudit() as audit:
        with KeyframePolicyExperiment(audit, "shadow"):
            observed = run_slam(tmp_path)
    np.testing.assert_array_equal(observed.poses, control.poses)
    np.testing.assert_array_equal(observed.map_points, control.map_points)
    assert observed.telemetry["events"] == control.telemetry["events"]
    assert observed.keyframe_indices == control.keyframe_indices
    for frame in audit.frames[1:]:
        assert set(frame["decision"]["experimental_decisions"]) == set(SELECTORS)


@pytest.mark.parametrize("name", ["parallax_120", "depth_040"])
def test_gap2_ablation_changes_geometry_cadence_without_relaxing_rotation(name):
    config = replace(SELECTORS[name], name=name + "_gap2", minimum_gap=2)
    signals = _signals()
    signals["last_reference"]["parallax_deg"]["median"] = 4.0
    signals["last_reference"]["baseline_over_depth"] = 0.08
    decision = evaluate_selector(config, SelectorState(), signals, 2, "kf", False, True)
    assert decision.selected and decision.reason == "geometry_growth"
    assert not _decide(name, signals, gap=2).selected


@pytest.mark.parametrize("name", ["parallax_120", "depth_040"])
def test_gap2_ablation_does_not_reintroduce_the_rejected_gap2_rotation_trigger(name):
    config = replace(SELECTORS[name], name=name + "_gap2", minimum_gap=2)
    signals = _signals()
    signals["rotation_deg"] = 5.0
    signals["last_reference"]["parallax_deg"]["median"] = 0.0
    signals["last_reference"]["baseline_over_depth"] = 0.0
    assert not evaluate_selector(config, SelectorState(), signals, 2, "kf", False, True).selected
