"""The shared override resolver's pins (M4/M6): the origin-module
resolution and the live-pin mechanism the editor's Advanced tab and
the harness's --override share."""
import pytest

import slam.utils.constants as const
import slam.core.tracker as tracker_mod
import slam.features.matcher as matcher_mod
from slam.utils.overrides import (apply_override, apply_overrides,
                                  current_value, resolve_module)


def test_origin_resolution_beats_the_star_importers():
    # A constants.py name resolves to the CONSTANTS module (the
    # origins win over the star-importing modules that hold copies).
    assert resolve_module("RELOCALIZATION_QUALITY_THRESHOLD") is const
    # The machinery-side names resolve to their own modules.
    assert resolve_module("ROTMODEL_BOOTSTRAP_MIN_POINTS") is \
        tracker_mod
    assert resolve_module("PNP_GUARD_TAU_K") is \
        __import__("slam.ba.bundle_adjustment_g2o", fromlist=["x"])
    assert resolve_module("ROTATION_MODEL_T_COHERENCE_REARM") is \
        matcher_mod


def test_apply_override_sets_the_origin_live():
    was = const.RELOCALIZATION_QUALITY_THRESHOLD
    holder, parsed = apply_override("RELOCALIZATION_QUALITY_THRESHOLD",
                                    "0.45")
    assert holder is const
    assert parsed == 0.45
    assert const.RELOCALIZATION_QUALITY_THRESHOLD == 0.45
    # The readers through the module alias see the change live.
    assert tracker_mod.const.RELOCALIZATION_QUALITY_THRESHOLD == 0.45
    # The star-import copies follow the origin now (the live-pin
    # contract) - restore them too, or the later pins inherit 0.45.
    sync = apply_override.__globals__["_sync_copy_holders"]
    const.__dict__["RELOCALIZATION_QUALITY_THRESHOLD"] = was
    sync("RELOCALIZATION_QUALITY_THRESHOLD", 0.45, was, const)
    assert const.RELOCALIZATION_QUALITY_THRESHOLD == was
    assert tracker_mod.RELOCALIZATION_QUALITY_THRESHOLD == was
    assert matcher_mod.RELOCALIZATION_QUALITY_THRESHOLD == was


def test_apply_overrides_matches_the_benchmarks_cli_contract():
    was_min = const.LOOP_MIN_DRIFT_M
    was_max = const.LOOP_MAX_DRIFT_M
    apply_overrides(["LOOP_MIN_DRIFT_M=0.5", "LOOP_MAX_DRIFT_M=25.0"])
    assert const.LOOP_MIN_DRIFT_M == 0.5
    assert const.LOOP_MAX_DRIFT_M == 25.0
    const.LOOP_MIN_DRIFT_M = was_min
    const.LOOP_MAX_DRIFT_M = was_max


def test_apply_override_syncs_the_star_import_copies():
    # tracker/matcher/map star-import constants: their import-time
    # copies read at CALL time - the live-pin contract requires the
    # copies to follow the origin's update (the user-reported live
    # no-op fix).
    name = "RELOCALIZATION_QUALITY_THRESHOLD"
    was_o = getattr(const, name)
    was_t = getattr(tracker_mod, name)
    was_m = getattr(matcher_mod, name)
    try:
        apply_override(name, "0.44")
        assert getattr(const, name) == 0.44
        assert getattr(tracker_mod, name) == 0.44, \
            "the star-import copy did not follow"
        assert getattr(matcher_mod, name) == 0.44
    finally:
        const.__dict__[name] = was_o
        tracker_mod.__dict__[name] = was_t
        matcher_mod.__dict__[name] = was_m
        apply_override.__globals__["_sync_copy_holders"](
            name, 0.44, was_o, const)
        assert getattr(const, name) == was_o and \
            getattr(tracker_mod, name) == was_o and \
            getattr(matcher_mod, name) == was_o


def test_unknown_name_and_bad_parse_fail_loud():
    with pytest.raises(KeyError):
        apply_override("NOT_A_REAL_CONSTANT", "1")
    with pytest.raises(ValueError):
        apply_override("RELOCALIZATION_QUALITY_THRESHOLD", "abc")
