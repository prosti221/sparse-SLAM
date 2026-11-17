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
    const.RELOCALIZATION_QUALITY_THRESHOLD = was
    assert const.RELOCALIZATION_QUALITY_THRESHOLD == was


def test_apply_overrides_matches_the_benchmarks_cli_contract():
    was_min = const.LOOP_MIN_DRIFT_M
    was_max = const.LOOP_MAX_DRIFT_M
    apply_overrides(["LOOP_MIN_DRIFT_M=0.5", "LOOP_MAX_DRIFT_M=25.0"])
    assert const.LOOP_MIN_DRIFT_M == 0.5
    assert const.LOOP_MAX_DRIFT_M == 25.0
    const.LOOP_MIN_DRIFT_M = was_min
    const.LOOP_MAX_DRIFT_M = was_max


def test_unknown_name_and_bad_parse_fail_loud():
    with pytest.raises(KeyError):
        apply_override("NOT_A_REAL_CONSTANT", "1")
    with pytest.raises(ValueError):
        apply_override("RELOCALIZATION_QUALITY_THRESHOLD", "abc")
