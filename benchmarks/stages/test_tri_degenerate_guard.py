"""Stage test: the degenerate-direction guard on the
triangulation creation gate (TRIANGULATION_DEGENERATE_GUARD_ENABLED).

A candidate whose direction vectors are non-normalizable has NO
parallax. The guarded path rejects it deterministically under the
parallax gate's attribution (code table unchanged); the unguarded
legacy path normalized zero/NaN vectors into NaN comparisons that
PASS every gate (NaN < thr is False) - the latent NaN-acceptance
hazard - and normalize_vec's 0/0 fired a RuntimeWarning (measured
on real legs at an aggressive intake; no firing instance on the
fresh production path).

Pins here: the shipped default; the hazard closed (a NaN point is
REJECTED, deterministically); the measured zero-at-camera event
(rejected by cheirality in BOTH paths; the guard adds the parallax
attribution and silences the 0/0 warning); bit-transparency on
every normalizable candidate; and kill-switch transparency (flag
pinned 0 in memory reproduces the legacy semantics EXACTLY,
acceptance-hazard and warning included).

Known scope limit (recorded, deferred): a NaN point still
fires frame.py project_point's own NaN/NaN divide on the way to
its reprojection error - the guard closes the gate SEMANTICS
(rejection instead of accidental acceptance); frame.py's divide
is a separate, unexercised path.
"""
import warnings

import cv2 as cv
import numpy as np
import pytest

import slam.utils.utils as utils_mod
from benchmarks.stages.helpers import (make_K, make_frame, project_gt,
                                       set_keypoints, yaw_pose)
from slam.utils.utils import is_valid_triangulated_point
from slam.utils.constants import TRIANGULATION_VALIDATION_CODE

K = make_K()


def _validated_pair(T1, T2, X):
    """(cur, prev) frames + matches for one world point - the
    test_triangulation.py fixture, mirrored."""
    prev_f = make_frame(T1, K)
    cur_f = make_frame(T2, K)
    uv1, _ = project_gt(T1, X, K)
    uv2, _ = project_gt(T2, X, K)
    set_keypoints(prev_f, [uv1])
    set_keypoints(cur_f, [uv2])
    cur_f.set_match_data([cv.DMatch(0, 0, 0.0)])
    return cur_f, prev_f


def _arbitrary_pair(T1, T2):
    """(cur, prev) frames + matches with fixed keypoints - for
    degenerate candidates whose projection is undefined."""
    prev_f = make_frame(T1, K)
    cur_f = make_frame(T2, K)
    set_keypoints(prev_f, [(329.0, 249.0)])
    set_keypoints(cur_f, [(329.0, 249.0)])
    cur_f.set_match_data([cv.DMatch(0, 0, 0.0)])
    return cur_f, prev_f


def test_guard_default_is_shipped():
    """The shipped state: the guard is ON;
    the flag is read at call time via the module global, so probes
    can pin it in memory for A/B)."""
    assert utils_mod.TRIANGULATION_DEGENERATE_GUARD_ENABLED == 1


def test_guard_rejects_nan_point_deterministically():
    """The hazard closed: a NaN candidate previously PASSED every
    gate (NaN comparisons are False) and was ACCEPTED with NaN
    geometry; the guard rejects it under the parallax attribution -
    the only gate that can catch it (cheirality/depth/reprojection
    comparisons with NaN are all False)."""
    T1 = yaw_pose(0.0, (0.0, 0.0, 0.0))
    T2 = yaw_pose(0.0, (1.0, 0.0, 0.0))
    cur_f, prev_f = _arbitrary_pair(T1, T2)
    point_4d = np.array([np.nan, np.nan, np.nan, 1.0])

    result = is_valid_triangulated_point(0, cur_f, prev_f, point_4d)
    assert result["validation_code"] == \
        TRIANGULATION_VALIDATION_CODE["MIN_PARALLAX_VIOLATION"]
    assert result["failed_codes"] == \
        [TRIANGULATION_VALIDATION_CODE["MIN_PARALLAX_VIOLATION"]]


def test_guard_silences_the_zero_vector_warning_at_camera():
    """The MEASURED event (real legs at an aggressive intake): a
    candidate exactly at a camera center - normalize_vec's 0/0.
    Guarded: no RuntimeWarning anywhere (project_point returns
    None at z=0 before dividing), cheirality still rejects first
    (z <= MIN_DEPTH = 0.0), and the guard adds the deterministic
    parallax attribution the legacy path never had (the NaN
    parallax comparison PASSED it)."""
    T1 = yaw_pose(0.0, (0.0, 0.0, 0.0))      # the candidate sits AT
    T2 = yaw_pose(0.0, (1.0, 0.0, 0.0))      # this camera's center
    cur_f, prev_f = _arbitrary_pair(T1, T2)
    point_4d = np.array([0.0, 0.0, 0.0, 1.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = is_valid_triangulated_point(0, cur_f, prev_f, point_4d)
    assert caught == [], \
        f"guarded path fired {len(caught)} warnings: " \
        f"{[str(w.message) for w in caught]}"
    assert result["validation_code"] == \
        TRIANGULATION_VALIDATION_CODE["MIN_DEPTH_VIOLATION"]
    assert result["failed_codes"] == [
        TRIANGULATION_VALIDATION_CODE["MIN_DEPTH_VIOLATION"],
        TRIANGULATION_VALIDATION_CODE["MIN_PARALLAX_VIOLATION"],
        TRIANGULATION_VALIDATION_CODE["MAX_REPROJECTION_ERROR_VIOLATION"],
    ]


def test_guard_is_bit_transparent_on_normalizable_candidates():
    """Every normalizable candidate computes IDENTICALLY either way
    (v / ||v||, the same arithmetic): the full result dict is
    bit-equal across the flag, on an accepted point AND on a
    low-parallax rejection."""
    T1 = yaw_pose(0.0, (0.0, 0.0, 0.0))
    T2 = yaw_pose(0.0, (1.0, 0.0, 0.0))
    X_good = np.array([0.2, 0.1, 5.0])
    X_low = np.array([0.2, 0.1, 5.0])        # identical poses below
    cases = [
        ("accept", _validated_pair(T1, T2, X_good), np.append(X_good, 1.0)),
        ("low-parallax", _validated_pair(T1, T1, X_low),
         np.append(X_low, 1.0)),
    ]
    for name, (cur_f, prev_f), point_4d in cases:
        on = is_valid_triangulated_point(0, cur_f, prev_f, point_4d)
        old = utils_mod.TRIANGULATION_DEGENERATE_GUARD_ENABLED
        try:
            utils_mod.TRIANGULATION_DEGENERATE_GUARD_ENABLED = 0
            off = is_valid_triangulated_point(0, cur_f, prev_f, point_4d)
        finally:
            utils_mod.TRIANGULATION_DEGENERATE_GUARD_ENABLED = old
        assert on == off, f"{name}: guard changed a normalizable result"


def test_flag_0_reproduces_legacy_semantics_exactly(monkeypatch):
    """Kill-switch transparency: flag pinned 0 = the legacy path,
    bit-for-bit - INCLUDING the two measured defects (they are the
    legacy contract the guard replaced, pinned so a future edit
    cannot silently change the switch's meaning):
    1. the NaN candidate is ACCEPTED (validation_code VALID - the
       latent NaN-acceptance hazard);
    2. the zero-at-camera candidate fires normalize_vec's 0/0
       RuntimeWarning (the measured event on real legs) and its
       failed_codes carry NO parallax attribution
       (the NaN comparison passed the gate)."""
    monkeypatch.setattr(utils_mod,
                        "TRIANGULATION_DEGENERATE_GUARD_ENABLED", 0)

    T1 = yaw_pose(0.0, (0.0, 0.0, 0.0))
    T2 = yaw_pose(0.0, (1.0, 0.0, 0.0))

    # (1) the NaN-acceptance hazard, as legacy semantics
    cur_f, prev_f = _arbitrary_pair(T1, T2)
    result = is_valid_triangulated_point(
        0, cur_f, prev_f, np.array([np.nan, np.nan, np.nan, 1.0]))
    assert result["validation_code"] == \
        TRIANGULATION_VALIDATION_CODE["VALID"]
    assert result["failed_codes"] == []

    # (2) the 0/0 warning + the parallax pass-through, as legacy
    cur_f, prev_f = _arbitrary_pair(T1, T2)
    with pytest.warns(RuntimeWarning, match="invalid value"):
        result = is_valid_triangulated_point(
            0, cur_f, prev_f, np.array([0.0, 0.0, 0.0, 1.0]))
    assert result["validation_code"] == \
        TRIANGULATION_VALIDATION_CODE["MIN_DEPTH_VIOLATION"]
    assert result["failed_codes"] == [
        TRIANGULATION_VALIDATION_CODE["MIN_DEPTH_VIOLATION"],
        TRIANGULATION_VALIDATION_CODE["MAX_REPROJECTION_ERROR_VIOLATION"],
    ]
