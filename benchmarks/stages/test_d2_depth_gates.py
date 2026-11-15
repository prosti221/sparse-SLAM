"""Stage tests: the D2b depth gates (the focal-adaptive parallax
floor and the scene-relative depth ceiling) in the triangulation
validator.

The floor helper is pinned as a pure function; the validator-level
rescue/tighten deltas are pinned on synthetic one-point pairs (the
failed D2 leg's far candidate is RESCUED at flag 1 - the user's
intent; a wide-angle camera tightens). Open3D is not involved; the
battery is untouched by construction.
"""
import cv2 as cv
import numpy as np
import pytest

from benchmarks.stages.helpers import (  # noqa: E402
    make_K, make_frame, project_gt, set_keypoints)
from slam.utils import constants as const  # noqa: E402
from slam.utils import utils as utils_mod  # noqa: E402


# ===== the floor helper =====

def test_parallax_floor_deg_focal_adaptive():
    f = utils_mod._parallax_floor_deg
    # the battery's focal: near-equivalent to the fixed 0.65
    assert f(450.0) == pytest.approx(0.6366, abs=1e-3)
    # the salem class (high fx): LOOSER - better depth per degree
    assert f(1449.0) == pytest.approx(0.1977, abs=1e-3)
    # the wide-angle class: TIGHTER
    assert f(300.0) == pytest.approx(0.9550, abs=1e-3)


# ===== the validator-level deltas =====

def _pair_fixture(point_3d, baseline, K=None):
    """Two frames (the current at the origin, the previous translated
    along +x by `baseline`) with the candidate's EXACT projected
    keypoints (zero reprojection error) and a matching DMatch. The
    poses are assigned WHOLE (Frame.pose is a copy-returning
    property - slice writes mutate a temporary)."""
    cur = make_frame(K=K)
    prev = make_frame(K=K)
    p = np.eye(4)
    p[:3, 3] = [baseline, 0.0, 0.0]
    prev.pose = p
    set_keypoints(cur, [project_gt(cur.pose, point_3d, cur.K)[0]])
    set_keypoints(prev, [project_gt(prev.pose, point_3d, prev.K)[0]])
    cur.matches = [cv.DMatch(0, 0, 0.)]
    return cur, prev


def _failed_codes(cur, prev, point_3d, scene_depth_median=None):
    result = utils_mod.is_valid_triangulated_point(
        0, cur, prev, np.array([*point_3d, 1.0]),
        scene_depth_median=scene_depth_median)
    return result["failed_codes"]


def _flip(flag_name, value):
    old = getattr(utils_mod, flag_name)
    setattr(utils_mod, flag_name, value)
    return old


def test_far_point_rescued_at_flag1():
    """The rescue case: a candidate in the (0.637, 0.65)-degree
    window (Z=100, perpendicular baseline B=1.123 -> parallax
    0.643 deg, sigma_Z/Z = 0.198): REJECTED by the fixed 0.65 floor
    at flag 0, RESCUED by the adaptive floor (0.637 deg at fx=450)
    at flag 1 - the user's intent."""
    point = (0.0, 0.0, 100.0)
    cur, prev = _pair_fixture(point, 1.123)

    old = _flip("TRIANGULATION_DEPTH_UNCERTAINTY_GATE", 0)
    try:
        codes = _failed_codes(cur, prev, point)
        assert codes == [const.TRIANGULATION_VALIDATION_CODE[
            'MIN_PARALLAX_VIOLATION']]
    finally:
        _flip("TRIANGULATION_DEPTH_UNCERTAINTY_GATE", old)

    _flip("TRIANGULATION_DEPTH_UNCERTAINTY_GATE", 1)
    try:
        assert _failed_codes(cur, prev, point) == []
    finally:
        _flip("TRIANGULATION_DEPTH_UNCERTAINTY_GATE", old)


def test_wide_angle_tightens_at_flag1():
    """The wide-angle case: a candidate at parallax 0.798 deg (Z=100,
    B=1.396) passes the fixed 0.65 floor at flag 0; at fx=300 the
    adaptive floor rises to ~0.955 deg and the same candidate is
    REJECTED at flag 1 (the correct wide-angle penalty)."""
    point = (0.0, 0.0, 100.0)
    K = make_K(fx=300.0)
    cur, prev = _pair_fixture(point, 1.396, K=K)

    old = _flip("TRIANGULATION_DEPTH_UNCERTAINTY_GATE", 0)
    try:
        codes = _failed_codes(cur, prev, point)
        assert codes == []      # the fixed 0.65 floor passes 0.798
    finally:
        _flip("TRIANGULATION_DEPTH_UNCERTAINTY_GATE", old)
    _flip("TRIANGULATION_DEPTH_UNCERTAINTY_GATE", 1)
    try:
        codes = _failed_codes(cur, prev, point)
        assert codes == [const.TRIANGULATION_VALIDATION_CODE[
            'MIN_PARALLAX_VIOLATION']]
    finally:
        _flip("TRIANGULATION_DEPTH_UNCERTAINTY_GATE", old)


def test_near_point_unaffected():
    """A well-conditioned candidate (Z=5, parallax ~14 deg): accepted
    at BOTH flag values (the adaptive floor must not touch
    well-conditioned candidates at either focal)."""
    point = (0.0, 0.0, 5.0)
    cur, prev = _pair_fixture(point, 1.25)

    for flag in (0, 1):
        old = _flip("TRIANGULATION_DEPTH_UNCERTAINTY_GATE", flag)
        try:
            assert _failed_codes(cur, prev, point) == []
        finally:
            _flip("TRIANGULATION_DEPTH_UNCERTAINTY_GATE", old)


def test_pure_rotation_pair_zero_parallax_rejects_at_both():
    """A zero baseline (both cameras at the same center): the rays
    are identical -> zero parallax -> the floor rejects at flag 0;
    the adaptive floor at flag 1 rejects too (theta_min > 0 for any
    finite focal - the infinite-uncertainty case)."""
    point = (0.0, 0.0, 5.0)
    cur, prev = _pair_fixture(point, 0.0)
    yaw = 0.05
    c, s = np.cos(yaw), np.sin(yaw)
    P = np.eye(4)
    P[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    prev.pose = P
    set_keypoints(prev, [project_gt(prev.pose, np.array(point),
                                    prev.K)[0]])

    PAR = const.TRIANGULATION_VALIDATION_CODE['MIN_PARALLAX_VIOLATION']
    for flag in (0, 1):
        old = _flip("TRIANGULATION_DEPTH_UNCERTAINTY_GATE", flag)
        try:
            codes = _failed_codes(cur, prev, point)
            assert codes == [PAR]
        finally:
            _flip("TRIANGULATION_DEPTH_UNCERTAINTY_GATE", old)


def test_scene_relative_ceiling_tightens_and_falls_back():
    """Flag 0: the fixed 1000 ceiling (a Z=100 candidate passes).
    Flag 1 with the scene median 5: the ceiling drops to 50 and the
    same candidate trips MAX_DEPTH_VIOLATION; the no-median fallback
    restores the fixed ceiling."""
    point = (0.0, 0.0, 100.0)
    cur, prev = _pair_fixture(point, 1.25)

    old = _flip("TRIANGULATION_SCENE_RELATIVE_MAX_DEPTH", 0)
    try:
        codes = _failed_codes(cur, prev, point)
        assert const.TRIANGULATION_VALIDATION_CODE[
            'MAX_DEPTH_VIOLATION'] not in codes
        # the no-median fallback: the fixed ceiling
        codes = _failed_codes(cur, prev, point,
                              scene_depth_median=None)
        assert const.TRIANGULATION_VALIDATION_CODE[
            'MAX_DEPTH_VIOLATION'] not in codes
    finally:
        _flip("TRIANGULATION_SCENE_RELATIVE_MAX_DEPTH", old)

    _flip("TRIANGULATION_SCENE_RELATIVE_MAX_DEPTH", 1)
    try:
        codes = _failed_codes(cur, prev, point,
                              scene_depth_median=5.0)
        assert const.TRIANGULATION_VALIDATION_CODE[
            'MAX_DEPTH_VIOLATION'] in codes
        # a generous scene median restores acceptance (Zmax 10x 20)
        codes = _failed_codes(cur, prev, point,
                              scene_depth_median=20.0)
        assert const.TRIANGULATION_VALIDATION_CODE[
            'MAX_DEPTH_VIOLATION'] not in codes
    finally:
        _flip("TRIANGULATION_SCENE_RELATIVE_MAX_DEPTH", old)


def test_flag_contract(monkeypatch):
    """Both flags are module-level INTs defaulting 0 and the
    --override machinery reaches them (utils resolves before its
    star-importers)."""
    assert isinstance(utils_mod.TRIANGULATION_DEPTH_UNCERTAINTY_GATE, int)
    assert utils_mod.TRIANGULATION_DEPTH_UNCERTAINTY_GATE == 0
    assert isinstance(utils_mod.TRIANGULATION_SCENE_RELATIVE_MAX_DEPTH, int)
    assert utils_mod.TRIANGULATION_SCENE_RELATIVE_MAX_DEPTH == 0

    from benchmarks.run import apply_overrides
    apply_overrides(["TRIANGULATION_DEPTH_UNCERTAINTY_GATE=1",
                     "TRIANGULATION_SCENE_RELATIVE_MAX_DEPTH=1"])
    try:
        assert utils_mod.TRIANGULATION_DEPTH_UNCERTAINTY_GATE == 1
        assert utils_mod.TRIANGULATION_SCENE_RELATIVE_MAX_DEPTH == 1
    finally:
        utils_mod.TRIANGULATION_DEPTH_UNCERTAINTY_GATE = 0
        utils_mod.TRIANGULATION_SCENE_RELATIVE_MAX_DEPTH = 0
    assert utils_mod.TRIANGULATION_DEPTH_UNCERTAINTY_GATE == 0
    assert utils_mod.TRIANGULATION_SCENE_RELATIVE_MAX_DEPTH == 0
