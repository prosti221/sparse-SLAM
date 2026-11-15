"""Stage tests: the D1 adaptive thresholds (the loop-RANSAC budget
and the noise-aware PnP acceptance guard).

The stopping rule and the guard tau are pinned as pure functions;
the loop-RANSAC's flag-0/flag-1 behavior is pinned through the real
(unbound) method on synthetic correspondences, counting the drawn
samples. Open3D is not involved; the battery is untouched by
construction.
"""
import math

import cv2 as cv
import numpy as np
import pytest

from slam import loop_closure as lc_mod  # noqa: E402
from slam.ba import bundle_adjustment_g2o as ba_mod  # noqa: E402
from slam.utils import constants as const  # noqa: E402


# ===== the stopping rule =====

def test_ransac_required_iters_formula():
    f = lc_mod._ransac_required_iters
    assert f(50, 100) == 108    # w = 0.5
    assert f(70, 100) == 26     # w = 0.7
    assert f(99, 100) == 3
    assert f(100, 100) == 1     # w -> 1
    assert f(0, 100) > 10 ** 8  # no model yet: keep sampling
    assert f(40, 100) == 267    # w = 0.4
    assert f(0, 0) > 10 ** 8

def test_ransac_required_iters_monotone_in_w():
    f = lc_mod._ransac_required_iters
    seq = [f(b, 100) for b in (1, 5, 10, 20, 40, 60, 80, 95, 100)]
    assert seq == sorted(seq, reverse=True)


# ===== the guard tau + EMA =====

def test_pnp_guard_tau_fallback_and_clamps():
    base = const.PNP_MAX_MEDIAN_REPROJECTION_ERROR
    tau = ba_mod._pnp_guard_tau
    assert tau(None, base) == base            # the fallback
    assert tau(0.001, base) == base           # never tighter than base
    assert tau(0.02, base) == pytest.approx(3 * 0.02)
    assert tau(0.03, base) == pytest.approx(0.08)   # the 4x upper clamp
    assert tau(0.05, base) == pytest.approx(0.08)


def test_pnp_guard_ema_update_accepted_only_shape():
    upd = ba_mod._pnp_guard_ema_update
    assert upd(None, 0.01) == pytest.approx(0.01)
    assert upd(0.02, 0.04) == pytest.approx(0.75 * 0.02 + 0.25 * 0.04)
    # idempotent convergence: repeated identical medians converge to them
    e = None
    for _ in range(20):
        e = upd(e, 0.03)
    assert e == pytest.approx(0.03)


# ===== the loop-RANSAC flag behavior through the real method =====

def _make_correspondences(rng, n, inliers, noise=0.002,
                          outlier_scale=50.0):
    """A synthetic loop-PnP problem: `inliers` of `n` points reproject
    from a known pose with tiny noise; the rest are gross outliers."""
    X = rng.uniform(-2.0, 2.0, (n, 3))
    X[:, 2] += 4.0                       # in front
    R_true = cv.Rodrigues(np.array([0.05, -0.03, 0.02]))[0]
    t_true = np.array([0.1, -0.05, 0.02])
    Xc = (R_true @ X.T).T + t_true
    uv = Xc[:, :2] / Xc[:, 2:3]
    uv_in = uv[:inliers] + rng.normal(0, noise, (inliers, 2))
    uv_out = uv[inliers:] + rng.normal(0, outlier_scale, (n - inliers, 2))
    img = np.vstack([uv_in, uv_out])
    return X, img, R_true, t_true


def _count_drawn_samples(monkeypatch, expected_calls_cap):
    """Patch the loop module's rng factory with a counting one."""
    counter = {"choices": 0}

    real_rng = np.random.default_rng

    class CountingRng:
        def __init__(self, inner):
            self._inner = inner

        def choice(self, *a, **kw):
            counter["choices"] += 1
            return self._inner.choice(*a, **kw)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def factory(seed):
        return CountingRng(real_rng(seed))

    monkeypatch.setattr(np.random, "default_rng", factory)
    return counter


def test_loop_ransac_adaptive_behavior(monkeypatch):
    """The unconditional adaptive rule: on a high-inlier problem the
    sampling stops at the formula's N (well under the old fixed-50
    budget), the quality gate holds, and a no-inlier problem runs
    exactly to the cap."""
    rng = np.random.default_rng(7)
    obj, img, R_true, t_true = _make_correspondences(rng, n=100,
                                                     inliers=80)
    fx = 500.0

    counter = _count_drawn_samples(monkeypatch, 200)
    r = lc_mod.LoopClosureDetector._pnp_ransac(None, obj, img, fx)
    # early stop: well under the old fixed-50 budget (the formula's
    # N at this fixture's w ~ 0.8 is ~14)
    assert counter["choices"] < 50
    assert r[2] is not None and len(r[2]) >= 60

    # A no-inlier problem: best_count stays 0, the rule never fires,
    # and the loop runs to the cap (the waste is explicit and bounded).
    rng = np.random.default_rng(11)
    obj = rng.uniform(-2, 2, (40, 3))
    img = rng.uniform(1e3, 2e3, (40, 2))   # absurd image coords
    counter = _count_drawn_samples(monkeypatch, 200)
    r = lc_mod.LoopClosureDetector._pnp_ransac(None, obj, img, 500.0)
    assert counter["choices"] == const.LOOP_RANSAC_MAX_ITERATIONS
    assert r[2] is None


# ===== the flag contracts =====

def test_flag_contract():
    """The gate flags are GONE (the mechanisms are unconditional since
    the D1 adoption - the kill-switch amendment's post-adoption
    convention); the ceiling constant stays pinned."""
    assert not hasattr(lc_mod, "LOOP_RANSAC_ADAPTIVE")
    assert not hasattr(ba_mod, "PNP_GUARD_NOISE_ADAPTIVE")
    assert const.LOOP_RANSAC_MAX_ITERATIONS == 200
