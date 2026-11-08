"""Stage test: phase 2j calibration arc.

Homes the pins for the calibration items as they land:
- 2j-0: the FOCAL_SCALE diagnostic (harness-side focal perturbation for
  pain measurement and recovery validation)
- 2j-2/3/4 (later): the GeoCalib initializer wrapper, the distortion
  boundaries (k1 == 0 keeps the literal identity path), and the
  intrinsics refit.

The DEFAULT pipeline must stay bit-identical: every pin here asserts the
identity/no-op semantics of the default path, not just the active one.
"""
import numpy as np
import pytest

import slam.utils.constants as const
from benchmarks.harness import load_calibration
from benchmarks.synthetic.generator import ensure_sequence


@pytest.fixture(scope="module")
def translation_seq():
    return ensure_sequence("translation")


def test_focal_scale_perturbs_focal_only(translation_seq, monkeypatch):
    """2j-0: --override FOCAL_SCALE=0.8 multiplies fx and fy by 0.8 and
    leaves the principal point untouched."""
    base = load_calibration(translation_seq)

    monkeypatch.setattr(const, "FOCAL_SCALE", 0.8, raising=True)
    scaled = load_calibration(translation_seq)

    assert scaled[0, 0] == pytest.approx(base[0, 0] * 0.8, rel=1e-12)
    assert scaled[1, 1] == pytest.approx(base[1, 1] * 0.8, rel=1e-12)
    # Principal point untouched: the perturbation is focal-only by
    # design (the 2j refit keeps c pinned at the center).
    assert scaled[0, 2] == base[0, 2]
    assert scaled[1, 2] == base[1, 2]


def test_focal_scale_default_is_literal_noop(translation_seq, monkeypatch):
    """2j-0: at the default 1.0 the multiply is SKIPPED entirely - the
    returned calibration is bit-identical to the uninstrumented one
    (the default battery must never move)."""
    base = load_calibration(translation_seq)

    monkeypatch.setattr(const, "FOCAL_SCALE", 1.0, raising=True)
    scaled = load_calibration(translation_seq)

    assert np.array_equal(scaled, base)
