"""[C0] the native bridge pins.

The extension must build/import on BOTH machines (arm64 laptop, x86-64
desktop), and the FP contract is executable: the C++ row norm is
bit-identical to numpy's elementwise sqrt(x*x + y*y) - the [26d] fold
norm's semantics, computed in strict scalar order with FMA contraction
disabled (benchmarks/CPP_BACKEND_MIGRATION.md, locked decision 2).
"""
import numpy as np
import pytest

slam_native = pytest.importorskip("slam_native")


def test_module_reports_version():
    assert slam_native.__version__


def test_row_norms_bit_identical_to_numpy_elementwise():
    from numpy.random import default_rng
    rng = default_rng(7)
    d = rng.normal(size=(100_000, 2))
    got = np.asarray(slam_native.row_norms(d))
    want = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
    # BIT-identical, deliberately NOT allclose: the FP contract.
    np.testing.assert_array_equal(got, want)


def test_row_norms_extracts_py_floats_exactly():
    d = np.array([[3.0, 4.0], [0.0, 0.0], [1e-300, 1e-300]])
    got = np.asarray(slam_native.row_norms(d))
    assert got[0] == 5.0 and got[1] == 0.0
    np.testing.assert_array_equal(
        got[2], np.sqrt(1e-300 * 1e-300 + 1e-300 * 1e-300))


def test_row_norms_rejects_bad_shape():
    with pytest.raises(ValueError):
        slam_native.row_norms(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        slam_native.row_norms(np.zeros(4))
