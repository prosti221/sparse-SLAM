"""Stage test: phase 2j calibration arc.

Homes the pins for the calibration items as they land:
- 2j-0: the FOCAL_SCALE diagnostic (harness-side focal perturbation for
  pain measurement and recovery validation)
- 2j-1: the calib_motion sequences + the generator's k1 render path
- 2j-2/3/4 (later): the GeoCalib initializer wrapper, the distortion
  boundaries (k1 == 0 keeps the literal identity path), and the
  intrinsics refit.

The DEFAULT pipeline must stay bit-identical: every pin here asserts the
identity/no-op semantics of the default path, not just the active one.
"""
import hashlib

import numpy as np
import pytest

import slam.utils.constants as const
from benchmarks.harness import load_calibration
from benchmarks.synthetic.generator import (FX, FY, CX, CY, SEQUENCES,
                                            _calib_motion_pose, _project,
                                            ensure_sequence)


@pytest.fixture(scope="module")
def translation_seq():
    return ensure_sequence("translation")


@pytest.fixture(scope="module")
def calib_seq():
    return ensure_sequence("calib_motion")


@pytest.fixture(scope="module")
def calib_k1_seq():
    return ensure_sequence("calib_motion_k1")


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


# --- 2j-1: generator k1 render path + calib_motion sequences -----------

K_450 = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])


def test_generator_projection_distortion_closed_form():
    """2j-1: _project at k1=0 is the exact pinhole closed form (the
    legacy path), and at k1 != 0 applies the simple-radial model in
    NORMALIZED coordinates - the same convention the SLAM side's
    undistort-at-normalization assumes (2j-3)."""
    pts = np.array([[0.30, -0.20, 4.0],
                    [-0.50, 0.40, 2.0],
                    [0.00, 0.00, 3.0],
                    [0.65, 0.50, 1.5]])   # near the image corner
    R_cw, t_cw = np.eye(3), np.zeros(3)

    xy = pts[:, :2] / pts[:, 2:3]
    expect_pinhole = np.column_stack(
        [xy[:, 0] * K_450[0, 0] + K_450[0, 2],
         xy[:, 1] * K_450[1, 1] + K_450[1, 2]])
    assert np.allclose(
        _project(pts, R_cw, t_cw, K_450, 0.0), expect_pinhole, atol=1e-12)

    k1 = -0.06
    r2 = xy[:, 0] ** 2 + xy[:, 1] ** 2
    d = xy * (1.0 + k1 * r2[:, None])
    expect_distorted = np.column_stack(
        [d[:, 0] * K_450[0, 0] + K_450[0, 2],
         d[:, 1] * K_450[1, 1] + K_450[1, 2]])
    assert np.allclose(
        _project(pts, R_cw, t_cw, K_450, k1), expect_distorted, atol=1e-12)


def test_calib_motion_pose_is_diverse_and_in_room():
    """2j-1: the trajectory is the observability-positive testbed - its
    INVARIANT is motion diversity: real translation along ALL THREE
    axes and real rotation around TWO different axes (DroidCalib's
    requirement; the existing sequences are all critical-motion). Also
    pins: valid rotations and a path strictly inside the room (walls at
    |x|=10, |y|=6, z=0/12)."""
    n = SEQUENCES["calib_motion"]["n_frames"]
    poses = [_calib_motion_pose(i, n) for i in range(n)]
    Rs = np.stack([R for R, _ in poses])
    ts = np.stack([t for _, t in poses])

    for R in Rs:  # valid rotations
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

    assert np.all(np.abs(ts[:, 0]) < 10.0)   # inside the x walls
    assert np.all(np.abs(ts[:, 1]) < 6.0)    # inside floor/ceiling
    assert np.all((ts[:, 2] > 0.0) & (ts[:, 2] < 12.0))  # inside z walls

    # Translation diversity: every axis carries real displacement.
    span = ts.max(axis=0) - ts.min(axis=0)
    assert np.all(span > 1.0), f"translation span per axis: {span}"

    # Rotation diversity: accumulated |rotation vector| per axis - the
    # pitch (x) and yaw (y) axes must both carry real rotation.
    from scipy.spatial.transform import Rotation as _Rot
    rots = np.stack([_Rot.from_matrix(R).as_rotvec() for R in Rs])
    acc = np.abs(rots).sum(axis=0)
    assert acc[0] > 0.5, "no meaningful pitch accumulated"
    assert acc[1] > 0.5, "no meaningful yaw accumulated"


def test_calib_k1_sequence_shares_trajectory_differs_frames(calib_seq,
                                                            calib_k1_seq):
    """2j-1: the k1 variant is the SAME trajectory with ONLY the render
    distortion differing - the property that isolates distortion in
    the 2j-5b recovery test. Frames must genuinely differ; calibration
    files must both be the true pinhole K (the SLAM side receives true
    f; k1 is what the arc must ESTIMATE)."""
    traj_a = (calib_seq / "trajectory.txt").read_bytes()
    traj_b = (calib_k1_seq / "trajectory.txt").read_bytes()
    assert traj_a == traj_b, "the k1 variant must share the trajectory"

    for seq in (calib_seq, calib_k1_seq):
        assert (seq / "calibration.txt").read_text().strip() == \
            f"{FX} {FY} {CX} {CY}"

    def _md5(path):
        return hashlib.md5(path.read_bytes()).hexdigest()

    assert _md5(calib_seq / "frames" / "frame_000100.png") != \
        _md5(calib_k1_seq / "frames" / "frame_000100.png"), \
        "distortion had no effect on the rendered frames"

