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


# --- 2j-2: GeoCalib initializer (all pins mocked - no torch/network) ----

import sys  # noqa: E402
import types  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from slam.calibration import geocalib_init as gi
from slam.utils import constants as calib_const
from slam.utils.utils import load_video


class _FT:
    """Minimal torch-tensor stand-in for mocked GeoCalib results."""

    def __init__(self, val):
        self.val = np.asarray(val, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.val

    def __getitem__(self, i):
        return _FT(self.val[i])


def _fake_result(fx=450.0, fy=455.0, k1=-0.05, rel_sigma=0.04):
    """Mirrors the INSTALLED package's calibrate() return: uncertainties
    at the TOP LEVEL (result["covariance"] is the raw matrix, not
    key-addressable - pinned after a real-package smoke caught the
    distinction)."""
    cam = SimpleNamespace(f=_FT([[fx, fy]]), k1=_FT([[k1]]))
    result = {"camera": cam, "covariance": _FT(np.zeros((1, 3, 3))),
              "focal_uncertainty": _FT([rel_sigma])}
    if rel_sigma is None:
        del result["focal_uncertainty"]   # GeoCalib declined to report
    return result


def test_extract_prior_maps_geocalib_result():
    """2j-2: the GeoCalib result maps to the prior: f (index 0 - the
    shared-intrinsics convention), c PINNED AT THE IMAGE CENTER
    (GeoCalib's own assumption), k1, and focal_uncertainty (TOP-LEVEL
    key of the result; the "covariance" entry is the raw matrix)
    interpreted as a RELATIVE 1-sigma of f."""
    prior = gi._extract_prior(_fake_result(), img_w=640, img_h=480)

    assert prior.fx == 450.0 and prior.fy == 455.0
    assert prior.cx == 320.0 and prior.cy == 240.0  # image center
    assert prior.k1 == -0.05
    assert prior.sigma_f == pytest.approx(0.04 * 452.5)
    assert prior.sigma_k1 == calib_const.CALIB_PRIOR_SIGMA_K1
    assert prior.source == "geocalib"
    # K is constructible directly (the slam.py wiring consumes this)
    K = prior.K
    assert K[0, 0] == 450.0 and K[1, 1] == 455.0
    assert K[0, 2] == 320.0 and K[1, 2] == 240.0

    # Absent uncertainty degrades the prior to anchor-only (no trust).
    prior = gi._extract_prior(_fake_result(rel_sigma=None),
                              img_w=640, img_h=480)
    assert prior.sigma_f == float("inf")


def test_estimate_intrinsics_prior_calls_geocalib_correctly(monkeypatch):
    """2j-2: with a mocked geocalib module, the wrapper calls
    calibrate(batch, shared_intrinsics=True, camera_model=...),
    converts the caller's BGR frames, and returns the mapped prior."""
    calls = {}

    class _FakeGeoCalib:
        def __init__(self, weights="pinhole"):
            calls["weights"] = weights

        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            return self

        def calibrate(self, batch, **kwargs):
            calls["kwargs"] = kwargs
            calls["batch_shape"] = tuple(batch.shape)
            return _fake_result()

    fake_mod = types.ModuleType("geocalib")
    fake_mod.GeoCalib = _FakeGeoCalib
    monkeypatch.setitem(sys.modules, "geocalib", fake_mod)

    frames = [np.zeros((48, 64, 3), np.uint8) for _ in range(4)]
    prior = gi.estimate_intrinsics_prior(frames)

    assert calls["weights"] == "distorted"
    assert calls["kwargs"]["shared_intrinsics"] is True
    assert calls["kwargs"]["camera_model"] == "simple_radial"
    assert calls["batch_shape"] == (4, 3, 48, 64)  # B, C, H, W
    assert prior.fx == 450.0 and prior.cx == 32.0  # center of 64-wide


def test_estimate_intrinsics_prior_requires_the_extra(monkeypatch):
    """2j-2: without the geocalib package the error names the install
    command (the optional 'calib' extra), not the raw ImportError."""
    monkeypatch.setitem(sys.modules, "geocalib", None)
    with pytest.raises(ImportError, match="uv sync --extra calib"):
        gi.estimate_intrinsics_prior(
            [np.zeros((48, 64, 3), np.uint8)])


class _FakeCap:
    def __init__(self, n_available, fps=10.0, w=64, h=48):
        self.n_available = n_available
        self.fps = fps
        self.read_count = 0
        self.size = (h, w)

    def get(self, prop):
        import cv2 as cv
        return self.fps if prop == cv.CAP_PROP_FPS else 0.0

    def read(self):
        if self.read_count >= self.n_available:
            return False, None
        self.read_count += 1
        return True, np.zeros((*self.size, 3), np.uint8)


def test_estimate_video_intrinsics_buffers_and_replays(monkeypatch):
    """2j-2: the video init buffers CALIB_INIT_SECONDS * fps frames
    (clamped to [min, max]) and returns them for replay - they have
    been consumed from the capture."""
    seen = {}

    def fake_estimate(frames):
        seen["n"] = len(frames)
        return gi.IntrinsicsPrior(500.0, 500.0, 32.0, 24.0,
                                  source="mock")

    monkeypatch.setattr(gi, "estimate_intrinsics_prior", fake_estimate)

    cap = _FakeCap(n_available=50, fps=10.0)
    prior, frames = gi.estimate_video_intrinsics(cap)

    expected_n = int(np.clip(round(calib_const.CALIB_INIT_SECONDS * 10.0),
                             calib_const.CALIB_INIT_MIN_FRAMES,
                             calib_const.CALIB_INIT_MAX_FRAMES))
    assert seen["n"] == expected_n
    assert len(frames) == expected_n
    assert cap.read_count == expected_n  # consumed from the capture
    assert prior.fx == 500.0

    # A video that ends before the minimum batch size fails loudly.
    with pytest.raises(RuntimeError, match="too few frames"):
        gi.estimate_video_intrinsics(_FakeCap(n_available=3, fps=10.0))


def _write_tiny_video(path, n_frames=5):
    import cv2 as cv
    w, h = 64, 48
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"MJPG"),
                            10.0, (w, h))
    for i in range(n_frames):
        writer.write(np.full((h, w, 3), 40 + 10 * i, np.uint8))
    writer.release()
    return str(path)


class _FakeConfig:
    def __init__(self, props):
        self.props = props

    def get_global_config_property(self, key):
        return self.props.get(key)

    def get_video_property(self, tag, key, default=None):
        return self.props.get(key, default)


def test_load_video_auto_signaling(tmp_path):
    """2j-2: 'fx: auto' (or an absent fx) signals auto-calibration by
    returning K=None - slam.py then runs estimate_video_intrinsics.
    A numeric fx keeps today's behavior exactly."""
    video = _write_tiny_video(tmp_path / "tiny.avi")

    cap, K = load_video(_FakeConfig(
        {"load_video": "v", "path": video, "fx": "auto"}))
    assert K is None
    cap.release()

    cap, K = load_video(_FakeConfig(
        {"load_video": "v", "path": video}))          # fx absent
    assert K is None
    cap.release()

    cap, K = load_video(_FakeConfig(
        {"load_video": "v", "path": video, "fx": 480}))
    assert K[0, 0] == 480.0          # fy defaults to fx
    assert K[1, 1] == 480.0
    assert K[0, 2] == 32.0           # cx defaults to W // 2
    assert K[1, 2] == 24.0           # cy defaults to H // 2
    cap.release()

