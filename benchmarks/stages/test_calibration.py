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
                                            _build_distortion_map,
                                            _calib_motion_pose,
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
    leaves the principal point (and k1 - normalized units, not a
    focal) untouched."""
    base_K, base_k1 = load_calibration(translation_seq)

    monkeypatch.setattr(const, "FOCAL_SCALE", 0.8, raising=True)
    scaled_K, scaled_k1 = load_calibration(translation_seq)

    assert scaled_K[0, 0] == pytest.approx(base_K[0, 0] * 0.8, rel=1e-12)
    assert scaled_K[1, 1] == pytest.approx(base_K[1, 1] * 0.8, rel=1e-12)
    # Principal point untouched: the perturbation is focal-only by
    # design (the 2j refit keeps c pinned at the center).
    assert scaled_K[0, 2] == base_K[0, 2]
    assert scaled_K[1, 2] == base_K[1, 2]
    assert scaled_k1 == base_k1 == 0.0


def test_focal_scale_default_is_literal_noop(translation_seq, monkeypatch):
    """2j-0: at the default 1.0 the multiply is SKIPPED entirely - the
    returned calibration is bit-identical to the uninstrumented one
    (the default battery must never move)."""
    base_K, _ = load_calibration(translation_seq)

    monkeypatch.setattr(const, "FOCAL_SCALE", 1.0, raising=True)
    scaled_K, _ = load_calibration(translation_seq)

    assert np.array_equal(scaled_K, base_K)


# --- 2j-1: generator k1 render path + calib_motion sequences -----------

K_450 = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])


def test_generator_distortion_map_closed_form():
    """2j-3: the per-pixel render distortion map is the exact inverse
    of the simple-radial forward model (x_d = x_u (1 + k1 r_u^2)): for
    sampled OUTPUT pixels, the map's source coordinate (undistorted,
    under the wider-FOV K_hi) must reproduce the output pixel under the
    forward model. (Replaces the 2j-1 tiled-render pin: the per-pixel
    remap superseded the tiling - see the 2j-3 record.)"""
    K = K_450
    k1 = -0.06
    W_hi = int(np.ceil(640 * 1.2))
    H_hi = int(np.ceil(480 * 1.2))
    K_hi = np.array([[FX, 0.0, W_hi / 2.0],
                     [0.0, FY, H_hi / 2.0],
                     [0.0, 0.0, 1.0]])

    map_x, map_y = _build_distortion_map(K, k1, K_hi)

    for u_out, v_out in ((0, 0), (639, 479), (320, 240),
                         (100, 400), (500, 50), (1, 478)):
        x_u = (map_x[v_out, u_out] - K_hi[0, 2]) / K_hi[0, 0]
        y_u = (map_y[v_out, u_out] - K_hi[1, 2]) / K_hi[1, 1]
        x_d = (u_out - K[0, 2]) / K[0, 0]
        y_d = (v_out - K[1, 2]) / K[1, 1]
        scale = 1.0 + k1 * (x_u * x_u + y_u * y_u)
        assert np.isclose(x_u * scale, x_d, atol=1e-9)
        assert np.isclose(y_u * scale, y_d, atol=1e-9)

    # Coverage: every output pixel's source lies inside the hi-res
    # canvas (the wider FOV exists for exactly this).
    assert np.all((map_x >= 0) & (map_x <= W_hi - 1)
                  & (map_y >= 0) & (map_y <= H_hi - 1))


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
        assert (seq / "calibration.txt").read_text().strip().split()[:4] == \
            [str(FX), str(FY), str(CX), str(CY)]

    # Phase 2j-3: the k1 variant's file additionally carries the true
    # k1 as the 5th token (the generator writes it only when nonzero).
    assert (calib_k1_seq / "calibration.txt").read_text().strip().split()[4] \
        == "-0.06"

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


# --- 2j-3: the distortion boundaries in Frame ---------------------------

import cv2 as cv  # noqa: E402

from benchmarks.harness import load_calibration as _load_calibration  # noqa: E402
from slam.core.frame import Frame  # noqa: E402
from slam.utils.utils import normalize  # noqa: E402


def _mk_frame(K, k1=0.0):
    return Frame(np.zeros((48, 64, 3), np.uint8), K, k1)


def test_frame_undistortion_round_trip():
    """2j-3: the undistort-at-normalization boundary is the EXACT
    inverse of the generator's forward model (x_d = x_u (1 + k1 r_u^2))
    - cv.undistortPoints implements that inverse. Round-tripped through
    PIXEL keypoints and set_features, the way the pipeline sees it."""
    K = K_450
    k1 = -0.06
    x_u = np.array([[0.30, -0.20], [-0.55, 0.40], [0.62, 0.47],
                    [0.0, 0.0], [-0.1, 0.33]])
    r2 = x_u[:, 0] ** 2 + x_u[:, 1] ** 2
    x_d = x_u * (1.0 + k1 * r2)[:, None]
    pixels = np.column_stack([x_d[:, 0] * K[0, 0] + K[0, 2],
                              x_d[:, 1] * K[1, 1] + K[1, 2]])

    frame = _mk_frame(K, k1)
    frame.set_features([cv.KeyPoint(float(p[0]), float(p[1]), 31.0)
                        for p in pixels], np.zeros((5, 32), np.uint8))

    assert np.allclose(frame.kp_pts_norm, x_u, atol=1e-6)


def test_frame_k1_zero_is_literal_identity():
    """2j-3: at k1=0 the normalization path IS the legacy `normalize`
    call - the default pipeline stays bit-identical (pinned at the
    array level, not just approximately)."""
    pixels = np.array([[100.0, 50.0], [320.0, 240.0], [500.0, 300.0]])
    frame = _mk_frame(K_450, 0.0)
    frame.set_features([cv.KeyPoint(float(p[0]), float(p[1]), 31.0)
                        for p in pixels], np.zeros((3, 32), np.uint8))

    assert np.array_equal(frame.kp_pts_norm, normalize(pixels, frame.Kinv))


def test_frame_pixel_projection_applies_distortion():
    """2j-3: pixel-mode projection applies the simple-radial forward
    model (the 1px triangulation gate compares in pixel space - the Q2
    record verified that gate load-bearing); k1=0 stays the exact
    pinhole path."""
    K = K_450
    k1 = -0.06
    frame = _mk_frame(K, k1)  # identity pose: world == camera coords
    Xs = np.array([[0.3, -0.2, 2.0], [-0.55, 0.4, 1.0], [0.0, 0.25, 4.0]])

    for X in Xs:
        x, y = X[0] / X[2], X[1] / X[2]
        s = 1.0 + k1 * (x * x + y * y)
        expect = np.array([K[0, 0] * x * s + K[0, 2],
                          K[1, 1] * y * s + K[1, 2]])
        assert np.allclose(frame.project_point(X, normalized=False),
                           expect, atol=1e-12)

    # The batch path matches the single-point path under k1.
    batch = frame.project_points(Xs, normalized=False)
    for i, X in enumerate(Xs):
        assert np.allclose(batch[i],
                           frame.project_point(X, normalized=False),
                           atol=1e-12)

    # k1 = 0 -> the exact pinhole closed form (the identity regression).
    frame0 = _mk_frame(K, 0.0)
    for X in Xs:
        x, y = X[0] / X[2], X[1] / X[2]
        expect = np.array([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]])
        assert np.allclose(frame0.project_point(X, normalized=False),
                           expect, atol=1e-12)


def test_load_calibration_k1_token(calib_k1_seq, translation_seq):
    """2j-3: the 5th calibration value carries k1 (the generator
    writes it only when nonzero); 4-value files parse to k1=0
    (backward compatible - existing sequences' files unchanged)."""
    K_k1, k1 = _load_calibration(calib_k1_seq)
    assert k1 == -0.06
    assert K_k1[0, 0] == FX

    K_pin, k1_pin = _load_calibration(translation_seq)
    assert k1_pin == 0.0
    assert K_pin[0, 0] == FX

