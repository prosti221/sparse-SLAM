"""Stage test: the calibration arc.

Homes the pins for the calibration items as they land:
- the FOCAL_SCALE diagnostic (harness-side focal perturbation for
  pain measurement and recovery validation)
- the calib_motion sequences + the generator's k1 render path
- the GeoCalib initializer wrapper, the distortion
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

    # The k1 variant's file additionally carries the true
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
    """'fx: auto' (or an absent fx) signals auto-calibration by
    returning K=None - slam.py then runs estimate_video_intrinsics.
    A numeric fx keeps today's behavior exactly. load_video
    returns a THIRD value, the config's k1 (0.0 when absent); the auto
    branch returns 0.0 because GeoCalib supplies its own k1 through
    the prior."""
    video = _write_tiny_video(tmp_path / "tiny.avi")

    cap, K, k1 = load_video(_FakeConfig(
        {"load_video": "v", "path": video, "fx": "auto"}))
    assert K is None
    assert k1 == 0.0
    cap.release()

    cap, K, k1 = load_video(_FakeConfig(
        {"load_video": "v", "path": video}))          # fx absent
    assert K is None
    assert k1 == 0.0
    cap.release()

    cap, K, k1 = load_video(_FakeConfig(
        {"load_video": "v", "path": video, "fx": 480}))
    assert K[0, 0] == 480.0          # fy defaults to fx
    assert K[1, 1] == 480.0
    assert K[0, 2] == 32.0           # cx defaults to W // 2
    assert K[1, 2] == 24.0           # cy defaults to H // 2
    assert k1 == 0.0                  # k1 defaults to the identity path
    cap.release()


def test_load_video_returns_configured_k1(tmp_path):
    """An explicit k1 flows out of load_video unchanged (the
    float cast accepts yaml ints/strings); an explicit k1 is IGNORED on
    the auto branch by design (GeoCalib supplies its own)."""
    video = _write_tiny_video(tmp_path / "tiny.avi")

    cap, K, k1 = load_video(_FakeConfig(
        {"load_video": "v", "path": video, "fx": 480, "k1": -0.0608}))
    assert K is not None
    assert k1 == -0.0608
    assert isinstance(k1, float)
    cap.release()

    # yaml parses -0.0608 as float; a quoted string must also work
    cap, K, k1 = load_video(_FakeConfig(
        {"load_video": "v", "path": video, "fx": 480, "k1": "-0.1"}))
    assert k1 == -0.1
    cap.release()

    # auto branch: an explicit k1 does NOT override GeoCalib's estimate
    cap, K, k1 = load_video(_FakeConfig(
        {"load_video": "v", "path": video, "fx": "auto", "k1": -0.06}))
    assert K is None and k1 == 0.0
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


# --- 2j-4: the intrinsics refit (the DroidCalib port) --------------------

from scipy.spatial.transform import Rotation  # noqa: E402

from benchmarks.stages.helpers import make_frame, se3, set_keypoints  # noqa: E402
from slam.calibration.refit import (IntrinsicsEstimate,  # noqa: E402
                                    IntrinsicsRefiner, gate_frees)
from slam.core.map import Map  # noqa: E402
from slam.core.observation import Observation  # noqa: E402
from slam.core.point import Point  # noqa: E402
from slam.core.tracker import Tracker  # noqa: E402
from slam.utils.utils import normalize, undistort_points_norm  # noqa: E402


def _refit_fixture(f_true=450.0, k1_true=-0.05, n_kf=12, n_pts=150):
    """A ground-truth map for the refit pins: keyframes on a baseline
    arc all looking at a SHARED point cloud (guaranteed mutual
    visibility + strong parallax - the basin search separates the true
    intrinsics basin by multi-view inconsistency, which needs the map
    over-determined; a 5-KF/80-point first fixture let a jointly-wrong
    (f, k1) beat the truth, and far-field keypoints past the
    simple-radial inversion horizon blew up the solve - both caught by
    the recovery pin). Observations live in BOTH registries as the
    same object (the registry-consistency semantics)."""
    K = np.array([[f_true, 0.0, 320.0],
                  [0.0, f_true, 240.0],
                  [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(11)
    m = Map()
    kfs = []

    def _look_at(pos, target):
        fwd = target - pos
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.eye(3)
        R[:, 0], R[:, 1], R[:, 2] = right, up, fwd
        return R

    anchor = np.array([2.8, 0.0, 8.5])
    for j in range(n_kf):
        pos = np.array([0.55 * j, 0.30 * np.sin(0.7 * j), 0.45 * j])
        # look-at toward the shared cloud + a small local pitch
        # oscillation (rotation diversity for the solve's conditioning)
        R = _look_at(pos, anchor) @ Rotation.from_euler(
            "x", 0.05 * np.sin(0.9 * j)).as_matrix()
        T = se3(R, pos)
        fr = make_frame(T, K)
        fr.k1 = k1_true
        kfs.append(fr)
        m.add_keyframe(fr)

    pts = []
    rng_attempts = 0
    while len(pts) < n_pts and rng_attempts < 8000:
        rng_attempts += 1
        X = anchor + np.array([rng.uniform(-1.1, 1.1),
                               rng.uniform(-0.8, 0.8),
                               rng.uniform(-1.1, 1.1)])
        # Guard (real pipelines never leave |x_norm| ~ 0.75; beyond
        # |x| ~ 4.5 the simple-radial INVERSION HORIZON for k1=-0.05
        # corrupts cv.undistortPoints).
        ok = True
        for fr in kfs:
            cam = fr.world_to_camera(X)
            if cam[2] <= 0.2:
                ok = False
                break
            u = f_true * cam[0] / cam[2] + 320.0
            v = f_true * cam[1] / cam[2] + 240.0
            if not (-40.0 <= u <= 680.0 and -40.0 <= v <= 520.0):
                ok = False
                break
        if ok:
            pts.append(Point(X, None, np.zeros(32, dtype=np.uint8)))
    assert len(pts) >= min(60, n_pts), \
        f"fixture could not collect in-frame points ({len(pts)})"

    for fr in kfs:
        uvs = []
        for pt in pts:
            cam = fr.world_to_camera(pt.pt_3d)
            x_u, y_u = cam[0] / cam[2], cam[1] / cam[2]
            scale = 1.0 + k1_true * (x_u ** 2 + y_u ** 2)
            uvs.append((f_true * x_u * scale + 320.0,
                        f_true * y_u * scale + 240.0))
        set_keypoints(fr, uvs)
        for i, pt in enumerate(pts):
            if fr.world_to_camera(pt.pt_3d)[2] <= 0.2:
                continue  # behind this camera: no observation
            obs = Observation(fr, pt, i)
            pt.add_observation(obs)
            fr.add_point_observation(obs)

    m.add_points(pts)
    return m, kfs, pts


def test_refit_recovers_intrinsics_from_perturbed_estimate():
    """2j-4 v3.1, THE core pin: the end-of-stream BASIN SEARCH recovers
    (f, k1) from a perturbed initialization on a ground-truth map. The
    grid resolves f to the fine-grid step around the coarse winner
    (~1% here); k1's additive grid contains the truth exactly."""
    m, _, _ = _refit_fixture()
    est = IntrinsicsEstimate(0.8 * 450.0, 0.8 * 450.0, 320.0, 240.0,
                             -0.05 + 0.03,
                             sigma_f=0.35 * 0.8 * 450.0,
                             sigma_k1=0.02, source="test")
    refiner = IntrinsicsRefiner(m, est)
    refiner.phase = "free"

    refiner.final_search()

    assert abs(est.f - 450.0) / 450.0 < 0.01, f"recovered f={est.f}"
    assert abs(est.k1 - (-0.05)) < 0.005, f"recovered k1={est.k1}"
    assert est.frozen, "the final search must freeze the estimate"


def test_hook_only_gates_search_runs_at_stream_end():
    """2j-4 v3.1: the insertion hook ONLY gate-checks - with the gate
    freed, hooks never touch the estimate (the mid-stream search
    variant was measured and rejected: it churns the map for no
    resolvable gain); the search runs once, at stream end. A gate that
    never freed is a no-op even at stream end."""
    m, _, _ = _refit_fixture()
    est = IntrinsicsEstimate(0.85 * 450.0, 0.85 * 450.0, 320.0, 240.0,
                             -0.05, sigma_f=0.35 * 0.85 * 450.0,
                             sigma_k1=0.02, source="test")
    refiner = IntrinsicsRefiner(m, est)

    # The gate never freed (few keyframes): even the final search is a
    # no-op - the provided intrinsics stand.
    refiner.final_search()
    assert est.f == 0.85 * 450.0 and not est.frozen

    # With the gate freed, the hooks alone change NOTHING...
    refiner.phase = "free"
    before = est.as_dict()
    for _ in range(5):
        refiner.on_keyframe_inserted()
    assert est.as_dict() == before, "the hook must not search mid-stream"

    # ...and the final search recovers + freezes.
    refiner.final_search()
    assert abs(est.f - 450.0) / 450.0 < 0.02
    assert est.frozen
    frozen_dict = est.as_dict()
    refiner.on_keyframe_inserted()
    refiner.final_search()   # frozen: no-op even at stream end
    assert est.as_dict() == frozen_dict


def test_gate_locks_critical_motion_and_frees_diverse():
    """2j-4: the observability gate rejects exactly DroidCalib's
    critical motions - single-axis translation (the `translation`
    profile), orbital motion (2 translation axes but yaw-only
    rotation, the `orbit` profile), planar forward+yaw (the
    `forward_rotation` profile) - and frees on diverse motion."""
    lateral = [make_frame(se3(np.eye(3), [1.5 * i, 0.0, 0.0]))
               for i in range(14)]
    assert not gate_frees(lateral, 1000)

    orbital = []
    for i in range(14):
        theta = 2.0 * np.pi * i / 14.0
        orbital.append(make_frame(se3(
            Rotation.from_euler("y", theta).as_matrix(),
            [4.0 * np.cos(theta), 0.0, 4.0 * np.sin(theta)])))
    assert not gate_frees(orbital, 1000)

    forward_yaw = [make_frame(se3(
        Rotation.from_euler("y", 0.25 * np.sin(i / 5.0)).as_matrix(),
        [0.0, 0.0, 1.2 * i])) for i in range(14)]
    assert not gate_frees(forward_yaw, 1000)

    diverse = [make_frame(se3(
        Rotation.from_euler("yx",
                            [0.25 * np.sin(i / 3.0), 0.3 * np.cos(i / 4.0)]
                            ).as_matrix(),
        [0.8 * i, 0.6 * np.sin(i / 2.0),
         3.0 + 2.5 * np.sin(i / 2.5)]))
        for i in range(14)]
    # NOTE: the translation components must be DECOUPLED - a first
    # fixture had x and z both linear in i (a straight diagonal line),
    # and the gate correctly counted that as ONE axis.
    assert gate_frees(diverse, 1000)

    # The floor conditions gate even diverse motion.
    assert not gate_frees(diverse, 100)        # too few map points
    assert not gate_frees(diverse[:6], 1000)   # too few keyframes


def test_gate_locks_noisy_orbit():
    """2j-5 negative-control regression: orbit's rough tracking
    accumulates pose-estimation NOISE on secondary rotation axes, and
    the first gate design summed |rotvec| per axis - LINEARLY - so
    the noise freed a gate that pure yaw must keep locked (Run A
    measured orbit's gate freeing and the search adopting f=463.5).
    The eigen form weighs each pair's rotation by its squared
    magnitude, so the noise contributes quadratically less than the
    real yaw rotation mass."""
    rng = np.random.default_rng(3)
    noisy_orbital = []
    yaw = 0.0
    for _ in range(16):
        yaw += 2.0 * np.pi / 16.0
        # The TRUE motion is pure yaw; pose-estimation noise adds a
        # ~1.5 deg wobble on the other axes (orbit's per-KF reality).
        R = Rotation.from_euler("xyz", [
            np.deg2rad(rng.normal(0.0, 1.5)),
            yaw + np.deg2rad(rng.normal(0.0, 1.5)),
            np.deg2rad(rng.normal(0.0, 1.5))]).as_matrix()
        noisy_orbital.append(make_frame(se3(
            R, [4.0 * np.cos(yaw), 0.0, 4.0 * np.sin(yaw)])))
    assert not gate_frees(noisy_orbital, 1000)


def test_search_keeps_true_incumbent_and_adopts_gross_error():
    """2j-5 regression: the adopt-margin guard. A TRUE init must be
    KEPT (the floor noise between plausible candidates is ~1-5% and
    biased - tum moved a true init +3.4% before the guard); a gross
    error must still be adopted (the measured 20%+ signal clears the
    10% margin)."""
    # True init on a ground-truth fixture: kept exactly.
    m, _, _ = _refit_fixture()
    est = IntrinsicsEstimate(450.0, 450.0, 320.0, 240.0, -0.05,
                             sigma_f=150.0, sigma_k1=0.02, source="test")
    refiner = IntrinsicsRefiner(m, est)
    refiner.phase = "free"
    refiner.final_search()
    assert est.f == 450.0 and est.k1 == -0.05, \
        f"a true init must be kept (got f={est.f}, k1={est.k1})"
    assert est.frozen

    # Gross error: adopted (and frozen at the winning basin).
    m, _, _ = _refit_fixture()
    est = IntrinsicsEstimate(0.8 * 450.0, 0.8 * 450.0, 320.0, 240.0,
                             -0.02, sigma_f=126.0, sigma_k1=0.02,
                             source="test")
    refiner = IntrinsicsRefiner(m, est)
    refiner.phase = "free"
    refiner.final_search()
    assert abs(est.f - 450.0) / 450.0 < 0.02, \
        f"a gross error must be adopted (got f={est.f})"
    assert est.frozen


def test_kept_incumbent_preserves_unequal_focals():
    """2j-5 regression: the shared-f design averaged TUM's unequal
    focals (517.31/516.47 -> 516.9) even on the KEPT path - a provided
    fx != fy must survive untouched (caught by the TUM run's
    final_intrinsics readout: f_final was 0.08% below the provided
    init for no reason). The estimate here carries unequal focals
    whose MEAN is the fixture's truth, so the incumbent is at the
    noise floor and nothing can clear the adopt margin."""
    m, _, _ = _refit_fixture()
    est = IntrinsicsEstimate(460.0, 440.0, 320.0, 240.0, -0.05,
                             sigma_f=150.0, sigma_k1=0.02, source="test")
    refiner = IntrinsicsRefiner(m, est)
    refiner.phase = "free"

    refiner.final_search()

    assert (est.fx == 460.0 and est.fy == 440.0
            and est.k1 == -0.05), (
        f"a kept incumbent must preserve fx/fy exactly "
        f"(got {est.fx}/{est.fy})")
    assert est.frozen


def test_refit_locked_is_noop():
    """With the gate locked (fewer keyframes than the floor), the
    hook changes NOTHING - the estimate, its values, and the phase are
    untouched. This is the property that makes the negative controls
    (orbit/translation/fr with the module ON) bit-identical to module
    OFF."""
    m, _, _ = _refit_fixture(n_kf=3, n_pts=30)
    est = IntrinsicsEstimate.from_K(K_450)
    before = est.as_dict()
    refiner = IntrinsicsRefiner(m, est)

    refiner.on_keyframe_inserted()

    assert refiner.phase == "locked"
    assert est.as_dict() == before


def test_propagation_updates_frames_and_caches():
    """2j-4: propagate lands the new intrinsics on every keyframe - K,
    k1, the Kinv cache, the kp_pts_norm frame cache (recomputed under
    the new intrinsics), the Observation.pt_2d_norm caches
    (invalidated - they were normalized coords baked under the OLD
    intrinsics) - and drops the persistent g2o graph (its edge
    MEASUREMENTS are normalized coords baked at edge creation)."""
    m, kfs, _ = _refit_fixture()  # a realistic observation set
    est = IntrinsicsEstimate(400.0, 400.0, 320.0, 240.0, -0.01,
                             sigma_f=150.0, sigma_k1=0.02, source="test")
    refiner = IntrinsicsRefiner(m, est)
    refiner.phase = "free"

    kf = kfs[0]
    obs = next(iter(kf.observed_points.values()))
    old_norm = obs.pt_2d_norm  # warm the cached_property
    assert "pt_2d_norm" in obs.__dict__
    _ = kf.Kinv.copy()

    refiner._propagate(443.0, -0.045)  # land new intrinsics directly

    new_f, new_k1 = est.f, est.k1
    assert kf.K[0, 0] == new_f and kf.K[1, 1] == new_f
    assert kf.k1 == new_k1
    assert np.allclose(kf.Kinv, np.linalg.inv(kf.K))

    if new_k1 != 0.0:
        expect = undistort_points_norm(kf.kp_pts, kf.K, new_k1)
    else:
        expect = normalize(kf.kp_pts, kf.Kinv)
    assert np.array_equal(kf.kp_pts_norm, expect)

    # The pt_2d_norm cached_property was POPPED by the propagation
    # (it was normalized coords baked under the OLD intrinsics) and is
    # recomputed on access under the NEW ones. NOTE: the value drift is
    # near-zero HERE because the solve recovered ~the true K the
    # fixture was built with - the invalidation is what matters, not
    # the drift.
    assert "pt_2d_norm" not in obs.__dict__
    new_norm = obs.pt_2d_norm
    assert "pt_2d_norm" in obs.__dict__

    px = obs.pt_2d  # the observed pixel: K-independent
    if new_k1 != 0.0:
        expect_norm = undistort_points_norm(
            np.asarray(px, dtype=np.float64).reshape(1, 2), kf.K, new_k1)[0]
    else:
        expect_norm = (kf.Kinv @ np.array([px[0], px[1], 1.0]))[:2]
    assert np.allclose(new_norm, expect_norm, atol=1e-9)
    _ = old_norm  # (kept for the narrative; see the note above)

    assert m.g2o_optimizer._local_opt is None, \
        "persistent graph must be dropped on intrinsics change"


def test_tracker_creates_no_refiner_by_default():
    """2j-4: the DEFAULT pipeline constructs no refiner (and never even
    imports the module) - the flag is the single activation point."""
    m = Map()
    m.intrinsics = IntrinsicsEstimate.from_K(K_450)
    tracker = Tracker(m)
    assert tracker.calib_refiner is None

