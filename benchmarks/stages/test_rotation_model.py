"""Stage tests: phase 13 — the rotation-dominant pair pose (finding 37).

The 13a-13c baseline arc (five measured gate candidates, four dead)
pinned the design: per-pair two-view rotation-vs-translation is
UNMEASURABLE in the pan's regime (planar / sub-noise-parallax pairs —
every per-pair statistic that fires on the pan also fires on slow real
translation), but TEMPORAL t-coherence separates the sources cleanly
(real translation keeps the committed E-t direction coherent pair
after pair; rotation-only motion commits noise directions). These pins
hold:

- the ROTATION MODEL (matcher): the numpy-RNG ray-Kabsch RANSAC
  recovers a pure-rotation pair pose exactly, with t = 0 BY
  CONSTRUCTION (translation is the unobservable component) and the
  consensus matches attached (the match_features contract);
- the GATE (tracker): the t-coherence EMA never engages on coherent
  translation, engages on incoherent directions, sticks (hysteresis),
  re-arms on recovered coherence, and keeps the legacy estimate
  whenever it has no basis (run start, failed pairs, no consensus);
- the SCOPE: rotation_model_enabled defaults OFF (the frozen battery
  and every harness-path diagnostic are bit-identical by
  construction); the module kill-switch reproduces legacy exactly;
- the 13c MEASURED pins (alpha / engage / re-arm / residual /
  min inliers) are guarded against accidental edits.
"""
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

import slam.core.tracker as tracker_mod
import slam.features.matcher as matcher_mod
from benchmarks.stages.helpers import make_frame, project_gt, yaw_pose
from slam.core.map import Map
from slam.core.tracker import Tracker
from slam.features.matcher import rotation_model_pose_between_frames
from slam.utils import telemetry

K = np.array([[450.0, 0.0, 320.0],
              [0.0, 450.0, 240.0],
              [0.0, 0.0, 1.0]])


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(
        np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def _rotation_pair(yaw_deg=3.0, n=120, seed=11):
    """Two frames viewing a random cloud from the SAME camera center,
    the second rotated by yaw_deg: a PURE-ROTATION pair (the exact
    model class of finding 37's pan). Each point carries its own
    random descriptor in both frames, so Lowe matching recovers the
    correspondence exactly."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-2.0, 2.0, n),
                         rng.uniform(-1.2, 1.2, n),
                         rng.uniform(6.0, 12.0, n)])
    desc = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    T_prev, T_cur = np.eye(4), yaw_pose(np.radians(yaw_deg))
    frames = {}
    keep = []
    for name, T in (('prev', T_prev), ('cur', T_cur)):
        f = make_frame(T, K)
        uvs, kept = [], []
        for i, Xi in enumerate(X):
            uv, _ = project_gt(T, Xi, K)
            if not (0 <= uv[0] < 640 and 0 <= uv[1] < 480):
                continue
            uvs.append(uv)
            kept.append(i)
        kps = [cv.KeyPoint(float(u), float(v), size=20) for u, v in uvs]
        f.set_features(kps, desc[kept])
        frames[name] = f
        keep.append(kept)
    return frames['cur'], frames['prev']


# ===== the rotation model (matcher helper) =====

def test_helper_recovers_pure_rotation_pose():
    """THE model pin: on a pure-rotation pair the helper commits the
    exact pair rotation with t = 0 — no fabricated translation — and
    attaches its consensus matches (the match_features contract)."""
    f_cur, f_prev = _rotation_pair(3.0)
    Rt = rotation_model_pose_between_frames(f_cur, f_prev, 'ORB')
    assert Rt is not None
    expected = yaw_pose(np.radians(3.0))[:3, :3]
    assert rot_angle_deg(Rt[:3, :3] @ expected.T) < 1e-3
    np.testing.assert_array_equal(Rt[:3, 3], np.zeros(3))
    assert len(f_cur.matches) >= matcher_mod.ROTATION_MODEL_MIN_INLIERS


def test_helper_is_deterministic():
    """Fixed numpy seed per call: same pair, same output — and no cv
    RNG anywhere in the helper (by construction; this pins the
    determinism half)."""
    f_cur, f_prev = _rotation_pair(-1.5, seed=13)
    a = rotation_model_pose_between_frames(f_cur, f_prev, 'ORB')
    b = rotation_model_pose_between_frames(f_cur, f_prev, 'ORB')
    np.testing.assert_array_equal(a, b)


def test_helper_no_basis_paths():
    """Featureless frames (the finding-35 class) and too-few matches
    return None — the caller keeps the legacy estimate."""
    f_cur, f_prev = _rotation_pair(3.0)
    blank = make_frame(np.eye(4), K)
    assert rotation_model_pose_between_frames(f_cur, blank, 'ORB') is None
    assert rotation_model_pose_between_frames(blank, f_prev, 'ORB') is None

    tiny_cur = make_frame(np.eye(4), K)
    tiny_prev = make_frame(np.eye(4), K)
    rng = np.random.default_rng(3)
    kps = [cv.KeyPoint(float(u), float(v), size=20)
           for u, v in rng.uniform(50, 600, (5, 2))]
    d = rng.integers(0, 256, (5, 32), dtype=np.uint8)
    tiny_cur.set_features(kps, d)
    tiny_prev.set_features(kps, d)
    assert rotation_model_pose_between_frames(
        tiny_cur, tiny_prev, 'ORB') is None


def test_helper_kill_switch_is_legacy(monkeypatch):
    """ROTATION_MODEL_ENABLED = 0: the helper returns before any
    work — legacy behavior, bit-for-bit."""
    monkeypatch.setattr(matcher_mod, 'ROTATION_MODEL_ENABLED', 0)
    f_cur, f_prev = _rotation_pair(3.0)
    assert rotation_model_pose_between_frames(f_cur, f_prev, 'ORB') is None


# ===== the gate (tracker) =====

def _tracker(enabled=True):
    t = Tracker(Map())
    t.rotation_model_enabled = enabled
    t.cur_frame = make_frame(np.eye(4), K)
    t.prev_frame = make_frame(yaw_pose(1.0), K)
    return t


def _rt(direction, angle_deg=0.5):
    Rt = np.eye(4)
    Rt[:3, :3] = Rotation.from_euler('y', np.radians(angle_deg)).as_matrix()
    Rt[:3, 3] = np.asarray(direction, float)
    return Rt


def test_gate_defaults_and_video_scope():
    """rotation_model_enabled defaults OFF: the frozen battery and
    every harness-path diagnostic keep the legacy pair pose
    bit-identical BY CONSTRUCTION (slam.py's video loop — and only it
    — enables the gate, tied to the intake gate's cadence)."""
    t = Tracker(Map())
    assert t.rotation_model_enabled is False
    assert t._rot_model_ema is None
    assert t._rot_model_engaged is False


def test_gate_disabled_returns_legacy_and_touches_no_state():
    t = _tracker(enabled=False)
    Rt = _rt([1.0, 0.0, 0.0])
    for _ in range(10):
        assert t._update_rotation_model_gate(Rt) is Rt
    assert t._rot_model_ema is None
    assert t._rot_model_engaged is False


def test_gate_kill_switch_returns_legacy(monkeypatch):
    monkeypatch.setattr(matcher_mod, 'ROTATION_MODEL_ENABLED', 0)
    t = _tracker(enabled=True)
    Rt = _rt([0.0, 1.0, 0.0])
    rng = np.random.default_rng(5)
    for _ in range(10):
        out = t._update_rotation_model_gate(_rt(rng.normal(size=3)))
        assert out.shape == (4, 4)
    assert t._rot_model_ema is None   # the switch gates the EMA feed too
    assert not t._rot_model_engaged


def test_gate_never_engages_on_coherent_translation():
    """Real translation keeps the committed t direction coherent pair
    after pair (13c: the kyoto_2_5 walking segment EMA never dips
    below 0.958; walking_salem never below 0.996). A coherent series
    must NEVER engage, and the legacy pose must pass through
    untouched (the same object)."""
    t = _tracker(enabled=True)
    base = np.array([0.9, 0.1, 0.3])
    base = base / np.linalg.norm(base)
    for k in range(30):
        jittered = base + 1e-3 * np.cos(0.1 * k) * np.array([0, 1, 0])
        Rt = _rt(jittered)
        assert t._update_rotation_model_gate(Rt) is Rt
    assert t._rot_model_engaged is False
    assert t._rot_model_ema > matcher_mod.ROTATION_MODEL_T_COHERENCE_REARM


def test_gate_engages_sticks_and_rearms(monkeypatch):
    """Incoherent committed directions (the pan's measured signature:
    forward / backward / backward / forward) engage the rotation
    model; the engagement STICKS through locally-coherent noise
    stretches (hysteresis: re-arm only above REARM); a sustained
    coherent stretch (real translation resumed) re-arms legacy. While
    engaged, the committed pair pose is the ROTATION MODEL's — the
    legacy estimate still feeds the EMA."""
    t = _tracker(enabled=True)
    rng = np.random.default_rng(9)  # noqa: F841 (kept for the rearm-phase jitter)

    # a fixed rotation-model pose the monkeypatched helper returns
    Rt_rot = np.eye(4)
    Rt_rot[:3, :3] = yaw_pose(0.7)[:3, :3]

    def _fake_rot_model(*args, **kwargs):
        return Rt_rot

    monkeypatch.setattr(tracker_mod, 'rotation_model_pose_between_frames',
                        _fake_rot_model)

    # --- incoherent (cycling orthogonal axes: |cos| = 0 between
    #     consecutive pairs): engages within a few pairs ---
    axes = np.eye(3)
    engaged_at = None
    for k in range(12):
        Rt = _rt(axes[k % 3] * (1.0 if k % 2 else -1.0))
        out = t._update_rotation_model_gate(Rt)
        if t._rot_model_engaged and engaged_at is None:
            engaged_at = k
            assert out is Rt_rot
    assert engaged_at is not None and engaged_at <= 6
    assert t._rot_model_ema < matcher_mod.ROTATION_MODEL_T_COHERENCE_ENGAGE

    # --- locally coherent noise must NOT re-arm (sticky below REARM) ---
    wob = np.array([1.0, 0.05, 0.0]) / np.linalg.norm([1.0, 0.05, 0.0])
    for _ in range(5):
        out = t._update_rotation_model_gate(_rt(wob))
        assert out is Rt_rot
        assert t._rot_model_engaged
    assert t._rot_model_ema < matcher_mod.ROTATION_MODEL_T_COHERENCE_REARM

    # --- sustained coherence re-arms legacy (alpha 0.9: 1 - 0.5*0.9^k
    #     crosses 0.95 at k ~ 22; use 30 to be safe) ---
    for _ in range(30):
        Rt = _rt(wob)
        out = t._update_rotation_model_gate(Rt)
        if not t._rot_model_engaged:
            assert out is Rt
            break
    assert not t._rot_model_engaged


def test_gate_no_basis_keeps_legacy():
    """No EMA basis (run start; failed pairs; zero-t estimates) never
    engages: the legacy estimate stands — the no-basis principle."""
    t = _tracker(enabled=True)
    assert t._update_rotation_model_gate(None) is None
    zero_t = np.eye(4)
    for _ in range(10):
        assert t._update_rotation_model_gate(zero_t) is zero_t
    assert t._rot_model_ema is None
    assert t._rot_model_engaged is False
    # a failed pair between good pairs leaves the EMA chain intact
    good = _rt([0.0, 0.0, 1.0])
    t._update_rotation_model_gate(good)
    t._update_rotation_model_gate(None)
    t._update_rotation_model_gate(good)
    assert t._rot_model_ema == 1.0


def test_13c_measured_pins():
    """The gate constants are MEASURED pins (phase13c_t_coherence.json
    traces: pan engaged 281/281 with EMA p90 0.93 below re-arm; real
    walking never engaged, EMA min 0.958) — guarded against
    accidental edits, never tuning targets."""
    assert matcher_mod.ROTATION_MODEL_EMA_ALPHA == 0.9
    assert matcher_mod.ROTATION_MODEL_T_COHERENCE_ENGAGE == 0.80
    assert matcher_mod.ROTATION_MODEL_T_COHERENCE_REARM == 0.95
    assert matcher_mod.ROTATION_MODEL_RESIDUAL == 3e-3
    assert matcher_mod.ROTATION_MODEL_MIN_INLIERS == 20
    assert matcher_mod.ROTATION_MODEL_ENABLED == 1


def test_engaged_pair_commits_rotation_only_pose(monkeypatch, ):
    """End-to-end on the gate: while engaged, update()'s committed
    pair step is the rotation model's [R | 0] — the unit-t injection
    is gone BY CONSTRUCTION — and the engaged/rearm telemetry fires
    only on the engaged path."""
    rec = telemetry.TelemetryRecorder()
    telemetry.set_recorder(rec)
    try:
        t = _tracker(enabled=True)
        rng = np.random.default_rng(21)

        def _fake_rot_model(frame1, frame2, method):
            return _rt([0.0, 0.0, 0.0], angle_deg=0.4)

        monkeypatch.setattr(tracker_mod, 'rotation_model_pose_between_frames',
                            _fake_rot_model)
        # structurally incoherent directions (cycling orthogonal axes:
        # |cos| = 0 between consecutive pairs - the engagement is by
        # construction, not seed luck)
        axes = np.eye(3)
        fed = [_rt(axes[k % 3]) for k in range(14)]
        for Rt in fed[:10]:
            t._update_rotation_model_gate(Rt)
        assert t._rot_model_engaged
        out = t._update_rotation_model_gate(fed[10])
        np.testing.assert_array_equal(out[:3, 3], np.zeros(3))
        assert rec.events.get('pair_rotation_model_engage') == 1
        assert 'pair_rotation_model_rearm' not in rec.events
    finally:
        telemetry.set_recorder(None)
