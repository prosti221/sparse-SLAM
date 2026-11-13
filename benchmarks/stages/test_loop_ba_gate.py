"""Stage test: phase 18 — the loop-correction combination's BA
acceptance gate and the normalized odometry information.

Pins the two mechanisms the [18b-2] A/B judges:

- the gate [18a-2]: a global BA on the correction path (gated=True)
  writes back ONLY when the objective it minimizes (total squared
  NORMALIZED reprojection error — the g2o EdgeProjectXYZ2UV residual,
  mirrored by _total_reprojection_sq_error) strictly improves; a
  diverged solve is rejected and the pre-solve state is restored
  EXACTLY (the re-anchored, consistent state in the combination —
  what makes the write-back rejectable at all); telemetry
  loop_global_ba_rejected. The max/median move ratio was measured
  NOT to discriminate (phase-18 plan) — the objective test replaced
  it, so no move-based pin here.
- the legacy path: global_bundle_adjustment() (gated defaults
  False) never reads the gate — the Map's PERIODIC global BA and
  every other existing caller stay exactly as they were (the
  battery-silence contract for the gate).
- the normalization [18a-1]: per-edge odometry information =
  LOOP_INFO_ODOMETRY_AGGREGATE / n_odometry_edges — the AGGREGATE
  is constant vs map size (legacy per-edge 10.0 made the aggregate
  grow ~100-200 -> ~2500 from young to big maps; phase 17c's two
  faces of one ratio defect). Flag OFF = the legacy per-edge
  constant.
- the flag defaults (measured adoption facts, phase-18 plan).
"""
import inspect
from contextlib import contextmanager

import numpy as np
from scipy.spatial.transform import Rotation

from benchmarks.stages.helpers import (make_K, make_frame,
                                       make_world_points, project_gt, se3,
                                       set_keypoints, yaw_pose)
from slam.ba import bundle_adjustment_g2o as ba_mod
from slam.ba.bundle_adjustment_g2o import G2OBundleAdjustment
from slam.core.map import Map
from slam.core.observation import Observation
from slam.core.point import Point
from slam.utils import constants as const, telemetry

N_PTS = 16  # 4 KFs x 16 pts = 64 observations >= the 50-obs minimum


def _traj_map(n_kf=4):
    """A straight-trajectory map with REAL measurements: every keyframe
    observes every point through its true pose, so the objective is
    EXACTLY zero at truth (the projections ARE the measurements) and
    any corruption of poses/points is measurable reprojection error."""
    K = make_K()
    m = Map()
    kfs = [make_frame(yaw_pose(0.0, (0.4 * i, 0.0, 0.0)), K)
           for i in range(n_kf)]
    X = make_world_points(N_PTS, z_range=(6.0, 9.0), rng_seed=11)
    points = [Point(x.copy(), None, np.zeros(32, dtype=np.uint8))
              for x in X]
    for kf in kfs:
        uvs = [project_gt(kf.pose, x, K)[0] for x in X]
        set_keypoints(kf, uvs)
        m.add_keyframe(kf)
    m.add_points(points)
    for pt_i, pt in enumerate(points):
        for kf in kfs:
            pt.add_observation(Observation(kf, pt, pt_i))
    return m, kfs, points


def _corrupt(kfs, points, seed=3):
    """Small deterministic NON-rigid corruption: per-KF pose noise +
    per-point structure noise. (A rigid gauge shift would leave the
    objective invariant — the residuals are relative — so the noise
    must differ per keyframe for the error to be measurable.)"""
    rng = np.random.default_rng(seed)
    for kf in kfs:
        kf.pose = kf.pose @ se3(
            Rotation.from_euler("xyz",
                                rng.uniform(-0.05, 0.05, 3)).as_matrix(),
            rng.uniform(-0.15, 0.15, 3))
    for pt in points:
        pt.pt_3d = pt.pt_3d + rng.uniform(-0.15, 0.15, 3)


def _diverging_solver(keyframes, pts, observations, *a, **kw):
    """A fake solve that DIVERGES — teleports every keyframe and
    inflates the structure: the phase-17 measured failure mode
    (21-160-unit KF teleports from a stale/inconsistent init),
    deterministic, no g2o."""
    for kf in keyframes:
        kf.pose = kf.pose @ se3(
            Rotation.from_euler("y", np.radians(30)).as_matrix(),
            np.array([50.0, 10.0, -20.0]))
    for pt in pts:
        pt.pt_3d = pt.pt_3d * 7.0 + 3.0
    return True


@contextmanager
def _recording():
    rec = telemetry.TelemetryRecorder()
    previous = telemetry._recorder
    telemetry.set_recorder(rec)
    try:
        yield rec
    finally:
        telemetry.set_recorder(previous)


def test_gate_accepts_improving_solve(monkeypatch):
    """A solve that improves the objective writes back: the fake lands
    exactly on truth (err ~ 0) from a corrupted pre-solve state — the
    accepted case's contract is poses WRITTEN, accepted telemetry,
    return True. The gate flag is pinned ON: the shipped default is
    OFF after the phase-18 rejection, and this pin tests the
    machinery."""
    m, kfs, points = _traj_map()
    ba = m.g2o_optimizer
    monkeypatch.setattr(ba_mod, "LOOP_BA_GATE_ENABLED", 1)
    truth_poses = [kf.pose.copy() for kf in kfs]
    truth_pts = [pt.pt_3d.copy() for pt in points]
    _corrupt(kfs, points)

    def improving(keyframes, pts, observations, *a, **kw):
        for kf, T in zip(keyframes, truth_poses):
            kf.pose = T.copy()
        for pt, X in zip(pts, truth_pts):
            pt.pt_3d = X.copy()
        return True

    monkeypatch.setattr(ba, "_optimize_with_g2o", improving)

    err_pre = ba._total_reprojection_sq_error(
        m.get_observations(m.keyframes))
    with _recording() as rec:
        assert ba.global_bundle_adjustment(gated=True)
    err_post = ba._total_reprojection_sq_error(
        m.get_observations(m.keyframes))
    assert err_pre > 0 and err_post < err_pre, \
        "the fake must land on a strictly better objective"
    for kf, T in zip(kfs, truth_poses):
        np.testing.assert_allclose(kf.pose, T, atol=1e-15, rtol=0), \
            "an accepted solve must keep its write-back"
    assert rec.events.get("loop_global_ba_accepted", 0) == 1
    assert rec.events.get("loop_global_ba_rejected", 0) == 0


def test_gate_rejects_diverging_solve_and_restores(monkeypatch):
    """THE gate pin: err_post >= err_pre (a diverged rearrangement, not
    a refinement) -> the solve is REJECTED, the pre-solve state is
    restored EXACTLY (poses AND points — the re-anchored consistent
    state), and the solve counts as failed. The gate flag is pinned
    ON: the shipped default is OFF after the phase-18 rejection, and
    this pin tests the machinery."""
    m, kfs, points = _traj_map()
    ba = m.g2o_optimizer
    monkeypatch.setattr(ba_mod, "LOOP_BA_GATE_ENABLED", 1)
    _corrupt(kfs, points)
    pre_poses = [kf.pose.copy() for kf in kfs]
    pre_pts = [pt.pt_3d.copy() for pt in points]
    monkeypatch.setattr(ba, "_optimize_with_g2o", _diverging_solver)

    with _recording() as rec:
        assert not ba.global_bundle_adjustment(gated=True)
    for kf, P in zip(kfs, pre_poses):
        np.testing.assert_array_equal(kf.pose, P), \
            "a rejected solve must restore the pre-solve pose exactly"
    for pt, X in zip(points, pre_pts):
        np.testing.assert_array_equal(pt.pt_3d, X), \
            "a rejected solve must restore the pre-solve point exactly"
    assert rec.events.get("loop_global_ba_rejected", 0) == 1
    assert rec.events.get("loop_global_ba_accepted", 0) == 0


def test_legacy_path_is_bit_transparent_by_default(monkeypatch):
    """The gate is OPT-IN per call site: global_bundle_adjustment()
    (gated=False default) never reads the gate even with the flag ON —
    the diverged write-back PERSISTS (no restore, no gate telemetry).
    The Map's PERIODIC global BA and every existing caller keep this
    path. And the kill-switch: LOOP_BA_GATE_ENABLED=0 makes even
    gated=True legacy."""
    m, kfs, points = _traj_map()
    ba = m.g2o_optimizer
    _corrupt(kfs, points)
    pre_poses = [kf.pose.copy() for kf in kfs]
    monkeypatch.setattr(ba, "_optimize_with_g2o", _diverging_solver)

    with _recording() as rec:
        assert ba.global_bundle_adjustment()
    for kf, P in zip(kfs, pre_poses):
        assert not np.array_equal(kf.pose, P), \
            "the legacy path must keep the solve's write-back as-is"
    assert rec.events.get("loop_global_ba_rejected", 0) == 0
    assert rec.events.get("loop_global_ba_accepted", 0) == 0

    for kf, P in zip(kfs, pre_poses):
        kf.pose = P.copy()
    monkeypatch.setattr(ba_mod, "LOOP_BA_GATE_ENABLED", 0)
    with _recording() as rec:
        assert ba.global_bundle_adjustment(gated=True)
    for kf, P in zip(kfs, pre_poses):
        assert not np.array_equal(kf.pose, P), \
            "flag OFF: gated=True must behave legacy too"
    assert rec.events.get("loop_global_ba_rejected", 0) == 0


def test_real_solver_refinement_passes_the_gate(monkeypatch):
    """With the REAL g2o solver: a small non-rigid corruption refined
    back toward truth must be ACCEPTED — the gate must never block a
    genuine refinement (phase 17 measured healthy repairs at move
    ratios 2.3-6.7; the objective test is what separates them). The
    gate flag is pinned ON: the shipped default is OFF after the
    phase-18 rejection, and this pin tests the machinery."""
    m, kfs, points = _traj_map()
    ba = m.g2o_optimizer
    monkeypatch.setattr(ba_mod, "LOOP_BA_GATE_ENABLED", 1)
    _corrupt(kfs, points)
    err_pre = ba._total_reprojection_sq_error(
        m.get_observations(m.keyframes))
    assert err_pre > 0

    with _recording() as rec:
        assert ba.global_bundle_adjustment(gated=True)
    err_post = ba._total_reprojection_sq_error(
        m.get_observations(m.keyframes))
    assert err_post < err_pre, \
        "a genuine refinement must improve the objective"
    assert rec.events.get("loop_global_ba_accepted", 0) == 1
    assert rec.events.get("loop_global_ba_rejected", 0) == 0


def test_odometry_information_is_normalized_by_edge_count(monkeypatch):
    """[18a-1] THE math pin: per-edge odometry information =
    AGGREGATE / n_odometry_edges, so the AGGREGATE over the chain is
    the constant 500.0 at ANY map size (the legacy defect: per-edge
    10.0 constant -> the aggregate grew with n — the young-map snap
    and the big-map no-op were the same ratio defect at opposite
    ends). The normalization flag is pinned ON: the shipped default
    is OFF after the phase-18 rejection, and this pin tests the
    normalization math."""
    m4, kfs4, _ = _traj_map(4)
    m8, kfs8, _ = _traj_map(8)
    monkeypatch.setattr(ba_mod, "LOOP_ODOMETRY_NORMALIZED_ENABLED", 1)
    v4 = m4.g2o_optimizer._odometry_information(
        {kf.frame_id: None for kf in kfs4})
    v8 = m8.g2o_optimizer._odometry_information(
        {kf.frame_id: None for kf in kfs8})
    np.testing.assert_allclose(
        v4, np.eye(6) * (500.0 / 3), atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        v8, np.eye(6) * (500.0 / 7), atol=1e-12, rtol=0)
    assert abs(3 * v4[0, 0] - 500.0) < 1e-9
    assert abs(7 * v8[0, 0] - 500.0) < 1e-9


def test_odometry_information_flag_off_is_legacy_constant(monkeypatch):
    """Kill-switch transparency: LOOP_ODOMETRY_NORMALIZED_ENABLED=0
    reproduces the legacy per-edge information EXACTLY — the
    LOOP_INFO_ODOMETRY constant per edge, independent of the edge
    count (the measured defect, retained verbatim for the A/B's OFF
    leg)."""
    assert const.LOOP_INFO_ODOMETRY == 10.0
    m4, kfs4, _ = _traj_map(4)
    m8, kfs8, _ = _traj_map(8)
    monkeypatch.setattr(ba_mod, "LOOP_ODOMETRY_NORMALIZED_ENABLED", 0)
    v4 = m4.g2o_optimizer._odometry_information(
        {kf.frame_id: None for kf in kfs4})
    v8 = m8.g2o_optimizer._odometry_information(
        {kf.frame_id: None for kf in kfs8})
    np.testing.assert_array_equal(v4, np.eye(6) * const.LOOP_INFO_ODOMETRY)
    np.testing.assert_array_equal(v8, v4)


def test_normalized_graph_keeps_an_exact_prior_exact(monkeypatch):
    """Real pose-graph machinery, normalization pinned ON (the
    shipped default is OFF after the phase-18 rejection): with EXACT
    odometry edges (the tautological chain prior,
    measured from the current poses) and an agreeing zero-drift loop
    edge (the production measurement convention), the solve is a
    no-op — the normalized stiffness keeps a consistent graph
    consistent (no over-bend at the young-map end, where the legacy
    aggregate let the loop edge dominate 1:1)."""
    m, kfs, _ = _traj_map(6)
    monkeypatch.setattr(ba_mod, "LOOP_ODOMETRY_NORMALIZED_ENABLED", 1)
    poses_before = [kf.pose.copy() for kf in kfs]
    older, reference = kfs[0], kfs[-1]
    relative = older.pose @ np.linalg.inv(reference.pose)
    assert m.g2o_optimizer.pose_graph_optimization(
        [(older.frame_id, reference.frame_id, relative)])
    for kf, P in zip(kfs, poses_before):
        np.testing.assert_allclose(kf.pose, P, atol=1e-6)


def test_phase18_flag_and_signature_defaults_pinned():
    """The measured adoption facts (the phase-18 record): the
    combination AND every pre-registered ablation (norm+gate,
    C2-only, gate-only) FAILED the amended contract on the desktop
    (phase18_t1_contract*.json), so both flags ship OFF; the
    re-anchor stays OFF (17d-1). The machinery + pins remain as
    flag-gated no-ops — bit-identity proven on the ladder bridge
    (_c18off ≡ phase-17 OFF evidence) and the battery. The gate is
    opt-in per call site — gated defaults False, the legacy
    bit-transparent contract for every existing caller."""
    assert ba_mod.LOOP_ODOMETRY_NORMALIZED_ENABLED == 0
    assert ba_mod.LOOP_INFO_ODOMETRY_AGGREGATE == 500.0
    assert ba_mod.LOOP_BA_GATE_ENABLED == 0
    sig = inspect.signature(G2OBundleAdjustment.global_bundle_adjustment)
    assert sig.parameters["gated"].default is False
