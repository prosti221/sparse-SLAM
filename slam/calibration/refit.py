"""Intrinsics refit in the BA loop (phase 2j-4) - the DroidCalib port,
v3: BASIN SEARCH.

Design record (the full probe trail is in benchmarks/IMPROVEMENT.md
"Phase 2j-4 record"):

v1 refined (f, k1) by plain least squares against the frozen map's
reprojection residuals, alternated with the g2o local BA. Six probes
killed it: the map built under a WRONG f absorbs the error completely
(wrong-f map reprojections at the NOISE FLOOR - wall-plane scene,
volume scene, wall-hug trajectory, and real TUM imagery alike),
making the wrong-f basin a critical point of BOTH the alternating
and the joint problem. No local method - reprojection gradient or
epipolar pairwise (measured: pure noise) - escapes it. The wrong-f
and true-f basins are DISCONNECTED for local descent; the true basin
is measurably better (15% lower converged cost on near-noiseless
synthetic; chi2 gap ~480 px^2 over 30k observations).

v3 therefore COMPARES basins instead of following gradients: for each
candidate f (coarse-to-fine grid around the estimate), propagate K/k1
to the map (the 2j-3 machinery - two cache recomputations, the g2o
graph holds no intrinsics), re-solve the WHOLE map into that
candidate's basin via the global g2o BA (point-side observations -
the 2f lesson; the built-in global path still uses the frame-side
registry), and measure the converged pixel-domain cost. Keep the best
basin. The crossing probe proved one BA call gets 98% of the way and
the ordering is decided by two orders of magnitude; per-candidate
costs converge under a small cap of BA calls.

Lifecycle (v3.1, end-of-stream form): the observability gate
(unchanged - critical motions stay locked) frees during the run ->
ONE basin search at STREAM END (run_slam post-loop; the mid-stream
variant was measured and rejected - the record has the numbers) ->
freeze. A gate that never freed is a no-op (critical-motion
sequences keep the provided intrinsics).

The k1 coordinate pass runs only when the current k1 is nonzero
(pinhole runs keep k1 pinned at EXACTLY 0 - the no-hallucination
property is structural, not measured).

LOCKED / frozen / between-searches are no-ops by construction: the
negative controls (orbit / translation / fr) are bit-identical to
module OFF. The DEFAULT pipeline never imports this module (the
tracker only constructs a refiner under CALIB_REFIT_ENABLED).
"""
import numpy as np
from scipy.spatial.transform import Rotation

from slam.utils import constants as const
from slam.utils import telemetry
from slam.utils.logger import info_log, warning_log
from slam.utils.utils import normalize, undistort_points_norm

LOG_TAG = "CalibRefit"

# The candidate grids (module-level, not --override-sweepable: they are
# search structure, not tunable thresholds).
_COARSE_F_GRID = (0.60, 0.75, 0.85, 0.95, 1.00, 1.05, 1.20, 1.40)
_REFINE_F_GRID = (0.92, 0.96, 1.00, 1.04, 1.08)
_REFINE2_F_GRID = (0.97, 0.985, 1.00, 1.015, 1.03)
_K1_ADDITIVE_GRID = (-0.06, -0.03, -0.012, 0.0, 0.012, 0.03, 0.06)


class IntrinsicsEstimate:
    """The LIVE intrinsics state (phase 2j-4): the value the frame-
    construction sites read every frame, and the object the refit
    mutates on convergence. Carries the INITIALIZATION's sigmas for
    provenance (the search needs no prior weights - it compares
    basins by data cost; the GeoCalib uncertainty informed the INIT,
    which is where it belongs).

    The principal point is pinned at the image center and never
    refined (GeoCalib's own assumption; recorded in the 2j plan).
    """

    def __init__(self, fx, fy, cx, cy, k1=0.0,
                 sigma_f=float("inf"), sigma_k1=float("inf"),
                 source="unknown"):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.k1 = float(k1)
        self.sigma_f = float(sigma_f)
        self.sigma_k1 = float(sigma_k1)
        self.source = source
        self.frozen = False

    @property
    def f(self):
        return 0.5 * (self.fx + self.fy)

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @classmethod
    def from_K(cls, K, k1=0.0, source="dataset"):
        """From a known/provided calibration (the benchmark path)."""
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        return cls(fx, fy, cx, cy, k1,
                   sigma_f=const.CALIB_PRIOR_SIGMA_F_FRAC * 0.5 * (fx + fy),
                   sigma_k1=const.CALIB_PRIOR_SIGMA_K1,
                   source=source)

    @classmethod
    def from_prior(cls, prior):
        """From a GeoCalib IntrinsicsPrior (the video 'fx: auto' path)."""
        return cls(prior.fx, prior.fy, prior.cx, prior.cy, prior.k1,
                   sigma_f=prior.sigma_f, sigma_k1=prior.sigma_k1,
                   source=prior.source)

    def as_dict(self):
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "k1": self.k1, "sigma_f": self.sigma_f,
                "sigma_k1": self.sigma_k1, "source": self.source,
                "frozen": self.frozen}

    def __repr__(self):
        return (f"IntrinsicsEstimate(f=({self.fx:.1f},{self.fy:.1f}) "
                f"k1={self.k1:.5f} source={self.source} "
                f"{'FROZEN' if self.frozen else 'live'})")


def gate_metrics(keyframes):
    """Observability metrics over the keyframes (recomputed from poses
    each check - no accumulator state; O(n_kf) at keyframe cadence).

    translation_axes: the number of significant eigen-directions of
    the keyframe-center covariance (>= 2 excludes single-axis pure
    translation - a critical motion). NOTE: the components must be
    DECOUPLED - a straight diagonal line counts as one axis (the
    first gate pin caught exactly that).
    rotation_axes: the number of axes whose accumulated |rotation
    vector| exceeds CALIB_GATE_MIN_ROT_DEG (>= 2 excludes planar
    rotation AND orbital motion, which is yaw-only).
    """
    n = len(keyframes)
    translation_axes = 0
    if n >= 2:
        centers = np.array([kf.pose[:3, 3] for kf in keyframes])
        centered = centers - centers.mean(axis=0)
        eig = np.linalg.eigvalsh(centered.T @ centered)
        eig[eig < 0] = 0.0
        translation_axes = int(np.sum(
            eig >= const.CALIB_GATE_TRANSL_AXIS_RATIO * eig[-1]))
    rot_acc = np.zeros(3)
    for a, b in zip(keyframes[:-1], keyframes[1:]):
        R_rel = a.pose[:3, :3].T @ b.pose[:3, :3]
        rot_acc += np.abs(Rotation.from_matrix(R_rel).as_rotvec())
    rotation_axes = int(np.sum(np.degrees(rot_acc)
                               >= const.CALIB_GATE_MIN_ROT_DEG))
    return translation_axes, rotation_axes, rot_acc


def gate_frees(keyframes, points_count):
    """The observability gate (all thresholds sweepable constants)."""
    if len(keyframes) < const.CALIB_GATE_MIN_KEYFRAMES:
        return False
    if points_count < const.CALIB_GATE_MIN_POINTS:
        return False
    translation_axes, rotation_axes, _ = gate_metrics(keyframes)
    if translation_axes < const.CALIB_GATE_MIN_TRANSL_AXES:
        return False
    if rotation_axes < const.CALIB_GATE_MIN_ROT_AXES:
        return False
    return True


class IntrinsicsRefiner:
    """Owns the gate -> final-search lifecycle (v3.1, end-of-stream
    mode - the mid-stream variant was measured and rejected: searching
    at keyframe cadence churns the map for ~53 s and cannot resolve f
    finely anyway; the record has the acceptance numbers). Created by
    the Tracker ONLY when CALIB_REFIT_ENABLED is set (an int flag -
    the --override parser is type(current)(value), and bool("0") is
    True); the module is not even imported otherwise."""

    def __init__(self, map, estimate):
        self.map = map
        self.estimate = estimate
        self.phase = "locked"

    # -- lifecycle -------------------------------------------------------

    def on_keyframe_inserted(self):
        """The tracker's post-insertion hook: GATE-CHECK ONLY (the
        search itself runs once, at stream end - final_search)."""
        est = self.estimate
        if est.frozen:
            telemetry.event("calib_hook_frozen")
            return

        if self.phase == "locked":
            if not gate_frees(self.map.keyframes, len(self.map.points)):
                telemetry.event("calib_gate_locked")
                return
            self.phase = "free"
            telemetry.event("calib_gate_freed")
            info_log(LOG_TAG, f"Observability gate freed at "
                     f"{len(self.map.keyframes)} keyframes "
                     f"(f={est.f:.1f}, k1={est.k1:.5f}) - the basin "
                     f"search will run at stream end")
            return

        telemetry.event("calib_hook_free")

    def final_search(self):
        """The one-shot basin search, called by the run entry points
        at STREAM END (run_slam post-loop; the app path may call it
        whenever the stream finishes). The gate never having freed is
        a no-op (critical-motion sequences keep the provided
        intrinsics - the module is inert for them by construction)."""
        est = self.estimate
        if est.frozen:
            telemetry.event("calib_final_skipped_frozen")
            return
        if self.phase == "locked":
            telemetry.event("calib_final_gate_locked")
            info_log(LOG_TAG, "stream end: the observability gate "
                     "never freed - keeping the provided intrinsics")
            return

        self._run_search()
        est.frozen = True
        telemetry.event("calib_gate_frozen")
        info_log(LOG_TAG, f"intrinsics frozen at stream end: "
                 f"f={est.f:.2f}, k1={est.k1:.5f}")

    # -- the basin search ---------------------------------------------------

    def _point_side_observations(self):
        """Keyframe observations from the POINT-SIDE registry
        (Point.observations - the 2f/finding-17 lesson: the frame-side
        registry additionally holds dedup-only matches)."""
        kf_ids = {kf.frame_id for kf in self.map.keyframes}
        obs_list = []
        for pt in self.map.points:
            for fid, obs in pt.observations.items():
                if fid in kf_ids:
                    obs_list.append((pt, obs))
        return obs_list

    def _cost(self, f, k1, obs_list):
        """Converged-basin cost: mean pixel-domain squared residual per
        observation over the CURRENT map state (the same quantity the
        probes scanned as J(f))."""
        X = np.array([pt.pt_3d for pt, _ in obs_list])
        R = np.array([obs.frame.pose[:3, :3].T for _, obs in obs_list])
        t = np.array([-obs.frame.pose[:3, :3].T @ obs.frame.pose[:3, 3]
                      for _, obs in obs_list])
        u_obs = np.array([obs.pt_2d[0] for _, obs in obs_list])
        v_obs = np.array([obs.pt_2d[1] for _, obs in obs_list])
        cam = np.einsum("kij,kj->ki", R, X) + t
        x_u = cam[:, 0] / cam[:, 2]
        y_u = cam[:, 1] / cam[:, 2]
        est = self.estimate
        ru = u_obs - (f * x_u * (1.0 + k1 * (x_u ** 2 + y_u ** 2))
                      + est.cx)
        rv = v_obs - (f * y_u * (1.0 + k1 * (x_u ** 2 + y_u ** 2))
                      + est.cy)
        return float(np.mean(ru ** 2 + rv ** 2))

    def _snapshot(self):
        return ([kf.pose.copy() for kf in self.map.keyframes],
                [pt.pt_3d.copy() for pt in self.map.points])

    def _restore(self, snap):
        poses, pts3d = snap
        for kf, pose in zip(self.map.keyframes, poses):
            kf.pose = pose
        for pt, X in zip(self.map.points, pts3d):
            pt.pt_3d = X

    def _solve_into_basin(self, obs_list):
        """Re-solve the whole map toward the CURRENT intrinsics basin:
        global g2o BA over point-side observations (the built-in
        global path still uses the frame-side registry - the 2f lesson
        applies here too), iterated until the cost stabilizes (the
        crossing probe: one call lands 98% of the way; a small cap
        converges the floor)."""
        if len(obs_list) < const.MINIMUM_GLOBAL_OBSERVATIONS_FOR_POINT:
            telemetry.event("calib_search_few_obs")
            return None
        # de-duplicated (point, frame) solver input
        observations = [obs for _, obs in obs_list]
        last = None
        for _ in range(const.CALIB_SEARCH_BA_CAP):
            ok = self.map.g2o_optimizer._optimize_with_g2o(
                self.map.keyframes, self.map.points, observations,
                fix_initial_poses=True, fix_points=False)
            if not ok:
                telemetry.event("calib_search_ba_failed")
                return None
            est = self.estimate
            cur = self._cost(est.f, est.k1, obs_list)
            if last is not None and \
                    abs(last - cur) <= const.CALIB_SEARCH_J_TOL * max(cur, 1e-12):
                return cur
            last = cur
        return last

    def _evaluate_candidate(self, f_cand, k1_cand, incumbent, obs_list):
        """One basin candidate: restore the incumbent state, propagate
        the candidate intrinsics, solve into the basin, return its
        converged cost (or None if the solve failed)."""
        self._restore(incumbent)
        self._propagate(f_cand, k1_cand)
        return self._solve_into_basin(obs_list)

    def _run_search(self):
        """The full coarse-to-fine basin search over (f, and k1 when
        the current k1 is nonzero), landing the map in the winning
        basin."""
        est = self.estimate
        f_init = est.f
        k1_init = est.k1
        incumbent = self._snapshot()
        obs_list = self._point_side_observations()

        # --- coarse f pass ---
        best_f, best_cost, best_k1 = f_init, None, k1_init
        for scale in _COARSE_F_GRID:
            f_c = f_init * scale
            cost = self._evaluate_candidate(f_c, k1_init, incumbent,
                                             obs_list)
            info_log(LOG_TAG, f"  candidate f={f_c:.1f}: cost={cost}")
            if cost is not None and (best_cost is None or cost < best_cost):
                best_f, best_cost = f_c, cost
        if best_cost is None:
            # No candidate produced a usable solve: keep everything.
            self._restore(incumbent)
            self._propagate(f_init, k1_init)
            warning_log(LOG_TAG, "basin search found no usable candidate")
            telemetry.event("calib_search_failed")
            return

        # --- fine f pass around the coarse winner ---
        for scale in _REFINE_F_GRID:
            f_c = best_f * scale
            cost = self._evaluate_candidate(f_c, k1_init, incumbent,
                                            obs_list)
            if cost is not None and cost < best_cost:
                best_f, best_cost = f_c, cost

        # --- second fine pass (1.5% steps): the coarse+fine grids
        #     resolve f to ~2% of the INIT; this brings the winner to
        #     ~0.5% (the recovery pin's tolerance) ---
        for scale in _REFINE2_F_GRID:
            f_c = best_f * scale
            cost = self._evaluate_candidate(f_c, k1_init, incumbent,
                                            obs_list)
            if cost is not None and cost < best_cost:
                best_f, best_cost = f_c, cost

        # --- k1 pass (only when the current k1 is nonzero; pinhole
        #     runs keep k1 pinned at exactly 0 by construction) ---
        if k1_init != 0.0:
            for k1_c in _K1_ADDITIVE_GRID:
                k1_cand = k1_init + k1_c
                cost = self._evaluate_candidate(best_f, k1_cand,
                                                 incumbent, obs_list)
                if cost is not None and cost < best_cost:
                    best_k1, best_cost = k1_cand, cost

        # --- land the winning basin from the incumbent state ---
        self._restore(incumbent)
        self._propagate(best_f, best_k1)
        final_cost = self._solve_into_basin(obs_list) or best_cost

        telemetry.event("calib_search_done")

        info_log(LOG_TAG,
                 f"basin search: f {f_init:.1f} -> {est.f:.1f}, "
                 f"k1 {k1_init:.5f} -> {est.k1:.5f} "
                 f"(cost {final_cost:.4f}, {len(obs_list)} obs)")

    # -- propagation -------------------------------------------------------

    def _propagate(self, f_new, k1_new):
        """Land new intrinsics: update the estimate, every keyframe's
        K/k1 + the two normalized-coordinate caches (kp_pts_norm,
        Observation.pt_2d_norm), and drop the persistent local g2o
        graph (its edge MEASUREMENTS are normalized coords baked at
        edge creation - stale under new intrinsics; the next local BA
        call rebuilds from the map). The g2o graphs hold no intrinsics
        of their own (the 2j survey), so nothing else needs
        invalidation."""
        est = self.estimate
        est.fx = est.fy = f_new
        est.k1 = k1_new
        K_new = est.K

        for kf in self.map.keyframes:
            kf.K = K_new
            kf.k1 = k1_new
            kf.__dict__.pop("Kinv", None)
            if kf.keypoints is not None and len(kf.keypoints) > 0:
                if k1_new != 0.0:
                    kf.kp_pts_norm = undistort_points_norm(
                        kf.kp_pts, K_new, k1_new)
                else:
                    kf.kp_pts_norm = normalize(kf.kp_pts, kf.Kinv)
                for obs in kf.observed_points.values():
                    obs.__dict__.pop("pt_2d_norm", None)

        self.map.g2o_optimizer.reset_persistent_graph()
        telemetry.event("calib_propagated")
