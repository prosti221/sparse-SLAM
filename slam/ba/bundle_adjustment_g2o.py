import g2o
import numpy as np
import time as _time
from slam.core.frame import Frame
from slam.core.point import Point
from slam.core.observation import Observation
from slam_native import FoldInputs, error_fold, BASolveInputs, solve_se3_ba
from typing import List, Tuple
from slam.utils.logger import *
from slam.utils.constants import *
from slam.utils import constants as const
from slam.utils import telemetry
import cv2 as cv

LOG_TAG = 'g2oBA'

# ---- BA flags: beside their machinery, module-level INT/FLOAT
# constants, read at CALL time so probes can pin them in memory for
# A/B. ----

# Scale-invariant pose-graph odometry information: every odometry
# edge currently carries LOOP_INFO_ODOMETRY (10.0), so the AGGREGATE
# odometry stiffness grows with map size (~100-200 on a young map vs
# ~2500 on a 250-KF map), which either lets the loop edge dominate
# 1:1 (lossy snaps on small maps) or drowns it entirely (drift 14x
# with the graph a no-op). Normalization keeps the AGGREGATE
# constant: per-edge = AGGREGATE / n_odometry_edges.
# MEASURED: the normalized graph alone FAILED its adoption contract
# (two sequences degraded final-KF-ATE by ~+0.11 vs control), and
# sweeping to rescue a rejected mechanism violates the no-rescue
# clause. Ships OFF; the IMPROVEMENT.md record holds the numbers.
LOOP_ODOMETRY_NORMALIZED_ENABLED = 0
LOOP_INFO_ODOMETRY_AGGREGATE = 500.0

# The BA acceptance gate on the correction path (opt-in per call
# site via gated=True; the Map's PERIODIC global BA stays ungated -
# the default is the legacy bit-transparent path). The gate: a
# solve is written back only if it improves the objective it
# minimizes (total squared NORMALIZED reprojection error over the
# map's observations), measured on the re-anchored pre-solve state
# vs the post-solve state; a rejection restores the pre-solve state
# and counts the solve failed (telemetry loop_global_ba_rejected).
# MEASURED: the objective criterion is INSUFFICIENT - 68 gated
# decisions, 67 accepted / 1 rejected, while 12 accepted solves
# degraded GT-aligned ATE by >0.02 (worst +0.129) and the single
# rejection did not save its run (+0.127 final). A solve that fits
# the map's own biased structure better is not a truer solve.
# Ships OFF; the machinery + pins remain for a redesigned criterion
# (the IMPROVEMENT.md record holds the numbers).
LOOP_BA_GATE_ENABLED = 0


# The BA stage's internal split (build vs native solve vs write-back),
# recorded into the telemetry VALUE store when the diagnostic flag is
# ON - observation-only, the [26d] measurement (the stage timer only
# sees the whole; the pip g2o package cannot host a batched API, so
# the fixable side must be identified by measurement). Flag 0 = no
# emission anywhere (byte-identical); read at CALL time.
BA_TIMING_ATTRIB_ENABLED = 0


class G2OBundleAdjustment:
    def __init__(self, map, verbose=False):
        self.verbose = verbose
        self.map = map
        # Persistent local-BA graph. Consecutive local BA calls
        # share ~90% of their problem; rebuilding it every call was ~60%
        # of BA time in per-edge pybind construction. The graph is
        # reconciled per call: new vertices are added with their
        # observation edges (the add rule needs no edge registry - every
        # involved keyframe is an observer of a local point, so a new
        # vertex's edge endpoints are guaranteed present), stale vertices
        # are removed (remove_vertex cascades their edges), and fixed
        # flags follow window membership (the gauge anchor is the oldest
        # involved keyframe, which slides). Between LOCAL solves no
        # estimate refresh is needed: local write-back leaves graph
        # and map equal. External pose-graph/global-BA write-back must
        # invalidate this cache, otherwise the next local solve starts
        # from pre-correction estimates. NOTE: this makes local BA a
        # WITHIN-BASELINES change (not bit-identical - the g2o solve is
        # vertex-id-order sensitive at ~1e-15; the IMPROVEMENT.md record
        # holds the numbers).
        # The persistent local-BA graph, NATIVE (migration C2b2b):
        # the g2o optimizer, the vertex caches, and the edge-pair
        # bookkeeping live in C++ across calls; the reconcile below is
        # the transactional driver (membership/anchor in Python, the
        # reconcile + solve + extraction native).
        from slam_native import PersistentLocalBA
        self._native_local_ba = PersistentLocalBA(self.verbose)

    def reset_persistent_graph(self):
        """Invalidate after an external state/measurement change.

        Pose-graph and fresh/global BA invalidate BEFORE write-back:
        the next local solve must rebuild from the corrected map,
        including after a partial write failure.

        Also called when intrinsics change mid-run (K/k1
        propagation). The persistent graph's edge
        MEASUREMENTS are normalized coordinates baked at edge creation
        (set_measurement(obs.pt_2d_norm)) - stale under new
        intrinsics - so the graph is dropped wholesale and the next
        local BA call rebuilds it from the map (the pre-2f path: a
        one-time rebuild cost per calibration round). The graph holds
        no intrinsics of its own (CameraParameters(1,0,0)), so nothing
        else needs invalidation."""
        self._native_local_ba.reset()

    def refine_pose_pnp(self, frame: Frame = None) -> Tuple[bool, int]:
        """Refine a frame's pose with PnP over its map-point observations.

        `frame=None` refines the map's current keyframe (the keyframe path
        via Map.optimize). The tracker passes the CURRENT frame: an
        earlier version read and wrote the PREVIOUS keyframe
        instead, so per-frame refinement never applied to the tracked
        trajectory - and every frame re-solved an already-converged stale
        problem (~500 wasted solves on TUM).

        Returns (success, input_count), NOT a robust inlier count. The
        iterative solve uses every supplied correspondence; no RANSAC or
        per-measurement inlier mask is computed here. input_count is zero
        if there is no frame or fewer than four observations (no solve).
        The legacy minimum/telemetry names containing "inliers" are kept
        for compatibility, but must not be used as an inlier statistic.

        Divergence guards: a solution that is non-finite,
        reprojects worse than PNP_MAX_MEDIAN_REPROJECTION_ERROR over its
        own observations, moves the camera more than
        PNP_MAX_CENTER_STEP_RATIO of the median observed depth, or rotates
        more than PNP_MAX_ROTATION_STEP_DEG from the seed is rejected - the
        frame keeps its seed pose.
        """
        if frame is None:
            frame = self.map.cur_keyframe
        if frame is None:
            telemetry.event("pnp_no_frame")
            warning_log(LOG_TAG, "No frame given for pose refinement")
            return False, 0

        points_3d = []
        points_2d = []

        for obs in frame.observed_points.values():
            points_3d.append(obs.pt_3d)
            points_2d.append(obs.pt_2d_norm)

        if len(points_3d) < 4:
            telemetry.event("pnp_few_points")
            warning_log(
                LOG_TAG, f"Not enough points for pose refinement ({len(points_3d)} provided).")
            return False, 0

        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)

        # Initial extrinsic guess: cv.solvePnP refines the world-to-camera
        # transform (X_cam = R @ X_world + t), so the guess must come from the
        # inverse of the camera-to-world pose stored on the frame. Seeding
        # with the pose directly inverts rotation/translation and can push
        # the iterative solver into a mirrored local minimum.
        seed_pose = frame.pose  # Twc
        T_cw_init = np.linalg.inv(seed_pose)
        R_init = T_cw_init[:3, :3]
        t_init = T_cw_init[:3, 3]

        rvec_init, _ = cv.Rodrigues(R_init)
        rvec_init = np.array(rvec_init, dtype=np.float32).reshape(3, 1)
        t_init_f = np.array(t_init, dtype=np.float32).reshape(3, 1)

        success, rvec, tvec = cv.solvePnP(
            points_3d,
            points_2d,
            np.eye(3),
            distCoeffs=np.zeros(5),
            rvec=rvec_init,
            tvec=t_init_f,
            useExtrinsicGuess=True,
            flags=cv.SOLVEPNP_ITERATIVE
        )
        input_count = len(points_3d)

        if not success or input_count < const.PNP_MINIMUM_INLIERS:
            telemetry.event("pnp_no_solution" if not success else "pnp_few_inliers")
            warning_log(
                LOG_TAG, f"PnP failed or not enough input correspondences: {input_count}")
            return False, input_count

        R_refined, _ = cv.Rodrigues(rvec)
        t_refined = np.asarray(tvec, dtype=np.float64).flatten()

        def _reject(reason: str):
            telemetry.event(f"pnp_guard_{reason}")
            warning_log(
                LOG_TAG, f"PnP guard rejected solution ({reason}) - keeping seed pose")
            return False, input_count

        # --- Divergence guards: reject and keep the seed pose ---
        if not (np.all(np.isfinite(R_refined)) and np.all(np.isfinite(t_refined))):
            return _reject("nonfinite")

        # Reprojection quality of the solution over its own observations
        # (median: robust to a minority of mismatched observations).
        pts64 = points_3d.astype(np.float64)
        obs2d64 = points_2d.astype(np.float64)
        X_cam = (R_refined @ pts64.T).T + t_refined
        front = X_cam[:, 2] > 0
        if int(front.sum()) < const.PNP_MINIMUM_INLIERS:
            return _reject("behind_camera")
        proj = X_cam[front, :2] / X_cam[front, 2:3]
        median_err = float(np.median(
            np.linalg.norm(proj - obs2d64[front], axis=1)))
        if median_err > const.PNP_MAX_MEDIAN_REPROJECTION_ERROR:
            return _reject("reprojection")

        # A refinement may not teleport or spin the camera: bounds relative
        # to the seed pose and the observed scene depth.
        T_cw_refined = np.eye(4)
        T_cw_refined[:3, :3] = R_refined
        T_cw_refined[:3, 3] = t_refined
        refined_pose = np.linalg.inv(T_cw_refined)  # Twc
        center_step = float(np.linalg.norm(
            refined_pose[:3, 3] - seed_pose[:3, 3]))

        X_cam_init = (R_init @ pts64.T).T + t_init
        positive = X_cam_init[:, 2] > 0
        if int(positive.sum()) == 0:
            return _reject("seed_behind_camera")
        med_depth = float(np.median(X_cam_init[positive, 2]))
        if center_step > const.PNP_MAX_CENTER_STEP_RATIO * med_depth:
            return _reject("center_step")

        cos_rot = (np.trace(R_init.T @ R_refined) - 1.0) / 2.0
        rot_step = float(np.degrees(np.arccos(np.clip(cos_rot, -1.0, 1.0))))
        if rot_step > const.PNP_MAX_ROTATION_STEP_DEG:
            return _reject("rotation_step")

        # Update frame pose
        frame.pose = refined_pose

        debug_log(LOG_TAG, f"Pose refined with {input_count} input correspondences "
                           f"(median reproj {median_err:.5f}, "
                           f"center step {center_step:.3f}, "
                           f"rot step {rot_step:.2f} deg).")
        telemetry.event("pnp_ok")

        return True, input_count

    def local_bundle_adjustment(self, reference_frame_id: int, window_size: int = 5, fix_points=True) -> bool:
        debug_log(
            LOG_TAG, f"Starting local BA with g2o around keyframe {reference_frame_id}")

        local_keyframes = self.map.get_local_keyframes(
            reference_frame_id, window_size)
        if not local_keyframes:
            telemetry.event("ba_no_local_keyframes")
            warning_log(LOG_TAG, "No local keyframes found for local BA")
            return False

        # Local points: every map point observed by a window keyframe.
        # Kept in map insertion order so the optimization is
        # deterministic. (A past experiment restricted this set to
        # points with >= 2 window observations - REJECTED:
        # fast-rotating sequences never re-observe points, so most
        # points have exactly one window observation by construction,
        # and orbit's map collapsed to 2 keyframes / surf 0%.)
        local_point_set = self.map.get_local_points(local_keyframes)
        local_points = [p for p in self.map.points if p in local_point_set]

        if self.map._point_registry is None or not local_points:
            telemetry.event("ba_insufficient_observations")
            warning_log(LOG_TAG, "Insufficient observations for BA: 0")
            return False

        # The C5a assembly: the fold + BA inputs derived from the obs
        # registry's rows in the EXACT Python orders (the local points
        # in map order, each point's rows in its observation insertion
        # order), one native call. All observations of the local points
        # INCLUDING keyframes outside the window and never-promoted
        # tracker frames - the same set the per-object list
        # comprehension built; the edges filter to keyframe frames at
        # the solve (the anchors: without the fixed out-of-window
        # keyframes the free window + points is a gauge-free subsystem).
        kf_ids = [kf.frame_id for kf in self.map.keyframes]
        local_ids = [p.point_id for p in local_points]
        rows_map = self.map._registry_rows
        local_rows = [rows_map[pid] for pid in local_ids]
        local_avg = [p.average_reprojection_error for p in local_points]
        asm = self.map._obs_registry.assemble_local_ba(
            kf_ids, [kf.frame_id for kf in local_keyframes],
            local_ids, local_rows,
            self.map._point_registry.positions_view(),
            np.array(local_avg, dtype=np.float64))
        fold_obs_frame = asm[0]

        if len(fold_obs_frame) < const.MINIMUM_LOCAL_OBSERVATIONS_FOR_POINT:
            telemetry.event("ba_insufficient_observations")
            warning_log(
                LOG_TAG,
                f"Insufficient observations for BA: {len(fold_obs_frame)}")
            return False
        else:
            debug_log(
                LOG_TAG,
                f"Using : {len(fold_obs_frame)} observations for BA optimization")

        return self._local_ba_persistent(local_points, asm,
                                         fix_points=fix_points)

    def _local_ba_persistent(self, local_points, asm, fix_points) -> bool:
        try:
            return self._reconcile_and_optimize(local_points, asm, fix_points)
        except Exception as e:
            telemetry.event("ba_exception")
            error_log(LOG_TAG, f"persistent local BA failed: {e}")
            import traceback
            error_log(LOG_TAG, traceback.format_exc())
            # The graph state may be corrupt - rebuild from scratch on
            # the next call.
            self._local_opt = None
            self._kf_vertex = {}
            self._pt_vertex = {}
            self._graph_edges_by_pt = {}
            self._graph_edges_by_kf = {}
            return False

    def _reconcile_and_optimize(self, local_points, asm, fix_points) -> bool:
        """The reconcile-and-solve driver: membership/anchor resolved
        by the C5a assembly (the obs registry's row space, the exact
        Python orders), the reconcile + solve + extraction in ONE
        native call (PersistentLocalBA - migration C2b2b), the
        write-back and the fold here. The native side preserves the
        Python reconcile's exact op order (stale cascade, point
        vertices in map order, KF vertices in map order, delta edges
        in the observations list order, the fixed-flag refresh), so
        the solve sequence is identical to the g2o-python path this
        ports.
        """
        t_attrib = _time.perf_counter() if BA_TIMING_ATTRIB_ENABLED else 0.0

        (fold_obs_frame, fold_obs_point, fold_norm, _fold_rows, fold_avg,
         fold_nobs, fold_slot, fold_frame_ids, ba_frame_ids, anchor_a,
         in_window, edge_frame_rows, edge_point_rows, edge_uv,
         positions) = asm
        anchor_row = int(anchor_a[0])
        frames = [self.map.keyframes_by_id[fid] for fid in ba_frame_ids]

        T_wc = np.stack([np.linalg.inv(kf.pose) for kf in frames])
        local_ids = [p.point_id for p in local_points]

        if BA_TIMING_ATTRIB_ENABLED:
            t_solve0 = _time.perf_counter()

        T_wc_new, pos_new, iterations, actual_edges = \
            self._native_local_ba.solve_local(
                frame_ids=ba_frame_ids, point_ids=local_ids,
                T_wc=T_wc, positions=positions, in_window=in_window,
                anchor_row=anchor_row,
                points_fixed=1 if fix_points else 0,
                obs_frame_row=edge_frame_rows,
                obs_point_row=edge_point_rows,
                obs_uv=edge_uv, max_iterations=30)
        telemetry.event("ba_iterations", int(iterations))

        # Safety net: every KEYFRAME observation must now be an edge.
        if len(edge_frame_rows) != actual_edges:
            telemetry.event("ba_edge_mismatch",
                            len(edge_frame_rows) - actual_edges)

        if BA_TIMING_ATTRIB_ENABLED:
            t_write0 = _time.perf_counter()

        # --- Write back (same semantics as the fresh build) ---
        for i, kf in enumerate(frames):
            kf.pose = np.linalg.inv(T_wc_new[i])

        if not fix_points:
            for j, p in enumerate(local_points):
                p.pt_3d = np.array(pos_new[j])
            # The C3b registry hook: the solved positions go native in
            # one bulk update (the registry is the geometry cache).
            reg = self.map._point_registry
            if reg is not None:
                reg.update_positions(
                    [p.point_id for p in local_points], pos_new)

        self._recompute_point_errors(local_points, assembly=asm)
        if BA_TIMING_ATTRIB_ENABLED:
            t_end = _time.perf_counter()
            telemetry.value("ba_build_s", t_solve0 - t_attrib)
            telemetry.value("ba_solve_s", t_write0 - t_solve0)
            telemetry.value("ba_writeback_s", t_end - t_write0)
        return True

    def global_bundle_adjustment(self, gated: bool = False) -> bool:
        """Global BA. `gated=True` (the loop-correction call site)
        enables the acceptance gate; the default False is the legacy
        bit-transparent path (the Map's PERIODIC global BA and every
        other existing caller stay exactly as they were - the gate
        is scoped to the correction path)."""
        debug_log(LOG_TAG, "Starting global BA with g2o")
        keyframes = self.map.keyframes
        points = self.map.points
        observations = self.map.get_observations(keyframes)

        if len(observations) < const.MINIMUM_GLOBAL_OBSERVATIONS_FOR_POINT:
            telemetry.event("ba_global_insufficient_observations")
            warning_log(
                LOG_TAG, f"Insufficient observations for global BA: {len(observations)}")
            return False

        if not (gated and LOOP_BA_GATE_ENABLED):
            return self._optimize_with_g2o(keyframes, points, observations, fix_initial_poses=True, fix_points=False)

        # ---- the acceptance gate ----
        # Snapshot the pre-solve state. On the correction path
        # this is the RE-ANCHORED state (graph-corrected poses +
        # points moved with them): CONSISTENT — so a rejected BA
        # leaves a coherent map, which is exactly what makes the
        # write-back rejectable (an earlier standalone re-anchor
        # failed its A/B partly because a rejection would have
        # left the moved-poses/stale-points mess).
        pre_poses = {kf.frame_id: kf.pose.copy() for kf in keyframes}
        pre_pts = [np.asarray(p.pt_3d).copy() for p in points]
        err_pre = self._total_reprojection_sq_error(observations)

        ok = self._optimize_with_g2o(keyframes, points, observations, fix_initial_poses=True, fix_points=False)
        if not ok:
            return False

        err_post = self._total_reprojection_sq_error(observations)
        if err_post >= err_pre:
            # The solve did not improve the objective it minimizes:
            # a diverged rearrangement, not a refinement. Restore
            # the pre-solve (re-anchored, consistent) state; the
            # per-point error stats were recomputed inside the solve
            # on the REJECTED estimates, so recompute them on the
            # restored state too. The persistent graph was already
            # invalidated inside the solve; it rebuilds from the
            # restored state.
            for kf in keyframes:
                kf.pose = pre_poses[kf.frame_id]
            for p, x in zip(points, pre_pts):
                p.pt_3d = x
            self.map.rebuild_point_coords()
            self._recompute_point_errors(observations)
            telemetry.event("loop_global_ba_rejected")
            warning_log(
                LOG_TAG,
                f"Global BA REJECTED by the acceptance gate "
                f"(objective {err_pre:.3f} -> {err_post:.3f}, not an "
                f"improvement) - keeping the re-anchored pre-solve state")
            return False
        telemetry.event("loop_global_ba_accepted")
        debug_log(
            LOG_TAG, f"Global BA accepted by the gate "
            f"(objective {err_pre:.3f} -> {err_post:.3f})")
        return True

    def _total_reprojection_sq_error(self, observations) -> float:
        """The g2o global-BA objective — total
        squared NORMALIZED reprojection error over the observations
        (the EdgeProjectXYZ2UV residual the solver actually
        minimizes: projection through inv(pose) with the unit
        camera, measured against pt_2d_norm). The scale-free
        before/after acceptance quantity; mirrors the solver's own
        residual including its behind-camera behavior (no
        cheirality special case — the same large error both
        sides). No RNG; corrections are rare so the per-observation
        loop is a one-time cost per correction event."""
        total = 0.0
        for obs in observations:
            T_cw = np.linalg.inv(obs.frame.pose)
            Xc = T_cw[:3, :3] @ obs.point.pt_3d + T_cw[:3, 3]
            z = Xc[2] if abs(Xc[2]) > 1e-12 else 1e-12
            du = Xc[0] / z - obs.pt_2d_norm[0]
            dv = Xc[1] / z - obs.pt_2d_norm[1]
            total += du * du + dv * dv
        return float(total)

    def _odometry_information(self, frame_to_vertex) -> np.ndarray:
        """The odometry edges' per-edge information
        matrix. Legacy: LOOP_INFO_ODOMETRY per edge, so AGGREGATE
        odometry stiffness grows with map size — ~100-200 on a young
        map (the loop edge competes 1:1 -> the measured lossy snap:
        fr1 e1 desktop +72% KF-ATE applying GOOD constraints) vs
        ~2500 on a 250-KF map (the loop edge drowns -> post-graph ==
        pre to four decimals while the chain drifts 14x; fr3, both
        hosts). Normalized: the AGGREGATE spread over the edge count,
        scale-invariant vs map size — one mechanism fixing both faces.
        500.0 lands the measured mild zone (mh_01's
        healthy repairs ran at aggregate odo:loop ~3-6:1). The flag is
        read at call time (probes pin it in memory for A/B)."""
        if LOOP_ODOMETRY_NORMALIZED_ENABLED:
            vertex_ids = set(frame_to_vertex)
            n_odo = sum(
                1 for kf1, kf2 in zip(self.map.keyframes,
                                      self.map.keyframes[1:])
                if kf1.frame_id in vertex_ids
                and kf2.frame_id in vertex_ids) or 1
            return np.eye(6) * (LOOP_INFO_ODOMETRY_AGGREGATE / n_odo)
        return np.eye(6) * const.LOOP_INFO_ODOMETRY

    def pose_graph_optimization(self, loop_constraints: List[Tuple[int, int, np.ndarray]]) -> bool:
        debug_log(
            LOG_TAG, f"Starting pose graph optimization with {len(loop_constraints)} loop constraints")

        if len(loop_constraints) == 0:
            debug_log(LOG_TAG, "No loop constraints to optimize")
            return True

        try:
            # Create optimizer with standard setup (but without camera params for pose graph)
            opt = g2o.SparseOptimizer()
            solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
            solver = g2o.OptimizationAlgorithmLevenberg(solver)
            opt.set_algorithm(solver)

            # Create pose graph vertices
            frame_to_vertex = self._create_keyframe_vertices(
                opt, self.map.keyframes, vertex_id_offset=0, fix_first=True)

            odometry_information = self._odometry_information(frame_to_vertex)

            # Add odometry edges (sequential constraints between
            # keyframes - the tautological-by-design CHAIN PRIOR:
            # computed from the current estimated poses, so they are
            # exact by construction and carry no corrective
            # information; the loop edge is the graph's only source of
            # correction. The comments used to invert the
            # Frame convention - frame.pose IS camera-to-world (Twc),
            # and the edge measurement form is M(a,b) =
            # pose_a @ inv(pose_b), matching the g2o ideal
            # inv(V_a) @ V_b with V = the vertex estimate.)
            odometry_edge_count = 0
            for i in range(len(self.map.keyframes) - 1):
                kf1 = self.map.keyframes[i]
                kf2 = self.map.keyframes[i + 1]

                if kf1.frame_id not in frame_to_vertex or kf2.frame_id not in frame_to_vertex:
                    continue

                # The relative pose kf1 -> kf2 in the graph's
                # measurement form: pose_1 @ inv(pose_2).
                relative_pose = kf1.pose @ np.linalg.inv(kf2.pose)
                R_rel = relative_pose[:3, :3]
                t_rel = relative_pose[:3, 3]
                se3_rel = g2o.SE3Quat(R_rel, t_rel)

                # Create odometry edge
                edge = g2o.EdgeSE3Expmap()
                edge.set_vertex(0, frame_to_vertex[kf1.frame_id])
                edge.set_vertex(1, frame_to_vertex[kf2.frame_id])
                edge.set_measurement(se3_rel)
                # The (possibly normalized) per-edge odometry
                # information computed above.
                edge.set_information(odometry_information)

                # Add robust kernel to handle odometry outliers gracefully
                edge.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.991)))
                opt.add_edge(edge)
                odometry_edge_count += 1

            # Add loop closure edges
            loop_edge_count = 0
            for from_id, to_id, relative_pose in loop_constraints:
                if from_id not in frame_to_vertex or to_id not in frame_to_vertex:
                    warning_log(
                        LOG_TAG, f"Loop constraint keyframes not found in vertex map: {from_id} -> {to_id}")
                    continue

                R_rel = relative_pose[:3, :3]
                t_rel = relative_pose[:3, 3]
                se3_rel = g2o.SE3Quat(R_rel, t_rel)

                # Create loop closure edge
                edge = g2o.EdgeSE3Expmap()
                edge.set_vertex(0, frame_to_vertex[from_id])
                edge.set_vertex(1, frame_to_vertex[to_id])
                edge.set_measurement(se3_rel)
                # Loop-edge information: in the g2o SE3Expmap error
                # order (rotation, translation), rotation information
                # > translation (a PnP constraint's rotation is
                # typically the more reliable part). The
                # values moved to the LOOP_INFO_* constants (an older
                # comment claimed "rotation should be
                # flexible" while setting rotation information HIGHER
                # - contradictory; the VALUES are unchanged).
                edge.set_information(np.diag(
                    [const.LOOP_INFO_ROTATION] * 3
                    + [const.LOOP_INFO_TRANSLATION] * 3))

                # Add robust kernel to handle outliers gracefully
                edge.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.991)))
                opt.add_edge(edge)
                loop_edge_count += 1

            debug_log(
                LOG_TAG, f"Added {odometry_edge_count} odometry edges and {loop_edge_count} loop closure edges")

            if loop_edge_count == 0:
                warning_log(LOG_TAG, "No valid loop closure edges added")
                return False

            # Optimize with more iterations for pose graph
            opt.initialize_optimization()
            opt.optimize(50)

            # The local graph still holds pre-correction estimates.
            # Invalidate BEFORE any map write, including a partial
            # failure; later global BA may fail or be skipped.
            self.reset_persistent_graph()

            # Update keyframe poses using shared helper
            self._update_keyframe_poses_from_vertices(frame_to_vertex)

            debug_log(
                LOG_TAG, f"Pose graph optimization completed successfully with {loop_edge_count} loop constraints")
            return True

        except Exception as e:
            error_log(LOG_TAG, f"Pose graph optimization failed: {e}")
            import traceback
            error_log(LOG_TAG, traceback.format_exc())
            return False

    def _optimize_with_g2o(self, keyframes: List[Frame], points: List[Point],
                           observations: List[Observation],
                           fix_initial_poses: bool = False,
                           fix_points: bool = False,
                           fixed_keyframe_ids: set = None) -> bool:
        """The global BA solve: the SOA assembly here, the build +
        Levenberg + extraction in ONE native call (`solve_se3_ba` - the
        C2b1-verified fresh-build kernel, bit-identical to the
        g2o-python path on identical inputs), the write-back and the
        fold here. The vertex/edge creation order and the fixed-flag
        semantics are the g2o-python path's exactly (KFs in list order
        with frame 0 fixed per fix_initial_poses plus any explicitly
        anchored ids, points after them, edges in the observations
        order skipping unregistered pairs).
        """
        try:
            frame_row = {kf.frame_id: i for i, kf in enumerate(keyframes)}
            point_row = {p.point_id: i for i, p in enumerate(points)}
            edge_obs = [obs for obs in observations
                        if obs.frame.frame_id in frame_row
                        and obs.point.point_id in point_row]
            if len(edge_obs) < 10:
                telemetry.event("ba_too_few_edges")
                warning_log(LOG_TAG, f"Too few edges: {len(edge_obs)}")
                return False

            fixed = np.zeros(len(keyframes), dtype=np.uint8)
            for i, kf in enumerate(keyframes):
                if ((i == 0 and fix_initial_poses) or
                        (fixed_keyframe_ids is not None and
                         kf.frame_id in fixed_keyframe_ids)):
                    fixed[i] = 1
            T_wc = np.stack([np.linalg.inv(kf.pose) for kf in keyframes])
            positions = np.array([p.pt_3d for p in points], dtype=np.float64)
            obs_frame_row = np.fromiter(
                (frame_row[obs.frame.frame_id] for obs in edge_obs),
                dtype=np.int64, count=len(edge_obs))
            obs_point_row = np.fromiter(
                (point_row[obs.point.point_id] for obs in edge_obs),
                dtype=np.int64, count=len(edge_obs))
            obs_uv = np.array([obs.pt_2d_norm for obs in edge_obs],
                              dtype=np.float64)

            T_wc_new, pos_new, iterations, _chi2 = solve_se3_ba(BASolveInputs(
                T_wc=T_wc, fixed=fixed, positions=positions,
                points_fixed=1 if fix_points else 0,
                obs_frame=obs_frame_row.tolist(),
                obs_point=obs_point_row.tolist(),
                obs_uv=obs_uv, max_iterations=30))
            telemetry.event("ba_iterations", int(iterations))
            # The cost audit's fresh-BA sampler reads the g2o objects
            # created by _create_keyframe_vertices; the native solve
            # has none - the same row shape, from this solve's inputs.
            self._last_global_graph = {
                "vertices": len(keyframes) + len(points),
                "pose_vertices": len(keyframes),
                "point_vertices": len(points),
                "fixed_pose_vertices": int(fixed.sum()),
                "edges": len(edge_obs),
            }

            # This fresh solve rewrites state outside the persistent
            # local graph. Drop that cache before write-back so the
            # next local solve cannot restore pre-correction estimates.
            self.reset_persistent_graph()

            for i, kf in enumerate(keyframes):
                kf.pose = np.linalg.inv(T_wc_new[i])

            if not fix_points:
                for j, point in enumerate(points):
                    point.pt_3d = np.array(pos_new[j])
                # The C3b registry hook (the bulk geometry update).
                reg = self.map._point_registry
                if reg is not None:
                    reg.update_positions(
                        [p.point_id for p in points], pos_new)

            self._recompute_point_errors(observations)
            return True

        except Exception as e:
            telemetry.event("ba_exception")
            error_log(LOG_TAG, f"g2o optimization failed: {e}")
            import traceback
            error_log(LOG_TAG, traceback.format_exc())
            return False

    def _create_g2o_optimizer(self):
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)

        terminate = g2o.SparseOptimizerTerminateAction()
        terminate.set_gain_threshold(const.BA_TERMINATE_GAIN_THRESHOLD)
        opt.set_algorithm(solver)
        # Early exit at convergence: this action was previously
        # created but never REGISTERED, so every BA ran the full
        # 30-iteration budget even once the relative chi gain per
        # iteration had fallen below the threshold. Local BA re-solves
        # largely converged maps on every keyframe insertion (the same
        # window points re-optimized every 2-3 frames), so most of that
        # budget was spent proving convergence, not improving the map.
        opt.add_post_iteration_action(terminate)

        # Add normalized camera parameters
        cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)
        cam.set_id(0)
        opt.add_parameter(cam)

        return opt

    def _create_keyframe_vertices(self, optimizer, keyframes: List[Frame],
                                  vertex_id_offset: int = 0, fix_first: bool = True,
                                  fixed_keyframe_ids: set = None):
        frame_to_vertex = {}

        for i, kf in enumerate(keyframes):
            # Use camera-to-world transformation directly
            pose = np.linalg.inv(kf.pose)
            R = pose[:3, :3]
            t = pose[:3, 3]
            se3 = g2o.SE3Quat(R, t)

            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(vertex_id_offset + i)
            v_se3.set_estimate(se3)
            # First keyframe is fixed when requested (gauge origin), plus any
            # explicitly anchored keyframes (e.g. local-BA window boundary).
            is_fixed = (i == 0 and fix_first) or (
                fixed_keyframe_ids is not None and kf.frame_id in fixed_keyframe_ids)
            v_se3.set_fixed(is_fixed)

            optimizer.add_vertex(v_se3)
            frame_to_vertex[kf.frame_id] = v_se3

        return frame_to_vertex

    def _update_keyframe_poses_from_vertices(self, frame_to_vertex):
        for kf_id, vertex in frame_to_vertex.items():
            # Find the keyframe by ID
            kf = self.map.get_keyframe_by_id(kf_id)
            if kf is None:
                continue

            est = vertex.estimate()
            R = est.rotation().matrix()
            t = est.translation()

            # Create new pose matrix (camera-to-world)
            new_pose = np.eye(4)
            new_pose[:3, :3] = R
            new_pose[:3, 3] = t

            # Update frame pose (world-to-camera)
            kf.pose = np.linalg.inv(new_pose)

    def _recompute_point_errors(
        self,
        observations,
        assembly=None,
    ) -> None:
        """Recompute every observation's reprojection error and fold it
        into each point's running average (the NATIVE error-fold
        kernel, migration C2a).

        The kernel is a pure function over the flat SOA this method
        assembles; the Python oracle lives in the stage tests only
        (`_reference_fold`). The kernel's op order mirrors the oracle:
        per-frame first-appearance grouping, a per-row 4-term
        index-order projection dot, the behind-camera/isfinite skips,
        the elementwise norm, and pass 2 applying errors in the
        ORIGINAL observations order with the running average (n = the
        point's GLOBAL observation count).

        UPDATE ORDER IS LOAD-BEARING and preserved by that pass-2
        order. The flag flush flags only points whose fold actually
        updated - the same idempotent set the interleaved flags
        produced (the flag is lazily consumed: Frame.tracking_quality
        recomputes on next read).

        Two input modes: the GLOBAL path passes the Observation list
        and the flat SOA is extracted per object here; the LOCAL path
        (migration C5a) passes the local Point objects plus the
        assembly tuple - the SOA comes from the obs registry's row
        space in the same orders, zero per-observation Python.
        """
        if assembly is not None:
            (fold_obs_frame, fold_obs_point, fold_norm, fold_rows, fold_avg,
             fold_nobs, fold_slot, fold_frame_ids, _ba_frame_ids, _anchor_a,
             _in_window, _e_fr, _e_pr, _e_uv, _positions) = assembly
            points = [observations[i] for i in fold_slot]
            poses_inv = np.stack(
                [self.map.frames_by_id[fid].pose_inv
                 for fid in fold_frame_ids])
            # The solve moved the geometry after the assembly ran: the
            # fold's positions must be the POST-write-back values, so
            # they re-gather from the live registry block here (the
            # write-back's bulk update put the solved positions in it).
            fold_pos = np.asarray(
                self.map._point_registry.positions_view())[fold_rows]
            err, new_avg, updated = error_fold(FoldInputs(
                poses_inv=poses_inv, obs_frame=fold_obs_frame,
                point_pos=fold_pos, obs_norm=fold_norm,
                obs_point=fold_obs_point, point_avg=fold_avg,
                point_nobs=fold_nobs))
            self._apply_fold(points, err, new_avg, updated)
            return

        O = len(observations)
        if O == 0:
            return

        # Flat extraction: unique frames/points in FIRST-APPEARANCE
        # order (dict preserves insertion), obs -> row mappings.
        frames = list(dict.fromkeys(ob.frame for ob in observations))
        points = list(dict.fromkeys(ob.point for ob in observations))
        frame_row = {f: i for i, f in enumerate(frames)}
        point_row = {p: i for i, p in enumerate(points)}

        obs_frame = np.fromiter(
            (frame_row[ob.frame] for ob in observations),
            dtype=np.int64, count=O)
        obs_point = np.fromiter(
            (point_row[ob.point] for ob in observations),
            dtype=np.int64, count=O)
        obs_norm = np.array([ob.pt_2d_norm for ob in observations],
                            dtype=np.float64)
        point_pos = np.array([p.pt_3d for p in points], dtype=np.float64)
        poses_inv = np.stack([f.pose_inv for f in frames])
        point_avg = np.array([p.average_reprojection_error for p in points],
                             dtype=np.float64)
        point_nobs = np.fromiter((len(p.observations) for p in points),
                                 dtype=np.int64, count=len(points))

        fold_in = FoldInputs(poses_inv=poses_inv, obs_frame=obs_frame,
                             point_pos=point_pos, obs_norm=obs_norm,
                             obs_point=obs_point, point_avg=point_avg,
                             point_nobs=point_nobs)
        err, new_avg, updated = error_fold(fold_in)

        self._apply_fold(points, err, new_avg, updated)

    def _apply_fold(self, points, err, new_avg, updated) -> None:
        """The fold's write-back: the running average per updated point
        (the idempotent flag flush - flags only points whose fold
        updated)."""
        for i, pt in enumerate(points):
            if updated[i]:
                pt.average_reprojection_error = new_avg[i]
                pt._update_observing_keyframe_quality()