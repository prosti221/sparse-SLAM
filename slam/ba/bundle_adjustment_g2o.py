import g2o
import numpy as np
import time as _time
from collections import defaultdict
from slam.core.frame import Frame
from slam.core.point import Point
from slam.core.observation import Observation
from typing import List, Tuple
from slam.utils.logger import *
from slam.utils.constants import *
from slam.utils import constants as const
from slam.utils import telemetry
from typing import List, Tuple
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
        self._local_opt = None
        self._kf_vertex = {}   # frame_id -> VertexSE3Expmap
        self._pt_vertex = {}   # point_id -> VertexPointXYZ
        # Incremental edge bookkeeping: the reconcile
        # rebuilt the existing-pair set from opt.edges() on EVERY call -
        # two pybind vertex() calls per edge (821k calls across the 39
        # BA calls of a translation run, ~21% of its BA time). These
        # two views of the graph's (point, keyframe) edge pairs are
        # maintained at edge creation and stale-vertex removal instead;
        # pair membership is O(1) and no per-call rebuild happens.
        self._graph_edges_by_pt = {}   # point_id -> set(frame_id)
        self._graph_edges_by_kf = {}   # frame_id -> set(point_id)
        self._local_edge_info = None
        self._local_robust_kernel = None
        self._local_vertex_id = 0

    def _next_vertex_id(self) -> int:
        vid = self._local_vertex_id
        self._local_vertex_id += 1
        return vid

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
        self._local_opt = None
        self._kf_vertex = {}
        self._pt_vertex = {}
        self._graph_edges_by_pt = {}
        self._graph_edges_by_kf = {}
        self._local_edge_info = None
        self._local_robust_kernel = None
        self._local_vertex_id = 0

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

        # All observations of the local points, INCLUDING observations from
        # keyframes outside the window. Those extra keyframes are added as
        # fixed vertices and anchor the optimization: without them, the free
        # window poses + points form a gauge-free subsystem that can drift
        # rigidly with no reprojection penalty. (The list can also contain
        # NON-keyframe tracker observations - frames never promoted; both
        # the fresh-build path and the persistent graph skip them at edge
        # construction.)
        observations = [obs for p in local_points
                        for obs in p.observations.values()]

        if len(observations) < const.MINIMUM_LOCAL_OBSERVATIONS_FOR_POINT:
            telemetry.event("ba_insufficient_observations")
            warning_log(
                LOG_TAG, f"Insufficient observations for BA: {len(observations)}")
            return False
        else:
            debug_log(
                LOG_TAG, f"Using : {len(observations)} observations for BA optimization")

        return self._local_ba_persistent(
            local_keyframes, local_points, observations,
            fix_points=fix_points)

    def _ensure_local_optimizer(self):
        """Lazily build the persistent optimizer (solver, terminate
        action, camera parameter, shared kernel / information)."""
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)

        terminate = g2o.SparseOptimizerTerminateAction()
        terminate.set_gain_threshold(const.BA_TERMINATE_GAIN_THRESHOLD)
        opt.set_algorithm(solver)
        opt.add_post_iteration_action(terminate)

        cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)
        cam.set_id(0)
        opt.add_parameter(cam)

        if self.verbose:
            opt.set_verbose(True)

        self._local_opt = opt
        self._local_robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
        self._local_edge_info = np.eye(2)
        self._kf_vertex = {}
        self._pt_vertex = {}
        self._graph_edges_by_pt = {}
        self._graph_edges_by_kf = {}
        return opt

    def _add_point_edge(self, obs):
        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_parameter_id(0, 0)
        edge.set_vertex(0, self._pt_vertex[obs.point.point_id])
        edge.set_vertex(1, self._kf_vertex[obs.frame.frame_id])
        edge.set_information(self._local_edge_info)
        edge.set_measurement(obs.pt_2d_norm)
        edge.set_robust_kernel(self._local_robust_kernel)
        # Record the pair only if the edge actually entered the graph
        # (add_edge returns False on failure): the bookkeeping must
        # mirror the graph exactly, and a failed edge must stay
        # "missing" so the next reconcile's delta pass re-creates it -
        # the same self-healing the 2f per-call rebuild provided.
        if self._local_opt.add_edge(edge):
            pid = obs.point.point_id
            fid = obs.frame.frame_id
            self._graph_edges_by_pt.setdefault(pid, set()).add(fid)
            self._graph_edges_by_kf.setdefault(fid, set()).add(pid)
        return edge

    def _local_ba_persistent(self, local_keyframes, local_points,
                             observations, fix_points) -> bool:
        try:
            return self._reconcile_and_optimize(
                local_keyframes, local_points, observations, fix_points)
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

    def _reconcile_and_optimize(self, local_keyframes, local_points,
                                 observations, fix_points) -> bool:
        t_attrib = _time.perf_counter() if BA_TIMING_ATTRIB_ENABLED else 0.0
        if self._local_opt is None:
            self._ensure_local_optimizer()
        opt = self._local_opt

        # --- Compute the target membership ---
        target_kf_ids = {obs.frame.frame_id for obs in observations}
        target_pt_ids = {p.point_id for p in local_points}
        local_kf_ids = {kf.frame_id for kf in local_keyframes}

        # Gauge anchor semantics preserved from the fresh build: the
        # OLDEST involved keyframe is fixed (involved_keyframes[0] in
        # _create_keyframe_vertices), plus every keyframe outside the
        # window.
        anchor_kf_id = None
        for kf in self.map.keyframes:
            if kf.frame_id in target_kf_ids:
                anchor_kf_id = kf.frame_id
                break

        # --- Remove stale vertices (cascade removes their edges) ---
        for frame_id in [fid for fid in self._kf_vertex
                         if fid not in target_kf_ids]:
            opt.remove_vertex(self._kf_vertex.pop(frame_id))
            # Maintain the edge bookkeeping: the cascade removed
            # exactly this keyframe's pairs.
            for pid in self._graph_edges_by_kf.pop(frame_id, ()):
                pt_pairs = self._graph_edges_by_pt.get(pid)
                if pt_pairs is not None:
                    pt_pairs.discard(frame_id)
                    if not pt_pairs:
                        del self._graph_edges_by_pt[pid]
        for point_id in [pid for pid in self._pt_vertex
                         if pid not in target_pt_ids]:
            opt.remove_vertex(self._pt_vertex.pop(point_id))
            for fid in self._graph_edges_by_pt.pop(point_id, ()):
                kf_pairs = self._graph_edges_by_kf.get(fid)
                if kf_pairs is not None:
                    kf_pairs.discard(point_id)
                    if not kf_pairs:
                        del self._graph_edges_by_kf[fid]

        # --- Add new point vertices (in map order). Edges are created in
        # a single delta pass below, from the observations list - the
        # POINT-side registry, exactly the constraint set the fresh
        # build optimizes. (The frame registry is NOT a valid edge
        # source: _filter_duplicate_keypoints registers observations
        # frame-side only, and overwrites frame-side entries with
        # objects that differ from the point-side ones - both variants
        # were measured to break the fresh-build parity.) ---
        for p in local_points:
            if p.point_id not in self._pt_vertex:
                v = g2o.VertexPointXYZ()
                v.set_id(self._next_vertex_id())
                v.set_estimate(p.pt_3d)
                v.set_fixed(fix_points)
                v.set_marginalized(True)
                opt.add_vertex(v)
                self._pt_vertex[p.point_id] = v

        # --- Add new keyframe vertices (in map order) ---
        for kf in self.map.keyframes:
            if kf.frame_id in target_kf_ids and \
                    kf.frame_id not in self._kf_vertex:
                T_cw = np.linalg.inv(kf.pose)
                v = g2o.VertexSE3Expmap()
                v.set_id(self._next_vertex_id())
                v.set_estimate(g2o.SE3Quat(T_cw[:3, :3], T_cw[:3, 3]))
                opt.add_vertex(v)
                self._kf_vertex[kf.frame_id] = v

        # --- Delta edges: one pass over the observations list, in the
        # same order the fresh build adds them, creating only pairs not
        # yet in the graph. Pair membership comes from
        # the incremental bookkeeping (_graph_edges_by_pt) instead of
        # rebuilding a set from opt.edges() per call; the iteration
        # order is unchanged, so the solve sequence is identical. (The
        # frame registry is NOT a valid edge source:
        # _filter_duplicate_keypoints registers observations frame-side
        # only, and overwrites frame-side entries with objects that
        # differ from the point-side ones - both variants were
        # measured to break the fresh-build parity.) ---
        for obs in observations:
            fid = obs.frame.frame_id
            if fid not in self._kf_vertex:
                continue
            pt_pairs = self._graph_edges_by_pt.get(obs.point.point_id)
            if pt_pairs is None or fid not in pt_pairs:
                self._add_point_edge(obs)

        # Safety net: every KEYFRAME observation must now be an edge.
        # (The observations list also contains non-keyframe tracker
        # observations - frames that were never promoted - which the
        # fresh build equally skips at edge construction.)
        expected_edges = sum(
            1 for obs in observations
            if obs.frame.frame_id in self._kf_vertex)
        actual_edges = len(self._local_opt.edges())
        if actual_edges != expected_edges:
            telemetry.event("ba_edge_mismatch",
                           expected_edges - actual_edges)

        # --- Refresh fixed flags (the anchor and boundary slide) ---
        for frame_id, v in self._kf_vertex.items():
            v.set_fixed(frame_id == anchor_kf_id or
                        frame_id not in local_kf_ids)

        # --- Optimize ---
        t_solve0 = _time.perf_counter() if BA_TIMING_ATTRIB_ENABLED \
            else 0.0
        opt.initialize_optimization()
        iterations = opt.optimize(30)
        telemetry.event("ba_iterations", int(iterations))

        # --- Write back (same semantics as the fresh build) ---
        t_write0 = _time.perf_counter() if BA_TIMING_ATTRIB_ENABLED \
            else 0.0
        for frame_id, v in self._kf_vertex.items():
            kf = self.map.get_keyframe_by_id(frame_id)
            if kf is None:
                continue
            est = v.estimate()
            new_pose = np.eye(4)
            new_pose[:3, :3] = est.rotation().matrix()
            new_pose[:3, 3] = est.translation()
            kf.pose = np.linalg.inv(new_pose)

        if not fix_points:
            for p in local_points:
                v = self._pt_vertex.get(p.point_id)
                if v is not None:
                    p.pt_3d = np.array(v.estimate())

        self._recompute_point_errors(observations)
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
        try:
            # Create optimizer with standard setup
            opt = self._create_g2o_optimizer()

            # Robust kernel for outlier rejection
            robust_kernel = g2o.RobustKernelHuber(
                np.sqrt(5.991))  # sqrt(chi2 95%)

            # Create keyframe vertices
            frame_to_vertex = self._create_keyframe_vertices(
                opt, keyframes, vertex_id_offset=0, fix_first=fix_initial_poses,
                fixed_keyframe_ids=fixed_keyframe_ids)

            # Create point vertices with interleaved IDs
            point_to_vertex = {}
            for i, point in enumerate(points):
                v_point = g2o.VertexPointXYZ()
                # Start after keyframe vertices
                v_point.set_id(len(keyframes) + i)
                v_point.set_estimate(point.pt_3d)
                v_point.set_fixed(fix_points)
                v_point.set_marginalized(True)
                opt.add_vertex(v_point)
                point_to_vertex[point] = v_point

            # Add observation edges. The 2x2 information matrix
            # is constant across edges - hoisted out of the loop instead
            # of allocating np.eye(2) once per edge (~700k allocations on
            # TUM). The dead best_indices sort (top-5 lowest-error point
            # indices, consumed by a commented-out set_fixed line) is
            # gone too.
            edge_information = np.eye(2)
            edge_count = 0
            for obs in observations:
                if obs.frame.frame_id not in frame_to_vertex or obs.point not in point_to_vertex:
                    continue

                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_parameter_id(0, 0)  # camera parameter
                edge.set_vertex(0, point_to_vertex[obs.point])  # point
                # camera pose
                edge.set_vertex(1, frame_to_vertex[obs.frame.frame_id])
                edge.set_information(edge_information)
                edge.set_measurement(obs.pt_2d_norm)
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)
                edge_count += 1

            if edge_count < 10:
                telemetry.event("ba_too_few_edges")
                warning_log(LOG_TAG, f"Too few edges: {edge_count}")
                return False

            if self.verbose:
                opt.set_verbose(True)

            # Optimize; the returned value is the number of LM iterations
            # actually performed (early-exit via the terminate action).
            opt.initialize_optimization()
            iterations = opt.optimize(30)
            telemetry.event("ba_iterations", int(iterations))

            # This fresh solve rewrites state outside the persistent
            # local graph. Drop that cache before write-back so the
            # next local solve cannot restore pre-correction estimates.
            self.reset_persistent_graph()

            # Update keyframe poses
            self._update_keyframe_poses_from_vertices(frame_to_vertex)

            # Update 3D points if they were optimized
            if not fix_points:
                for point, vertex in point_to_vertex.items():
                    point.pt_3d = np.array(vertex.estimate())

            # Update reprojection errors
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
        observations: List[Observation]
    ) -> None:
        # Batch the per-frame projection. The original loop
        # re-projected every observed point individually (~1.5M single-
        # point GEMVs on TUM per run); Frame.project_points does the
        # same math as one GEMM per frame group.
        #
        # UPDATE ORDER IS LOAD-BEARING and is preserved exactly by the
        # two-pass structure: Point.update_reprojection_error keeps a
        # running average, so each point's own errors must be applied in
        # the OBSERVATIONS-LIST order. A naive by-frame grouping changes
        # that order (a later-listed point's earlier frame can be pulled
        # forward by another point's grouping) and measurably diverges
        # orbit/tum. Pass 1 only READS geometry (project + norm); pass 2
        # applies the errors in the original order.
        #
        # Both bit-identical: (a) pass 1 computes the
        # norms vectorized per frame group (the same per-row reduction
        # as the per-observation np.linalg.norm call it replaces), and
        # (b) pass 2 hoists the quality-flag storm - update_reprojection
        # _error used to flag ALL of a point's observers on EVERY fold
        # step (quadratic in observation count; ~780k idempotent
        # attribute-sets per BA call on translation). The flag is
        # idempotent and lazily consumed (Frame.tracking_quality
        # recomputes on next read), so folding with update_quality=False
        # and flagging each point's observers once - only points whose
        # fold actually updated, exactly as the interleaved flags would
        # have - leaves the observable state identical. The local-BA
        # observations list is point-consecutive (one flag pass per
        # point); the global-BA list is frame-grouped, where the
        # change-detection flags each point once per contiguous segment
        # - idempotent, so still correct.
        by_frame = defaultdict(list)
        for obs in observations:
            by_frame[obs.frame].append(obs)

        error_by_obs = {}
        for frame, group in by_frame.items():
            pts_3d = np.array([obs.pt_3d for obs in group],
                              dtype=np.float64)

            projected = frame.project_points(pts_3d, normalized=True)

            # project_points writes NaN for behind-camera points;
            # the old path skipped them (project_point -> None).
            # The finite mask is vectorized, but the norm stays a
            # per-observation 1-D call: np.linalg.norm(x, axis=1)
            # reduces via add.reduce while the 1-D norm reduces via
            # dot - measurably different at the last ULP (stage-pinned),
            # and the fold's bits are load-bearing (stage-pinned).
            finite = np.all(np.isfinite(projected), axis=1)
            if finite.any():
                # The vectorized equivalent of the per-observation
                # np.linalg.norm call: elementwise products + a
                # 2-element sum in index order - the same arithmetic
                # per row as sqrt(dot(x, x)), no cross-element
                # reduction to reorder (the fold's bits are pinned to
                # the reference fold by the stage tests).
                obs2d = np.array([obs.pt_2d_norm for obs in group],
                                 dtype=np.float64)
                d = projected - obs2d
                norms = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
                for obs, ok, err in zip(group, finite, norms):
                    if ok:
                        error_by_obs[obs] = err

        # Pass 2: apply the errors in the ORIGINAL observations order,
        # flagging each point's observers once after its fold segment.
        flushed = None
        flushed_updated = False

        def _flush_flags():
            if flushed is not None and flushed_updated:
                flushed._update_observing_keyframe_quality()

        for obs in observations:
            if obs.point is not flushed:
                _flush_flags()
                flushed = obs.point
                flushed_updated = False
            error = error_by_obs.get(obs)
            if error is not None:
                obs.point.update_reprojection_error(
                    error, update_quality=False)
                flushed_updated = True
        _flush_flags()