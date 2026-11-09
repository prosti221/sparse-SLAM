import g2o
import numpy as np
from collections import defaultdict
from slam.core.frame import Frame
from slam.core.point import Point
from slam.core.observation import Observation
from typing import List, Tuple
from slam.utils.logger import *
from slam.utils.constants import *
from slam.utils import constants as const
from slam.utils import telemetry
from uuid import UUID
from typing import List, Optional, Tuple, List, Dict
import cv2 as cv

LOG_TAG = 'g2oBA'


class G2OBundleAdjustment:
    def __init__(self, map, verbose=False):
        self.verbose = verbose
        self.map = map
        # Phase 2f: persistent local-BA graph. Consecutive local BA calls
        # share ~90% of their problem; rebuilding it every call was ~60%
        # of BA time in per-edge pybind construction. The graph is
        # reconciled per call: new vertices are added with their
        # observation edges (the add rule needs no edge registry - every
        # involved keyframe is an observer of a local point, so a new
        # vertex's edge endpoints are guaranteed present), stale vertices
        # are removed (remove_vertex cascades their edges), and fixed
        # flags follow window membership (the gauge anchor is the oldest
        # involved keyframe, which slides). No estimate refresh is
        # needed: the post-BA write-back leaves the persistent graph's
        # estimates identical to the map state.
        # NOTE: this makes local BA a WITHIN-BASELINES change (not
        # bit-identical - the g2o solve is vertex-id-order sensitive at
        # ~1e-15; see benchmarks/IMPROVEMENT.md "Phase 2f spike").
        self._local_opt = None
        self._kf_vertex = {}   # frame_id -> VertexSE3Expmap
        self._pt_vertex = {}   # point_id -> VertexPointXYZ
        # Phase 2i (P1): incremental edge bookkeeping. The 2f reconcile
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
        """Phase 2j-4: called when the intrinsics change mid-run (the
        2j-4 refit's K/k1 propagation). The persistent graph's edge
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

    def refine_pose_pnp(self, frame: Frame = None):
        """Refine a frame's pose with PnP over its map-point observations.

        `frame=None` refines the map's current keyframe (the keyframe path
        via Map.optimize). The tracker passes the CURRENT frame: finding 9
        was precisely this call reading and writing the PREVIOUS keyframe
        instead, so per-frame refinement never applied to the tracked
        trajectory - and every frame re-solved an already-converged stale
        problem (~500 wasted solves on TUM).

        Divergence guards (phase 1b): a solution that is non-finite,
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
        inliers = np.arange(len(points_3d)).reshape(-1, 1)

        if not success or len(inliers) < const.PNP_MINIMUM_INLIERS:
            telemetry.event("pnp_no_solution" if not success else "pnp_few_inliers")
            warning_log(
                LOG_TAG, f"PnP failed or not enough inliers: {len(inliers)}")
            return False, len(inliers)

        R_refined, _ = cv.Rodrigues(rvec)
        t_refined = np.asarray(tvec, dtype=np.float64).flatten()

        def _reject(reason: str):
            telemetry.event(f"pnp_guard_{reason}")
            warning_log(
                LOG_TAG, f"PnP guard rejected solution ({reason}) - keeping seed pose")
            return False, len(inliers)

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

        debug_log(LOG_TAG, f"Pose refined with {len(inliers)} inliers "
                           f"(median reproj {median_err:.5f}, "
                           f"center step {center_step:.3f}, "
                           f"rot step {rot_step:.2f} deg).")
        telemetry.event("pnp_ok")

        return True, len(inliers)

    def local_bundle_adjustment(self, reference_frame_id: UUID, window_size: int = 5, fix_points=True) -> bool:
        debug_log(
            LOG_TAG, f"Starting local BA with g2o around keyframe {reference_frame_id}")

        local_keyframes = self.map.get_local_keyframes(
            reference_frame_id, window_size)
        if not local_keyframes:
            telemetry.event("ba_no_local_keyframes")
            warning_log(LOG_TAG, "No local keyframes found for local BA")
            return False

        # Local points: every map point observed by a window keyframe.
        # Kept in map insertion order so the optimization is deterministic.
        # (A phase-2d experiment restricted this set to points with >= 2
        # window observations - REJECTED: fast-rotating sequences never
        # re-observe points, so most points have exactly one window
        # observation by construction, and orbit's map collapsed to 2
        # keyframes / surf 0%. See benchmarks/IMPROVEMENT.md "Phase 2d".)
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
        # yet in the graph. Phase 2i (P1): pair membership comes from
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
        opt.initialize_optimization()
        iterations = opt.optimize(30)
        telemetry.event("ba_iterations", int(iterations))

        # --- Write back (same semantics as the fresh build) ---
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
        return True

    def global_bundle_adjustment(self) -> bool:
        debug_log(LOG_TAG, "Starting global BA with g2o")
        keyframes = self.map.keyframes
        points = self.map.points
        observations = self.map.get_observations(keyframes)

        if len(observations) < const.MINIMUM_GLOBAL_OBSERVATIONS_FOR_POINT:
            telemetry.event("ba_global_insufficient_observations")
            warning_log(
                LOG_TAG, f"Insufficient observations for global BA: {len(observations)}")
            return False

        return self._optimize_with_g2o(keyframes, points, observations, fix_initial_poses=True, fix_points=False)

    def pose_graph_optimization(self, loop_constraints: List[Tuple[UUID, UUID, np.ndarray]]) -> bool:
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

            # Add odometry edges (sequential constraints between keyframes)
            odometry_edge_count = 0
            for i in range(len(self.map.keyframes) - 1):
                kf1 = self.map.keyframes[i]
                kf2 = self.map.keyframes[i + 1]

                if kf1.frame_id not in frame_to_vertex or kf2.frame_id not in frame_to_vertex:
                    continue

                # Compute relative pose between consecutive keyframes
                # kf.pose = world-to-camera (Twc), we need camera-to-world (Tcw)
                Tcw1 = np.linalg.inv(kf1.pose)  # camera-to-world for kf1
                Tcw2 = np.linalg.inv(kf2.pose)  # camera-to-world for kf2
                relative_pose = np.linalg.inv(Tcw1) @ Tcw2  # Tcw2 @ inv(Tcw1)
                R_rel = relative_pose[:3, :3]
                t_rel = relative_pose[:3, 3]
                se3_rel = g2o.SE3Quat(R_rel, t_rel)

                # Create odometry edge
                edge = g2o.EdgeSE3Expmap()
                edge.set_vertex(0, frame_to_vertex[kf1.frame_id])
                edge.set_vertex(1, frame_to_vertex[kf2.frame_id])
                edge.set_measurement(se3_rel)
                # Odometry edges have moderate information
                edge.set_information(np.eye(6) * 10.0)

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
                # Use softer constraints - rotation especially should be flexible
                # Lower information = higher uncertainty = more flexible
                loop_info = np.diag([5, 5, 5, 1, 1, 1])  # Softer than odometry
                edge.set_information(loop_info)

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

            # Add observation edges. Phase 2e: the 2x2 information matrix
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
        # Early exit at convergence (phase 2a): this action was previously
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
        # Phase 2e: batch the per-frame projection. The original loop
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
        # Phase 2i (P1), both bit-identical: (a) pass 1 computes the
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
            # and the fold's bits are load-bearing (finding 15).
            finite = np.all(np.isfinite(projected), axis=1)
            for obs, proj, ok in zip(group, projected, finite):
                if ok:
                    error_by_obs[obs] = np.linalg.norm(
                        proj - obs.pt_2d_norm)

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

    def _extract_camera_params(self):
        if self.map.cur_keyframe is None or not hasattr(self.map.cur_keyframe, 'K'):
            raise ValueError(
                "Current keyframe or camera matrix K not available")

        K = self.map.cur_keyframe.K
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        return {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'K': K
        }
