import g2o
import numpy as np
from slam.core.frame import Frame
from slam.core.point import Point
from slam.core.observation import Observation
from typing import List, Tuple
from slam.utils.logger import *
from slam.utils.constants import *
from uuid import UUID
from typing import List, Optional, Tuple, List, Dict
import cv2 as cv

LOG_TAG = 'g2oBA'


class G2OBundleAdjustment:
    def __init__(self, map, verbose=False):
        self.verbose = verbose
        self.map = map

    def refine_pose_pnp(self):
        points_3d = []
        points_2d = []

        for obs in self.map.cur_keyframe.observed_points.values():
            points_3d.append(obs.pt_3d)
            points_2d.append(obs.pt_2d_norm)

        if len(points_3d) < 4:
            warning_log(
                LOG_TAG, f"Not enough points for pose refinement ({len(points_3d)} provided).")
            return False, 0

        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)

        # Initial pose guess from current frame pose
        R_init = self.map.cur_keyframe.pose[:3, :3]
        t_init = self.map.cur_keyframe.pose[:3, 3]

        rvec_init, _ = cv.Rodrigues(R_init)
        rvec_init = np.array(rvec_init, dtype=np.float32).reshape(3, 1)
        t_init = np.array(t_init, dtype=np.float32).reshape(3, 1)

        success, rvec, tvec = cv.solvePnP(
            points_3d,
            points_2d,
            np.eye(3),
            distCoeffs=np.zeros(5),
            rvec=rvec_init,
            tvec=t_init,
            useExtrinsicGuess=True,
            flags=cv.SOLVEPNP_ITERATIVE
        )
        inliers = np.arange(len(points_3d)).reshape(-1, 1)

        if not success or len(inliers) < PNP_MINIMUM_INLIERS:
            warning_log(
                LOG_TAG, f"PnP failed or not enough inliers: {len(inliers)}")
            return False, len(inliers)

        # Update frame pose
        R_refined, _ = cv.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3] = R_refined
        pose[:3, 3] = tvec.flatten()
        self.map.cur_keyframe.pose = np.linalg.inv(pose)

        debug_log(LOG_TAG, f"Pose refined with {len(inliers)} inliers.")

        return True, len(inliers)

    def local_bundle_adjustment(self, reference_frame_id: UUID, window_size: int = 5, fix_points=True) -> bool:
        debug_log(
            LOG_TAG, f"Starting local BA with g2o around keyframe {reference_frame_id}")

        local_keyframes = self.map.get_local_keyframes(
            reference_frame_id, window_size)
        observations = self.map.get_observations(local_keyframes)

        keyframes = self.map.keyframes
        points = self.map.points

        if len(observations) < MINIMUM_LOCAL_OBSERVATIONS_FOR_POINT:
            warning_log(
                LOG_TAG, f"Insufficient observations for BA: {len(observations)}")
            return False
        else:
            debug_log(
                LOG_TAG, f"Using : {len(observations)} observations for BA optimization")

        return self._optimize_with_g2o(keyframes, points, observations, fix_initial_poses=True, fix_points=fix_points)

    def global_bundle_adjustment(self) -> bool:
        debug_log(LOG_TAG, "Starting global BA with g2o")
        keyframes = self.map.keyframes
        points = self.map.points
        observations = self.map.get_observations(keyframes)

        if len(observations) < MINIMUM_GLOBAL_OBSERVATIONS_FOR_POINT:
            warning_log(
                LOG_TAG, f"Insufficient observations for global BA: {len(observations)}")
            return False

        return self._optimize_with_g2o(keyframes, points, observations, fix_initial_poses=False, fix_points=False)

    def _optimize_with_g2o(self, keyframes: List[Frame], points: List[Point],
                           observations: List[Observation],
                           fix_initial_poses: bool = False,
                           fix_points: bool = False) -> bool:
        try:
            # Create optimizer
            opt = g2o.SparseOptimizer()
            solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
            solver = g2o.OptimizationAlgorithmLevenberg(solver)

            terminate = g2o.SparseOptimizerTerminateAction()
            terminate.set_gain_threshold(1e-6)
            opt.set_algorithm(solver)

            # Add normalized camera parameters
            """
            camera_params = self._extract_camera_params()
            cam = g2o.CameraParameters(
                camera_params['fx'],
                (camera_params['cx'], camera_params['cy']),
                0
            )
            """
            cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)
            cam.set_id(0)
            opt.add_parameter(cam)

            # Robust kernel for outlier rejection
            robust_kernel = g2o.RobustKernelHuber(
                np.sqrt(5.991))  # sqrt(chi2 95%)

            # Create ID mappings
            frame_to_vertex = {}
            point_to_vertex = {}
            # Add keyframe vertices with unique IDs
            for i, kf in enumerate(keyframes):
                # Use camera-to-world transformation directly
                pose = np.linalg.inv(kf.pose)
                R = pose[:3, :3]
                t = pose[:3, 3]
                se3 = g2o.SE3Quat(R, t)

                v_se3 = g2o.VertexSE3Expmap()
                v_se3.set_id(i * 2)
                v_se3.set_estimate(se3)
                v_se3.set_fixed(i == 0 and fix_initial_poses)

                opt.add_vertex(v_se3)
                frame_to_vertex[kf] = v_se3

            # Find the best point based on reprojection error
            n = 5
            best_indices = sorted(
                range(len(points)), key=lambda i: points[i].average_reprojection_error, reverse=False)[:n]

            # Add point vertices with unique IDs
            for i, point in enumerate(points):
                v_point = g2o.VertexPointXYZ()
                v_point.set_id(i * 2 + 1)
                v_point.set_estimate(point.pt_3d)
                v_point.set_fixed(fix_points or i in best_indices)
                v_point.set_marginalized(True)
                opt.add_vertex(v_point)
                point_to_vertex[point] = v_point

            edge_count = 0
            for obs in observations:
                if obs.frame not in frame_to_vertex or obs.point not in point_to_vertex:
                    continue

                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_parameter_id(0, 0)  # camera parameter
                edge.set_vertex(0, point_to_vertex[obs.point])  # point
                # camera pose
                edge.set_vertex(1, frame_to_vertex[obs.frame])
                edge.set_information(np.eye(2))
                edge.set_measurement(obs.pt_2d_norm)
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)
                edge_count += 1

            if edge_count < 10:
                warning_log(LOG_TAG, f"Too few edges: {edge_count}")
                return False

            if self.verbose:
                opt.set_verbose(True)

            # Optimize with initialization
            opt.initialize_optimization()
            opt.optimize(30)

            # Update keyframe poses
            for kf, vertex in frame_to_vertex.items():
                est = vertex.estimate()
                R = est.rotation().matrix()
                t = est.translation()

                # Create new pose matrix
                new_pose = np.eye(4)
                new_pose[:3, :3] = R
                new_pose[:3, 3] = t

                # Update frame pose
                kf.pose = np.linalg.inv(new_pose)

            # Update 3D points if they were optimized
            if not fix_points:
                for point, vertex in point_to_vertex.items():
                    # Get updated point position
                    new_pt = np.array(vertex.estimate())

                    # Update point in map
                    point.pt_3d = new_pt

            # Update the point reprojection errors in all observed keyframes after optimization
            self._recompute_point_errors(observations)
            return True

        except Exception as e:
            error_log(LOG_TAG, f"g2o optimization failed: {e}")
            import traceback
            error_log(LOG_TAG, traceback.format_exc())
            return False

    def _recompute_point_errors(
        self,
        observations: List[Observation]
    ) -> None:
        for obs in observations:
            projected_point = obs.frame.project_point(obs.pt_3d)
            if projected_point is None:
                continue
            else:
                error = np.linalg.norm(projected_point - obs.pt_2d_norm)

            obs.point.update_reprojection_error(error)

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
