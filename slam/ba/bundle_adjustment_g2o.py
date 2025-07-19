import g2o
import numpy as np
from slam.core.map import Map
from slam.core.frame import Frame
from slam.core.point import Point
from typing import List, Tuple
from slam.utils.logger import *
from slam.utils.constants import *
from uuid import UUID
from typing import List, Optional, Tuple, List, Dict

LOG_TAG = 'g2oBA'


class G2OBundleAdjustment:
    def __init__(self, map: Map, verbose=False):
        self.verbose = verbose
        self.map: Map = map

    def local_bundle_adjustment(self, reference_frame_id: UUID, window_size: int = 5) -> bool:
        debug_log(
            LOG_TAG, f"Starting local BA with g2o around keyframe {reference_frame_id}")

        local_keyframes = self.map.get_local_keyframes(
            reference_frame_id, window_size)
        local_points = self.map.get_local_points(
            local_keyframes, min_observations=2)

        if len(local_keyframes) < MINIMUM_LOCAL_KEYFRAMES or len(local_points) < MINIMUM_LOCAL_POINTS:
            warning_log(
                LOG_TAG, f"Insufficient data for BA: {len(local_keyframes)} keyframes, {len(local_points)} points")
            return False

        observations = self.map.get_observations(
            local_keyframes, local_points, normalize_points=True)

        if len(observations) < MINIMUM_LOCAL_OBSERVATIONS_FOR_POINT:
            warning_log(
                LOG_TAG, f"Insufficient observations for BA: {len(observations)}")
            return False

        return self._optimize_with_g2o(local_keyframes, local_points, observations, fix_points=True)

    def global_bundle_adjustment(self) -> bool:
        debug_log(LOG_TAG, "Starting global BA with g2o")
        keyframes = self.map.keyframes
        points = self.map.points
        observations = self.map.get_observations(
            keyframes, points, normalize_points=True)

        if len(observations) < MINIMUM_GLOBAL_OBSERVATIONS_FOR_POINT:
            warning_log(
                LOG_TAG, f"Insufficient observations for global BA: {len(observations)}")
            return False

        return self._optimize_with_g2o(keyframes, points, observations, fix_points=False)

    def _optimize_with_g2o(self, keyframes: List[Frame], points: List[Point],
                           observations: List[Tuple[int, int, np.ndarray]],
                           fix_points: bool = False) -> bool:
        try:
            # Create optimizer
            opt = g2o.SparseOptimizer()
            solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
            solver = g2o.OptimizationAlgorithmLevenberg(solver)
            opt.set_algorithm(solver)

            # Add normalized camera parameters
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
                pose = np.linalg.inv(kf.pose.copy())
                R = pose[:3, :3]
                t = pose[:3, 3]
                se3 = g2o.SE3Quat(R, t)

                v_se3 = g2o.VertexSE3Expmap()
                v_se3.set_id(i)
                v_se3.set_estimate(se3)
                v_se3.set_fixed(i == 0)

                opt.add_vertex(v_se3)
                frame_to_vertex[kf] = v_se3

            # Find the best point based on reprojection error
            best_point_idx, lowest_point_error = 0, float("inf")
            for i, point in enumerate(points):
                if point.average_reprojection_error < lowest_point_error:
                    best_point_idx, lowest_point_error = i, point.average_reprojection_error

            # Add point vertices with unique IDs
            for i, point in enumerate(points):
                v_point = g2o.VertexPointXYZ()
                v_point.set_id(len(keyframes) + i)
                v_point.set_estimate(point.pt_3d)
                v_point.set_marginalized(True)
                # Fix the best point
                v_point.set_fixed(fix_points or i == best_point_idx)
                opt.add_vertex(v_point)
                point_to_vertex[point] = v_point

            edge_count = 0
            for point_idx, kf_idx, pt_2d in observations:
                if kf_idx >= len(keyframes) or point_idx >= len(points):
                    continue

                kf = keyframes[kf_idx]
                point = points[point_idx]

                if kf not in frame_to_vertex or point not in point_to_vertex:
                    continue

                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_parameter_id(0, 0)  # camera parameter
                edge.set_vertex(0, point_to_vertex[point])  # point
                edge.set_vertex(1, frame_to_vertex[kf])     # camera pose
                edge.set_information(np.eye(2))
                edge.set_measurement(pt_2d)
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
            opt.optimize(50)

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
                kf.is_pose_optimized = True
                kf.optimization_iterations += 1

            # Update 3D points if they were optimized
            if not fix_points:
                for point, vertex in point_to_vertex.items():
                    # Get updated point position
                    new_pt = np.array(vertex.estimate())
                    # point.update_reprojection_error(...)

                    # Update point in map
                    point.pt_3d = new_pt

            # Update the point reprojection errors in all observed keyframes after optimization
            self.recompute_point_errors(keyframes, points, observations)
            return True

        except Exception as e:
            error_log(LOG_TAG, f"g2o optimization failed: {e}")
            import traceback
            error_log(LOG_TAG, traceback.format_exc())
            return False

    def recompute_point_errors(
        self,
        keyframes: List[Frame],
        points: List[Point],
        observations: List[Tuple[int, int, np.ndarray]]
    ) -> None:
        for point_idx, kf_idx, pt_2d_observed in observations:
            if kf_idx >= len(keyframes) or point_idx >= len(points):
                continue

            kf = keyframes[kf_idx]
            point = points[point_idx]

            projected_point = kf.project_point(point.pt_3d)
            if projected_point is None:
                error = 10  # Punish heavily if the point is being projected behind the camera
            else:
                error = np.linalg.norm(projected_point - pt_2d_observed)

            # TODO: This is terrible, figure out how to unify quality updates.
            point.update_reprojection_error(error)
            kf.compute_tracking_quality()
            self.map.update_tracking_quality(kf.frame_id, kf.tracking_quality)
