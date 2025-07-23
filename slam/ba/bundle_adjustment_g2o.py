import g2o
import numpy as np
# from slam.core.map import Map
from slam.core.frame import Frame
from slam.core.point import Point
from slam.core.observation import Observation
from typing import List, Tuple
from slam.utils.logger import *
from slam.utils.constants import *
from uuid import UUID
from typing import List, Optional, Tuple, List, Dict

LOG_TAG = 'g2oBA'


class G2OBundleAdjustment:
    def __init__(self, map, verbose=False):
        self.verbose = verbose
        self.map = map

    def local_bundle_adjustment(self, reference_frame_id: UUID, window_size: int = 5, fix_points=True) -> bool:
        debug_log(
            LOG_TAG, f"Starting local BA with g2o around keyframe {reference_frame_id}")

        local_keyframes = self.map.get_local_keyframes(
            reference_frame_id, window_size)
        # local_keyframes = self.map.keyframes[-window_size:]
        local_points = self.map.get_local_points(
            local_keyframes, min_observations=2)

        if len(local_keyframes) < MINIMUM_LOCAL_KEYFRAMES or len(local_points) < MINIMUM_LOCAL_POINTS:
            warning_log(
                LOG_TAG, f"Insufficient data for BA: {len(local_keyframes)} keyframes, {len(local_points)} points")
            return False

        observations = self.map.get_observations(local_keyframes)

        if len(observations) < MINIMUM_LOCAL_OBSERVATIONS_FOR_POINT:
            warning_log(
                LOG_TAG, f"Insufficient observations for BA: {len(observations)}")
            return False
        else:
            debug_log(
                LOG_TAG, f"Using : {len(observations)} observations for BA optimization")

        return self._optimize_with_g2o(local_keyframes, local_points, observations, fix_initial_poses=True, fix_points=fix_points)

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
                pose = np.linalg.inv(kf.pose)
                R = pose[:3, :3]
                t = pose[:3, 3]
                se3 = g2o.SE3Quat(R, t)

                v_se3 = g2o.VertexSE3Expmap()
                v_se3.set_id(i)
                v_se3.set_estimate(se3)
                v_se3.set_fixed(i <= 1 and fix_initial_poses)

                opt.add_vertex(v_se3)
                frame_to_vertex[kf] = v_se3

            # Add point vertices with unique IDs
            for i, point in enumerate(points):
                v_point = g2o.VertexPointXYZ()
                v_point.set_id(len(keyframes) + i)
                v_point.set_estimate(point.pt_3d)
                v_point.set_fixed(fix_points)
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
            self.recompute_point_errors(observations)
            return True

        except Exception as e:
            error_log(LOG_TAG, f"g2o optimization failed: {e}")
            import traceback
            error_log(LOG_TAG, traceback.format_exc())
            return False

    def recompute_point_errors(
        self,
        observations: List[Observation]
    ) -> None:
        for obs in observations:
            projected_point = obs.frame.project_point(obs.pt_3d)
            if projected_point is None:
                continue
            else:
                error = np.linalg.norm(projected_point - obs.pt_2d_norm)

            # TODO: This is terrible, figure out how to unify quality updates and move the logic over to the map.
            obs.point.update_reprojection_error(error)
            obs.frame.compute_tracking_quality()
            self.map.update_tracking_quality(
                obs.frame_id, obs.frame.tracking_quality)
