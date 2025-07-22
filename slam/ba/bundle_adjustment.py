# from slam.core.map import Map
from slam.core.point import Point
from slam.core.frame import Frame
from slam.utils.constants import *
from slam.utils.logger import debug_log, error_log, warning_log

import time
from uuid import UUID
from collections import defaultdict
from typing import Optional, List, Tuple, Dict

import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


LOG_TAG = 'BundleAdjustment'


# TODO: Make these static methods, no need to instantiate the class.
class BundleAdjustment:
    def __init__(self, map, max_iterations: int = 50, ftol: float = 1e-7, xtol: float = 1e-7):
        self.max_iterations = max_iterations
        self.ftol = ftol
        self.xtol = xtol
        self.map = map
        self.fixed_frame = None

    def local_bundle_adjustment(self, reference_frame_id: UUID, window_size: int = 5, fix_points=False) -> bool:
        debug_log(
            LOG_TAG, f"Starting local BA around keyframe {reference_frame_id}")
        self.fixed_frame = reference_frame_id

        # Get local keyframes and points
        local_keyframes = self.map.get_local_keyframes(
            reference_frame_id, window_size)
        local_points = self.map.get_local_points(
            local_keyframes, min_observations=2)

        if len(local_keyframes) < MINIMUM_LOCAL_KEYFRAMES or len(local_points) < MINIMUM_LOCAL_POINTS:
            warning_log(
                LOG_TAG, f"Insufficient data for BA: {len(local_keyframes)} keyframes, {len(local_points)} points")
            return False

        # Get observations (point_idx, frame_idx, 2d_point)
        observations = self.map.get_observations(
            local_keyframes, local_points, normalize_points=True)

        if len(observations) < MINIMUM_LOCAL_OBSERVATIONS_FOR_POINT:
            warning_log(
                LOG_TAG, f"Insufficient observations for BA: {len(observations)}")
            return False

        debug_log(
            LOG_TAG, f"Local BA with {len(local_keyframes)} keyframes, {len(local_points)} points, {len(observations)} observations")

        # Setup optimization problem
        return self._optimize_bundle(local_keyframes, local_points, observations)

    def global_bundle_adjustment(self, max_keyframes: Optional[int] = None) -> bool:
        keyframes = self.map.keyframes
        if max_keyframes and len(keyframes) > max_keyframes:
            # Take most recent keyframes
            keyframes = keyframes[-max_keyframes:]
        points = self.map.points

        observations = self.map.get_observations(
            keyframes, points, normalize_points=True)

        if len(observations) < MINIMUM_GLOBAL_OBSERVATIONS_FOR_POINT:
            warning_log(
                LOG_TAG, f"Insufficient observations for global BA: {len(observations)}")
            return False

        debug_log(
            LOG_TAG, f"Global BA with {len(keyframes)} keyframes, {len(points)} points, {len(observations)} observations")

        return self._optimize_bundle(keyframes, points, observations)

    def _optimize_bundle(
        self,
        keyframes: List[Frame],
        points: List[Point],
        observations: List[Tuple[int, int, np.ndarray]]
    ) -> bool:
        try:
            # Pack parameters
            x0 = self._pack_parameters(keyframes, points)

            # Setup residual function
            def residual_function(x):
                return self._compute_residuals(x, keyframes, points, observations)

            # Setup Jacobian sparsity pattern
            jac_sparsity = self._get_jacobian_sparsity(
                keyframes, points, observations)

            # Perform optimization
            start_time = time.time()
            result = least_squares(
                residual_function,
                x0,
                jac_sparsity=jac_sparsity,
                max_nfev=self.max_iterations,
                ftol=self.ftol,
                xtol=self.xtol,
                method='trf',
                loss='huber',
                f_scale=LEAST_SQUARES_F_SCALE
            )

            optimization_time = time.time() - start_time

            if result.success:
                # Unpack optimized parameters
                self._unpack_parameters(
                    result.x, keyframes, points)
                debug_log(
                    LOG_TAG, f"BA converged in {result.nfev} iterations, {optimization_time:.3f}s")

                avg_reproj_error = np.sqrt(result.cost / len(observations))
                debug_log(LOG_TAG, f"Final cost: {avg_reproj_error:.6f}")

                self.recompute_point_errors(keyframes, points, observations)

                return True
            else:
                warning_log(
                    LOG_TAG, f"BA failed to converge: {result.message}")
                return False

        except Exception as e:
            error_log(LOG_TAG, f"BA optimization error: {e}")
            return False

    def _pack_parameters(self, keyframes: List[Frame], points: List[Point]) -> np.ndarray:
        """
        Pack keyframe poses and 3D points into optimization vector
        """
        params = []

        # Pack keyframe poses (6DOF each, skip first keyframe as reference)
        for i, kf in enumerate(keyframes):
            if kf.frame_id == self.fixed_frame:  # Keep first keyframe fixed as reference
                continue
            pose_6dof = kf.pose_6dof
            params.extend(pose_6dof)

        # Pack 3D points (3DOF each)
        for point in points:
            params.extend(point.pt_3d)

        return np.array(params)

    def _unpack_parameters(self, x: List[np.ndarray], keyframes: List[Frame], points: List[Point]):
        """
        Unpack optimization vector back to keyframe poses and 3D points
        """
        idx = 0

        # Unpack keyframe poses (skip first keyframe)
        for i, kf in enumerate(keyframes):
            if kf.frame_id == self.fixed_frame:  # First keyframe is fixed
                continue
            pose_6dof = x[idx:idx+6]
            kf.pose_6dof = pose_6dof
            idx += 6

        # Unpack 3D points
        for point in points:
            point.pt_3d = x[idx:idx+3]
            idx += 3

    def _compute_residuals(
        self,
        x: List[np.ndarray],
        keyframes: List[Frame],
        points: List[Point],
        observations: List[Tuple[int, int, np.ndarray]]
    ) -> np.ndarray:
        # Temporarily unpack parameters
        keyframes_copy = [kf for kf in keyframes]  # Shallow copy
        points_copy = [Point(p.pt_3d.copy())
                       for p in points]  # Deep copy points

        self._unpack_parameters(x, keyframes_copy, points_copy)

        residuals = []
        for point_idx, kf_idx, observed_2d in observations:
            kf = keyframes_copy[kf_idx]
            point_3d = points_copy[point_idx].pt_3d

            # Project 3D point to image pixel coordinates (u, v)
            projected_2d = kf.project_point(point_3d)

            if projected_2d is None:  # Point behind camera
                residuals.extend([10.0, 10.0])  # Large error
                points[point_idx].update_reprojection_error(10.0)
            else:
                # Compute reprojection error
                error = projected_2d - observed_2d
                points[point_idx].update_reprojection_error(
                    np.linalg.norm(error)
                )
                residuals.extend(error)

        return np.array(residuals)

    def _get_jacobian_sparsity(
        self,
        keyframes: List[Frame],
        points: List[Point],
        observations: List[Tuple[int, int, np.ndarray]]
    ) -> lil_matrix:
        # Build keyframe index mapping (skip fixed frame)
        pose_param_indices = {}
        pose_param_counter = 0
        for kf in keyframes:
            if kf.frame_id == self.fixed_frame:
                continue
            pose_param_indices[kf.frame_id] = pose_param_counter
            pose_param_counter += 1

        num_kf_params = pose_param_counter * 6
        num_point_params = len(points) * 3
        total_params = num_kf_params + num_point_params
        num_residuals = len(observations) * 2

        sparsity = lil_matrix((num_residuals, total_params), dtype=bool)

        for obs_idx, (point_idx, kf_idx, _) in enumerate(observations):
            residual_idx = obs_idx * 2
            kf = keyframes[kf_idx]
            if kf.frame_id != self.fixed_frame:
                pose_offset = pose_param_indices[kf.frame_id] * 6
                sparsity[residual_idx:residual_idx+2,
                         pose_offset:pose_offset+6] = True

            # Point parameters
            point_offset = num_kf_params + point_idx * 3
            sparsity[residual_idx:residual_idx+2,
                     point_offset:point_offset+3] = True

        return sparsity

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

            # TODO: This is terrible, figure out how to unify quality updates and move the logic over to the map.
            point.update_reprojection_error(error)
            kf.compute_tracking_quality()
            self.map.update_tracking_quality(kf.frame_id, kf.tracking_quality)
