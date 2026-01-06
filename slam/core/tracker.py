import numpy as np
import cv2 as cv
from scipy.spatial import cKDTree
from collections import defaultdict
from typing import List, Tuple
from uuid import UUID

from slam.core.point import Point
from slam.core.frame import Frame
from slam.core.map import Map
from slam.core.observation import Observation
from slam.utils.utils import *
from slam.utils.constants import *
from slam.utils.logger import *
from slam.features.matcher import *
from slam.features.feature_extractor import FeatureExtractor
from slam.loop_closure import LoopClosureCandidate

LOG_TAG = 'Tracker'


class Tracker:
    def __init__(self, map: Map, feature_extraction_method: str = ORB_EXTRACTOR_NAME, enable_multiscale=False, enable_ba=True):
        self.feature_extraction_method = feature_extraction_method
        self.step = 0
        self.cur_frame: Frame = None
        self.prev_frame: Frame = None
        self.map: Map = map

        self.velocity = np.eye(4)
        self.iterations_since_last_keyframe = 0
        self.consecutive_successful_relocalizations = 0
        self.consecutive_relocalization_failures = 0

        self.enable_ba = enable_ba

        self.feature_extractor = FeatureExtractor(feature_extraction_method)
        self.feature_extractor.set_multiscale_enabled(enable_multiscale)

    def update(self, new_frame: Frame) -> bool:
        # Extract features from the new frame
        new_kps, new_descs = self.feature_extractor.extract(
            new_frame.gray_image)
        new_frame.set_features(new_kps, new_descs)

        # If this is the first frame, initialize the current frame
        if not self.cur_frame:
            self.cur_frame = new_frame
            self.map.add_keyframe(self.cur_frame)
            return True

        self.prev_frame = self.cur_frame
        self.cur_frame = new_frame

        # Predict the initial pose estimate using constant velocity model (more stable)
        self.cur_frame.pose = self._get_next_pose_estimate(
            use_kinematic_model=True)

        # Try to refine with feature matching if possible
        Rt = match_features_between_frames(
            self.cur_frame, self.prev_frame, self.feature_extraction_method)
        if Rt is not None:
            # Refine the kinematic estimate with feature-based estimate
            refined_pose = Rt @ self.prev_frame.pose
            self.cur_frame.pose = refined_pose
            debug_log(LOG_TAG, "Pose estimate refined with feature matching")
        else:
            debug_log(
                LOG_TAG, "Using kinematic pose estimate (feature matching failed)")

        # Project the visible map points onto current frame for matching by projection
        projected_points = self._project_visible_map_points()

        # Match the projected points with the current frame's keypoints
        observations = self._match_projected_points(projected_points)

        # Refine pose using motion-only bundle adjustment if we have enough observations
        if len(observations) >= 10:
            pose_refined = self.map.g2o_optimizer.refine_pose_pnp()
            if pose_refined:
                debug_log(
                    LOG_TAG, f"Pose refined with PnP using {len(observations)} observations")
            else:
                debug_log(LOG_TAG, "Pose refinement failed")

        # Check & handle keyframe insertion criteria
        self._handle_keyframe_insertion(observations)

        # Periodic loop closure detection
        self._handle_loop_closure_detection()

        self.step += 1
        self.iterations_since_last_keyframe += 1
        self.map.update_tracking_quality()

        # Update the velocity based on the current and previous frame poses
        self.velocity = np.linalg.inv(
            self.prev_frame.pose) @ self.cur_frame.pose

        # Check for relocalization if tracking quality is consistently low
        self._handle_relocalization_check()

        return self.cur_frame.is_keyframe

    def _initialize(self):
        self.cur_frame.pose = self._get_next_pose_estimate()

        self.map.add_keyframe(self.prev_frame)
        self.map.add_keyframe(self.cur_frame)

        self._on_keyframe_inserted()

    def _on_keyframe_inserted(self):
        self.map.optimize(self.enable_ba)

        triangulated_points = self._triangulate()
        self.map.add_points(triangulated_points)

    def _filter_duplicate_keypoints(self):
        for match in self.cur_frame.matches:
            cur_keyframe_kp = self.map.cur_keyframe.keypoints[match.trainIdx]
            if cur_keyframe_kp not in self.map.cur_keyframe.keypoint_to_point_map:
                continue

            point = self.map.cur_keyframe.get_point_by_keypoint(
                cur_keyframe_kp)

            if point is None:
                continue

            self.cur_frame.kp_unique_mask[match.queryIdx] = False

            self.cur_frame.add_point_observation(
                Observation(self.cur_frame, point, match.queryIdx))

        # Filter out matches that are already triangulated in the global map.
        pre_filter_length = len(self.cur_frame.matches)
        self.cur_frame.matches = [
            m for m in self.cur_frame.matches if self.cur_frame.kp_unique_mask[m.queryIdx]]

        debug_log(
            LOG_TAG, f"Removing duplicate points before triangulation. From {pre_filter_length} to {len(self.cur_frame.matches)}")

    def _handle_keyframe_insertion(self, observations: List[Observation]):
        # Update the observations retrieved from matching by projection
        for obs in observations:
            obs.point.add_observation(obs)

        should_insert, result = should_insert_keyframe(
            self.cur_frame, self.map.cur_keyframe)

        if not should_insert and self.iterations_since_last_keyframe < MAX_NUMBER_OF_FRAMES_BETWEEN_KEYFRAMES:
            debug_log(
                LOG_TAG, f"Skipping keyframe insertion criteria not met. {result}")
            return
        debug_log(
            LOG_TAG, f"Inserting keyframe after {self.iterations_since_last_keyframe} iterations.")

        # Match features between current frame and previous keyframe
        match_features_between_frames(
            self.cur_frame, self.map.cur_keyframe, self.feature_extraction_method)

        # We filter out matches that have already been triangulated.
        self._filter_duplicate_keypoints()

        # Promote current frame to keyframe
        self.map.add_keyframe(self.cur_frame)

        self.iterations_since_last_keyframe = 0
        self._on_keyframe_inserted()

    def _project_visible_map_points(self) -> List[Tuple[Point, np.ndarray]]:
        projected_points = []

        for mp in self.map.points:
            pt_3d = mp.pt_3d
            # Project the 3D point to the current frame as pixel image coordinates (u, v)
            projected_point = self.cur_frame.project_point(
                pt_3d, normalized=False)
            if projected_point is None:
                continue
            projected_points.append(
                (mp, projected_point))

        return projected_points

    def _match_projected_points(
        self, projected_points: List[Tuple[Point, np.ndarray]],
        dist_thresh: int = 5
    ) -> List[Observation]:
        observations = []
        _, descriptors = self.cur_frame.get_keypoints_descriptors()
        kp_coords = self.cur_frame.kp_pts

        if kp_coords is None or len(kp_coords) == 0:
            warning_log(
                LOG_TAG, "No keypoints available for projection matching")
            return np.array([]), np.array([])

        tree = cKDTree(kp_coords)

        for mp, (u_proj, v_proj) in projected_points:
            # If the points are out of bounds, we skip.
            if not self.cur_frame.is_point_visible(mp.pt_3d):
                continue

            if self.cur_frame in mp.observations:
                continue

            # Find keypoints within threshold
            nearby_indices = tree.query_ball_point(
                [u_proj, v_proj], r=dist_thresh)

            best_idx = -1
            best_dist = float('inf')

            for i in nearby_indices:
                d1 = mp.descriptor
                d2 = descriptors[i]
                if d1 is None or d2 is None:
                    continue
                desc_dist = cv.norm(
                    d1, d2, cv.NORM_HAMMING if self.feature_extraction_method in BINARY_DESCRIPTION_METHODS else cv.NORM_L2)
                if desc_dist < best_dist:
                    best_dist = desc_dist
                    best_idx = i

            if best_idx != -1 and best_dist < 20:  # Stricter threshold for better matches
                repro_error = np.linalg.norm(
                    kp_coords[best_idx] - np.array([u_proj, v_proj]))

                # Update observation relationships
                observation = Observation(self.cur_frame, mp, best_idx)
                observations.append(observation)
                mp.update_reprojection_error(repro_error)
                self.cur_frame.kp_unique_mask[best_idx] = False

        debug_log(
            LOG_TAG, f"Found {len(observations)} projected matches after filtering")

        return observations

    def _triangulate(self) -> List[Point]:
        if not (self.map.cur_keyframe and self.map.prev_keyframe):
            warning_log(LOG_TAG, "Not enough keyframes to triangulate points.")
            return []

        # Triangulate points
        points_4d = compute_triangulation(
            self.map.cur_keyframe, self.map.prev_keyframe, use_optimization=False)

        valid_points = points_4d[:, 3] != 0
        points_4d = points_4d[valid_points]
        points_4d = points_4d / points_4d[:, 3:]

        points = []
        valid_indices = np.where(valid_points)[0]

        rejected_points = 0
        validation_results = defaultdict(lambda: 0)

        for i, idx in enumerate(valid_indices):
            if idx >= len(self.map.cur_keyframe.matches):
                error_log(LOG_TAG, f"Index {idx} out of bounds for matches.")
                break

            point_4d = points_4d[i]

            point_validation_result = is_valid_triangulated_point(
                idx, self.map.cur_keyframe, self.map.prev_keyframe, point_4d)

            if (code := point_validation_result['validation_code']) != TRIANGULATION_VALIDATION_CODE["VALID"]:
                rejected_points += 1
                validation_results[TRIANGULATION_VALIDATION_CODE[code]] += 1
                continue

            point = self._create_point_and_register_observations(
                point_4d, idx)
            point.update_reprojection_error(
                point_validation_result['reprojection_error'])

            points.append(point)

        validation_str = "".join(
            [f'{k}: {v / rejected_points * 100:.2f}%, ' for k, v in validation_results.items()])
        debug_log(
            LOG_TAG,
            f"Triangulated {len(points)} points, rejected {rejected_points} points. Validation results: {validation_str}"
        )
        return points

    def _create_point_and_register_observations(self, point_4d: np.ndarray, point_idx: int) -> Point:
        match = self.map.cur_keyframe.matches[point_idx]
        point = Point(
            np.array(point_4d)[:3],
            self.cur_frame.get_color_value_for_keypoint(match.queryIdx),
            self.map.cur_keyframe.get_keypoints_descriptors()[
                1][match.queryIdx],
        )

        # Current keyframe observation
        kp_idx_cur = match.queryIdx
        cur_frame_obs = Observation(self.map.cur_keyframe, point, kp_idx_cur)
        point.add_observation(cur_frame_obs)

        # Previous keyframe observation
        kp_idx_prev = match.trainIdx
        prev_frame_obs = Observation(
            self.map.prev_keyframe, point, kp_idx_prev)
        point.add_observation(prev_frame_obs)

        return point

    def _get_next_pose_estimate(self, use_kinematic_model=False):
        if use_kinematic_model:
            return self.prev_frame.pose @ self.velocity

        Rt = match_features_between_frames(
            self.cur_frame, self.prev_frame, self.feature_extraction_method)

        if Rt is None:
            return self.prev_frame.pose @ self.velocity

        return Rt @ self.prev_frame.pose

    def _handle_loop_closure_detection(self):
        return  # Disable loop closure for now
        if self.step % 10 != 0 or len(self.map.keyframes) <= 10:
            return

        current_keyframe = self.map.cur_keyframe
        if current_keyframe is None:
            return

        loop_candidates = self.map.detect_loop_closures(current_keyframe)
        if not loop_candidates:
            debug_log(LOG_TAG, "No loop closure candidates detected")
            return

        debug_log(
            LOG_TAG, f"Detected {len(loop_candidates)} potential loop closures")

        # Set query keyframe ID for candidates
        for i, cand in enumerate(loop_candidates):
            updated_cand = LoopClosureCandidate(
                query_keyframe_id=current_keyframe.frame_id,
                match_keyframe_id=cand.match_keyframe_id,
                similarity_score=cand.similarity_score
            )
            loop_candidates[i] = updated_cand

        verified_loops = self.map.loop_detector.verify_loop_candidates(
            loop_candidates)
        if not verified_loops:
            debug_log(LOG_TAG, "No loop closures verified")
            return

        debug_log(
            LOG_TAG, f"Verified {len(verified_loops)} loop closures - triggering optimization")

        # Create pose graph constraints from verified loops
        loop_constraints = []
        for verified_loop in verified_loops:
            constraint = self.map.pose_graph_optimizer.create_constraint_from_verified_loop(
                verified_loop)
            loop_constraints.append(constraint)

        success = self.map.pose_graph_optimizer.optimize_with_loop_constraints(
            loop_constraints)
        if not success:
            warning_log(LOG_TAG, "Pose graph optimization failed")
            return

        debug_log(
            LOG_TAG, f"Successfully optimized pose graph with {len(verified_loops)} loop constraints")

        # After pose graph optimization, run full global BA to fix 3D points
        # This is critical because pose correction makes existing triangulations invalid
        debug_log(
            LOG_TAG, "Running full global bundle adjustment to fix 3D points after pose correction")
        global_ba_success = self.map.g2o_optimizer.global_bundle_adjustment()
        if global_ba_success:
            debug_log(
                LOG_TAG, "Global bundle adjustment completed successfully after loop closure")
        else:
            warning_log(
                LOG_TAG, "Global bundle adjustment failed after loop closure - map may be inconsistent")

    def _handle_relocalization_check(self):
        """Check for relocalization using the map's recovery state"""
        if not self.map.needs_recovery():
            return

        # Check for too many consecutive successful relocalizations BEFORE attempting
        debug_log(
            LOG_TAG, f"Checking consecutive successful relocs: {self.consecutive_successful_relocalizations}/{MAX_CONSECUTIVE_SUCCESSFUL_RELOCALIZATIONS}")
        if self.consecutive_successful_relocalizations >= MAX_CONSECUTIVE_SUCCESSFUL_RELOCALIZATIONS:
            warning_log(
                LOG_TAG, f"Too many consecutive successful relocalizations ({self.consecutive_successful_relocalizations}) - map may be unstable, resetting")
            self._reset_system()
            return

        debug_log(LOG_TAG, f"Map needs recovery - attempting relocalization")
        if self._attempt_relocalization():
            debug_log(LOG_TAG, "Relocalization successful!")
            return

        debug_log(LOG_TAG, "Relocalization failed, trying aggressive approach")
        if self._attempt_aggressive_relocalization():
            debug_log(LOG_TAG, "Aggressive relocalization successful!")
            return

        # Relocalization failed completely
        self.map.consecutive_relocalization_failures += 1
        warning_log(
            LOG_TAG, f"Relocalization failed - tracking lost ({self.map.consecutive_relocalization_failures}/{MAX_CONSECUTIVE_RELOCALIZATION_FAILURES})")

        # Check if we should reset the system due to consecutive failures
        if self.map.consecutive_relocalization_failures >= MAX_CONSECUTIVE_RELOCALIZATION_FAILURES:
            self._reset_system()
            return

        warning_log(
            LOG_TAG, "Continuing with degraded tracking - try returning to previously mapped areas")

    def _attempt_relocalization(self) -> bool:
        # Search through recent keyframes (last N)
        recent_keyframes = self.map.keyframes[-RELOCALIZATION_WINDOW:] if len(
            self.map.keyframes) > RELOCALIZATION_WINDOW else self.map.keyframes

        for kf in reversed(recent_keyframes):
            pose, inliers = self._match_frame_to_keyframe(self.cur_frame, kf)
            if inliers >= MIN_RELOCALIZATION_INLIERS:
                # Success! Reset pose and continue
                self.cur_frame.pose = pose
                self.consecutive_successful_relocalizations += 1
                self.map.on_relocalization_successful(kf.frame_id)
                return True
        return False

    def _attempt_aggressive_relocalization(self) -> bool:
        # Search ALL keyframes with more permissive matching
        for kf in reversed(self.map.keyframes):
            pose, inliers = self._match_frame_to_keyframe_permissive(
                self.cur_frame, kf)
            if inliers >= MIN_AGGRESSIVE_RELOCALIZATION_INLIERS:
                # Reset pose and potentially clear recent bad keyframes
                self.map.remove_keyframes_after(kf.frame_id)
                self.cur_frame.pose = pose
                self.consecutive_successful_relocalizations += 1
                self.map.on_relocalization_successful(kf.frame_id)
                return True
        return False

    def _match_frame_to_keyframe(self, frame: Frame, keyframe: Frame) -> Tuple[Optional[np.ndarray], int]:
        matches, Rt = match_features(
            frame, keyframe, self.feature_extraction_method)

        if len(matches) < MIN_RELOCALIZATION_MATCHES:
            return None, 0

        # Extract matched points
        indices_f1 = [m.queryIdx for m in matches]
        indices_f2 = [m.trainIdx for m in matches]

        pts_f1_norm = frame.kp_pts_norm[indices_f1]
        pts_f2_norm = keyframe.kp_pts_norm[indices_f2]

        # Estimate pose using RANSAC
        E, mask, _ = estimate_essential_matrix(pts_f1_norm, pts_f2_norm)
        if E is None:
            return None, 0

        inliers = np.sum(mask) if mask is not None else 0

        if inliers >= MIN_RELOCALIZATION_INLIERS:
            pose = extractRt(E)
            # Transform relative to keyframe pose
            return pose @ keyframe.pose, inliers

        return None, 0

    def _match_frame_to_keyframe_permissive(self, frame: Frame, keyframe: Frame) -> Tuple[Optional[np.ndarray], int]:
        # Temporarily adjust matcher constants for more permissive matching
        original_threshold = MATCHER_RANSAC_THRESHOLD
        original_min_inliers = MATCHER_RANSAC_MINIMUM_INLIERS

        # Make matching more permissive
        import slam.utils.constants as const
        const.MATCHER_RANSAC_THRESHOLD = 0.01  # More lenient
        const.MATCHER_RANSAC_MINIMUM_INLIERS = 5  # Lower minimum

        try:
            result = self._match_frame_to_keyframe(frame, keyframe)
        finally:
            # Restore original values
            const.MATCHER_RANSAC_THRESHOLD = original_threshold
            const.MATCHER_RANSAC_MINIMUM_INLIERS = original_min_inliers

        return result

    def _reset_system(self):
        """Reset the entire SLAM system and start fresh with current frame"""
        info_log(LOG_TAG, "System reset triggered - clearing map and restarting")

        # Reset the map completely
        self.map.reset()

        # Reset tracker state
        self.step = 0
        self.velocity = np.eye(4)
        self.iterations_since_last_keyframe = 0
        self.consecutive_successful_relocalizations = 0
        self.consecutive_relocalization_failures = 0

        # Set current frame as new origin (identity pose)
        self.cur_frame.pose = np.eye(4)
        self.prev_frame = None

        # Add current frame as first keyframe of new map
        self.map.add_keyframe(self.cur_frame)

        info_log(LOG_TAG, "System reset complete - starting new map")
