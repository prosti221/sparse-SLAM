import numpy as np
import cv2 as cv
from scipy.spatial import cKDTree
from collections import defaultdict
from typing import List, Tuple

from slam.core.point import Point
from slam.core.frame import Frame
from slam.core.map import Map
from slam.core.observation import Observation
from slam.utils.utils import *
from slam.utils.constants import *
from slam.utils.logger import *
from slam.features.matcher import *
from slam.features.feature_extractor import FeatureExtractor

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
            return False

        # Check if this is during initialization
        should_initialize = self.prev_frame is None
        self.prev_frame = self.cur_frame
        self.cur_frame = new_frame

        if should_initialize:
            self._initialize()
        else:
            # Predict the initial pose estimate for current frame
            self.cur_frame.pose = self._get_next_pose_estimate()

            # Project the visible map points onto current frame for matching by projection
            projected_points = self._project_visible_map_points()

            # Match the projected points with the current frame's keypoints
            observations = self._match_projected_points(projected_points)

            # Check & handle keyframe insertion criteria
            self._handle_keyframe_insertion(observations)

        self.step += 1
        self.iterations_since_last_keyframe += 1
        self.map.update_tracking_quality()

        # Update the velocity based on the current and previous frame poses
        self.velocity = np.linalg.inv(
            self.prev_frame.pose) @ self.cur_frame.pose

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
        should_insert, result = should_insert_keyframe(
            self.cur_frame, self.map.cur_keyframe)

        if not should_insert and self.iterations_since_last_keyframe < MAX_NUMBER_OF_FRAMES_BETWEEN_KEYFRAMES:
            debug_log(
                LOG_TAG, f"Skipping keyframe insertion criteria not met. {result}")
            return
        debug_log(
            LOG_TAG, f"Inserting keyframe after {self.iterations_since_last_keyframe} iterations.")

        # Update the observations retrieved from matching by projection
        for obs in observations:
            obs.point.add_observation(obs)

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

            if best_idx != -1 and best_dist < 40:
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
