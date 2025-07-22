import numpy as np
import cv2 as cv
from scipy.spatial import cKDTree
from collections import defaultdict
from typing import List, Tuple
import matplotlib.pyplot as plt

from slam.core.point import Point
from slam.core.frame import Frame
from slam.core.map import Map
from slam.core.observation import Observation
from slam.utils.utils import *
from slam.utils.constants import *
from slam.utils.logger import *
from slam.features.matcher import *
from slam.features.feature_extractor import FeatureExtractor
from slam.ba.bundle_adjustment_g2o import G2OBundleAdjustment

LOG_TAG = 'Tracker'


class Tracker:
    def __init__(self, map: Map, feature_extraction_method: str = ORB_EXTRACTOR_NAME):
        self.feature_extraction_method = feature_extraction_method
        self.step = 0
        self.cur_frame: Frame = None
        self.prev_frame: Frame = None
        self.map: Map = map

        self.velocity = np.eye(4)
        self.iterations_since_last_keyframe = 0

        self.bundle_adjustment = G2OBundleAdjustment(map)
        self.feature_extractor = FeatureExtractor(feature_extraction_method)

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
            # Optical flow gives mixed results, sometimes good, sometimes shit.
            Rt = match_features_between_frames(
                self.cur_frame, self.prev_frame, self.feature_extraction_method)
            self.cur_frame.pose = np.dot(
                Rt, self.prev_frame.pose)
            # Match projected points from the map to the current frame
            # self.cur_frame.pose = self._predict_pose_with_optical_flow()
            projected_points = self._project_visible_map_points()

            # Match the projected points with the current frame's keypoints
            self._match_projected_points(projected_points)

            # Check & handle keyframe insertion criteria
            self._handle_keyframe_insertion()

        # Update the velocity based on the current and previous frame poses
        self.velocity = self.cur_frame.pose @ np.linalg.inv(
            self.prev_frame.pose)

        self.step += 1
        self.iterations_since_last_keyframe += 1

        return self.cur_frame.is_keyframe

    def _initialize(self):
        Rt = match_features_between_frames(
            self.cur_frame, self.prev_frame, self.feature_extraction_method)
        if Rt is not None:
            self.cur_frame.pose = np.dot(
                Rt, self.prev_frame.pose)

        self.map.add_keyframe(self.prev_frame)
        self.map.add_keyframe(self.cur_frame)

        self._on_keyframe_inserted()

    def _on_keyframe_inserted(self):
        self.map.optimize()

        triangulated_points = self._triangulate()
        self.map.add_points(triangulated_points)

    def _filter_duplicate_keypoints(self):
        for match in self.cur_frame.matches:
            prev_keyframe_kp = self.map.prev_keyframe.keypoints[match.trainIdx]
            if prev_keyframe_kp not in self.map.prev_keyframe.keypoint_to_point_map:
                continue

            point_id = self.map.prev_keyframe.keypoint_to_point_map[prev_keyframe_kp]
            point = self.map.get_point_by_id(point_id)

            if point is None:
                continue

            self.map.cur_keyframe.kp_unique_mask[match.queryIdx] = False

            self.map.cur_keyframe.add_point_observation(
                Observation(self.map.cur_keyframe, point, match.queryIdx))

        # Filter out matches that are already triangulated in the global map.
        pre_filter_length = len(self.map.cur_keyframe.matches)
        self.map.cur_keyframe.matches = [
            m for m in self.map.cur_keyframe.matches if self.map.cur_keyframe.kp_unique_mask[m.queryIdx]]

        debug_log(
            LOG_TAG, f"Removing duplicate points before triangulation. From {pre_filter_length} to {len(self.map.cur_keyframe.matches)}")

    def _handle_keyframe_insertion(self):
        should_insert, result = should_insert_keyframe(
            self.map, self.cur_frame, self.map.cur_keyframe)

        if not should_insert and self.iterations_since_last_keyframe < MAX_NUMBER_OF_FRAMES_BETWEEN_KEYFRAMES:
            debug_log(
                LOG_TAG, f"Skipping keyframe insertion criteria not met. {result}")
            return
        debug_log(
            LOG_TAG, f"Inserting keyframe after {self.iterations_since_last_keyframe} iterations.")

        self.map.add_keyframe(self.cur_frame)

        match_features_between_frames(
            self.map.cur_keyframe, self.map.prev_keyframe, self.feature_extraction_method)

        # We filter out matches that have already been triangulated.
        self._filter_duplicate_keypoints()

        self.iterations_since_last_keyframe = 0
        self._on_keyframe_inserted()

    def _project_visible_map_points(self) -> List[Tuple[Point, np.ndarray]]:
        projected_points = []
        local_keyframes = self.map.get_local_keyframes(
            self.map.cur_keyframe.frame_id, LOCAL_MAP_WINDOW_SIZE)
        local_map_points = self.map.get_local_points(local_keyframes)

        for mp in local_map_points:
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
        dist_thresh: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        matches = []
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

            if best_idx != -1 and best_dist < 30:
                repro_error = np.linalg.norm(
                    kp_coords[best_idx] - np.array([u_proj, v_proj]))

                matches.append((mp, best_idx))

                # Update observation relationships
                observation = Observation(self.cur_frame, mp, best_idx)

                mp.add_observation(observation)
                mp.update_reprojection_error(repro_error)

                self.cur_frame.add_point_observation(observation)
                self.cur_frame.kp_unique_mask[best_idx] = False

        debug_log(
            LOG_TAG, f"Found {len(matches)} projected matches after filtering")

        return matches

    def _triangulate(self) -> List[Point]:
        if not (self.map.cur_keyframe and self.map.prev_keyframe):
            warning_log(LOG_TAG, "Not enough keyframes to triangulate points.")
            return []

        # Triangulate points
        points_4d = compute_triangulation(
            self.map.cur_keyframe, self.map.prev_keyframe, use_optimization=True)

        valid_points = points_4d[:, 3] != 0
        points_4d = points_4d[valid_points]
        points_4d = points_4d / points_4d[:, 3:]

        points = []
        valid_indices = np.where(valid_points)[0]

        rejected_points = 0
        validation_results = defaultdict(lambda: 0)
        # set_dynamic_triangulation_depths(
        #    self.map.cur_keyframe, self.map.prev_keyframe, points_4d[:, :3], valid_indices)

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
                1][match.queryIdx]
        )

        # Current keyframe observation
        kp_idx_cur = match.queryIdx
        cur_frame_obs = Observation(self.map.cur_keyframe, point, kp_idx_cur)
        point.add_observation(cur_frame_obs)
        self.map.cur_keyframe.add_point_observation(cur_frame_obs)

        # Previous keyframe observation
        kp_idx_prev = match.trainIdx
        prev_frame_obs = Observation(
            self.map.prev_keyframe, point, kp_idx_prev)
        point.add_observation(prev_frame_obs)
        self.map.prev_keyframe.add_point_observation(prev_frame_obs)

        return point

    def _predict_pose_with_optical_flow(self) -> np.ndarray:
        if self.prev_frame is None or self.cur_frame is None:
            error_log(LOG_TAG, "Missing frames for optical flow prediction.")
            return np.eye(4)

        # Step 1: Get grayscale images
        prev_img = self.prev_frame.gray_image
        cur_img = self.cur_frame.gray_image

        # Step 2: Get keypoints from previous frame
        if self.prev_frame.keypoints is None or len(self.prev_frame.keypoints) < 8:
            warning_log(
                LOG_TAG, f"Not enough keypoints: {len(self.prev_frame.keypoints) if self.prev_frame.keypoints else 0}")
            return self.velocity @ self.prev_frame.pose

        # Ensure keypoints are in correct format (float32)
        prev_kps = np.array(self.prev_frame.kp_pts, dtype=np.float32)

        next_kps, status, error = cv.calcOpticalFlowPyrLK(
            prev_img, cur_img, prev_kps, None,
        )

        if next_kps is None or status is None:
            return self.velocity @ self.prev_frame.pose

        status = status.flatten()
        valid_mask = (status == 1)

        # Additional filtering based on tracking error
        if error is not None:
            error = error.flatten()
            error_threshold = np.median(error)
            error_mask = error < error_threshold
            valid_mask = valid_mask & error_mask

        matched_prev = prev_kps[valid_mask]
        matched_next = next_kps[valid_mask]

        debug_log(
            LOG_TAG, f"Valid optical flow matches for RANSAC: {np.sum(valid_mask)}/{len(status)}")

        if len(matched_prev) < 8:
            warning_log(
                LOG_TAG, f"Too few optical flow matches: {len(matched_prev)}")
            return self.velocity @ self.prev_frame.pose

        matched_prev_norm = normalize(matched_prev, self.prev_frame.Kinv)
        matched_next_norm = normalize(matched_next, self.cur_frame.Kinv)

        E, mask = cv.findEssentialMat(
            matched_next_norm, matched_prev_norm,
            method=cv.RANSAC,
            prob=0.999,
            threshold=0.0005,
            maxIters=1000
        )

        if E is None or mask is None:
            warning_log(LOG_TAG, "Essential matrix estimation failed.")
            return self.velocity @ self.prev_frame.pose

        # Step 7: Filter inliers
        inliers = mask.flatten() == 1

        debug_log(
            LOG_TAG, f"RANSAC inliers: {np.sum(inliers)}/{len(matched_prev)} ({np.sum(inliers)/len(matched_prev)*100:.1f}%)")

        if len(inliers) < 5:
            warning_log(LOG_TAG, f"Too few inliers: {np.sum(inliers)}")
            return self.velocity @ self.prev_frame.pose

        Rt = extractRtFromE(matched_next_norm,
                            matched_prev_norm, E, self.cur_frame.Kinv)

        predicted_pose = Rt @ self.prev_frame.pose

        return predicted_pose
