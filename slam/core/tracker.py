import numpy as np
import cv2 as cv
from scipy.spatial import cKDTree
from collections import defaultdict
from typing import List, Tuple

from slam.utils.utils import *
from slam.ba.bundle_adjustment_g2o import G2OBundleAdjustment
# from slam.ba.bundle_adjustment import BundleAdjustment
from slam.features.matcher import *
from slam.core.point import Point
from slam.utils.constants import *
from slam.utils.logger import *
from slam.features.feature_extractor import FeatureExtractor
from slam.core.frame import Frame
from slam.core.map import Map

LOG_TAG = 'Tracker'


class Tracker:
    def __init__(self, map: Map, feature_extraction_method: str = ORB_EXTRACTOR_NAME):
        self.feature_extraction_method = feature_extraction_method
        self.step = 0
        self.cur_frame: Frame = None
        self.prev_frame: Frame = None
        self.map: Map = map

        self.cur_keyframe: Frame = None
        self.prev_keyframe: Frame = None

        self.velocity = np.eye(4)
        self.iterations_since_last_keyframe = 0

        self.bundle_adjustment = G2OBundleAdjustment()
        # self.bundle_adjustment = BundleAdjustment()
        self.feature_extractor = FeatureExtractor(feature_extraction_method)

    def update(self, new_frame: Frame) -> None:
        # Extract features from the new frame
        new_kps, new_descs = self.feature_extractor.extract(
            new_frame.get_gray_image())
        new_frame.set_features(new_kps, new_descs)

        # If this is the first frame, initialize the current frame
        if not self.cur_frame:
            self.cur_frame = new_frame
            return

        # Check if this is during initialization
        should_initialize = self.prev_frame is None
        self.prev_frame = self.cur_frame
        self.cur_frame = new_frame

        # Get initial predicted pose
        Rt = match_features_between_frames(
            self.cur_frame, self.prev_frame, self.feature_extraction_method)
        self.cur_frame.pose = np.dot(Rt, self.prev_frame.pose)

        # Optical flow gives mixed results, sometimes good, sometimes shit.
        # self.cur_frame.pose = self._predict_pose_with_optical_flow()

        if should_initialize:
            self._initialize()
        else:
            # Match projected points from the map to the current frame
            projected_points = self._project_visible_map_points()

            # Match the projected points with the current frame's keypoints
            self._match_projected_points(projected_points)

            # Check & handle keyframe insertion criteria
            self._handle_keyframe_insertion()

        # Update the velocity based on the current and previous frame poses
        """
        self.velocity = self.cur_frame.pose @ np.linalg.inv(
            self.prev_frame.pose)
        """

        self.step += 1
        self.iterations_since_last_keyframe += 1

    def _initialize(self):

        self.prev_keyframe = self.prev_frame
        self.cur_keyframe = self.cur_frame

        self.map.add_keyframe(self.prev_keyframe)
        self.map.add_keyframe(self.cur_keyframe)

        self._on_keyframe_inserted()

    def _on_keyframe_inserted(self):
        """ 
        if (self.map.should_perform_global_bundle_adjustment(GLOBAL_BUNDLE_ADJUSTMENT_KEYFRAME_INTERVAL)):
            self.bundle_adjustment.global_bundle_adjustment(self.map)
        else:
            self.bundle_adjustment.local_bundle_adjustment(
                self.map,
                self.prev_keyframe.frame_id,
                window_size=LOCAL_BUNDLE_ADJUSTMENT_WINDOW_SIZE
            )
        """
        triangulated_points = self._triangulate()
        self.map.add_points(triangulated_points)

    def _handle_keyframe_insertion(self):
        should_insert, result = should_insert_keyframe(
            self.map, self.cur_frame, self.cur_keyframe)

        if not should_insert and self.iterations_since_last_keyframe < MAX_NUMBER_OF_FRAMES_BETWEEN_KEYFRAMES:
            debug_log(
                LOG_TAG, f"Skipping keyframe insertion criteria not met. {result}")
            return
        debug_log(
            LOG_TAG, f"Inserting keyframe after {self.iterations_since_last_keyframe} iterations.")
        self.prev_keyframe = self.cur_keyframe
        self.cur_keyframe = self.cur_frame
        match_features_between_frames(
            self.cur_keyframe, self.prev_keyframe, self.feature_extraction_method)
        self.map.add_keyframe(self.cur_keyframe)
        self.iterations_since_last_keyframe = 0
        self._on_keyframe_inserted()

    def _project_visible_map_points(self) -> List[Tuple[Point, np.ndarray]]:
        projected_points = []

        for mp in self.map.points:
            pt_3d = mp.pt_3d
            # Project the 3D point to the current frame pixel coordinates (u, v)
            projected_point = self.cur_frame.project_point(pt_3d)
            if projected_point is None:
                continue
            projected_points.append(
                (mp, projected_point))

        return projected_points

    def _match_projected_points(
        self, projected_points: List[Tuple[Point, np.ndarray]],
        dist_thresh: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        matched_3d = []
        matched_2d = []
        _, descriptors = self.cur_frame.get_keypoints_descriptors()
        kp_coords = self.cur_frame.kp_pts

        if len(kp_coords) == 0:
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

            if best_idx != -1 and best_dist < 50:
                matched_3d.append(mp.pt_3d)
                matched_2d.append(kp_coords[best_idx])

                # Update observation relationships
                pt_2d = kp_coords[best_idx]
                mp.add_observation(self.cur_frame.frame_id, best_idx, pt_2d)
                self.cur_frame.add_point_observation(
                    mp.point_id, best_idx, pt_2d)

        debug_log(
            LOG_TAG, f"Found {len(matched_2d)} projected matches after filtering")

        return np.array(matched_3d), np.array(matched_2d)

    def _triangulate(self) -> List[Point]:
        if not (self.cur_keyframe and self.prev_keyframe):
            warning_log(LOG_TAG, "Not enough keyframes to triangulate points.")
            return []

        # Filter out already observed matches from current frame
        pre_filter_match_count = len(self.cur_frame.matches)
        self.cur_frame.set_match_data([
            m for m in self.cur_frame.matches
            if m.queryIdx not in self.cur_frame.keypoint_to_point_map
        ])
        post_filter_match_count = len(self.cur_frame.matches)
        debug_log(
            LOG_TAG, f"Removing duplicate points before triangulation. From {pre_filter_match_count} to {post_filter_match_count}")

        # Triangulate points
        points_4d = compute_triangulation(
            self.cur_keyframe, self.prev_keyframe, use_optimization=True)

        valid_points = points_4d[:, 3] != 0
        points_4d = points_4d[valid_points]
        points_4d = points_4d / points_4d[:, 3:]

        points = []
        valid_indices = np.where(valid_points)[0]

        rejected_points = 0
        validation_results = defaultdict(lambda: 0)
        for i, idx in enumerate(valid_indices):
            if idx >= len(self.cur_keyframe.matches):
                error_log(LOG_TAG, f"Index {idx} out of bounds for matches.")
                break

            point_4d = points_4d[i]

            if (code := is_valid_triangulated_point(idx, self.cur_keyframe, self.prev_keyframe, point_4d)) != TRIANGULATION_VALIDATION_CODE["VALID"]:
                rejected_points += 1
                validation_results[TRIANGULATION_VALIDATION_CODE[code]] += 1
                continue

            point = self._create_point_and_register_observations(
                point_4d, idx)
            points.append(point)

        validation_str = "".join(
            [f'{k}: {v / rejected_points * 100:.2f}%, ' for k, v in validation_results.items()])
        debug_log(
            LOG_TAG,
            f"Triangulated {len(points)} points, rejected {rejected_points} points. Validation results: {validation_str}"
        )
        return points

    def _create_point_and_register_observations(self, point_4d: np.ndarray, point_idx: int) -> Point:
        match = self.cur_keyframe.matches[point_idx]
        point = Point(
            np.array(point_4d)[:3],
            self.cur_frame.get_color_value_for_keypoint(match.queryIdx),
            self.cur_keyframe.get_keypoints_descriptors()[1][match.queryIdx]
        )
        # Current keyframe observation
        kp_idx_cur = match.queryIdx
        pt_2d_cur = self.cur_keyframe.keypoints[kp_idx_cur].pt
        point.add_observation(self.cur_keyframe.frame_id,
                              kp_idx_cur, pt_2d_cur)
        self.cur_keyframe.add_point_observation(
            point.point_id, kp_idx_cur, pt_2d_cur)

        # Previous keyframe observation
        kp_idx_prev = match.trainIdx
        pt_2d_prev = self.prev_keyframe.keypoints[kp_idx_prev].pt
        point.add_observation(self.prev_keyframe.frame_id,
                              kp_idx_prev, pt_2d_prev)
        self.prev_keyframe.add_point_observation(
            point.point_id, kp_idx_prev, pt_2d_prev)

        return point

    """
    def _predict_pose_with_optical_flow(self) -> np.ndarray:
        if self.prev_frame is None or self.cur_frame is None:
            debug_log(LOG_TAG, "Missing frames for optical flow prediction.")
            return np.eye(4)

        # Step 1: Get grayscale images
        prev_img = self.prev_frame.get_gray_image()
        cur_img = self.cur_frame.get_gray_image()

        # Step 2: Get keypoints from previous frame
        if self.prev_frame.keypoints is None or len(self.prev_frame.keypoints) < 8:
            debug_log(LOG_TAG, "Not enough keypoints for optical flow.")
            return self.velocity @ self.prev_frame.pose

        prev_kps = np.array(
            [kp.pt for kp in self.prev_frame.keypoints], dtype=np.float32)

        # Step 3: Compute optical flow (KLT)
        next_kps, status, _ = cv.calcOpticalFlowPyrLK(
            prev_img, cur_img, prev_kps, None)

        if next_kps is None or status is None:
            debug_log(LOG_TAG, "Optical flow failed.")
            return self.velocity @ self.prev_frame.pose

        # Step 4: Filter valid matches
        status = status.flatten()
        matched_prev = prev_kps[status == 1]
        matched_next = next_kps[status == 1]

        matched_prev_norm = normalize(matched_prev, self.prev_frame.Kinv)
        matched_next_norm = normalize(matched_next, self.cur_frame.Kinv)

        if len(matched_prev) < 8:
            debug_log(LOG_TAG, "Too few optical flow matches.")
            return self.velocity @ self.prev_frame.pose

        # Step 5: Estimate essential matrix
        E, mask = cv.findEssentialMat(
            matched_next_norm, matched_prev_norm,
            method=cv.RANSAC, prob=0.999, threshold=0.005
        )

        if E is None or mask is None:
            debug_log(LOG_TAG, "Essential matrix estimation failed.")
            return self.velocity @ self.prev_frame.pose

        inliers = mask.flatten() == 1
        matched_prev_norm = matched_prev_norm[inliers]
        matched_next_norm = matched_next_norm[inliers]

        if len(matched_prev) < 5:
            debug_log(
                LOG_TAG, "Too few inliers after essential matrix estimation.")
            return self.velocity @ self.prev_frame.pose

        Rt = extractRt(E)

        predicted_pose = Rt @ self.prev_frame.pose

        debug_log(
            LOG_TAG, f"Predicted pose using optical flow with {len(matched_prev)} inliers.")

        return predicted_pose
    """
