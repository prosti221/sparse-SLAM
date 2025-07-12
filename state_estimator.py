"""
The state estimator will only take in cur features, and use these to compute all of our states.
"""
import numpy as np
import cv2 as cv
from utils import *
from matcher import *
from point import Point
from scipy.spatial import cKDTree
from bundle_adjustment import BundleAdjustment
from constants import *
from logger import debug_log, info_log, error_log, warning_log

LOG_TAG = 'StateEstimator'


class StateEstimator:
    def __init__(self):
        self.cur_frame = None
        self.prev_frame = None

        self.cur_keyframe = None
        self.prev_keyframe = None

        self.velocity = np.eye(4)
        self.iterations_since_last_keyframe = 0

        self.bundle_adjustment = BundleAdjustment()

    def update(self, new_frame, map):
        if not self.cur_frame:
            self.cur_frame = new_frame
            return

        # Check if this is during initialization
        should_initialize = self.prev_frame is None
        self.prev_frame = self.cur_frame
        self.cur_frame = new_frame

        self.iterations_since_last_keyframe += 1

        if should_initialize:
            self._initialize(map)
        else:
            # Get initial predicted pose
            predicted_pose = self.velocity @ self.prev_frame.pose
            self.cur_frame.pose = predicted_pose

            # Match projected points from the map to the current frame
            projected_points = self._project_visible_map_points(map)

            # Match the projected points with the current frame's keypoints
            matched_3d, matched_2d = self._match_projected_points(
                projected_points)

            # Estimate the refined pose using matched 3D and 2D points
            inliers = 0
            if len(matched_3d) >= MINIMUM_NUMBER_OF_INLIERS_FOR_NEW_KEYFRAME:
                inliers = self._estimate_refined_pose(matched_3d, matched_2d)

            self._handle_keyframe_insertion(inliers, map)

        # Update the velocity based on the current and previous frame poses
        self.velocity = self.cur_frame.pose @ np.linalg.inv(
            self.prev_frame.pose)

    def _initialize(self, map):
        E = match_features_between_frames(
            self.prev_frame, self.cur_frame)
        Rt = extractRt(E)
        self.cur_frame.pose = Rt @ self.prev_frame.pose
        self.prev_keyframe = self.prev_frame
        self.cur_keyframe = self.cur_frame

        map.add_keyframe(self.prev_keyframe)
        map.add_keyframe(self.cur_keyframe)

        self._on_keyframe_inserted(map)

    def _on_keyframe_inserted(self, map):
        triangulated_points = self._triangulate()
        map.add_points(triangulated_points)
        if (map.should_perform_global_bundle_adjustment(GLOBAL_BUNDLE_ADJUSTMENT_KEYFRAME_INTERVAL)):
            self.bundle_adjustment.global_bundle_adjustment(map)
        else:
            self.bundle_adjustment.local_bundle_adjustment(
                map,
                self.cur_keyframe.frame_id,
                window_size=LOCAL_BUNDLE_ADJUSTMENT_WINDOW_SIZE
            )

        self.iterations_since_last_keyframe = 0

    def _handle_keyframe_insertion(self, number_of_matched_points, map):
        number_of_new_points_ratio = number_of_matched_points / \
            len(self.cur_frame.keypoints)
        number_of_new_points_is_significant = number_of_new_points_ratio < NEW_POINTS_THRESHOLD and len(
            self.cur_frame.keypoints) >= MINIMUM_NUMBER_OF_INLIERS_FOR_NEW_KEYFRAME

        translation_distance = compute_translation_distance(
            self.cur_frame.pose, self.cur_keyframe.pose)
        camera_moved_significantly = translation_distance > MINIMUM_TRANSLATION_THRESHOLD

        R1 = self.cur_frame.pose[:3, :3]
        R2 = self.cur_keyframe.pose[:3, :3]
        rotation_angle = compute_rotation_angle(R1, R2)
        camera_rotated_significantly = rotation_angle > MINIMUM_ROTATION_THRESHOLD

        criterias_met = int(camera_moved_significantly) + int(
            camera_rotated_significantly) + int(number_of_new_points_is_significant)

        if criterias_met >= MINIMUM_NUMBER_OF_KEY_FRAME_CRITERIAS_MET or self.iterations_since_last_keyframe >= MAX_NUMBER_OF_FRAMES_BETWEEN_KEYFRAMES:
            self.prev_keyframe = self.cur_keyframe
            self.cur_keyframe = self.cur_frame
            match_features_between_frames(
                self.prev_keyframe, self.cur_keyframe)
            map.add_keyframe(self.cur_keyframe)
            self._on_keyframe_inserted(map)

    def _project_visible_map_points(self, map):
        projected_points = []

        for mp in map.points:
            pt_3d = mp.pt_3d
            projected_point = self.cur_frame.project_point(pt_3d)
            if projected_point is None:
                continue
            projected_points.append(
                (mp, self.cur_frame.project_point(pt_3d)))

        return projected_points

    def _match_projected_points(self, projected_points, dist_thresh=5):
        matched_3d = []
        matched_2d = []
        keypoints, descriptors = self.cur_frame.get_keypoints_descriptors()

        if len(keypoints) == 0:
            return np.array([]), np.array([])

        kp_coords = np.array([kp.pt for kp in keypoints])
        tree = cKDTree(kp_coords)

        for mp, (u_proj, v_proj) in projected_points:
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
                desc_dist = cv.norm(d1, d2, cv.NORM_HAMMING)
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

        return np.array(matched_3d), np.array(matched_2d)

    def _estimate_refined_pose(self, matched_3d, matched_2d):
        success, R, t, inliers = cv.solvePnPRansac(
            matched_3d.astype(np.float32),
            matched_2d.astype(np.float32),
            self.cur_frame.K.astype(np.float32),
            None,  # No distortion
            flags=cv.SOLVEPNP_ITERATIVE,
            iterationsCount=PNP_ITERATIONS_COUNT,
            reprojectionError=PNP_REPROJECTION_ERROR
        )

        if success and len(inliers) >= PNP_MINIMUM_INLIERS:
            R, _ = cv.Rodrigues(R)
            # Convert world-to-camera back to camera-to-world
            world_to_cam = np.eye(4)
            world_to_cam[:3, :3] = R
            world_to_cam[:3, 3] = t.flatten()

            self.cur_frame.pose = np.linalg.inv(world_to_cam)
            return len(inliers)
        else:
            warning_log(LOG_TAG, "PnP failed, keeping predicted pose.")
            return 0

    def _triangulate(self):
        if not (self.cur_keyframe and self.prev_keyframe):
            warning_log(LOG_TAG, "Not enough keyframes to triangulate points.")
            return []

        # Triangulate points
        points_4d = compute_linear_dlt(self.prev_keyframe, self.cur_keyframe)
        valid_points = points_4d[:, 3] != 0
        points_4d = points_4d[valid_points]
        points_4d = points_4d / points_4d[:, 3:]

        points = []
        valid_indices = np.where(valid_points)[0]

        rejected_points = 0
        for idx, i in enumerate(valid_indices):
            if idx >= len(self.cur_keyframe.matches):
                break

            point_4d = points_4d[idx]

            if not is_valid_triangulated_point(idx, self.cur_keyframe, self.prev_keyframe, point_4d):
                rejected_points += 1
                continue

            point = self._create_point_and_register_observations(
                point_4d, idx)
            points.append(point)

        debug_log(
            LOG_TAG, f"Triangulated {len(points)} points, rejected {rejected_points} points.")
        return points

    def _create_point_and_register_observations(self, point_4d, point_idx):
        match = self.cur_keyframe.matches[point_idx]
        point = Point(
            np.array(point_4d)[:3],
            self.cur_keyframe.matched_pts_colors[point_idx],
            self.cur_keyframe.get_keypoints_descriptors()[1][match.trainIdx]
        )
        # Current keyframe observation
        kp_idx_cur = match.trainIdx
        pt_2d_cur = self.cur_keyframe.keypoints[kp_idx_cur].pt
        point.add_observation(self.cur_keyframe.frame_id,
                              kp_idx_cur, pt_2d_cur)
        self.cur_keyframe.add_point_observation(
            point.point_id, kp_idx_cur, pt_2d_cur)

        # Previous keyframe observation
        kp_idx_prev = match.queryIdx
        pt_2d_prev = self.prev_keyframe.keypoints[kp_idx_prev].pt
        point.add_observation(self.prev_keyframe.frame_id,
                              kp_idx_prev, pt_2d_prev)
        self.prev_keyframe.add_point_observation(
            point.point_id, kp_idx_prev, pt_2d_prev)

        return point
