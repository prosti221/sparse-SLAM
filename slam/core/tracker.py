"""
The state estimator will only take in cur features, and use these to compute all of our states.
"""
import numpy as np
import cv2 as cv
from scipy.spatial import cKDTree
from collections import defaultdict

from slam.utils.utils import *
from slam.ba.bundle_adjustment import BundleAdjustment
from slam.features.matcher import *
from slam.core.point import Point
from slam.utils.constants import *
from slam.utils.logger import *
from slam.features.feature_extractor import FeatureExtractor
from slam.core.frame import Frame

LOG_TAG = 'Tracker'


class Tracker:
    def __init__(self, feature_extraction_method=ORB_EXTRACTOR_NAME):
        self.use_dnn = feature_extraction_method == DNN_EXTRACTOR_NAME
        self.step = 0
        self.cur_frame = None
        self.prev_frame = None

        self.cur_keyframe = None
        self.prev_keyframe = None

        self.velocity = np.eye(4)
        self.iterations_since_last_keyframe = 0

        self.bundle_adjustment = BundleAdjustment()
        self.feature_extractor = FeatureExtractor(feature_extraction_method)

    def update(self, img, K, map):
        # Construct the new Frame object with the current image, features and camera matrix
        gray_img = cv.cvtColor(img, cv.IMREAD_GRAYSCALE)
        new_kps, new_descs = self.feature_extractor.extract(gray_img)

        new_frame = Frame(img, K)
        new_frame.set_features(new_kps, new_descs)

        # If this is the first frame, initialize the current frame
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
            if len(matched_3d) >= MINIMUM_NUMBER_OF_INLIERS_FOR_PROJECTION_MATCHING:
                self._estimate_refined_pose(matched_3d, matched_2d)

            self._handle_keyframe_insertion(map)

        # Update the velocity based on the current and previous frame poses
        self.velocity = self.cur_frame.pose @ np.linalg.inv(
            self.prev_frame.pose)

        self.step += 1

    def _initialize(self, map):
        E = match_features_between_frames(
            self.prev_frame, self.cur_frame, is_binary_desc=(not self.use_dnn))
        self.cur_frame.pose = self._predict_pose_with_optical_flow()
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

    def _handle_keyframe_insertion(self, map):
        should_insert, result = should_insert_keyframe(
            map, self.cur_frame, self.cur_keyframe)

        if not should_insert and self.iterations_since_last_keyframe < MAX_NUMBER_OF_FRAMES_BETWEEN_KEYFRAMES:
            debug_log(
                LOG_TAG, f"Skipping keyframe insertion criteria not met. {result}")
            return
        debug_log(
            LOG_TAG, f"Inserting keyframe after {self.iterations_since_last_keyframe} iterations.")
        self.prev_keyframe = self.cur_keyframe
        self.cur_keyframe = self.cur_frame
        match_features_between_frames(
            self.prev_keyframe, self.cur_keyframe, is_binary_desc=(not self.use_dnn))
        map.add_keyframe(self.cur_keyframe)
        self.iterations_since_last_keyframe = 0
        self._on_keyframe_inserted(map)

    def _project_visible_map_points(self, map):
        projected_points = []

        for mp in map.points:
            pt_3d = mp.pt_3d
            projected_point = self.cur_frame.project_point(pt_3d)
            if projected_point is None:
                continue
            projected_points.append(
                (mp, projected_point))

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
                desc_dist = cv.norm(
                    d1, d2, cv.NORM_L2 if self.use_dnn else cv.NORM_HAMMING)
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
        points_4d = compute_triangulation(
            self.prev_keyframe, self.cur_keyframe, use_optimization=True)
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

    def _predict_pose_with_optical_flow(self):
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

        if len(matched_prev) < 8:
            debug_log(LOG_TAG, "Too few optical flow matches.")
            return self.velocity @ self.prev_frame.pose

        # Step 5: Estimate essential matrix
        E, mask = cv.findEssentialMat(
            matched_next, matched_prev,
            self.cur_frame.K, method=cv.RANSAC, prob=0.999, threshold=0.005
        )

        if E is None or mask is None:
            debug_log(LOG_TAG, "Essential matrix estimation failed.")
            return self.velocity @ self.prev_frame.pose

        inliers = mask.flatten() == 1
        matched_prev = matched_prev[inliers]
        matched_next = matched_next[inliers]

        if len(matched_prev) < 5:
            debug_log(
                LOG_TAG, "Too few inliers after essential matrix estimation.")
            return self.velocity @ self.prev_frame.pose

        # Step 6: Recover pose
        _, R, t, _ = cv.recoverPose(
            E, matched_next, matched_prev, self.cur_frame.K)

        # Step 7: Compose predicted pose
        relative_pose = np.eye(4)
        relative_pose[:3, :3] = R
        relative_pose[:3, 3] = t.flatten()

        predicted_pose = relative_pose @ self.prev_frame.pose

        debug_log(
            LOG_TAG, f"Predicted pose using optical flow with {len(matched_prev)} inliers.")

        return predicted_pose
