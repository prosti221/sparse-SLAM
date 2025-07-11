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

        self.velocity = np.zeros((4, 4))

        self.bundle_adjustment = BundleAdjustment()

    def update(self, new_frame, map):
        if not self.cur_frame:
            self.cur_frame = new_frame
            return

        # Check if this is during initialization
        should_initialization = self.prev_frame is None
        self.prev_frame = self.cur_frame
        self.cur_frame = new_frame

        E = match_features_between_frames(
            self.prev_frame, self.cur_frame)

        if should_initialization:
            self.initialize(map, E)
        else:
            # Get initial predicted pose
            predicted_pose = self.velocity @ self.prev_frame.pose
            self.cur_frame.pose = predicted_pose

            # Match projected points from the map to the current frame
            projected_points = self.project_visible_map_points(map)

            # Match the projected points with the current frame's keypoints
            matched_3d, matched_2d = self.match_projected_points(
                projected_points)

            # Estimate the refined pose using matched 3D and 2D points
            inliers = 0
            if len(matched_3d) >= MINIMUM_NUMBER_OF_INLIERS_FOR_NEW_KEYFRAME:
                inliers = self.estimate_refined_pose(matched_3d, matched_2d)

            if (self.should_insert_keyframe(inliers)):
                self.prev_keyframe = self.cur_keyframe
                self.cur_keyframe = self.cur_frame
                match_features_between_frames(
                    self.prev_keyframe, self.cur_keyframe)
                map.add_keyframe(self.cur_keyframe)
                self.on_keyframe_inserted(map)

        # Update the velocity based on the current and previous frame poses
        self.velocity = self.cur_frame.pose @ np.linalg.inv(
            self.prev_frame.pose)

    def initialize(self, map, E):
        Rt = extractRt(E)
        self.cur_frame.pose = Rt @ self.prev_frame.pose
        self.prev_keyframe = self.prev_frame
        self.cur_keyframe = self.cur_frame
        map.add_keyframe(self.prev_keyframe)
        map.add_keyframe(self.cur_keyframe)
        self.on_keyframe_inserted(map)

    def on_keyframe_inserted(self, map):
        triangulated_points = self.triangulate()
        map.add_points(triangulated_points)
        if (map.should_perform_global_bundle_adjustment(GLOBAL_BUNDLE_ADJUSTMENT_KEYFRAME_INTERVAL)):
            self.bundle_adjustment.global_bundle_adjustment(map)
        else:
            self.bundle_adjustment.local_bundle_adjustment(
                map,
                self.cur_keyframe.frame_id,
                window_size=LOCAL_BUNDLE_ADJUSTMENT_WINDOW_SIZE
            )

    def should_insert_keyframe(self, number_of_matched_points):
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

        return number_of_new_points_is_significant or camera_moved_significantly or camera_rotated_significantly

    def project_visible_map_points(self, map):
        projected_points = []

        for mp in map.points:
            pt_3d = mp.pt_3d
            projected_point = self.cur_frame.project_point(pt_3d)
            if projected_point is None:
                continue
            projected_points.append(
                (mp, self.cur_frame.project_point(pt_3d)))

        return projected_points

    def match_projected_points(self, projected_points, dist_thresh=5):
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

    def estimate_refined_pose(self, matched_3d, matched_2d):
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

    def compute(self):
        ret = np.zeros((self.cur_keyframe.matched_pts.shape[0], 4))

        # Convert camera-to-world poses to world-to-camera for triangulation
        pose1 = np.linalg.inv(self.cur_keyframe.pose)  # world-to-camera
        pose2 = np.linalg.inv(self.prev_keyframe.pose)  # world-to-camera

        pts1 = normalize(self.cur_keyframe.matched_pts, self.cur_keyframe.Kinv)
        pts2 = normalize(self.cur_keyframe.matched_pts_prev_frame,
                         self.prev_keyframe.Kinv)

        for i, p in enumerate(zip(pts1, pts2)):
            A = np.zeros((4, 4))
            A[0] = p[0][0] * pose1[2] - pose1[0]
            A[1] = p[0][1] * pose1[2] - pose1[1]
            A[2] = p[1][0] * pose2[2] - pose2[0]
            A[3] = p[1][1] * pose2[2] - pose2[1]

            _, _, vt = np.linalg.svd(A)
            ret[i] = vt[3]

        return ret

    def triangulate(self):
        if not (self.cur_keyframe and self.prev_keyframe):
            warning_log(LOG_TAG, "Not enough keyframes to triangulate points.")
            return []

        # Check baseline - cameras should be far enough apart
        """
        baseline = np.linalg.norm(
            self.cur_keyframe.pose[:3, 3] - self.prev_keyframe.pose[:3, 3])

        if baseline < MIN_BASELINE_THRESHOLD:  # Minimum baseline threshold
            warning_log(LOG_TAG,
                        f"Baseline too small: {baseline:.3f}, skipping triangulation")
            return []
        """

        # Get poses in world-to-camera coordinates
        pose1 = np.linalg.inv(self.prev_keyframe.pose)
        pose2 = np.linalg.inv(self.cur_keyframe.pose)

        # Normalize points
        pts1 = normalize(self.cur_keyframe.matched_pts_prev_frame,
                         self.prev_keyframe.Kinv)
        pts2 = normalize(self.cur_keyframe.matched_pts,
                         self.cur_keyframe.Kinv)

        # Triangulate points
        points_4d = self.compute()
        valid_points = points_4d[:, 3] != 0
        points_4d = points_4d[valid_points]
        points_4d = points_4d / points_4d[:, 3:]

        points = []
        world_to_cam_cur = np.linalg.inv(self.cur_keyframe.pose)
        world_to_cam_prev = np.linalg.inv(self.prev_keyframe.pose)

        valid_indices = np.where(valid_points)[0]

        for idx, i in enumerate(valid_indices):
            if idx >= len(self.cur_keyframe.matches):
                break

            m = self.cur_keyframe.matches[idx]
            point_4d = points_4d[idx]

            # Compute parallax angle
            parallax = compute_parallax_angle(
                pts1[i], pts2[i], pose1, pose2)
            if parallax < np.radians(MINIMUM_PARALLAX_THRESHOLD):
                continue

            # Transform world point to camera coordinates
            pl1 = world_to_cam_cur @ point_4d
            pl2 = world_to_cam_prev @ point_4d

            if pl1[2] < MIN_DEPTH or pl1[2] > MAX_DEPTH:
                continue
            if pl2[2] < MIN_DEPTH or pl2[2] > MAX_DEPTH:
                continue

            # Reprojection error check
            pp1 = self.cur_keyframe.K @ pl1[:3]
            pp2 = self.prev_keyframe.K @ pl2[:3]

            pp1 = (pp1[0:2] / pp1[2]) - \
                self.cur_keyframe.get_keypoints_descriptors()[0][m.trainIdx].pt
            pp2 = (pp2[0:2] / pp2[2]) - \
                self.prev_keyframe.get_keypoints_descriptors()[
                0][m.queryIdx].pt

            pp1 = np.sum(pp1**2)
            pp2 = np.sum(pp2**2)

            if pp1 > MAX_SQUARED_REPROJECTION_ERROR or pp2 > MAX_SQUARED_REPROJECTION_ERROR:
                continue

            point = Point(
                np.array(point_4d)[:3],
                self.cur_keyframe.matched_pts_colors[idx],
                self.cur_keyframe.get_keypoints_descriptors()[1][m.trainIdx]
            )

            # Register observations in both keyframes
            kp_idx_cur = m.trainIdx
            kp_idx_prev = m.queryIdx

            # For current keyframe
            pt_2d_cur = self.cur_keyframe.keypoints[kp_idx_cur].pt
            point.add_observation(
                self.cur_keyframe.frame_id, kp_idx_cur, pt_2d_cur)
            self.cur_keyframe.add_point_observation(
                point.point_id, kp_idx_cur, pt_2d_cur)

            # For previous keyframe
            pt_2d_prev = self.prev_keyframe.keypoints[kp_idx_prev].pt
            point.add_observation(
                self.prev_keyframe.frame_id, kp_idx_prev, pt_2d_prev)
            self.prev_keyframe.add_point_observation(
                point.point_id, kp_idx_prev, pt_2d_prev)

            points.append(point)

        return points
