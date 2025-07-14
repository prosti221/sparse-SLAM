import uuid
import numpy as np
import cv2 as cv
from logger import debug_log
from constants import MAX_SQUARED_REPROJECTION_ERROR

LOG_TAG = 'Frame'


class Frame:
    def __init__(self, image, K):
        self.frame_id = uuid.uuid4()
        self.image = image

        self.K = K
        self.Kinv = np.linalg.inv(self.K)

        self.keypoints = None
        self.descriptors = None

        self.pose = np.eye(4)

        # Matching data
        self.matches = None
        self.matched_pts = None
        self.matched_pts_prev_frame = None
        self.matched_pts_colors = None

        # Enhanced for bundle adjustment
        self.observed_points = {}  # {point_id: (keypoint_idx, 2d_point)}
        self.keypoint_to_point_map = {}  # {keypoint_idx: point_id}

        # Track pose uncertainty and optimization status
        self.is_pose_optimized = False
        self.optimization_iterations = 0

        # Track keyframe quality metrics
        self.num_tracked_features = 0
        self.tracking_quality = 0.0

    def set_features(self, keypoints, descriptors):
        self.keypoints = keypoints
        self.descriptors = descriptors

    def set_match_data(self, matches, matched_pts, matched_pts_prev_frame):
        self.matches = matches
        self.matched_pts = matched_pts
        self.matched_pts_prev_frame = matched_pts_prev_frame
        self.num_tracked_features = len(matches) if matches else 0

        self._set_color_values_for_matched_points()

    def add_point_observation(self, point_id, keypoint_idx, pt_2d):
        self.observed_points[point_id] = (keypoint_idx, pt_2d)
        self.keypoint_to_point_map[keypoint_idx] = point_id

    def remove_point_observation(self, point_id):
        if point_id in self.observed_points:
            keypoint_idx, _ = self.observed_points[point_id]
            del self.observed_points[point_id]
            if keypoint_idx in self.keypoint_to_point_map:
                del self.keypoint_to_point_map[keypoint_idx]

    def get_camera_center(self):
        # Camera center is -R^T * t
        R = self.pose[:3, :3]
        t = self.pose[:3, 3]
        return -R.T @ t

    def world_to_camera(self, point_3d):
        """Transform 3D point from world to camera coordinates"""
        # Convert to homogeneous coordinates
        point_homo = np.append(point_3d, 1.0)

        # Transform to camera coordinates
        # Camera coordinates = R * (world_point - camera_center)
        world_to_cam = np.linalg.inv(self.pose)
        cam_coords = world_to_cam @ point_homo

        return cam_coords[:3]

    def get_projection_matrix(self):
        """Get 3x4 projection matrix"""
        # P = K * [R|t] where [R|t] is world-to-camera transformation
        world_to_cam = np.linalg.inv(self.pose)
        return self.K @ world_to_cam[:3, :]

    def get_observed_points(self):
        return list(self.observed_points.keys())

    def get_point_observation(self, point_id):
        return self.observed_points.get(point_id, None)

    def get_keypoint_point_id(self, keypoint_idx):
        return self.keypoint_to_point_map.get(keypoint_idx, None)

    def project_point(self, point_3d):
        """
        Project a 3D point to this frame's image coordinates

        Args:
            point_3d: 3D point in world coordinates

        Returns:
            2D point in image coordinates, or None if behind camera
        """
        # Transform to camera coordinates
        pt_cam = self.world_to_camera(point_3d)

        if pt_cam[2] <= 0:  # Behind camera
            return None

        # Project to image
        pt_img = self.K @ pt_cam
        u, v = pt_img[0] / pt_img[2], pt_img[1] / pt_img[2]

        return np.array([u, v])

    def is_point_visible(self, point_3d, margin=10):
        """
        Check if a 3D point is visible in this frame

        Args:
            point_3d: 3D point in world coordinates
            margin: Pixel margin from image borders

        Returns:
            True if point is visible, False otherwise
        """
        projected = self.project_point(point_3d)
        if projected is None:
            debug_log(
                LOG_TAG, f"Point {point_3d} is not visible in frame {self.frame_id}")
            return False

        H, W = self.image.shape[:2]
        u, v = projected

        return (margin <= u < W - margin and
                margin <= v < H - margin)

    def compute_reprojection_error(self, point_3d, observed_2d):
        projected = self.project_point(point_3d)
        if projected is None:
            debug_log(
                LOG_TAG, f"Point {point_3d} cannot be projected in frame {self.frame_id}")
            return float('inf')

        return np.linalg.norm(projected - observed_2d)

    def get_pose_6dof(self):
        """Get pose as 6DOF vector [rx, ry, rz, tx, ty, tz]"""
        R = self.pose[:3, :3]
        t = self.pose[:3, 3]

        # Convert rotation matrix to rotation vector
        rvec, _ = cv.Rodrigues(R)

        return np.concatenate([rvec.flatten(), t])

    def set_pose_from_6dof(self, pose_6dof):
        """Set pose from 6DOF vector"""
        rvec = pose_6dof[:3]
        t = pose_6dof[3:6]

        # Convert rotation vector to rotation matrix
        R, _ = cv.Rodrigues(rvec)

        self.pose = np.eye(4)
        self.pose[:3, :3] = R
        self.pose[:3, 3] = t

    def get_pose(self):
        return self.pose.copy()

    def compute_tracking_quality(self, map_obj):
        if len(self.keypoints) == 0:
            return 0.0

        # Count visible map points
        visible_count = 0
        high_quality_count = 0

        for point in map_obj.points:
            if self.frame_id in point.observations:
                visible_count += 1
                # Check if point has good quality (multiple observations, low reprojection error)
                if point.num_observations >= 3 and point.average_reprojection_error < 1.0:
                    high_quality_count += 1

        if visible_count == 0:
            return 0.0

        # Quality score combines visibility ratio and point quality
        visibility_ratio = visible_count / len(self.keypoints)
        quality_ratio = high_quality_count / visible_count if visible_count > 0 else 0.0

        self.tracking_quality = 0.7 * visibility_ratio + 0.3 * quality_ratio

        return self.tracking_quality

    def get_keyframe_statistics(self):
        stats = {
            'frame_id': self.frame_id,
            'num_keypoints': len(self.keypoints) if self.keypoints else 0,
            'num_matches': len(self.matches) if self.matches else 0,
            'num_observed_points': len(self.observed_points),
            'tracking_quality': self.tracking_quality,
            'is_pose_optimized': self.is_pose_optimized,
            'optimization_iterations': self.optimization_iterations
        }
        return stats

    def get_keypoints_descriptors(self):
        return self.keypoints, self.descriptors

    def get_gray_image(self):
        return cv.cvtColor(self.image, cv.IMREAD_GRAYSCALE)

    def _set_color_values_for_matched_points(self):
        # Sets the color values for the matched points in the current frame
        if self.matched_pts is not None:
            self.matched_pts_colors = np.array(
                [self.image[int(pt[1]), int(pt[0])][::-1] for pt in self.matched_pts])
        else:
            debug_log(
                LOG_TAG, "Matched points are None, cannot set colors")
            self.matched_pts_colors = None
