import uuid
import numpy as np
import cv2 as cv


class Frame:
    # TODO: Create a seperate class for Keyframe that inherits from Frame
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
        self.pose_covariance = None
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

        self.__set_color_values_for_matched_points()

    def add_point_observation(self, point_id, keypoint_idx, pt_2d):
        """Add observation of a 3D point in this frame"""
        self.observed_points[point_id] = (keypoint_idx, pt_2d)
        self.keypoint_to_point_map[keypoint_idx] = point_id

    def remove_point_observation(self, point_id):
        """Remove observation of a 3D point from this frame"""
        if point_id in self.observed_points:
            keypoint_idx, _ = self.observed_points[point_id]
            del self.observed_points[point_id]
            if keypoint_idx in self.keypoint_to_point_map:
                del self.keypoint_to_point_map[keypoint_idx]

    def get_observed_points(self):
        """Get list of 3D point IDs observed in this frame"""
        return list(self.observed_points.keys())

    def get_point_observation(self, point_id):
        """Get the 2D observation of a specific 3D point"""
        return self.observed_points.get(point_id, None)

    def get_keypoint_point_id(self, keypoint_idx):
        """Get the 3D point ID associated with a keypoint"""
        return self.keypoint_to_point_map.get(keypoint_idx, None)

    def project_point(self, point_3d):
        """
        Project a 3D point to this frame's image coordinates

        Args:
            point_3d: 3D point in world coordinates

        Returns:
            2D point in image coordinates, or None if behind camera
        """
        # Convert camera-to-world pose to world-to-camera
        world_to_cam = np.linalg.inv(self.pose)
        R, t = world_to_cam[:3, :3], world_to_cam[:3, 3]

        # Transform to camera coordinates
        pt_cam = R @ point_3d + t

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
            return False

        H, W = self.image.shape[:2]
        u, v = projected

        return (margin <= u < W - margin and
                margin <= v < H - margin)

    def compute_reprojection_error(self, point_3d, observed_2d):
        """
        Compute reprojection error for a 3D point

        Args:
            point_3d: 3D point in world coordinates
            observed_2d: Observed 2D point in image

        Returns:
            Reprojection error in pixels
        """
        projected = self.project_point(point_3d)
        if projected is None:
            return float('inf')

        return np.linalg.norm(projected - observed_2d)

    def update_pose_with_covariance(self, new_pose, covariance=None):
        """Update pose with uncertainty information"""
        self.pose = new_pose
        self.pose_covariance = covariance
        self.is_pose_optimized = True

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

    def compute_tracking_quality(self):
        """Compute tracking quality metrics"""
        if not self.matches:
            self.tracking_quality = 0.0
            return

        # Simple quality metric based on number of matches
        # Could be enhanced with reprojection errors, feature distribution, etc.
        max_features = 500  # Expected maximum number of features
        self.tracking_quality = min(1.0, len(self.matches) / max_features)

    def get_keyframe_statistics(self):
        """Get statistics about this keyframe"""
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

    def keypoints_descriptors(self):
        return self.keypoints, self.descriptors

    def get_gray_image(self):
        return cv.cvtColor(self.image, cv.IMREAD_GRAYSCALE)

    def __set_color_values_for_matched_points(self):
        # Sets the color values for the matched points in the current frame
        if self.matched_pts is not None:
            self.matched_pts_colors = np.array(
                [self.image[int(pt[1]), int(pt[0])][::-1] for pt in self.matched_pts])
        else:
            self.matched_pts_colors = None
