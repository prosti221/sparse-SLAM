import uuid
from uuid import UUID
import numpy as np
import cv2 as cv
from utils.logger import debug_log
from utils.constants import MAX_SQUARED_REPROJECTION_ERROR
from core.point import Point
from typing import List, Tuple, Dict

LOG_TAG = 'Frame'


class Frame:
    def __init__(self, image: np.ndarray, K: np.ndarray):
        self.frame_id: UUID = uuid.uuid4()
        self.image = image
        self.H, self.W = self.image.shape[:2]

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

        self.observed_points: Dict[UUID, Tuple[int, np.ndarray, Point]] = {}
        self.keypoint_to_point_map: Dict[int, UUID] = {}

        # Track pose uncertainty and optimization status
        self.is_pose_optimized = False
        self.optimization_iterations = 0

        # Track keyframe quality metrics
        self.num_tracked_features = 0
        self.tracking_quality = 0.0

    def set_features(self, keypoints: List[cv.KeyPoint], descriptors: np.ndarray):
        self.keypoints = keypoints
        self.descriptors = descriptors

    def set_match_data(
        self,
        matches: List[cv.DMatch],
        matched_pts: np.ndarray,
        matched_pts_prev_frame: np.ndarray
    ):
        self.matches = matches
        self.matched_pts = matched_pts
        self.matched_pts_prev_frame = matched_pts_prev_frame
        self.num_tracked_features = len(matches) if matches else 0

        self._set_color_values_for_matched_points()

    def add_point_observation(self, point: Point, keypoint_idx: int, pt_2d: np.ndarray):
        self.observed_points[point.point_id] = (keypoint_idx, pt_2d, point)
        self.keypoint_to_point_map[keypoint_idx] = point.point_id

    def remove_point_observation(self, point_id: UUID):
        if point_id in self.observed_points:
            keypoint_idx, _, _ = self.observed_points[point_id]
            del self.observed_points[point_id]
            if keypoint_idx in self.keypoint_to_point_map:
                del self.keypoint_to_point_map[keypoint_idx]

    def get_camera_center(self) -> np.ndarray:
        # Camera center is -R^T * t
        R = self.pose[:3, :3]
        t = self.pose[:3, 3]
        return -R.T @ t

    def world_to_camera(self, point_3d: np.ndarray) -> np.ndarray:
        """Transform 3D point from world to camera coordinates"""
        # Convert to homogeneous coordinates
        point_homo = np.append(point_3d, 1.0)

        # Transform to camera coordinates
        # Camera coordinates = R * (world_point - camera_center)
        world_to_cam = np.linalg.inv(self.pose)
        cam_coords = world_to_cam @ point_homo

        return cam_coords[:3]

    def get_projection_matrix(self) -> np.ndarray:
        """Get 3x4 projection matrix"""
        # P = K * [R|t] where [R|t] is world-to-camera transformation
        world_to_cam = np.linalg.inv(self.pose)
        return self.K @ world_to_cam[:3, :]

    def get_observed_points(self) -> List[int]:
        return list(self.observed_points.keys())

    def get_point_observation(self, point_id: UUID) -> Tuple[int, np.ndarray]:
        return self.observed_points.get(point_id, None)

    def get_keypoint_point_id(self, keypoint_idx: int) -> UUID:
        return self.keypoint_to_point_map.get(keypoint_idx, None)

    def project_point(self, point_3d: np.ndarray) -> np.ndarray:
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

    def is_point_visible(self, point_3d: np.ndarray, margin: int = 10) -> bool:
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

    def compute_reprojection_error(self, point_3d: np.ndarray, observed_2d: np.ndarray) -> float:
        projected = self.project_point(point_3d)
        if projected is None:
            debug_log(
                LOG_TAG, f"Point {point_3d} cannot be projected in frame {self.frame_id}")
            return float('inf')

        return np.linalg.norm(projected - observed_2d)

    def get_pose_6dof(self) -> np.ndarray:
        """Get pose as 6DOF vector [rx, ry, rz, tx, ty, tz]"""
        R = self.pose[:3, :3]
        t = self.pose[:3, 3]

        # Convert rotation matrix to rotation vector
        rvec, _ = cv.Rodrigues(R)

        return np.concatenate([rvec.flatten(), t])

    def set_pose_from_6dof(self, pose_6dof: np.ndarray):
        """Set pose from 6DOF vector"""
        rvec = pose_6dof[:3]
        t = pose_6dof[3:6]

        # Convert rotation vector to rotation matrix
        R, _ = cv.Rodrigues(rvec)

        self.pose = np.eye(4)
        self.pose[:3, :3] = R
        self.pose[:3, 3] = t

    def get_pose(self) -> np.ndarray:
        return self.pose.copy()

    def compute_tracking_quality(self) -> float:
        if len(self.keypoints) == 0:
            return 0.0

        # Count visible map points
        visible_count = len(self.observed_points)
        if visible_count == 0:
            return 0.0

        high_quality_count = 0
        for _, _, point in self.observed_points.values():
            # Check if point has good quality (multiple observations, low reprojection error)
            if point.num_observations >= 3 and point.average_reprojection_error < 1.0:
                high_quality_count += 1

        # Quality score combines visibility ratio and point quality
        visibility_ratio = visible_count / len(self.keypoints)
        quality_ratio = high_quality_count / visible_count if visible_count > 0 else 0.0

        self.tracking_quality = 0.7 * visibility_ratio + 0.3 * quality_ratio

        return self.tracking_quality

    def get_keypoints_descriptors(self) -> Tuple[List[cv.KeyPoint], np.ndarray]:
        return self.keypoints, self.descriptors

    def get_gray_image(self) -> np.ndarray:
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

    def get_keyframe_statistics(self) -> Dict:
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
