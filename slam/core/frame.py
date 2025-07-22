import numpy as np
import cv2 as cv
import uuid
from uuid import UUID
from slam.utils.logger import debug_log, error_log, warning_log
from slam.utils.constants import MAX_SQUARED_REPROJECTION_ERROR
from slam.utils.utils import normalize
from slam.core.point import Point
from typing import List, Tuple, Dict


LOG_TAG = 'Frame'


class Frame:
    def __init__(self, image: np.ndarray, K: np.ndarray):
        self.frame_id: UUID = uuid.uuid4()
        self.image = image
        self.H, self.W = image.shape[:2]

        self.K = K

        self.keypoints = None
        self.kp_pts = None
        self.kp_pts_norm = None
        self.kp_unique_mask = None
        self.descriptors = None

        self._pose = np.eye(4)

        # Matching data
        self.matches = None

        # TODO: Could be refactored to use a shared observations structure
        self.observed_points: Dict[UUID,
                                   Tuple[int, np.ndarray, Point]] = {}

        self.keypoint_to_point_map: Dict[Tuple[int, int], UUID] = {}

        self.is_keyframe = False

        # Track pose uncertainty and optimization status
        self.is_pose_optimized = False
        self.optimization_iterations = 0

        # Track keyframe quality metrics
        self.num_tracked_features = 0
        self.tracking_quality = 0.0

    def get_color_value_for_keypoint(self, keypoint_idx):
        DEFAULT_COLOR = np.array([255, 255, 255], dtype=np.uint8)

        if self.keypoints is None or keypoint_idx >= len(self.keypoints):
            warning_log(LOG_TAG, f"Invalid keypoint index: {keypoint_idx}")
            return DEFAULT_COLOR

        try:
            # Get keypoint coordinates
            kp = self.keypoints[keypoint_idx]
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))

            # Check image bounds
            if 0 <= x < self.W and 0 <= y < self.H:
                # Get BGR color and convert to RGB
                bgr_color = self.image[y, x]
                return np.array([bgr_color[2], bgr_color[1], bgr_color[0]], dtype=np.uint8)
            return DEFAULT_COLOR

        except Exception as e:
            error_log(
                LOG_TAG, f"Error getting color for keypoint {keypoint_idx}: {str(e)}")
            return DEFAULT_COLOR

    def get_point_observation(self, point_id):
        return self.observed_points.get(point_id, None)

    def get_keypoint_point_id(self, keypoint_idx):
        return self.keypoint_to_point_map.get(keypoint_idx, None)

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

    def get_keypoints_descriptors(self) -> Tuple[List[cv.KeyPoint], np.ndarray]:
        return self.keypoints, self.descriptors

    def set_features(self, keypoints: List[cv.KeyPoint], descriptors: np.ndarray) -> None:
        self.keypoints = keypoints
        self.descriptors = descriptors
        if len(keypoints) > 0:
            self.kp_pts = np.array(
                [kp.pt for kp in keypoints], dtype=np.float32)
            self.kp_pts_norm = normalize(self.kp_pts, self.Kinv)
            self.kp_unique_mask = [True] * len(self.kp_pts)

    def set_match_data(self, matches: List[cv.DMatch]) -> None:
        self.matches = matches
        self.num_tracked_features = len(matches) if matches else 0

    def add_point_observation(
        self,
        point: Point,
        keypoint_idx: int,
        pt_2d: np.ndarray
    ) -> None:
        self.observed_points[point.point_id] = (keypoint_idx, pt_2d, point)
        self.keypoint_to_point_map[self.keypoints[keypoint_idx]
                                   ] = point.point_id

    def remove_point_observation(self, point_id: UUID) -> None:
        if point_id in self.observed_points:
            keypoint_idx, _ = self.observed_points[point_id]
            del self.observed_points[point_id]
            if self.keypoints[keypoint_idx] in self.keypoint_to_point_map:
                del self.keypoint_to_point_map[self.keypoints[keypoint_idx]]

    def world_to_camera(self, point_3d: np.ndarray) -> np.ndarray:
        # Convert to homogeneous coordinates
        point_homo = np.append(point_3d, 1.0)

        # Transform to camera coordinates
        # Camera coordinates = R * (world_point - camera_center)
        world_to_cam = np.linalg.inv(self.pose)
        cam_coords = world_to_cam @ point_homo

        return cam_coords[:3]

    def project_point(self, point_3d: np.ndarray, normalized: bool = True) -> np.ndarray:
        # Transform to camera coordinates
        pt_cam = self.world_to_camera(point_3d)

        if pt_cam[2] <= 0:  # Behind camera
            return None

        # Normalized image coordinates (x/z and y/z)
        x = pt_cam[0] / pt_cam[2]
        y = pt_cam[1] / pt_cam[2]

        if normalized:
            return np.array([x, y])
        else:
            # Pixel coordinates (u, v)
            u = self.K[0, 0] * x + self.K[0, 2]
            v = self.K[1, 1] * y + self.K[1, 2]
            return np.array([u, v])

    def is_point_visible(self, point_3d: np.ndarray, margin: int = 10) -> bool:
        projected = self.project_point(point_3d, normalized=False)
        if projected is None:
            debug_log(
                LOG_TAG, f"Point {point_3d} is not visible in frame {self.frame_id}")
            return False

        H, W = self.image.shape[:2]
        # projected gives us pixel coordinates (u, v)
        u, v = projected

        return (margin <= u < W - margin and
                margin <= v < H - margin)

    def compute_reprojection_error(self, point_3d: np.ndarray, observed_2d: np.ndarray) -> float:
        """Compute reprojection error for a 3D point projected to this frame
        Args:
            point_3d: 3D point in world coordinates
            observed_2d: 2D point in image pixel oordinates (u, v)
        """
        projected = self.project_point(point_3d, normalized=True)
        if projected is None:
            debug_log(
                LOG_TAG, f"Point {point_3d} cannot be projected in frame {self.frame_id}")
            return float('inf')

        return np.linalg.norm(projected - self.normalize_keypoint(observed_2d))

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
            if point.num_observations >= 2 and point.average_reprojection_error < MAX_SQUARED_REPROJECTION_ERROR:
                high_quality_count += 1

        # Quality score combines visibility ratio and point quality
        visibility_ratio = visible_count / len(self.keypoints)
        quality_ratio = high_quality_count / visible_count if visible_count > 0 else 0.0

        self.tracking_quality = 0.7 * visibility_ratio + 0.3 * quality_ratio

        return self.tracking_quality

    def normalize_keypoint(self, pt_2d: np.ndarray) -> np.ndarray:
        if self.Kinv is None:
            raise ValueError("Camera intrinsic matrix Kinv is not set.")

        return (self.Kinv @ np.array([pt_2d[0], pt_2d[1], 1.0])).astype(np.float32)[:2]

    # Properties
    @property
    def camera_center(self):
        # Camera center is -R^T * t
        R = self.pose[:3, :3]
        t = self.pose[:3, 3]

        return -R.T @ t

    @property
    def projection_matrix(self):
        # P = K * [R|t] where [R|t] is world-to-camera transformation
        world_to_cam = np.linalg.inv(self.pose)

        return self.K @ world_to_cam[:3, :]

    @property
    def observed_points_list(self):
        return list(self.observed_points.keys())

    @property
    def pose(self):
        return self._pose.copy()

    @pose.setter
    def pose(self, value):
        self._pose = value

    @property
    def pose_6dof(self):
        """Get pose as 6DOF vector [rx, ry, rz, tx, ty, tz]"""
        R = self._pose[:3, :3]
        t = self._pose[:3, 3]

        # Convert rotation matrix to rotation vector
        rvec, _ = cv.Rodrigues(R)

        return np.concatenate([rvec.flatten(), t])

    @pose_6dof.setter
    def pose_6dof(self, pose_6dof: np.ndarray) -> None:
        rvec = pose_6dof[:3]
        t = pose_6dof[3:6]

        # Convert rotation vector to rotation matrix
        R, _ = cv.Rodrigues(rvec)

        self._pose = np.eye(4)
        self._pose[:3, :3] = R
        self._pose[:3, 3] = t

    @property
    def gray_image(self) -> np.ndarray:
        if len(self.image.shape) == 3:
            if self.image.shape[2] == 4:  # RGBA
                return cv.cvtColor(self.image.copy(), cv.COLOR_RGBA2GRAY)
            elif self.image.shape[2] == 3:  # RGB
                return cv.cvtColor(self.image.copy(), cv.COLOR_RGB2GRAY)
        return self.image.copy()  # Already grayscale

    @property
    def Kinv(self):
        return np.linalg.inv(self.K)
