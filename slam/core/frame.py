from functools import cached_property
import numpy as np
import cv2 as cv
# u64 identity (migration locked decision 3): native handles ARE
# these ids; creation is main-thread-only and count() is atomic
# under the GIL.
import itertools
_next_frame_id = itertools.count(1)

from slam.utils.logger import debug_log, error_log, warning_log
from slam.utils.constants import MAX_POINT_AVG_REPROJECTION_ERROR
from slam.utils.utils import normalize, undistort_points_norm
from slam.core.point import Point
from slam.core.observation import Observation
from typing import List, Tuple, Dict


LOG_TAG = 'Frame'


class Frame:
    def __init__(self, image: np.ndarray, K: np.ndarray, k1: float = 0.0):
        self.frame_id: int = next(_next_frame_id)
        self.image = image
        self.H, self.W = image.shape[:2]

        self.K = K
        # One simple-radial distortion parameter, kept PER FRAME next
        # to K (the refit updates both and propagates). 0.0 is the
        # default and the LITERAL identity path everywhere it is
        # read.
        self.k1 = float(k1)

        self.keypoints = None
        self.kp_pts = None
        self.kp_pts_norm = None
        self.kp_unique_mask = None
        self.descriptors = None

        self._pose = np.eye(4)
        self._pose_inv = np.eye(4)

        # Matching data
        self.matches = None

        self.observed_points: Dict[int, Observation] = {}

        self.keypoint_to_point_map: Dict[int, Point] = {}

        self.is_keyframe = False

        self._tracking_quality = 0.0
        self.tracking_quality_needs_update = False

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

    def get_point_by_keypoint(self, keypoint_idx):
        return self.keypoint_to_point_map.get(keypoint_idx, None)

    def get_keypoints_descriptors(self) -> Tuple[List[cv.KeyPoint], np.ndarray]:
        return self.keypoints, self.descriptors

    def set_features(self, keypoints: List[cv.KeyPoint], descriptors: np.ndarray) -> None:
        self.keypoints = keypoints
        self.descriptors = descriptors
        if len(keypoints) > 0:
            self.kp_pts = cv.KeyPoint.convert(keypoints)
            # k1 == 0 keeps the literal identity path. k1 != 0
            # undistorts at the normalization boundary - the ONLY
            # change the downstream needs: matching, triangulation,
            # PnP and the g2o BA all work in undistorted normalized
            # coordinates and are untouched.
            if self.k1 != 0.0:
                self.kp_pts_norm = undistort_points_norm(
                    self.kp_pts, self.K, self.k1)
            else:
                self.kp_pts_norm = normalize(self.kp_pts, self.Kinv)
            self.kp_unique_mask = [True] * len(self.kp_pts)

    def set_match_data(self, matches: List[cv.DMatch]) -> None:
        self.matches = matches

    def add_point_observation(self, observation: Observation) -> None:
        self.observed_points[observation.point_id] = observation
        self.keypoint_to_point_map[observation.keypoint] = observation.point

    def world_to_camera(self, point_3d: np.ndarray) -> np.ndarray:
        # Convert to homogeneous coordinates
        point_homo = np.append(point_3d, 1.0)

        # Transform to camera coordinates
        # Camera coordinates = R * (world_point - camera_center)
        # Uses the cached inverse: this runs once per map point per frame,
        # and recomputing a 4x4 inverse every call dominated run time on
        # large maps.
        cam_coords = self._pose_inv @ point_homo

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

        # Pixel coordinates (u, v). With k1 != 0 the pixel
        # prediction applies the simple-radial forward model - the 1px
        # triangulation gate compares in PIXEL space (the Q2 record),
        # and every other pixel-domain consumer must see the same
        # model. k1 == 0 is the literal pinhole path (bit-identical).
        if self.k1 != 0.0:
            scale = 1.0 + self.k1 * (x * x + y * y)
            x *= scale
            y *= scale

        u = self.K[0, 0] * x + self.K[0, 2]
        v = self.K[1, 1] * y + self.K[1, 2]
        return np.array([u, v])

    def project_points(self, points_3d: np.ndarray, normalized: bool = True) -> np.ndarray:
        """
        Project multiple 3D points to image coordinates.

        Args:
            points_3d: Array of 3D points, shape (N, 3)
            normalized: If True, return normalized coordinates, else pixel coordinates

        Returns:
            Array of projected points, shape (N, 2). Invalid projections (behind camera) are None.
        """
        if points_3d.ndim != 2 or points_3d.shape[1] != 3:
            raise ValueError("points_3d must be shape (N, 3)")

        # Convert to homogeneous coordinates and transform to camera frame
        points_homo = np.column_stack([points_3d, np.ones(len(points_3d))])
        world_to_cam = self._pose_inv
        points_cam = (world_to_cam @ points_homo.T).T

        # Check which points are in front of camera
        valid_mask = points_cam[:, 2] > 0

        # Initialize result array
        result = np.full((len(points_3d), 2), np.nan)

        if np.any(valid_mask):
            # Normalized coordinates for valid points
            x_norm = points_cam[valid_mask, 0] / points_cam[valid_mask, 2]
            y_norm = points_cam[valid_mask, 1] / points_cam[valid_mask, 2]

            if normalized:
                result[valid_mask] = np.column_stack([x_norm, y_norm])
            else:
                # Pixel mode applies the simple-radial
                # forward model (same as project_point; see the note
                # there). k1 == 0 is the literal pinhole path.
                if self.k1 != 0.0:
                    scale = 1.0 + self.k1 * (x_norm ** 2 + y_norm ** 2)
                    x_norm = x_norm * scale
                    y_norm = y_norm * scale

                # Convert to pixel coordinates
                u = self.K[0, 0] * x_norm + self.K[0, 2]
                v = self.K[1, 1] * y_norm + self.K[1, 2]
                result[valid_mask] = np.column_stack([u, v])

        return result

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
        for observation in self.observed_points.values():
            # Check if point has good quality (multiple observations, low reprojection error)
            if observation.point.num_observations >= 2 and observation.point.average_reprojection_error < MAX_POINT_AVG_REPROJECTION_ERROR:
                high_quality_count += 1

        # Quality score combines visibility ratio and point quality
        visibility_ratio = visible_count / len(self.keypoints)
        quality_ratio = high_quality_count / visible_count if visible_count > 0 else 0.0

        self._tracking_quality = 0.7 * visibility_ratio + 0.3 * quality_ratio
        self.tracking_quality_needs_update = False

    def normalize_keypoint(self, pt_2d: np.ndarray) -> np.ndarray:
        if self.Kinv is None:
            raise ValueError("Camera intrinsic matrix Kinv is not set.")

        # k1 != 0 undistorts at this boundary too (the
        # same convention as set_features); k1 == 0 is the literal
        # legacy path.
        if self.k1 != 0.0:
            return undistort_points_norm(
                np.asarray(pt_2d, dtype=np.float64).reshape(1, 2),
                self.K, self.k1)[0]

        return (self.Kinv @ np.array([pt_2d[0], pt_2d[1], 1.0])).astype(np.float32)[:2]

    @cached_property
    def gray_image(self) -> np.ndarray:
        if len(self.image.shape) == 3:
            if self.image.shape[2] == 4:  # RGBA
                return cv.cvtColor(self.image.copy(), cv.COLOR_RGBA2GRAY)
            elif self.image.shape[2] == 3:  # RGB
                return cv.cvtColor(self.image.copy(), cv.COLOR_RGB2GRAY)
        return self.image.copy()  # Already grayscale

    @cached_property
    def Kinv(self):
        return np.linalg.inv(self.K)

    # Properties
    @property
    def tracking_quality(self):
        if (self.tracking_quality_needs_update):
            self.compute_tracking_quality()

        return self._tracking_quality

    @property
    def camera_center(self):
        # For a camera-to-world pose (Twc = [R | t]), the camera center in
        # world coordinates IS the translation component: Twc @ origin = t.
        # (-R^T t would be the camera center of a world-to-camera pose.)
        return self._pose[:3, 3]

    @property
    def observed_points_list(self):
        return list(self.observed_points.keys())

    @property
    def pose(self):
        return self._pose.copy()

    @property
    def pose_inv(self):
        """The cached world-to-camera transform - exactly what
        project_points multiplies with (the native error-fold consumes
        it; a copy guards against aliasing)."""
        return self._pose_inv.copy()

    @pose.setter
    def pose(self, value):
        # Copy to guard against aliasing, and cache the inverse: it is
        # needed once per map point per frame (world_to_camera).
        self._pose = np.array(value)
        self._pose_inv = np.linalg.inv(self._pose)

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

        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        # Route through the pose property so the cached inverse stays fresh.
        self.pose = pose

    @property
    def num_tracked_features(self):
        return sum(self.kp_unique_mask)

    @property
    def statistics(self):
        stats = {
            'frame_id': self.frame_id,
            'num_keypoints': len(self.keypoints) if self.keypoints else 0,
            'num_matches': len(self.matches) if self.matches else 0,
            'num_observed_points': len(self.observed_points),
            'tracking_quality': self.tracking_quality,
        }
        return stats
