"""Shared helpers for stage-level benchmark tests."""
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from slam.core.frame import Frame


def make_K(fx: float = 450.0, fy: float = 450.0,
           cx: float = 320.0, cy: float = 240.0) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def yaw_pose(yaw: float, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Camera-to-world pose with a yaw-only rotation."""
    return se3(Rotation.from_euler("y", yaw).as_matrix(), np.asarray(t, float))


def make_frame(pose: np.ndarray = None, K: np.ndarray = None,
               w: int = 640, h: int = 480) -> Frame:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    frame = Frame(img, K if K is not None else make_K())
    frame.pose = pose if pose is not None else np.eye(4)
    return frame


def project_gt(T_wc: np.ndarray, X_world: np.ndarray, K: np.ndarray):
    """Ground-truth projection: world point -> (pixel uv, camera coords)."""
    Xc = (np.linalg.inv(T_wc) @ np.append(X_world, 1.0))[:3]
    assert Xc[2] > 0, "test point behind camera"
    uv = K @ (Xc / Xc[2])
    return uv[:2], Xc


def set_keypoints(frame: Frame, uvs, descriptors=None) -> Frame:
    """Attach synthetic keypoints at the given pixel coordinates."""
    kps = [cv.KeyPoint(float(u), float(v), size=20) for u, v in uvs]
    if descriptors is None:
        descriptors = np.zeros((len(kps), 32), dtype=np.uint8)
    frame.set_features(kps, descriptors)
    return frame


def make_world_points(n: int = 40, span: float = 0.8,
                      z_range=(7.0, 10.0), rng_seed: int = 7) -> np.ndarray:
    """Deterministic scatter of world points in front of cameras at z<5.5."""
    rng = np.random.default_rng(rng_seed)
    xy = rng.uniform(-span, span, size=(n, 2))
    z = rng.uniform(*z_range, size=(n, 1))
    return np.hstack([xy, z])
