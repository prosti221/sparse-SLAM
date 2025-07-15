import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from slam.utils.logger import *
from slam.utils.constants import *

LOG_TAG = 'Utils'


def load_video(video_name, config):
    VIDEO_PATH = config.get_video_property(video_name, 'path')
    cap = get_video_cap(VIDEO_PATH)
    W = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    Fx = config.get_video_property(video_name, 'fx', default=0)
    Fy = config.get_video_property(video_name, 'fy', default=Fx)
    Cx = config.get_video_property(video_name, 'cx', default=W//2)
    Cy = config.get_video_property(video_name, 'cy', default=H//2)
    K = construct_K(Fx, Fy, Cx, Cy)

    info_log(LOG_TAG, f"Loaded video: {VIDEO_PATH} with properties:\n"
             f"  Width: {W}, Height: {H}\n"
             f"  Focal Lengths: Fx={Fx}, Fy={Fy}\n"
             f"  Principal Point: Cx={Cx}, Cy={Cy}")

    return cap, K


def pt_obj_to_array(pts):
    points_3d = np.array([pt.pt_3d for pt in pts])
    color = np.array([pt.color for pt in pts])

    color = color / 255.0
    return points_3d, color


def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def normalize(pts, Kinv):
    return np.dot(Kinv, add_ones(pts).T).T[:, :2]


def denormalize(points, K):
    ret = np.dot(K, np.array([points[0], points[1], 1.0]))
    return int(round(ret[0])), int(round(ret[1]))


def extractRt(E):
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0

    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)

    t = U[:, 2]
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t

    return ret


def get_video_cap(path):
    # FRAME_RATE = 15
    cap = cv.VideoCapture(path)
    # cap.set(cv.CAP_PROP_FPS, FRAME_RATE)

    return cap


def construct_K(focal_x, focal_y, c_x, c_y):

    return np.array([[focal_x, 0, c_x],
                     [0, focal_y, c_y],
                     [0, 0, 1]])


def draw_keypoints(frame, keypoints):
    img_cpy = frame.copy()
    keypoint_frame = cv.drawKeypoints(
        frame, keypoints, img_cpy, color=(0, 255, 0), flags=0
    )

    return img_cpy


def draw_matches(prev_img, prev_keypoints, cur_img, cur_keypoints, matches):
    prev_img = prev_img.copy()
    cur_img = cur_img.copy()

    match_img = cv.drawMatchesKnn(
        prev_img, prev_keypoints,
        cur_img, cur_keypoints,
        [matches],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return match_img


def compute_translation_distance(Tcw1, Tcw2):
    t1 = Tcw1[:3, 3]
    t2 = Tcw2[:3, 3]
    return np.linalg.norm(t1 - t2)


def compute_rotation_angle(Rcw1, Rcw2):
    R_rel = Rcw1 @ Rcw2.T
    angle_axis = Rotation.from_matrix(R_rel).as_rotvec()
    return np.linalg.norm(angle_axis)  # Angle in radians


def is_valid_triangulated_point(point_idx, cur_frame, prev_frame, point_4d):
    # 1. Check if point is in front of both cameras
    point_3d = point_4d[:3]

    # Transform to camera coordinates for both frames
    cur_cam_coords = cur_frame.world_to_camera(point_3d)
    prev_cam_coords = prev_frame.world_to_camera(point_3d)

    if cur_cam_coords[2] <= MIN_DEPTH or prev_cam_coords[2] <= MIN_DEPTH:
        return TRIANGULATION_VALIDATION_CODE['MIN_DEPTH_VIOLATION']

    # 2. Check depth bounds
    if cur_cam_coords[2] > MAX_DEPTH or prev_cam_coords[2] > MAX_DEPTH:
        return TRIANGULATION_VALIDATION_CODE['MAX_DEPTH_VIOLATION']

    # 3. Check parallax angle
    if not check_parallax_angle(cur_frame, prev_frame, point_3d, MINIMUM_TRIANGULATION_PARALLAX_THRESHOLD):
        return TRIANGULATION_VALIDATION_CODE['MIN_PARALLAX_VIOLATION']

    # 4. Check reprojection error
    if not check_reprojection_error(point_idx, cur_frame, prev_frame, point_3d, MAX_SQUARED_REPROJECTION_ERROR):
        return TRIANGULATION_VALIDATION_CODE['MAX_REPROJECTION_ERROR_VIOLATION']

    # 5. Check if point is well-conditioned (not at infinity)
    if np.abs(point_4d[3]) < 1e-6:
        return TRIANGULATION_VALIDATION_CODE['WELL_CONDITIONED_VIOLATION']

    return TRIANGULATION_VALIDATION_CODE['VALID']


def check_parallax_angle(cur_frame, prev_frame, point_3d, min_parallax_deg):
    """Check if the parallax angle is sufficient for good triangulation"""

    # Get camera centers
    cur_center = cur_frame.get_camera_center()
    prev_center = prev_frame.get_camera_center()

    # Compute viewing rays
    ray1 = point_3d - cur_center
    ray2 = point_3d - prev_center

    # Normalize rays
    ray1_norm = ray1 / np.linalg.norm(ray1)
    ray2_norm = ray2 / np.linalg.norm(ray2)

    # Compute angle between rays
    cos_angle = np.dot(ray1_norm, ray2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_deg = np.degrees(np.arccos(cos_angle))

    return angle_deg >= min_parallax_deg


def check_reprojection_error(point_idx, cur_frame, prev_frame, point_3d, max_error):
    """Check reprojection error for both frames"""

    # Get matched 2D points
    match = cur_frame.matches[point_idx]
    cur_2d = cur_frame.keypoints[match.trainIdx].pt
    prev_2d = prev_frame.keypoints[match.queryIdx].pt

    # Project 3D point to both frames
    cur_proj = cur_frame.project_point(point_3d)
    prev_proj = prev_frame.project_point(point_3d)

    if cur_proj is None or prev_proj is None:
        return False

    # Compute reprojection errors
    cur_error = np.linalg.norm(np.array(cur_proj) - np.array(cur_2d))
    prev_error = np.linalg.norm(np.array(prev_proj) - np.array(prev_2d))

    return cur_error <= max_error and prev_error <= max_error


######## TESTING NEW TRIANGULATION  ########


def compute_triangulation(frame1, frame2, use_optimization=True):
    """
    Compute robust triangulation using multiple methods

    Args:
        frame1: First frame with matches
        frame2: Second frame with matches
        use_optimization: Whether to use iterative optimization

    Returns:
        np.array: Triangulated 3D points in homogeneous coordinates
    """
    if not frame2.matches or len(frame2.matches) == 0:
        warning_log(LOG_TAG, "No matches found for triangulation")
        return np.array([])

    # Extract matched points
    pts1 = np.array([frame1.keypoints[m.queryIdx].pt for m in frame2.matches])
    pts2 = np.array([frame2.keypoints[m.trainIdx].pt for m in frame2.matches])

    # Normalize points
    pts1_norm = pts1
    pts2_norm = pts2

    # Get projection matrices
    P1 = frame1.get_projection_matrix()
    P2 = frame2.get_projection_matrix()

    # Method 1: OpenCV triangulation (fast but basic)
    points_4d_cv = cv.triangulatePoints(P1, P2, pts1_norm.T, pts2_norm.T).T

    if len(points_4d_cv) == 0:
        warning_log(LOG_TAG,
                    "No valid points found in OpenCV triangulation")

    if not use_optimization:
        return points_4d_cv

    # Method 2: Iterative optimization for better accuracy
    points_4d_optimized = []
    optimized_points_count = 0
    for i in range(len(pts1)):
        pt1 = pts1_norm[i]
        pt2 = pts2_norm[i]

        # Use CV result as initial guess
        initial_guess = points_4d_cv[i][:3] / points_4d_cv[i][3]

        # Optimize using least squares
        result = optimize_triangulation(pt1, pt2, P1, P2, initial_guess)

        if result is not None:
            # Convert back to homogeneous coordinates
            points_4d_optimized.append(np.append(result, 1.0))
            optimized_points_count += 1
        else:
            # Fall back to OpenCV result
            points_4d_optimized.append(points_4d_cv[i])

    debug_log(
        LOG_TAG, f"Optimized {optimized_points_count} out of {len(pts1)} points")

    return np.array(points_4d_optimized)


def optimize_triangulation(pt1, pt2, P1, P2, initial_guess, max_iterations=10):
    """
    Optimize triangulation using iterative least squares

    Args:
        pt1, pt2: Normalized 2D points
        P1, P2: Projection matrices
        initial_guess: Initial 3D point estimate
        max_iterations: Maximum optimization iterations

    Returns:
        np.array: Optimized 3D point or None if failed
    """

    def residual_function(point_3d):
        # Project 3D point to both cameras
        point_homo = np.append(point_3d, 1.0)

        proj1 = P1 @ point_homo
        proj2 = P2 @ point_homo

        # Normalize projections
        if proj1[2] != 0:
            proj1 = proj1[:2] / proj1[2]
        else:
            proj1 = np.array([1e6, 1e6])

        if proj2[2] != 0:
            proj2 = proj2[:2] / proj2[2]
        else:
            proj2 = np.array([1e6, 1e6])

        # Compute residuals
        residual1 = pt1 - proj1
        residual2 = pt2 - proj2

        return np.concatenate([residual1, residual2])

    try:
        result = least_squares(
            residual_function,
            initial_guess,
            max_nfev=max_iterations * 10,
            ftol=1e-8,
            xtol=1e-8,
            method='lm'
        )

        if result.success:
            return result.x
        else:
            return None

    except Exception as e:
        warning_log(LOG_TAG, f"Optimization failed: {e}")
        return None


##### Testting keyframe selection #####

def compute_baseline_distance(frame1, frame2):
    """Compute baseline distance between two frames"""
    center1 = frame1.get_camera_center()
    center2 = frame2.get_camera_center()
    return np.linalg.norm(center2 - center1)


def compute_average_parallax(cur_frame, prev_keyframe, map_obj):
    """Compute average parallax angle for visible map points"""

    visible_points = []

    # Get points visible in both frames
    for point in map_obj.points:
        if (cur_frame.frame_id in point.observations and
                prev_keyframe.frame_id in point.observations):
            visible_points.append(point)

    if len(visible_points) < 10:
        return 0.0

    parallax_angles = []

    for point in visible_points:
        angle = compute_point_parallax(cur_frame, prev_keyframe, point.pt_3d)
        if angle > 0:
            parallax_angles.append(angle)

    if len(parallax_angles) == 0:
        return 0.0

    return np.mean(parallax_angles)


def compute_point_parallax(frame1, frame2, point_3d):
    """Compute parallax angle for a specific 3D point"""

    center1 = frame1.get_camera_center()
    center2 = frame2.get_camera_center()

    # Rays from cameras to point
    ray1 = point_3d - center1
    ray2 = point_3d - center2

    # Normalize rays
    ray1_norm = ray1 / np.linalg.norm(ray1)
    ray2_norm = ray2 / np.linalg.norm(ray2)

    # Compute angle
    cos_angle = np.dot(ray1_norm, ray2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


def compute_new_points_ratio(cur_frame, map_obj):
    if len(cur_frame.keypoints) == 0:
        return 0.0

    # Count how many keypoints in current frame are matched to existing map points
    matched_count = 0

    # Check if cur_frame has a point_observations attribute
    if hasattr(cur_frame, 'point_observations'):
        matched_count = len(cur_frame.point_observations)
    else:
        # Fallback: count from map perspective but more carefully
        matched_keypoint_indices = set()
        for point in map_obj.points:
            if cur_frame.frame_id in point.observations:
                # Get the keypoint index for this observation
                obs_data = point.observations[cur_frame.frame_id]
                if isinstance(obs_data, tuple) and len(obs_data) >= 2:
                    kp_idx = obs_data[0]  # Assuming (kp_idx, pt_2d) format
                    matched_keypoint_indices.add(kp_idx)
        matched_count = len(matched_keypoint_indices)

    new_points_count = len(cur_frame.keypoints) - matched_count
    return max(0.0, new_points_count / len(cur_frame.keypoints))


def should_insert_keyframe(map_obj, cur_frame, prev_keyframe, is_recovery=False):
    # 1. Check baseline constraints
    baseline_distance = compute_baseline_distance(cur_frame, prev_keyframe)
    baseline_ok = MINIMUM_BASELINE_THRESHOLD <= baseline_distance

    # 2. Check parallax constraints
    avg_parallax = compute_average_parallax(
        cur_frame, prev_keyframe, map_obj)
    parallax_ok = avg_parallax >= MINIMUM_KEYFRAME_PARALLAX_THRESHOLD

    # 3. Check new points ratio
    new_points_ratio = compute_new_points_ratio(cur_frame, map_obj)
    new_points_ok = new_points_ratio >= MINIMUM_NUMBER_OF_NEW_POINTS

    # 4. Check tracking quality
    tracking_quality = cur_frame.compute_tracking_quality(map_obj)
    quality_ok = tracking_quality >= MINIMUM_KEYFRAME_QUALITY_THRESHOLD

    # Decision logic
    reasons = []

    if not baseline_ok:
        if baseline_distance < MINIMUM_BASELINE_THRESHOLD:
            reasons.append(
                f"Baseline too small: {baseline_distance:.3f} < {MINIMUM_BASELINE_THRESHOLD}")
    if not parallax_ok:
        reasons.append(
            f"Parallax too small: {avg_parallax:.2f}° < {MINIMUM_KEYFRAME_PARALLAX_THRESHOLD}°")

    if not new_points_ok:
        reasons.append(
            f"Not enough new points: {new_points_ratio:.2f} < {MINIMUM_NUMBER_OF_NEW_POINTS}")

    if not quality_ok:
        reasons.append(
            f"Tracking quality too low: {tracking_quality:.2f} < {MINIMUM_KEYFRAME_QUALITY_THRESHOLD}")

    # Keyframe insertion criteria
    critical_conditions = [baseline_ok, parallax_ok]
    quality_conditions = [new_points_ok, quality_ok]

    # Insert keyframe if:
    # - All critical conditions are met AND at least one quality condition is met
    # - OR tracking quality is very low (emergency keyframe)
    should_insert = (all(critical_conditions) and any(
        quality_conditions))

    if should_insert:
        debug_log(LOG_TAG, "Keyframe statistics: "
                  f"Baseline: {baseline_distance:.3f}, "
                  f"Parallax: {avg_parallax:.2f}°, "
                  f"New Points Ratio: {new_points_ratio:.2f}, "
                  f"Tracking Quality: {tracking_quality:.2f}, ")
        return True, "Quality criteria met"
    else:
        return False, "; ".join(reasons)
