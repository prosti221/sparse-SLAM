import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from slam.utils.logger import *
from slam.utils.constants import *
import slam.utils.constants

LOG_TAG = 'Utils'


def load_video(config):
    video_tag = config.get_global_config_property("load_video")
    VIDEO_PATH = config.get_video_property(video_tag, 'path')

    cap = cap = cv.VideoCapture(VIDEO_PATH)
    W = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    Fx = config.get_video_property(video_tag, 'fx', default=0)
    Fy = config.get_video_property(video_tag, 'fy', default=Fx)
    Cx = config.get_video_property(video_tag, 'cx', default=W//2)
    Cy = config.get_video_property(video_tag, 'cy', default=H//2)
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


def extractRt(F):
    W = np.asmatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(F)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0

    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)

    t = U[:, 2]
    if t[2] < 0:
        t *= -1

    t = U[:, 2]
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t

    return ret


def extractRtFromE(pts1, pts2, E, K):

    # Extract rotation and translation
    _, R, t, _ = cv.recoverPose(E, pts1, pts2, K)

    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t.flatten()

    return ret


def construct_K(focal_x, focal_y, c_x, c_y):

    return np.array([[focal_x, 0, c_x],
                     [0, focal_y, c_y],
                     [0, 0, 1]])


def is_valid_triangulated_point(point_idx, cur_frame, prev_frame, point_4d):
    # 1. Check if point is in front of both cameras
    point_3d = point_4d[:3]

    # Transform to camera coordinates for both frames
    cur_cam_coords = cur_frame.world_to_camera(point_3d)
    prev_cam_coords = prev_frame.world_to_camera(point_3d)

    cur_reprojection_error, prev_reprojection_error = get_reprojection_error(
        point_idx, cur_frame, prev_frame, point_3d)

    # return cur_error <= max_error and prev_error <= max_error
    validation_code = TRIANGULATION_VALIDATION_CODE['VALID']

    if cur_cam_coords[2] <= slam.utils.constants.MIN_DEPTH or prev_cam_coords[2] <= slam.utils.constants.MIN_DEPTH:
        validation_code = TRIANGULATION_VALIDATION_CODE['MIN_DEPTH_VIOLATION']

    # 2. Check depth bounds
    if cur_cam_coords[2] > slam.utils.constants.MAX_DEPTH or prev_cam_coords[2] > slam.utils.constants.MAX_DEPTH:
        validation_code = TRIANGULATION_VALIDATION_CODE['MAX_DEPTH_VIOLATION']

    """
    # 3. Check baseline between the two cameras
    baseline_distance = compute_baseline_distance(cur_frame, prev_frame)
    point_distance = np.linalg.norm(point_3d - cur_frame.camera_center)
    # Baseline-to-distance ratio should be above minimum threshold
    # This prevents triangulating very distant points with insufficient baseline
    baseline_ratio = baseline_distance / max(point_distance, 1e-6)

    if baseline_ratio < slam.utils.constants.MIN_BASELINE_RATIO:
        validation_code = TRIANGULATION_VALIDATION_CODE['MIN_BASELINE_VIOLATION']
    """

    # 4. Check reprojection error
    if cur_reprojection_error > MAX_SQUARED_REPROJECTION_ERROR or prev_reprojection_error > MAX_SQUARED_REPROJECTION_ERROR:
        validation_code = TRIANGULATION_VALIDATION_CODE['MAX_REPROJECTION_ERROR_VIOLATION']

    # 5. Check if point is well-conditioned (not at infinity)
    if np.abs(point_4d[3]) < 1e-6:
        validation_code = TRIANGULATION_VALIDATION_CODE['WELL_CONDITIONED_VIOLATION']

    result = {
        'validation_code': validation_code,
        'reprojection_error': (cur_reprojection_error + prev_reprojection_error) / 2,
    }

    return result


def get_reprojection_error(point_idx, cur_frame, prev_frame, point_3d):
    match = cur_frame.matches[point_idx]

    # Pixel coordinates of the matched keypoints
    cur_2d = np.array(cur_frame.keypoints[match.queryIdx].pt)
    prev_2d = np.array(prev_frame.keypoints[match.trainIdx].pt)

    # Project the 3D point to both frames
    cur_proj = cur_frame.project_point(point_3d, normalized=False)
    prev_proj = prev_frame.project_point(point_3d, normalized=False)

    if cur_proj is None or prev_proj is None:
        return float('inf'), float('inf')

    # Compute reprojection errors in pixel image space
    cur_error = np.linalg.norm(cur_proj - cur_2d)
    prev_error = np.linalg.norm(prev_proj - prev_2d)

    return cur_error, prev_error


def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret


def compute_triangulation(frame1, frame2, use_optimization=True):
    if not frame1.matches or len(frame1.matches) == 0:
        warning_log(LOG_TAG, "No matches found for triangulation")
        return np.array([])

    # Extract matched points
    pts1 = np.array(
        [frame1.kp_pts_norm[m.queryIdx] for m in frame1.matches])
    pts2 = np.array(
        [frame2.kp_pts_norm[m.trainIdx] for m in frame1.matches])

    # Get projection matrices
    P1 = np.linalg.inv(frame1.pose)[:3, :]
    P2 = np.linalg.inv(frame2.pose)[:3, :]

    # points_4d_cv = cv.triangulatePoints(P1, P2, pts1.T, pts2.T).T
    points_4d = triangulate(P1, P2, pts1, pts2)

    if len(points_4d) == 0:
        warning_log(LOG_TAG,
                    "No valid points found in OpenCV triangulation")

    if not use_optimization:
        return points_4d

    points_4d_optimized = []
    optimized_points_count = 0
    for i in range(len(pts1)):
        pt1 = pts1[i]
        pt2 = pts2[i]

        # Use CV result as initial guess
        initial_guess = points_4d[i][:3] / points_4d[i][3]

        # Optimize using least squares
        result = optimize_triangulation(pt1, pt2, P1, P2, initial_guess)

        if result is not None:
            # Convert back to homogeneous coordinates
            points_4d_optimized.append(np.append(result, 1.0))
            optimized_points_count += 1
        else:
            # Fall back to OpenCV result
            points_4d_optimized.append(points_4d[i])

    debug_log(
        LOG_TAG, f"Optimized {optimized_points_count} out of {len(pts1)} points")

    return np.array(points_4d_optimized)


def optimize_triangulation(pt1, pt2, P1, P2, initial_guess, max_iterations=10):
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
            ftol=1e-15,
            xtol=1e-15,
            method='lm'
        )

        if result.success:
            return result.x
        else:
            warning_log(
                LOG_TAG, "Least squares optimization failed for triangulation")
            return None

    except Exception as e:
        warning_log(LOG_TAG, f"Least squares optimization error: {e}")
        return None


def compute_baseline_distance(frame1, frame2):
    center1 = frame1.camera_center
    center2 = frame2.camera_center

    return np.linalg.norm(center2 - center1)


def compute_new_points_ratio(cur_frame):
    if len(cur_frame.keypoints) == 0:
        return 0.0

    # We compute the ratio of non-observed matched points vs total matches with prev-frame
    return cur_frame.num_tracked_features / len(cur_frame.matches)


def should_insert_keyframe(cur_frame, prev_keyframe):
    # 1. Check baseline constraints
    baseline_distance = compute_baseline_distance(cur_frame, prev_keyframe)
    baseline_ok = MINIMUM_BASELINE_THRESHOLD <= baseline_distance

    # 2. Check new points ratio
    new_points_ratio = compute_new_points_ratio(cur_frame)
    new_points_ok = new_points_ratio >= MINIMUM_NEW_POINTS_RATIO

    # 3. Check tracking quality
    tracking_quality = cur_frame.compute_tracking_quality()
    quality_ok = tracking_quality >= MINIMUM_KEYFRAME_QUALITY_THRESHOLD

    # Decision logic
    reasons = []

    if not baseline_ok:
        if baseline_distance < MINIMUM_BASELINE_THRESHOLD:
            reasons.append(
                f"Baseline too small: {baseline_distance:.3f} < {MINIMUM_BASELINE_THRESHOLD}")
    if not new_points_ok:
        reasons.append(
            f"Not enough new points: {new_points_ratio:.2f} < {MINIMUM_NEW_POINTS_RATIO}")

    if not quality_ok:
        reasons.append(
            f"Tracking quality too low: {tracking_quality:.2f} < {MINIMUM_KEYFRAME_QUALITY_THRESHOLD}")

    # Keyframe insertion criteria
    critical_conditions = [baseline_ok]
    quality_conditions = [new_points_ok, quality_ok]

    # Insert keyframe if:
    # - All critical conditions are met AND at least one quality condition is met
    # - OR tracking quality is very low (emergency keyframe)
    should_insert = (all(critical_conditions) and any(
        quality_conditions))

    if should_insert:
        debug_log(LOG_TAG, "Keyframe statistics: "
                  f"Baseline: {baseline_distance:.3f}, "
                  f"New Points Ratio: {new_points_ratio:.2f}, "
                  f"Tracking Quality: {tracking_quality:.2f}, ")
        return True, "Quality criteria met"
    else:
        return False, "; ".join(reasons)

# Testing


def set_dynamic_triangulation_depths(cur_frame, prev_frame, pts_3d, valid_indices):
    # Extract depths for valid points only
    valid_depths = np.array([pts_3d[i][2] for i in valid_indices])
    # Filter out non-positive depths
    valid_depths = valid_depths[valid_depths > 0]
    if len(valid_depths) > 0:
        min_depth, max_depth = np.percentile(
            valid_depths, 20), np.percentile(valid_depths, 75)
        slam.utils.constants.MIN_DEPTH = max(
            0.0, min_depth)
        slam.utils.constants.MAX_DEPTH = max_depth

        # 2. Update baseline ratio constraints (new functionality)
    if len(valid_indices) > 10:  # Need sufficient samples for reliable statistics
        baseline_distance = compute_baseline_distance(cur_frame, prev_frame)

        # Compute baseline ratios for all valid points
        baseline_ratios = []

        for idx in valid_indices:
            point_3d = pts_3d[idx][:3]
            point_distance = np.linalg.norm(point_3d - cur_frame.camera_center)

            if point_distance > 0:
                baseline_ratio = baseline_distance / point_distance
                baseline_ratios.append(baseline_ratio)

        # Update baseline ratio threshold using MAD
        if len(baseline_ratios) > 5:
            baseline_ratios = np.array(baseline_ratios)
            median_ratio = np.median(baseline_ratios)
            # Set minimum baseline ratio as median - 2*MAD_STD, but with reasonable bounds
            dynamic_min_ratio = np.percentile(baseline_ratios, 45)
            slam.utils.constants.MIN_BASELINE_RATIO = max(
                0.01,  # Absolute minimum (1:200 ratio)
                min(dynamic_min_ratio, 0.1)  # Don't make it too restrictive
            )

        debug_log(
            LOG_TAG,
            f"Dynamic triangulation constraints updated:\n"
            f"  Depth: min={slam.utils.constants.MIN_DEPTH:.3f}m, max={slam.utils.constants.MAX_DEPTH:.3f}m\n"
            f"  Baseline ratio: min={slam.utils.constants.MIN_BASELINE_RATIO:.4f} "
            f"(from {len(baseline_ratios)} ratios, median={median_ratio:.4f}, percentile={dynamic_min_ratio:.4f})\n"
            f"  Stats: baseline_distance={baseline_distance:.3f}m, "
            f"  depth_range=[{np.min(valid_depths):.3f}, {np.max(valid_depths):.3f}]m"
        )


def visualize_matches(frame1, frame2, matches=None, save_path=None):
    if matches is None:
        matches = frame1.matches
    if matches is None or len(matches) == 0:
        return

    img1 = frame1.image
    img2 = frame2.image
    kp1 = frame1.keypoints
    kp2 = frame2.keypoints

    img_matches = cv.drawMatches(
        img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if save_path:
        cv.imwrite(save_path, img_matches)
    else:
        cv.imshow('Matches', img_matches)
        cv.waitKey(0)
        cv.destroyAllWindows()
