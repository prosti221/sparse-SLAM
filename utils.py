import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation
from logger import info_log
from constants import *

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


def compute_reprojection_error(keyframe, point, kp_idx):
    proj = keyframe.K @ point[:3]
    proj = proj[:2] / proj[2]
    kp = keyframe.get_keypoints_descriptors()[0][kp_idx].pt
    error_vec = proj - np.array(kp)
    return np.sum(error_vec ** 2)


def is_valid_triangulated_point(point_idx, cur_keyframe, prev_keyframe, point_4d):
    # Normalize points
    pts1 = normalize(cur_keyframe.matched_pts_prev_frame,
                     prev_keyframe.Kinv)
    pts2 = normalize(cur_keyframe.matched_pts,
                     cur_keyframe.Kinv)
    pt1 = pts1[point_idx]
    pt2 = pts2[point_idx]
    # Get poses in world-to-camera coordinates
    world_to_cam_cur = np.linalg.inv(cur_keyframe.pose)
    world_to_cam_prev = np.linalg.inv(prev_keyframe.pose)

    match = cur_keyframe.matches[point_idx]

    # Check parallax
    parallax = compute_parallax_angle(
        pt1, pt2, world_to_cam_prev, world_to_cam_cur)
    if parallax < np.radians(MINIMUM_PARALLAX_THRESHOLD):
        return False

    # Check depth in both cameras
    pl1 = world_to_cam_cur @ point_4d
    pl2 = world_to_cam_prev @ point_4d
    if not (MIN_DEPTH < pl1[2] < MAX_DEPTH and MIN_DEPTH < pl2[2] < MAX_DEPTH):
        return False

    # Check reprojection error
    reproj_error_cur = compute_reprojection_error(
        cur_keyframe, pl1, match.trainIdx)
    reproj_error_prev = compute_reprojection_error(
        prev_keyframe, pl2, match.queryIdx)

    if reproj_error_cur > MAX_SQUARED_REPROJECTION_ERROR or reproj_error_prev > MAX_SQUARED_REPROJECTION_ERROR:
        return False

    return True


def compute_rotation_angle(Rcw1, Rcw2):
    R_rel = Rcw1 @ Rcw2.T
    angle_axis = Rotation.from_matrix(R_rel).as_rotvec()
    return np.linalg.norm(angle_axis)  # Angle in radians


def compute_parallax_angle(pt1, pt2, pose1, pose2):
    # Convert to bearing vectors
    bearing1 = np.array([pt1[0], pt1[1], 1.0])
    bearing1 /= np.linalg.norm(bearing1)

    bearing2 = np.array([pt2[0], pt2[1], 1.0])
    bearing2 /= np.linalg.norm(bearing2)

    # Transform to world coordinates
    R1 = pose1[:3, :3]
    bearing1_world = R1.T @ bearing1  # Camera to world rotation

    R2 = pose2[:3, :3]
    bearing2_world = R2.T @ bearing2

    # Compute angle between rays
    cos_angle = np.dot(bearing1_world, bearing2_world)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def compute_linear_dlt(prev_keyframe, cur_keyframe):
    ret = np.zeros((cur_keyframe.matched_pts.shape[0], 4))

    # Convert camera-to-world poses to world-to-camera for triangulation
    pose1 = np.linalg.inv(cur_keyframe.pose)  # world-to-camera
    pose2 = np.linalg.inv(prev_keyframe.pose)  # world-to-camera

    pts1 = normalize(cur_keyframe.matched_pts, cur_keyframe.Kinv)
    pts2 = normalize(cur_keyframe.matched_pts_prev_frame, prev_keyframe.Kinv)

    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]

        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]

    return ret
