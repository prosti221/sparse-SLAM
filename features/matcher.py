"""
Takes a set of extracted features from two frames and uses a matching algorithm of choice to match corresponding features

"""
import cv2 as cv
import numpy as np
from utils.utils import *
from utils.constants import *
from utils.logger import debug_log, error_log, warning_log
import time

LOG_TAG = 'Matcher'


def match_features(prev_frame, cur_frame, is_binary_desc=True):
    #matcher = cv.BFMatcher(
    #    cv.NORM_HAMMING if is_binary_desc else cv.NORM_L2, crossCheck=False)
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
    prev_kp, prev_desc = prev_frame.get_keypoints_descriptors()
    cur_kp, cur_desc = cur_frame.get_keypoints_descriptors()

    st_time = time.time()
    matches = matcher.knnMatch(prev_desc, cur_desc, k=2)
    end_time = time.time()
    warning_log(LOG_TAG, f"Time taken for knnMatch: {end_time - st_time:.2f} seconds")

    good_matches = []
    for m, n in matches:
        if m.distance < 0.60 * n.distance:
            good_matches.append(m)

    # Filter using RANSAC
    if len(good_matches) > MATCHER_RANSAC_MINIMUM_INLIERS:
        pts_cur = np.array([cur_kp[m.trainIdx].pt for m in good_matches])
        pts_prev = np.array([prev_kp[m.queryIdx].pt for m in good_matches])

        pts_cur_norm = normalize(pts_cur, cur_frame.Kinv)
        pts_prev_norm = normalize(pts_prev, prev_frame.Kinv)

        E, mask = cv.findEssentialMat(
            pts_prev_norm, pts_cur_norm, cur_frame.K, method=cv.RANSAC, prob=MATCHER_RANSAC_PROBABILITY, threshold=MATCHER_RANSAC_THRESHOLD)
        filtered_matches = [m for i, m in enumerate(
            good_matches) if mask[i] == 1]

        debug_log(
            LOG_TAG, f"Found {len(filtered_matches)} filtered matches, {len(filtered_matches) / len(matches) * 100:.2f}% inliers")
        return filtered_matches, E
    else:
        warning_log(LOG_TAG, "Not enough matches found for RANSAC")

    return matches


def match_features_between_frames(frame1, frame2, is_binary_desc=True):
    matches, E = match_features(frame1, frame2, is_binary_desc)

    if len(matches) == 0:
        warning_log(
            LOG_TAG, f"No matches found between frames: {frame1.frame_id} and {frame2.frame_id}")
        return None, None

    matched_pts = np.float64(
        [frame2.keypoints[m.trainIdx].pt for m in matches])
    prev_frame_matched_pts = np.float64(
        [frame1.keypoints[m.queryIdx].pt for m in matches])

    debug_log(
        LOG_TAG, f"Found {len(matches)} matches between frames: {frame1.frame_id} and {frame2.frame_id}")

    frame2.set_match_data(
        matches, matched_pts, prev_frame_matched_pts)

    return E
