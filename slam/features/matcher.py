import cv2 as cv
import numpy as np
from typing import List
from slam.utils.utils import *
from slam.utils.constants import *
from slam.utils.logger import debug_log, error_log, warning_log
from slam.core.frame import Frame

LOG_TAG = 'Matcher'


def match_features(frame1: Frame, frame2: Frame, feature_extraction_method: str) -> List[cv.DMatch]:
    if feature_extraction_method == DNN_EXTRACTOR_NAME:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    else:
        is_binary_desc = feature_extraction_method in BINARY_DESCRIPTION_METHODS
        matcher = cv.BFMatcher(
            cv.NORM_HAMMING if is_binary_desc else cv.NORM_L2, crossCheck=False)

    f1_kp, f1_desc = frame1.get_keypoints_descriptors()
    f2_kp, f2_desc = frame2.get_keypoints_descriptors()

    matches = matcher.knnMatch(f1_desc, f2_desc, k=2)

    good_matches = []
    indices_f1_set, indices_f2_set = set(), set()
    for m, n in matches:
        if m.distance < LOWE_RATIO * n.distance and m.queryIdx not in indices_f1_set and m.trainIdx not in indices_f2_set:
            indices_f1_set.add(m.queryIdx)
            indices_f2_set.add(m.trainIdx)

            good_matches.append(m)

    # Filter using RANSAC
    if len(good_matches) > MATCHER_RANSAC_MINIMUM_INLIERS:
        pts_f1 = np.array([f1_kp[m.queryIdx].pt for m in good_matches])
        pts_f2 = np.array([f2_kp[m.trainIdx].pt for m in good_matches])

        pts_f1_norm = normalize(pts_f1, frame1.Kinv)
        pts_f2_norm = normalize(pts_f2, frame2.Kinv)

        E, mask = cv.findEssentialMat(
            pts_f1_norm, pts_f2_norm, method=cv.RANSAC, prob=MATCHER_RANSAC_PROBABILITY, threshold=MATCHER_RANSAC_THRESHOLD)

        filtered_matches = [m for i, m in enumerate(
            good_matches) if mask[i] == 1]

        pts_f1_filtered = np.array(
            [f1_kp[m.queryIdx].pt for m in filtered_matches])
        pts_f2_filtered = np.array(
            [f2_kp[m.trainIdx].pt for m in filtered_matches])

        Rt = extractRtFromE(pts_f1_filtered, pts_f2_filtered, E, frame1.K)

        debug_log(
            LOG_TAG, f"Found {len(filtered_matches)} filtered matches, {len(filtered_matches) / len(matches) * 100:.2f}% inliers")
        return filtered_matches, Rt
    else:
        warning_log(LOG_TAG, "Not enough matches found for RANSAC")

    return matches


def match_features_between_frames(frame1: Frame, frame2: Frame, feature_extraction_method: str) -> np.ndarray:
    matches, Rt = match_features(frame1, frame2, feature_extraction_method)

    if len(matches) == 0:
        warning_log(
            LOG_TAG, f"No matches found between frames: {frame1.frame_id} and {frame2.frame_id}")
        return None, None

    debug_log(
        LOG_TAG, f"Found {len(matches)} matches between frames: {frame1.frame_id} and {frame2.frame_id}")

    frame1.set_match_data(matches)

    return Rt
