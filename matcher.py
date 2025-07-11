"""
Takes a set of extracted features from two frames and uses a matching algorithm of choice to match corresponding features

"""
import cv2 as cv
import numpy as np
from utils import *
from constants import *


def match_features(prev_frame, cur_frame, ratio_thresh):
    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    prev_kp, prev_desc = prev_frame.get_keypoints_descriptors()
    cur_kp, cur_desc = cur_frame.get_keypoints_descriptors()

    # match descriptors of the two images
    matches = matcher.knnMatch(prev_desc, cur_desc, k=KNN_K_VALUE)

    # Filter using RANSAC
    if len(matches) > MATCHER_RANSAC_MINIMUM_INLIERS:
        pts_cur = np.array([cur_kp[m.trainIdx].pt for m, _ in matches])
        pts_prev = np.array([prev_kp[m.queryIdx].pt for m, _ in matches])

        pts_cur_norm = normalize(pts_cur, cur_frame.Kinv)
        pts_prev_norm = normalize(pts_prev, prev_frame.Kinv)

        E, mask = cv.findEssentialMat(
            pts_prev_norm, pts_cur_norm, cur_frame.K, method=cv.RANSAC, prob=MATCHER_RANSAC_PROBABILITY, threshold=MATCHER_RANSAC_THRESHOLD)
        filtered_matches = [m for i, (m, _) in enumerate(
            matches) if mask[i] == 1]

        return filtered_matches, E

    return matches


def match_features_between_frames(frame1, frame2):
    matches, E = match_features(frame1, frame2, LOWE_RATIO)

    if len(matches) == 0:
        print("[Matcher] No matches found.")
        return None, None

    matched_pts = np.float64(
        [frame2.keypoints[m.trainIdx].pt for m in matches])
    prev_frame_matched_pts = np.float64(
        [frame1.keypoints[m.queryIdx].pt for m in matches])

    frame2.set_match_data(
        matches, matched_pts, prev_frame_matched_pts)

    return E
