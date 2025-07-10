"""
Takes a set of extracted features from two frames and uses a matching algorithm of choice to match corresponding features

"""
import cv2 as cv
import numpy as np
from utils import *
from constants import *


def match_features(prev_frame, cur_frame, ratio_thresh):
    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    prev_kp, prev_desc = prev_frame.keypoints_descriptors()
    cur_kp, cur_desc = cur_frame.keypoints_descriptors()

    # match descriptors of the two images
    matches = matcher.knnMatch(prev_desc, cur_desc, k=KNN_K_VALUE)

    idx1, idx2 = [], []
    idx1s, idx2s = set(), set()

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh*n.distance:
            if m.distance < LOWE_DISTANCE_THRESHOLD:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                idx1s.add(m.queryIdx)
                idx2s.add(m.trainIdx)
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

        return filtered_matches, E

    return good_matches


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
