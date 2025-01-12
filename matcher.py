"""
Takes a set of extracted features from two frames and uses a matching algorithm of choice to match corresponding features

"""
import cv2 as cv
import numpy as np
from utils import *

def match_features(prev_features, cur_features, K, matcher_type='bf', ratio_thresh=0.75):
    if matcher_type == 'bf':
        # initialize a Brute-Force Matcher
        matcher = cv.BFMatcher(cv.NORM_HAMMING)

    elif matcher_type == 'flann':
        # initialize a FLANN Matcher
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
        search_params = dict(checks=50)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError(f"Invalid matcher type: {matcher_type}")

    prev_kp, prev_desc = prev_features
    cur_kp, cur_desc = cur_features
    Kinv = np.linalg.inv(K)

    # match descriptors of the two images
    matches = matcher.knnMatch(prev_desc, cur_desc, k=2)

    idx1, idx2 = [], []
    idx1s, idx2s = set(), set()

    good_matches = []
    for m,n in matches:
        if m.distance < ratio_thresh*n.distance:
            if m.distance < 32:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                idx1s.add(m.queryIdx)
                idx2s.add(m.trainIdx)
                good_matches.append(m)

    # Filter using RANSAC
    if len(good_matches) > 8:
        pts_cur = np.array([cur_kp[m.trainIdx].pt for m in good_matches])
        pts_prev = np.array([prev_kp[m.queryIdx].pt for m in good_matches])

        pts_cur_norm = normalize(pts_cur, Kinv)
        pts_prev_norm = normalize(pts_prev, Kinv)

        # Using the Fundamental Matrix seems to get better results for the initial frames
        F, mask = cv.findFundamentalMat(pts_prev_norm, pts_cur_norm, cv.FM_RANSAC, 0.005, 0.999) 

        #E, mask = cv.findEssentialMat(pts_prev_norm, pts_cur_norm, K, method=cv.RANSAC, prob=0.999, threshold=0.003)
        filtered_matches = [m for i, m in enumerate(good_matches) if mask[i] == 1]

        print(f"Number of raw points: {len(good_matches)}")
        print(f"Number of filtered points: {len(filtered_matches)}")
        print()
        
        #return filtered_matches, E
        return filtered_matches, F 
    
    return good_matches
