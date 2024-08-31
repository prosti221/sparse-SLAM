import cv2 as cv
import numpy as np
from utils import *

def extract_features(frame):
    """
    Extracts ORB features from a given frame
    """
    detector = cv.ORB_create() 
    keypoints, descriptors = detector.detectAndCompute(frame, None)

    return keypoints, descriptors

def match_features(prev_features, cur_features, matcher_type='bf', ratio_thresh=0.75):
    if matcher_type == 'bf':
        # initialize a Brute-Force Matcher
        #matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matcher = cv.BFMatcher()

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

    # match descriptors of the two images
    matches = matcher.knnMatch(prev_desc, cur_desc, k=2)
    # filter matches using Lowe's ratio test

    good_matches = []
    # Lowe's ratio test sucks
    for m,n in matches:
        if m.distance < ratio_thresh*n.distance:
            good_matches.append(m)
    
    return good_matches

def estimate_pose(prev_features, cur_features, matches, ransac=False):
    """
    Estimate the Essential matrix and pose using the 5-point algorithm
    """
    prev_kp, _ = prev_features
    cur_kp, _ = cur_features
    K = construct_K(200, 200, 0, 0)
    Kinv = np.linalg.inv(K)

    prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    cur_pts = np.float32([cur_kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    """
    good_points = []
    if ransac: 
        # Filter using RANSAC
        if len(matches) >= 4:
            homography, mask = cv.findHomography(prev_pts, cur_pts, cv.RANSAC, 7.0)
            matches_mask = mask.ravel().tolist()
            good_points = [cur_kp[m.trainIdx].pt for i, m in enumerate(matches) if matches_mask[i]]
            #good_matches = [m for i, m in enumerate(matches) if matches_mask[i]]

    #print(f"Number of raw points: {len(matches)}")
    #print(f"Number of filtered points: {len(good_points)}")
    """

    E, mask = cv.findEssentialMat(prev_pts, cur_pts, K, cv.RANSAC, prob=.99999, threshold=.1)
    F = Kinv.T @ E @ Kinv # solve for fundamental matrix
    _, R, t, _ = cv.recoverPose(E, prev_pts, cur_pts, K, mask=mask)

    prev_pts = prev_pts[mask.ravel() == 1]
    cur_pts = cur_pts[mask.ravel() == 1]
    
    return F, E, R, t, prev_pts, cur_pts 

def triangulate():
    pass


if __name__ == '__main__':
    VIDEO_PATH = './videos/vid1.mp4'
    cap = get_video_cap(VIDEO_PATH)
    prev_features = None
    prev_img = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray_img = cv.cvtColor(frame, cv.IMREAD_GRAYSCALE)
        keypoints, descriptors = extract_features(gray_img)
        keypoint_img = draw_keypoints(gray_img, keypoints)

        if prev_features is not None:
            cur_features = (keypoints, descriptors)
            good_matches = match_features(prev_features, cur_features)
            #match_img = draw_matches(prev_img, prev_features[0], frame, cur_features[0], good_matches)
            F, E, R, t, prev_pts, cur_pts = estimate_pose(prev_features, cur_features, good_matches, ransac=True)
            

        #if prev_features: cv.imshow('matches', match_img)
        cv.imshow('raw_keypoints', keypoint_img)
        prev_features = (keypoints, descriptors)
        prev_img = frame
        if cv.waitKey(1) == ord('q'):
            break
        