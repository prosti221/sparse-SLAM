import cv2 as cv
import numpy as np
from utils import *
from renderer import Renderer
from stateEstimator import StateEstimator
import matplotlib.pyplot as plt

def extract_features(frame):
    """
    Extracts ORB features from a given frame
    """
    detector = cv.ORB_create() 
    pts = cv.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 4000, qualityLevel=0.01, minDistance=7)

    # extraction
    kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    keypoints, descriptors = detector.compute(frame, kps)

    return keypoints, descriptors

if __name__ == '__main__':
    VIDEO_PATH = './videos/vid2.mp4'
    #VIDEO_PATH = './videos/vid2_rev.mp4'
    #VIDEO_PATH = './videos/drone.mp4'

    cap = get_video_cap(VIDEO_PATH)
    W = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    K = construct_K(620, 620, W//2, H//2)

    renderer = Renderer(K)
    renderer.start()

    stateEstimator = StateEstimator(K)

    prev_img = None
    count = 0 
    orient_camera = True
    while True:
        if cv.waitKey(1) == ord('q'):
            renderer.stop()
            break
        count += 1
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Extract features
        gray_img = cv.cvtColor(frame, cv.IMREAD_GRAYSCALE)
        keypoints, descriptors = extract_features(gray_img)
        keypoint_img = draw_keypoints(gray_img, keypoints)
        cur_features = (keypoints, descriptors)

        # Update state estimator with new features
        stateEstimator.update(cur_features)

        # If we have at least 2 frames, triangulate points and estimate camera pose
        if count > 2:

            #  Triangulate points
            points = stateEstimator.triangulate()
            Rt = stateEstimator.get_camera_pose()

            # Render point cloud and camera poses
            renderer.update_points(points)
            renderer.update_camera(Rt)

        #stateEstimator.visualize_matches(frame)
        cv.imshow('raw_keypoints', keypoint_img)

        prev_img = frame


    