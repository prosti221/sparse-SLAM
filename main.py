import cv2 as cv
from utils import *
from renderer import Renderer
from state_estimator import StateEstimator
from feature_extractor import FeatureExtractor
from config.parser import Parser

def load_video(video_name, config):
    VIDEO_PATH = config.get_video_property(video_name, 'path')
    cap = get_video_cap(VIDEO_PATH)
    W = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    F = config.get_video_property(VIDEO, 'focal_length')
    K = construct_K(F, F, W//2, H//2)

    return cap, K

if __name__ == '__main__':
    VIDEO = 'swiz'
    config = Parser('config/config.yaml')

    cap, K = load_video(VIDEO, config)

    featureExtractor = FeatureExtractor()
    stateEstimator = StateEstimator(K)

    renderer = Renderer(K)
    renderer.start()

    prev_img = None
    count = 0 
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
        cur_features = featureExtractor.extract(gray_img)
        keypoint_img = draw_keypoints(gray_img, cur_features[0])

        # Update state estimator with new features
        stateEstimator.update(cur_features, frame)

        # If we have at least 2 frames, triangulate points and estimate camera pose
        if count > 2:
            #  Triangulate points
            points = stateEstimator.triangulate()
            Rt = stateEstimator.get_camera_pose()

            # Render point cloud and camera poses
            renderer.update_points(points)
            renderer.update_camera(Rt)

        #stateEstimator.visualize_matches(frame)
        #cv.imshow('raw_keypoints', keypoint_img)
        cv.imshow('frame', frame)
        prev_img = frame
