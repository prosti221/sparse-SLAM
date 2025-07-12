import cv2 as cv
from utils import *
from renderer import Renderer
from state_estimator import StateEstimator
from feature_extractor import FeatureExtractor
from config.parser import Parser
from frame import Frame
from map import Map
from logger import debug_log, info_log, error_log

LOG_TAG = 'Main'

if __name__ == '__main__':
    VIDEO = 'greece'
    config = Parser('config/config.yaml')

    cap, K = load_video(VIDEO, config)

    feature_extractor = FeatureExtractor()
    state_estimator = StateEstimator()
    global_map = Map()
    renderer = Renderer(K)

    renderer.start()

    prev_img = None
    keyframe_count = 0
    while True:
        renderer.vis.poll_events()
        renderer.vis.update_renderer()

        if renderer.is_paused():
            continue  # Skip SLAM updates

        ret, frame = cap.read()
        if not ret:
            error_log(LOG_TAG, "Can't receive frame (stream end?). Exiting ...")
            break

        # Extract features
        gray_img = cv.cvtColor(frame, cv.IMREAD_GRAYSCALE)

        cur_kps, cur_descs = feature_extractor.extract(gray_img)
        keypoint_img = draw_keypoints(gray_img, cur_kps)

        cur_frame = Frame(frame, K)
        cur_frame.set_features(cur_kps, cur_descs)

        # Update state estimator with new features
        state_estimator.update(cur_frame, global_map)

        # Render the point cloud and camera poses if a new keyframe is detected
        if keyframe_count != len(global_map.keyframes):
            debug_log(
                LOG_TAG, f"Updating visualizer with {len(global_map.points)} points and {len(global_map.keyframes)} keyframes ")
            # Render point cloud and camera poses
            renderer.update_points(global_map.points)
            renderer.update_poses(global_map.keyframes)

            keyframe_count = len(global_map.keyframes)

        cv.imshow('frame', frame)
        prev_img = frame
