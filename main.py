import cv2 as cv
from utils import *
from renderer import Renderer
from state_estimator import StateEstimator
from feature_extractor import FeatureExtractor
from config.parser import Parser
from frame import Frame
from map import Map


if __name__ == '__main__':
    VIDEO = 'calibrated1'
    config = Parser('config/config.yaml')

    cap, K = load_video(VIDEO, config)

    feature_extractor = FeatureExtractor()
    state_estimator = StateEstimator()
    global_map = Map()
    renderer = Renderer(K)

    renderer.start()

    prev_img = None
    frame_count = 0
    keyframe_count = 0
    while True:
        if cv.waitKey(1) == ord('q'):
            renderer.stop()
            break
        frame_count += 1
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Extract features
        gray_img = cv.cvtColor(frame, cv.IMREAD_GRAYSCALE)

        cur_kps, cur_descs = feature_extractor.extract(gray_img)
        keypoint_img = draw_keypoints(gray_img, cur_kps)

        cur_frame = Frame(frame, K, frame_id=frame_count)
        cur_frame.set_features(cur_kps, cur_descs)

        # Update state estimator with new features
        state_estimator.update(cur_frame, global_map)

        # If we have at least 2 frames, triangulate points and estimate camera pose
        # TODO: This should only run if we have inserted a new keyframe
        if keyframe_count != len(global_map.keyframes):
            #  Triangulate points
            points = state_estimator.triangulate()
            Rt = state_estimator.get_camera_pose()

            # Update global map with new points
            global_map.add_points(points)

            # Render point cloud and camera poses
            renderer.update_points(global_map.points)
            renderer.update_camera(Rt)

            keyframe_count = len(global_map.keyframes)

        # stateEstimator.visualize_matches(frame)
        # cv.imshow('raw_keypoints', keypoint_img)
        cv.imshow('frame', frame)
        prev_img = frame
