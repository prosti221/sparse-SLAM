import cv2 as cv
from utils import *
from renderer import Renderer
from state_estimator import StateEstimator
from feature_extractor import FeatureExtractor
from config.parser import Parser
from frame import Frame
from map import Map


if __name__ == '__main__':
    VIDEO = 'road'
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
        if cv.waitKey(1) == ord('q'):
            renderer.stop()
            break
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
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
            # Render point cloud and camera poses
            Rt = state_estimator.get_camera_pose()
            renderer.update_points(global_map.points)
            renderer.update_camera(Rt)

            keyframe_count = len(global_map.keyframes)

            print(f"Point cloud size: {len(global_map.points)}")
            print()

        cv.imshow('frame', frame)
        prev_img = frame
