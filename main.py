import cv2 as cv
from utils import *
from renderer import Renderer
from state_estimator import StateEstimator
from feature_extractor import FeatureExtractor
from config.parser import Parser
from frame import Frame
from map import Map


if __name__ == '__main__':
    VIDEO = 'swiz'
    config = Parser('config/config.yaml')

    cap, K = load_video(VIDEO, config)

    feature_extractor = FeatureExtractor()
    state_estimator = StateEstimator()
    global_map = Map()
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

        cur_kps, cur_descs = feature_extractor.extract(gray_img)
        keypoint_img = draw_keypoints(gray_img, cur_kps)

        cur_frame = Frame(frame, K)
        cur_frame.set_features(cur_kps, cur_descs)

        # Update state estimator with new features
        state_estimator.update(cur_frame)

        # If we have at least 4 frames, triangulate points and estimate camera pose
        if count > 1:
            #  Triangulate points
            points = state_estimator.triangulate()
            Rt = state_estimator.get_camera_pose()

            # Update global map with new points
            global_map.add_points(points)

            # Render point cloud and camera poses
            # TODO: We should re-render the entire point cloud based on the current state of the map
            # This way the renderer does not need to keep track of points that might be removed in the future.
            #
            # The current poses should also be updated based on the current keyframe provided by the map.
            renderer.update_points(points)
            renderer.update_camera(Rt)

        # stateEstimator.visualize_matches(frame)
        # cv.imshow('raw_keypoints', keypoint_img)
        cv.imshow('frame', frame)
        prev_img = frame
