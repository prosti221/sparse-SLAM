import cv2 as cv
from slam.utils.utils import *
from slam.viz.renderer import Renderer
from slam.core.tracker import Tracker
from slam.config.parser import Parser
from slam.core.map import Map
from slam.utils.logger import *
from slam.utils.constants import *

LOG_TAG = 'Main'

if __name__ == '__main__':
    VIDEO = 'greece'
    config = Parser('slam/config/config.yaml')

    cap, K = load_video(VIDEO, config)

    global_map = Map()
    tracker = Tracker(
        global_map, feature_extraction_method=ORB_EXTRACTOR_NAME)
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

        # Update state estimator with new features
        tracker.update(frame, K)

        # Render the point cloud and camera poses if a new keyframe is detected
        if keyframe_count != len(global_map.keyframes):
            keyframe_count = len(global_map.keyframes)
            info_log(
                LOG_TAG, f"Updating renderer with {len(global_map.points)} points and {len(global_map.keyframes)} keyframes ")
            # Render point cloud and camera poses
            renderer.update_points(global_map.points)
            renderer.update_poses(global_map.keyframes)

            info_log(
                LOG_TAG, f"Map tracking quality is {global_map.get_avg_tracking_quality()}")

        cv.imshow('frame', frame)
        prev_img = frame
