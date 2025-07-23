from slam.utils.utils import *
from slam.viz.renderer import Renderer
from slam.core.tracker import Tracker
from slam.config.parser import Parser
from slam.core.map import Map
from slam.core.frame import Frame
from slam.utils.logger import *
from slam.utils.constants import *
import cv2 as cv

LOG_TAG = 'Main'

# TODO: Add this as part of the global config
RECORD_SESSION = True

if __name__ == '__main__':
    config = Parser('slam/config/config.yaml')
    info_log(LOG_TAG, f"Starting SLAM with parameters: {config}")

    FEATURE_EXTRACTOR = config.get_global_config_property("feature_extractor")
    RECORD_SESSION = config.get_global_config_property("record_session")
    ENABLE_MULTISCALE = config.get_global_config_property(
        "enable_multiscale_features")

    cap, K = load_video(config)

    global_map = Map()

    renderer = Renderer(global_map, K)

    tracker = Tracker(
        global_map, FEATURE_EXTRACTOR, ENABLE_MULTISCALE)

    renderer.start()

    if RECORD_SESSION:
        renderer.start_recording_session()

    while True:
        renderer.vis.poll_events()
        renderer.vis.update_renderer()

        if renderer.is_paused():
            continue  # Skip SLAM updates

        ret, img = cap.read()
        if not ret:
            error_log(LOG_TAG, "Can't receive frame (stream end?).")
            renderer.stop()
            break

        frame = Frame(img, K)

        # Update state estimator with new features
        is_new_keyframe = tracker.update(frame)

        # Render the point cloud and camera poses if a new keyframe is detected
        if is_new_keyframe:
            info_log(
                LOG_TAG, f"Updating renderer with {len(global_map.points)} points and {len(global_map.keyframes)} keyframes. Map tracking quality is: {global_map.avg_tracking_quality}")
            renderer.update()

        cv.imshow('frame', img)
        # Save session data and stop the renderer

    renderer.save_session_data()
    renderer.stop()
