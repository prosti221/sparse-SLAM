from slam.utils.utils import *
from slam.viz.renderer import Renderer
from slam.core.tracker import Tracker
from slam.config.parser import Parser
from slam.core.map import Map
from slam.core.frame import Frame
from slam.utils.logger import *
from slam.utils.constants import *
import cv2 as cv

LOG_TAG = 'Slam'

if __name__ == '__main__':
    config = Parser('slam/config/config.yaml')
    info_log(LOG_TAG, f"Starting SLAM with parameters: {config}")

    FEATURE_EXTRACTOR = config.get_global_config_property("feature_extractor")
    RECORD_SESSION = config.get_global_config_property("record_session")
    DISPLAY_CAPTURE = config.get_global_config_property("display_capture")
    ENABLE_MULTISCALE = config.get_global_config_property(
        "enable_multiscale_features")
    ENABLE_BA = config.get_global_config_property(
        "enable_ba")

    cap, K = load_video(config)
    W, H = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    global_map = Map()
    renderer = Renderer(
        global_map,
        K,
        W=W,
        H=H
    )
    tracker = Tracker(
        global_map, FEATURE_EXTRACTOR, ENABLE_MULTISCALE, ENABLE_BA)

    renderer.start()

    if RECORD_SESSION:
        renderer.start_recording_session()

    slam_in_progress = renderer_is_active = True
    prev_img = np.zeros((H, W, 3), dtype=np.uint8)
    while slam_in_progress or renderer_is_active:
        renderer.vis.update_renderer()
        if not renderer.vis.poll_events():
            renderer_is_active = False
            slam_in_progress = False
            continue

        # Check if quit was requested
        if renderer.should_quit():
            slam_in_progress = renderer_is_active = False
            break

        # Capture and write the current renderer frame if recording is enabled
        if RECORD_SESSION:
            renderer.capture_frame(complement_frame=prev_img.copy())

        # Check if we are in a paused state or if slam is not running
        if renderer.is_paused() or not slam_in_progress:
            cv.destroyAllWindows()
            continue  # Skip SLAM updates

        # Get the next image
        ret, img = cap.read()
        if not ret:
            info_log(
                LOG_TAG, f"SLAM processing finished with {len(global_map.points)} keyframes and {len(global_map.keyframes)} points in the map!")
            slam_in_progress = False
            continue

        # Update state estimator with new frame
        frame = Frame(img, K)
        is_new_keyframe = tracker.update(frame)

        # Render the point cloud and camera poses if a new keyframe is detected
        if is_new_keyframe:
            info_log(
                LOG_TAG, f"Updating renderer with {len(global_map.points)} points and {len(global_map.keyframes)} keyframes. Map tracking quality is: {global_map.avg_tracking_quality}")
            renderer.update()

        if DISPLAY_CAPTURE:
            cv.imshow('frame', img)
        prev_img = img

    # Save session data
    cv.destroyAllWindows()
    renderer.save_session_data()
    renderer.stop()
