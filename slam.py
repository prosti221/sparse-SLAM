import cv2 as cv

from slam.config.config_parser import ConfigParser
from slam.core.frame import Frame
from slam.core.map import Map
from slam.core.tracker import Tracker
from slam.utils.constants import *
from slam.utils.logger import *
from slam.utils.utils import *
from slam.viz.renderer import Renderer

LOG_TAG = "Slam"

if __name__ == "__main__":
    config = ConfigParser("slam/config/config.yaml")
    info_log(LOG_TAG, f"Starting SLAM with parameters: {config}")

    FEATURE_EXTRACTOR = config.get_global_config_property("feature_extractor")
    RECORD_SESSION = config.get_global_config_property("record_session")
    DISPLAY_CAPTURE = config.get_global_config_property("display_capture")
    ENABLE_MULTISCALE = config.get_global_config_property("enable_multiscale_features")
    ENABLE_BA = config.get_global_config_property("enable_ba")

    cap, K = load_video(config)
    CAP_W, CAP_H = (
        int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
    )

    # Phase 2j-2: 'fx: auto' - GeoCalib shared-intrinsics estimation
    # from the first ~CALIB_INIT_SECONDS of video (ViPE-style). The
    # buffered frames were consumed from the capture and are replayed
    # into the loop below; the prior (with its uncertainties) feeds the
    # 2j-4 intrinsics refit once that lands.
    init_frames = None
    intrinsics_prior = None
    if K is None:
        from slam.calibration import estimate_video_intrinsics
        intrinsics_prior, init_frames = estimate_video_intrinsics(cap)
        K = intrinsics_prior.K
        info_log(LOG_TAG,
                 f"Auto-calibrated intrinsics: {intrinsics_prior}")

    global_map = Map()
    renderer = Renderer(global_map, K, W=1920, H=1080)

    tracker = Tracker(global_map, FEATURE_EXTRACTOR, ENABLE_MULTISCALE, ENABLE_BA)

    renderer.start()

    if RECORD_SESSION:
        renderer.start_recording_session()

    slam_in_progress = renderer_is_active = True
    prev_img = np.zeros((CAP_H, CAP_W, 3), dtype=np.uint8)
    while slam_in_progress or renderer_is_active:
        renderer.vis.update_renderer()
        if not renderer.vis.poll_events() or renderer.should_quit():
            renderer_is_active = slam_in_progress = False
            continue

        # Capture and write the current renderer frame if recording is enabled
        if RECORD_SESSION:
            renderer.capture_frame(complement_frame=prev_img.copy())

        if renderer.is_paused() or not slam_in_progress:
            cv.destroyAllWindows()
            continue  # Skip SLAM updates

        if init_frames:
            # Phase 2j-2: replay the auto-calibration init batch
            # (consumed from the capture by estimate_video_intrinsics).
            img = init_frames.pop(0)
        else:
            ret, img = cap.read()
            if not ret:
                info_log(
                    LOG_TAG,
                    f"SLAM processing finished with {len(global_map.keyframes)} keyframes and {len(global_map.points)} points in the map!",
                )
                slam_in_progress = False
                continue

        frame = Frame(img, K)
        is_new_keyframe = tracker.update(frame)

        if is_new_keyframe:
            info_log(
                LOG_TAG,
                f"Updating renderer with {len(global_map.keyframes)} keyframes and {len(global_map.points)} points. Map tracking quality is: {global_map.avg_tracking_quality}",
            )
            renderer.update()

        if DISPLAY_CAPTURE:
            cv.imshow("frame", img)
        prev_img = img

    cv.destroyAllWindows()
    renderer.save_session_data()
    renderer.stop()
