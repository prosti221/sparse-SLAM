import cv2 as cv
import numpy as np

from slam.calibration import IntrinsicsEstimate
from slam.config.config_parser import ConfigParser
from slam.core.frame import Frame
from slam.core import intake as intake_mod
from slam.core.intake import FrameIntakeGate
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

    # The source resolver returns (cap, K|None, k1): K is None
    # only on the `fx: auto` sentinel, where the GeoCalib
    # branch below estimates (and now writes through to the
    # intrinsics cache); a cache or config hit serves K directly.
    # The intrinsics provenance flow (M1b): the source opens through
    # the resolver; a cache hit serves the stored intrinsics (no
    # re-estimation on restart); the GeoCalib branch writes through on
    # estimation; a config-intrinsics miss seeds the cache
    # (provenance=config).
    from slam.core.source import open_source, spec_from_config
    from slam.core import intrinsics_cache
    spec = spec_from_config(config)
    resolved = open_source(spec)
    if resolved is None:
        raise FileNotFoundError(f"Could not open the source: {spec.location}")
    cap, K, k1, _source_meta = resolved
    cached = intrinsics_cache.lookup(spec.location)
    if cached is not None:
        K = intrinsics_cache.cached_K(cached)
        k1 = float(cached.get("k1", 0.0))
        info_log(LOG_TAG,
                 f"Intrinsics cache hit ({cached.get('provenance')}): "
                 f"fx={cached.get('fx')} fy={cached.get('fy')}")
    CAP_W, CAP_H = (
        int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
    )

    # 'fx: auto' - GeoCalib shared-intrinsics estimation
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
        k1_prior = float(getattr(intrinsics_prior, "k1", 0.0))
        info_log(LOG_TAG,
                 f"Auto-calibrated intrinsics: {intrinsics_prior}")
        if intrinsics_cache.lookup(spec.location) is None:
            intrinsics_cache.store(
                spec.location, "geocalib", fx=float(K[0, 0]),
                fy=float(K[1, 1]), cx=float(K[0, 2]),
                cy=float(K[1, 2]), k1=k1_prior)
        k1 = 0.0
    elif intrinsics_cache.lookup(spec.location) is None:
        intrinsics_cache.store(spec.location, "config",
                               **_source_meta["intrinsics"])

    global_map = Map()
    # The live intrinsics estimate. From the GeoCalib
    # prior when 'fx: auto' ran; otherwise from the provided K (a
    # static estimate unless the refit is enabled). The frame loop
    # reads it every frame.
    if intrinsics_prior is not None:
        global_map.intrinsics = IntrinsicsEstimate.from_prior(
            intrinsics_prior)
    else:
        global_map.intrinsics = IntrinsicsEstimate.from_K(
            K, k1=k1, source="config")
    # The capture's fps, for the TIME normalization of the
    # frame-count cadence constants (recovery window, rotation-gap
    # floor, forced-insert backstop). 30.0 is the identity anchor;
    # cap.get returns 0.0 when the container reports none.
    container_fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    global_map.video_fps = container_fps
    # Displacement-gated frame intake: feed
    # only frames that carry measurable motion signal, so the pose
    # chain never noise-fits on sub-pixel displacement. Default ON;
    # 'enable_adaptive_frame_intake: false' in the config disables it
    # for A/B. The benchmark harness feeds frame FILES and never
    # constructs a gate - the frozen battery is untouched.
    intake_gate = None
    if config.get_global_config_property("enable_adaptive_frame_intake", True):
        intake_gate = FrameIntakeGate(
            FEATURE_EXTRACTOR, ENABLE_MULTISCALE, container_fps)

    # The extraction process: the worker owns the video source and
    # runs decode+extract ONE AHEAD (unconditional with the gate since
    # the C7b adoption); constructed before the GeoCalib init batch so
    # the worker's torch import + model load overlap it. The worker
    # decodes the same stream from 0 (its own capture), so the replay
    # frames line up with its items. The construction decision reads
    # the worker-thread bound once and pins the whole run's decode
    # path.
    extract_pipeline = None
    if intake_gate is not None:
        from slam.core.intake import ExtractionProcess
        video_path = config.get_video_property(
            config.get_global_config_property("load_video"), 'path')
        extract_pipeline = ExtractionProcess(
            video_path, CAP_W, CAP_H, FEATURE_EXTRACTOR, ENABLE_MULTISCALE,
            worker_threads=intake_mod.INTAKE_EXTRACT_WORKER_THREADS)
    renderer = Renderer(global_map, K, W=1920, H=1080)

    tracker = Tracker(global_map, FEATURE_EXTRACTOR, ENABLE_MULTISCALE, ENABLE_BA)

    # The rotation-dominant pair-pose gate is
    # VIDEO-PATH scoped and - the measured part - tied to the INTAKE
    # GATE's cadence: the t-coherence signal only exists when fed
    # pairs carry >= FRAME_INTAKE_MIN_DISPLACEMENT_PX of matched-
    # feature displacement (above the ORB quantization noise); at a
    # stride-1 every-frame feed (the harness, or the config's
    # 'enable_adaptive_frame_intake: false') the signal is not
    # measurable (13c: the synthetic stride-1 sequences read
    # incoherent on real translation). The benchmark harness never
    # reaches this line - the frozen battery is bit-identical by
    # construction.
    tracker.rotation_model_enabled = intake_gate is not None

    renderer.start()

    if RECORD_SESSION:
        renderer.start_recording_session()

    slam_in_progress = renderer_is_active = True
    prev_img = np.zeros((CAP_H, CAP_W, 3), dtype=np.uint8)
    while slam_in_progress or renderer_is_active:
        renderer.vis.update_renderer()
        if not renderer.vis.poll_events():
            renderer_is_active = slam_in_progress = False
            continue

        # Fly camera: apply held-key movement for this iteration. Runs
        # in every loop state (paused, gate-skipping, post-SLAM - all
        # of them re-enter here); dt is wall-clock based. Quit is
        # Open3D's native Esc / window close (poll_events returns False).
        renderer.process_fly_input()

        # Capture and write the current renderer frame if recording is enabled
        if RECORD_SESSION:
            renderer.capture_frame(complement_frame=prev_img.copy())

        if renderer.is_paused() or not slam_in_progress:
            cv.destroyAllWindows()
            continue  # Skip SLAM updates

        if extract_pipeline is not None:
            item = extract_pipeline.next_frame()
            if item is None:
                info_log(
                    LOG_TAG,
                    f"SLAM processing finished with {len(global_map.keyframes)} keyframes and {len(global_map.points)} points in the map!",
                )
                extract_pipeline.close()
                slam_in_progress = False
                continue
            img, preextracted = item[1], (item[2], item[3])
            if init_frames:
                # Replay phase: the pipeline item's stream position IS
                # the replayed batch frame (the worker decodes the same
                # capture from 0); the batch entry only marks the phase.
                init_frames.pop(0)
        elif init_frames:
            # Replay the auto-calibration init batch
            # (consumed from the capture by estimate_video_intrinsics).
            img = init_frames.pop(0)
            preextracted = None
        else:
            ret, img = cap.read()
            if not ret:
                info_log(
                    LOG_TAG,
                    f"SLAM processing finished with {len(global_map.keyframes)} keyframes and {len(global_map.points)} points in the map!",
                )
                slam_in_progress = False
                continue
            preextracted = None

        if intake_gate is not None:
            if not intake_gate.should_feed(img, preextracted):
                continue

        frame = Frame(img, global_map.intrinsics.K,
                       global_map.intrinsics.k1)
        if intake_gate is not None:
            # The gate's extraction of THIS frame is the tracker's
            # (same gray image; [26c] dedup - the [26a] probe measured
            # every fed frame paying the extraction twice). The
            # INTAKE_CARRY_EXTRACTION flag 0 keeps the pre-dedup
            # behavior for the A/B.
            if intake_mod.INTAKE_CARRY_EXTRACTION:
                frame.set_features(*intake_gate.last_extraction)
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
    if extract_pipeline is not None:
        # Covers the mid-run exits (the EOS break already closed it -
        # close() is idempotent); the worker is a daemon, so a hard
        # crash cannot leave it running.
        extract_pipeline.close()
    renderer.save_session_data()
    renderer.stop()
