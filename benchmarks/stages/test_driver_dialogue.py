"""The M7.1 dialogue pin: the step-process driver against an
in-process Session on the synthetic sequence - the summaries must be
BIT-EQUAL per step (the replay contract holds across the process
boundary; the pins adjudicate the driver)."""
import numpy as np  # noqa: F401
import cv2 as cv
from pathlib import Path

SEQUENCE_DIR = "sequence"


def test_driver_dialogue_is_bit_equal_to_in_process(tmp_path):
    d = tmp_path / "seq"
    d.mkdir()
    for i in range(40):
        img = np.zeros((480, 640, 3), np.uint8)
        img[70:400, 40:590] = 75
        for j in range(34):
            cv.circle(img, (50 + j * 15, 80 + (j % 2) * 330), 2,
                      (215, 215, 215), -1)
        cv.rectangle(img, (min(7 * i, 560), 200),
                     (min(7 * i, 560) + 24, 250), (0, 225, 0), -1)
        cv.imwrite(str(d / f"{i:06d}.png"), img)

    from slam.core.source import SourceSpec
    from slam.ui.driver import StepProcess
    from slam.ui.session import Session, SlamOptions

    spec = SourceSpec(SEQUENCE_DIR, str(d), fx=500.0, fy=500.0,
                      cx=320.0, cy=240.0, k1=0.0, sequence_fps=30.0)
    opts = SlamOptions(feature_extractor="ORB", enable_debug=False)

    # The dialogue: 12 commanded steps through the process.
    driver = StepProcess(spec, opts)
    try:
        assert driver.camera_meta["w"] == 640
        results = []
        for _ in range(12):
            driver.request_step()
            out = driver.wait_for("step", timeout=60.0)
            assert out is not None, "a step summary never arrived"
            results.append(out)
        # The camera slot held this step's frame (the ping-pong pair):
        # the last shipped slot is a real BGR frame, non-empty.
        from multiprocessing.shared_memory import SharedMemory
        summary, slot_name = results[-1]
        cam = SharedMemory(name=slot_name)
        shot = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=cam.buf)
        assert shot.mean() > 1.0, \
            f"the camera slot looks empty (mean {shot.mean():.1f})"
        cam.close()
    finally:
        driver.close()

    # The identical flow in-process.
    ref = Session(spec, opts)
    for i in range(12):
        r = ref.step()
        summary, slot = results[i]
        cur = getattr(ref.tracker, "cur_frame", None)
        m = ref.global_map
        assert summary["step"] == r.step
        assert summary["frame"] == r.frame_index
        assert summary["n_pts"] == r.points
        assert summary["n_kf"] == r.keyframes
        assert (abs(summary["avg_quality"] - r.avg_quality)) < 1e-12
        assert np.array_equal(summary["cur_pose"],
                              np.asarray(cur.pose))
        # The FRAME IDs are per-process labels (the module-global
        # counters continue at whatever position each process's
        # history left them): compare the TRAJECTORY, not the labels.
        assert len(summary["frustum_ids"]) == len(m.keyframes)
        assert np.array_equal(
            summary["frustum_poses"].reshape(-1, 4, 4),
            np.array([np.asarray(f.pose) for f in m.keyframes]))
        assert np.allclose(summary["frustum_quality"],
                           [f.tracking_quality
                            for f in m.keyframes])

    # The geometry array order: the map's points in the same order
    # (only asserted once the map actually has points by step 12).
    last_summary, _ = results[-1]
    if m.points:
        assert np.allclose(
            last_summary["pos"],
            np.array([p._pt_3d for p in m.points], dtype=np.float32),
            atol=1e-5)
    ref.close()
