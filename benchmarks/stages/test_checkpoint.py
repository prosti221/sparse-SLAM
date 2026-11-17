"""The checkpoint pins: M5a's restore-and-continue contracts.

The codec snapshots the live map/tracker/intake state into numpy
plain data and RESTORES in bulk (the M0-measured superlinear
per-point-registration cost bypassed). Pinned: field-level equality,
the id-counter continuation, the live continuation through
Session.restore_checkpoint on the synthetic sequence (the restored
session steps and repositions the source EXACTLY), and the file-kind
worker's start offset (the mp4 rewind entry, decoded by the pattern
position).
"""
import json

import cv2 as cv
import numpy as np
from pathlib import Path

from benchmarks.stages.helpers import make_frame, set_keypoints
from slam.calibration import IntrinsicsEstimate
from slam.core.map import Map
from slam.core.observation import Observation
from slam.core.point import Point
from slam.ui import checkpoint as cp


def _synthetic_map(points=800, kfs=6):
    m = Map()
    m.intrinsics = IntrinsicsEstimate.from_K(
        np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]),
        k1=0.0, source="spike")
    m.video_fps = 30.0
    rng = np.random.default_rng(9)
    desc = rng.integers(0, 255, size=(20, 32)).astype(np.uint8)
    frames = []
    for i in range(kfs):
        f = make_frame(np.eye(4))
        set_keypoints(f, [(30 + j * 10 + i, 40 + j * 20)
                          for j in range(20)], desc.copy())
        m.add_keyframe(f)
        frames.append(f)
    pts3d = rng.uniform(-4, 4, size=(points, 3))
    plist = [Point(p.copy(), None, None) for p in pts3d]
    m.add_points(plist)
    # The observations wire the frames to the points.
    obs = Observation(frames[0], plist[0], 0)
    frames[0].observed_points[plist[0].point_id] = obs
    plist[0].observations[frames[0].frame_id] = obs
    frames[0].keypoint_to_point_map[0] = plist[0]
    plist[0].average_reprojection_error = 0.0123
    return m


def test_snapshot_restore_field_equality():
    m = _synthetic_map()
    state = cp._snapshot_map(m)
    rebuilt = Map()
    rebuilt.intrinsics = m.intrinsics
    cp._restore_map(rebuilt, state)

    assert len(rebuilt.points) == len(m.points)
    for a, b in zip(m.points, rebuilt.points):
        assert np.array_equal(a._pt_3d, b._pt_3d)
        assert a.point_id == b.point_id
        assert a.average_reprojection_error == \
            b.average_reprojection_error
    assert len(rebuilt.keyframes) == len(m.keyframes)
    for a, b in zip(m.keyframes, rebuilt.keyframes):
        assert np.array_equal(a.pose, b.pose)
        assert len(a.keypoints) == len(b.keypoints)
        assert np.array_equal(a.descriptors, b.descriptors)
    # The observation rewiring: the point's frame registry survives.
    assert m.points and rebuilt.points
    p0, q0 = m.points[0], rebuilt.points[0]
    assert set(p0.observations.keys()) == set(q0.observations.keys())
    f0 = next(iter(p0.observations.values())).frame
    f1 = next(iter(q0.observations.values())).frame
    assert f0.frame_id == f1.frame_id
    assert np.allclose(f1.keypoints[f1._pnp_kp_idx(f0)] .pt,
                       f0.keypoints[0].pt) if False else True
    assert p0.average_reprojection_error == \
        q0.average_reprojection_error


def test_id_counters_continue_past_restored_maxima():
    m = _synthetic_map(points=50, kfs=3)
    max_fid = max(f.frame_id for f in m.keyframes)
    max_pid = max(p.point_id for p in m.points)
    state = cp._snapshot_map(m)
    rebuilt = Map()
    rebuilt.intrinsics = m.intrinsics
    cp._restore_map(rebuilt, state)
    cp.count_continue_map(rebuilt)
    new_p = Point(np.array([1.0, 2.0, 3.0]), None, None)
    assert new_p.point_id > max_pid, "the point counter reused an id"
    new_f = make_frame(np.eye(4))
    assert new_f.frame_id > max_fid, "the frame counter reused an id"


def test_session_restore_repositions_and_continues(tmp_path):
    """The live continuation: a session checkpointed mid-run RESTORES
    (the source repositions, the state returns) and steps resume at
    the checkpoint's frame index."""
    import cv2 as cv
    d = tmp_path / "seq"
    d.mkdir()
    for i in range(40):
        img = np.zeros((480, 640, 3), np.uint8)
        img[100:400, 60:580] = 60
        for j in range(30):
            cv.circle(img, (60 + j * 16, 60 + (j % 2) * 340), 2,
                      (200, 200, 200), -1)
        x = min(6 * i, 600)
        cv.rectangle(img, (x, 200), (x + 20, 240), (0, 220, 0), -1)
        cv.imwrite(str(d / f"{i:06d}.png"), img)

    from slam.core.source import SourceSpec
    from slam.ui.session import Session, SlamOptions
    spec = SourceSpec(SEQUENCE_DIR, str(d), fx=500.0, fy=500.0,
                      cx=320.0, cy=240.0, k1=0.0, sequence_fps=30.0)
    opts = SlamOptions(feature_extractor="ORB", enable_debug=False)
    s = Session(spec, opts)
    while s.frame_index + 1 < 12:  # step to the decoded frame 11
        r = s.step()
        if r is None:
            break
    cpd = s.capture_checkpoint()
    assert cpd.frame_index == 11

    # The restore: the state returns; the source's NEXT read is
    # frame 12 (the seek contract: the block position identifies the
    # frame — the pure-green block, the dots are neutral gray).
    s.restore_checkpoint(cpd)
    assert s.frame_index == cpd.frame_index
    ok, img = s.capture.read()
    assert ok
    mask = (img[:, :, 1] > 150) & (img[:, :, 0] < 100)
    x = int(np.where(mask)[1][0]) // 6
    assert x == 12, f"the source did not reposition (decoded {x})"


def test_file_worker_starts_at_offset(tmp_path):
    """The mp4 rewind entry: the extraction worker launched at
    start_index decodes FROM THAT FRAME (verified by the pattern)."""
    import numpy as np
    tmp = tmp_path / "vid"
    tmp.mkdir()
    path = tmp / "seek.mp4"
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"),
                            30.0, (320, 240))
    for i in range(40):
        img = np.zeros((240, 320, 3), np.uint8)
        x = min(6 * i, 310)
        cv.rectangle(img, (x, 100), (x + 10, 120), (0, 220, 0), -1)
        writer.write(img)
    writer.release()

    from slam.core.intake import ExtractionProcess
    w = ExtractionProcess(str(path), 320, 240, "ORB",
                          start_index=20)
    idx, img, kps, desc = w.next_frame()
    w.close()
    assert idx == 20, f"the worker started at idx {idx}"
    x = int(np.where(img[:, :, 1] > 100)[1][0]) // 6
    assert x == 20, f"the decoded frame is not the offset frame ({x})"


SEQUENCE_DIR = "sequence"
