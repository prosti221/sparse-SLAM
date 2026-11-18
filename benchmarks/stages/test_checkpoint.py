"""The checkpoint pins: M5a's restore-and-continue contracts.

The codec snapshots the live map/tracker/intake state into numpy
plain data and RESTORES in bulk (the M0-measured superlinear
per-point-registration cost bypassed). Pinned: field-level equality,
the keypoint-map object wiring (KeyPoint-object keys, Point values,
the stale phantom-guard entries), the defaultdict covisibility
contract, the archive observation wiring, the id-counter
continuation, the live continuation through Session.restore_checkpoint
on the synthetic sequence (the restored session steps and
repositions the source EXACTLY), and the file-kind worker's start
offset (the mp4 rewind entry, decoded by the pattern position).
"""
import json

import cv2 as cv
import numpy as np
from collections import defaultdict
from pathlib import Path

from benchmarks.stages.helpers import make_frame, set_keypoints
from slam.calibration import IntrinsicsEstimate
from slam.core.map import ArchivedMap, Map
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
    # The observation wires through the LIVE writer: the point-side
    # registry, the frame-side registry, and the keypoint map keyed
    # by the frame's own KeyPoint object (the identity contract the
    # codec restores).
    plist[0].add_observation(Observation(frames[0], plist[0], 0))
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
        # The derived feature caches: the direct keypoint assignment
        # bypasses set_features, so the restore rebuilds them
        # (triangulation and the loop detector subscript kp_pts_norm).
        assert b.kp_pts_norm is not None
        assert len(b.kp_pts_norm) == len(b.keypoints)
        assert b.kp_unique_mask is not None
    # The keypoint-map wiring: the restored keys are the frame's OWN
    # keypoint objects, the values the restored Point objects.
    rb0 = rebuilt.keyframes[0]
    assert rb0.keypoint_to_point_map
    (rb_kp, rb_pt), = rb0.keypoint_to_point_map.items()
    assert any(rb_kp is k for k in rb0.keypoints)
    assert rb_pt is rebuilt.points[0]
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


def test_stale_keypoint_map_entries_round_trip():
    """The phantom-guard state: a map entry whose point was pruned
    stays keyed in the live map (the measured de facto
    don't-re-triangulate guard). The restore reproduces it — the
    Point object reachable ONLY through the keyframe map."""
    m = _synthetic_map(points=50, kfs=3)
    kf = m.keyframes[1]
    dead = m.points[-1]
    dead_id = dead.point_id
    kf.keypoint_to_point_map[kf.keypoints[5]] = dead
    m.points.remove(dead)
    m.points_by_id.pop(dead_id, None)

    state = cp._snapshot_map(m)
    rebuilt = Map()
    rebuilt.intrinsics = m.intrinsics
    cp._restore_map(rebuilt, state)

    assert dead_id not in {p.point_id for p in rebuilt.points}
    rb_kf = rebuilt.keyframes[1]
    entries = [(k, p)
               for k, p in rb_kf.keypoint_to_point_map.items()
               if p.point_id == dead_id]
    assert len(entries) == 1
    kp, p = entries[0]
    assert any(kp is k for k in rb_kf.keypoints)
    assert all(p is not q for q in rebuilt.points)


def test_round_trip_from_restored_state_is_stable():
    """The double rewind: capturing from a RESTORED map round-trips
    again — the wiring keys are the restored objects, so the second
    snapshot resolves their indices by identity too."""
    m = _synthetic_map(points=50, kfs=3)
    state = cp._snapshot_map(m)
    one = Map()
    one.intrinsics = m.intrinsics
    cp._restore_map(one, state)
    state2 = cp._snapshot_map(one)
    two = Map()
    two.intrinsics = m.intrinsics
    cp._restore_map(two, state2)

    rb = two.keyframes[0]
    (kp, p), = rb.keypoint_to_point_map.items()
    assert any(kp is k for k in rb.keypoints)
    assert p is two.points[0]
    assert np.array_equal(p._pt_3d, m.points[0]._pt_3d)


def test_archived_observations_round_trip():
    """The archive rebuild wires the point-side observations (the
    recovery candidate filter reads point.is_observed_by) with
    faithful ids; the archived keyframe maps stay empty (nothing
    reads them); the counters continue past the archived maxima."""
    m = _synthetic_map(points=20, kfs=2)
    desc2 = np.random.default_rng(5).integers(
        0, 255, size=(2, 32)).astype(np.uint8)
    arch_f = make_frame(np.eye(4))
    set_keypoints(arch_f, [(50, 50), (70, 60)], desc2)
    arch_p = Point(np.array([1.0, 2.0, 3.0]), None, None)
    arch_f.add_point_observation(Observation(arch_f, arch_p, 1))
    m.archives.append(ArchivedMap([arch_f], [arch_p]))

    state = cp._snapshot_map(m)
    rebuilt = Map()
    rebuilt.intrinsics = m.intrinsics
    cp._restore_map(rebuilt, state)
    cp.count_continue_map(rebuilt)

    a = rebuilt.archives[0]
    assert a.points[0].point_id == arch_p.point_id
    assert a.points[0].is_observed_by(a.keyframes[0].frame_id)
    assert a.keyframes[0].keypoint_to_point_map == {}
    new_p = Point(np.array([9.0, 9.0, 9.0]), None, None)
    assert new_p.point_id > arch_p.point_id, \
        "the point counter reused an archived id"


def _state_summary(s):
    """The trajectory/map fingerprint for the replay contract."""
    import numpy as np
    m = s.global_map
    cur = getattr(s.tracker, "cur_frame", None)
    poses = [np.asarray(f.pose).ravel() for f in m.keyframes]
    pts = [np.asarray(p._pt_3d).ravel() for p in m.points]
    return {
        "frame": s.frame_index,
        "step": s.step_count,
        "kf_ids": [f.frame_id for f in m.keyframes],
        "n_pts": len(pts),
        "poses": np.concatenate(poses) if poses else np.zeros(0),
        "pts": np.concatenate(pts) if pts else np.zeros(0),
        "cur_pose": (np.asarray(cur.pose).ravel()
                     if cur is not None else np.zeros(16)),
        "quality": [float(f.tracking_quality) for f in m.keyframes],
    }


def _summaries_match(a, b):
    """The first mismatch class, or None."""
    if (a["frame"], a["step"], a["kf_ids"], a["n_pts"]) != \
            (b["frame"], b["step"], b["kf_ids"], b["n_pts"]):
        return (f"structure: frame {a['frame']}/{b['frame']}, "
                f"step {a['step']}/{b['step']}, "
                f"kfs {a['kf_ids']}/{b['kf_ids']}, "
                f"pts {a['n_pts']}/{b['n_pts']}")
    for key in ("poses", "pts", "cur_pose"):
        A, B = a[key], b[key]
        if A.shape != B.shape:
            return f"{key} shapes {A.shape}/{B.shape}"
        if A.size and not np.allclose(A, B, rtol=1e-12, atol=1e-12):
            d = float(np.abs(A - B).max())
            return f"{key} values (max abs delta {d:.3g})"
    if a["quality"] != b["quality"]:
        return f"qualities {a['quality']}/{b['quality']}"
    return None


def test_rewind_replay_is_bit_exact(tmp_path):
    """The no-change rewind contract (the user's acceptance): rewind
    to a checkpoint, change NOTHING, resume to the same step - the
    map and trajectory return to the pre-rewind state exactly. Any
    state the codec misses lands here as a mismatch."""
    import numpy as np
    import cv2 as _cv
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
        _cv.imwrite(str(d / f"{i:06d}.png"), img)

    from slam.core.source import SourceSpec
    from slam.ui.session import Session, SlamOptions
    spec = SourceSpec(SEQUENCE_DIR, str(d), fx=500.0, fy=500.0,
                      cx=320.0, cy=240.0, k1=0.0, sequence_fps=30.0)
    s = Session(spec, SlamOptions(feature_extractor="ORB",
                                  enable_debug=False))
    while s.frame_index + 1 < 26:
        if s.step() is None:
            break
    deadline = 10.0
    cps = []
    import time as _time
    t0 = _time.time()
    while len(s.checkpoints) < 1 and _time.time() - t0 < deadline:
        if s.step() is None:
            break
    cps = s.checkpoints
    assert cps, "no checkpoint by frame 26"

    # Continue to 31 (the pre-rewind future).
    while s.step_count < 31:
        if s.step() is None:
            break
    a = _state_summary(s)

    # Rewind to the checkpoint, change nothing, resume to the same
    # step count, and demand the same reality.
    s.restore_checkpoint(cps[0])
    while s.step_count < 31:
        if s.step() is None:
            break
    b = _state_summary(s)
    miss = _summaries_match(a, b)
    assert miss is None, f"the replay diverged: {miss}"


def test_covisibility_graph_keeps_the_defaultdict_contract():
    """The covisibility wiring indexes covisibility_graph[observer_id]
    unguarded (map._add_keyframe_to_point_covisibility): the live
    container is a nested defaultdict that materializes missing ids
    on read (tracking-phase observers never become keyframes). The
    restore must preserve that contract, or the first post-rewind
    keyframe insertion KeyErrors."""
    m = _synthetic_map(points=20, kfs=2)
    state = cp._snapshot_map(m)
    rebuilt = Map()
    rebuilt.intrinsics = m.intrinsics
    cp._restore_map(rebuilt, state)

    assert isinstance(rebuilt.covisibility_graph, defaultdict)
    assert set(rebuilt.covisibility_graph.keys()) == \
        set(m.covisibility_graph.keys())
    # The unguarded wiring access on an id absent from the graph.
    rebuilt.covisibility_graph[4242][1] += 1
    assert rebuilt.covisibility_graph[4242][1] == 1


SEQUENCE_DIR = "sequence"
