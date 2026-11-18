"""The timeline's checkpoint/restore machinery (M5a).

A checkpoint captures the LIVE pipeline state at a frame boundary:
the map (keyframes + points + the observation registries + the
covisibility + the quality/streak state + the pre-reset archives),
the tracker (velocity, counters, the rotation-model EMAs), the intake
gate, and the source stream position. `restore` rebuilds the live
pipeline IN PLACE with a BULK map reconstruction (the per-point
bookkeeping bypassed — the M0-measured 18s superlinear restore is the
alternative this path must beat) and re-positions the source.

Two honesty rules carried from M0's spike:
- the cv global RNG is NOT captured (not exposed in Python): the
  restore re-seeds with an explicit branch seed, making every branch
  resuming from a checkpoint share the same post-restore stream (the
  tweak is the only delta); a rewind is branch exploration, not
  bit-replay;
- the keyframe IMAGES are dropped (their only consumer is the
  tracker's CURRENT frame, which the checkpoint replaces wholesale).

The id counters are module-global itertools.count objects — the
restore continues them past the restored maxima.
"""
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import cv2 as cv
import numpy as np

from slam.core import frame as frame_mod
from slam.core import point as point_mod
from slam.core.frame import Frame
from slam.core.map import ArchivedMap, Map
from slam.core.observation import Observation
from slam.core.point import Point
from slam.loop_closure import KeyframeFeatureDatabase
from slam.utils.logger import info_log

LOG_TAG = "Checkpoint"

KF_KPS_FIELDS = ("pt_x", "pt_y", "size", "angle", "response",
                 "octave", "class_id")


@dataclass
class Checkpoint:
    frame_index: int                 # the decoded-stream position
    step_count: int
    map_state: dict
    tracker_state: dict
    tracker_frame: Optional[dict]
    intake_state: Optional[dict]
    branch_seed: int = 42
    label: str = ""
    forked: bool = False


# ---------------------------------------------------------------- map
def _snapshot_keyframe(f: Frame, live_pids: set) -> dict:
    kps = [(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave,
            k.class_id) for k in (f.keypoints or [])]
    # The live map is {KeyPoint object: Point} (add_point_observation
    # keys with the frame's OWN keypoint objects; _filter_duplicate_
    # keypoints looks them up by identity). The codec stores
    # {keypoint index: point id}, resolved by object identity, and
    # the restore rebuilds the object wiring. Stale entries (their
    # point already pruned while the map entry stayed - the measured
    # phantom-guard behavior) snapshot their Points separately: the
    # objects are reachable ONLY through this map.
    if f.keypoint_to_point_map:
        id_index = {id(k): i for i, k in enumerate(f.keypoints or [])}
        kp_to_point = {id_index[id(k)]: p.point_id
                       for k, p in f.keypoint_to_point_map.items()}
        stale = {p.point_id: p
                 for p in f.keypoint_to_point_map.values()
                 if p.point_id not in live_pids}
    else:
        kp_to_point, stale = {}, {}
    return {
        "frame_id": f.frame_id,
        "pose": np.asarray(f.pose, dtype=float),
        "is_keyframe": f.is_keyframe,
        "keypoints": np.array(kps, dtype=float) if kps else
        np.zeros((0, len(KF_KPS_FIELDS))),
        "descriptors": (np.asarray(f.descriptors)
                        if f.descriptors is not None else None),
        "observed": {pid: obs.kp_idx
                     for pid, obs in f.observed_points.items()},
        "kp_to_point": kp_to_point,
        "stale_points": [_snapshot_point(p) for p in stale.values()],
        "tracking_quality": float(f._tracking_quality),
        "image_dropped": True,
    }


def _snapshot_point(p: Point) -> dict:
    return {
        "point_id": p.point_id,
        "pt_3d": np.asarray(p._pt_3d, dtype=float),
        "color": (np.asarray(p.color) if p.color is not None else None),
        "descriptor": (np.asarray(p.descriptor)
                       if p.descriptor is not None else None),
        "error": float(p.average_reprojection_error),
        # The observation frame-ids rebuild the Observation objects on
        # restore (frame refs first, then point refs).
        "observed_frames": sorted(p.observations.keys()),
    }


def _snapshot_map(m: Map) -> dict:
    live_pids = {p.point_id for p in m.points}
    return {
        "keyframes": [_snapshot_keyframe(f, live_pids)
                      for f in m.keyframes],
        "frames_by_id": [_snapshot_keyframe(f, live_pids) for f
                         in m.frames_by_id.values()],
        "points": [_snapshot_point(p) for p in m.points],
        "quality_history": dict(m.tracking_quality_history),
        "consecutive_low_quality_frames": m.
        consecutive_low_quality_frames,
        "covisibility_graph": {k: dict(v)
                               for k, v in
                               (m.covisibility_graph or {}).items()},
        "covisibility_cache": {str(k): v
                               for k, v in
                               (m._covisibility_score_cache or {})
                               .items()},
        "registry_rows": dict(m._registry_rows or {}),
        "point_coords": list(m.point_coords or set()),
        "extractor_name": m.extractor_name,
        "video_fps": m.video_fps,
        "intrinsics_state": dict(m.intrinsics.__dict__),
        "point_registry": (None if m._point_registry is None or
                           m._point_registry.size() == 0 else
                           {"ids": list(m._point_registry.ids()),
                            "pos": np.array(
                                m._point_registry.positions_view(),
                                dtype=np.float64),
                            "desc": np.array(
                                m._point_registry.descriptors_view(),
                                dtype=np.uint8),
                            "has": [int(m._point_registry.has(pid))
                                    for pid in
                                    m._point_registry.ids()]}),
        "archives": [
            {"keyframes": [_snapshot_keyframe(f, arch_pids)
                           for f in a.keyframes],
             "points": [_snapshot_point(p) for p in a.points]}
            for a, arch_pids in
            ((a, {p.point_id for p in a.points})
             for a in (getattr(m, "archives", None) or []))],
        "g2o_ema": getattr(m.g2o_optimizer, "_pnp_median_ema", None),
    }


def _restore_keyframe(state: dict, K, k1,
                      pid_to_point: Optional[dict]) -> Frame:
    f = Frame(np.zeros((1, 1, 3), dtype=np.uint8), K, k1)
    f.frame_id = state["frame_id"]
    f.pose = state["pose"]
    f.is_keyframe = state["is_keyframe"]
    arr = state["keypoints"]
    f.keypoints = [cv.KeyPoint(float(r[0]), float(r[1]), float(r[2]),
                               float(r[3]), float(r[4]),
                               int(r[5]), int(r[6])) for r in arr]
    f.descriptors = state["descriptors"]
    # The direct keypoint assignment above bypasses set_features, so
    # the derived caches (kp_pts/kp_pts_norm/kp_unique_mask) must be
    # rebuilt the same way: triangulation (tracker) and the loop
    # detector subscript keyframe.kp_pts_norm directly.
    if f.keypoints:
        f.set_features(f.keypoints, f.descriptors)
    f._tracking_quality = state["tracking_quality"]
    f.keypoint_to_point_map = {}
    if pid_to_point is not None:
        for i, pid in state["kp_to_point"].items():
            p = pid_to_point.get(int(pid))
            if p is not None and 0 <= int(i) < len(f.keypoints):
                # The keys are the frame's OWN keypoint objects (the
                # live identity contract).
                f.keypoint_to_point_map[f.keypoints[int(i)]] = p
    return f


def _restore_point(state: dict) -> Point:
    p = Point(np.asarray(state["pt_3d"], dtype=float).copy(),
              (state["color"].copy() if state["color"] is not None
               else None),
              (state["descriptor"].copy()
               if state["descriptor"] is not None else None))
    p.point_id = state["point_id"]
    p.average_reprojection_error = float(state["error"])
    return p


def _restore_map(m: Map, state: dict) -> None:
    K = m.intrinsics.K
    k1 = m.intrinsics.k1

    # The points first: the keyframe maps and the observations wire
    # against them.
    points = {}
    for ps in state["points"]:
        p = _restore_point(ps)
        points[p.point_id] = p
    pid_to_point = dict(points)
    # The stale map entries' Points (pruned from the live list while
    # still keyed): rebuilt as bare objects, reachable only through
    # the keyframe maps — the live phantom-guard state reproduced.
    for ks in state["frames_by_id"] + state["keyframes"]:
        for sps in ks.get("stale_points", ()):
            pid = sps["point_id"]
            if pid not in pid_to_point:
                pid_to_point[pid] = _restore_point(sps)

    # The frames next (the keyframes are a subset by id), then the
    # cross-object wiring.
    frames = {}
    keyframes = []
    for ks in state["frames_by_id"]:
        f = _restore_keyframe(ks, K, k1, pid_to_point)
        frames[f.frame_id] = f
    for ks in state["keyframes"]:
        if ks["frame_id"] in frames:
            keyframes.append(frames[ks["frame_id"]])
        else:
            f = _restore_keyframe(ks, K, k1, pid_to_point)
            frames[f.frame_id] = f
            keyframes.append(f)

    # The observations: the FRAME's observed_points registry is the
    # authoritative side (pid -> kp_idx); the objects wire both ways.
    for ks in state["frames_by_id"] + state["keyframes"]:
        f = frames.get(ks["frame_id"])
        if f is None:
            continue
        for pid, kp_idx in ks["observed"].items():
            p = points.get(pid)
            if p is None:
                continue
            obs = Observation(f, p, int(kp_idx))
            f.observed_points[int(pid)] = obs
            p.observations[f.frame_id] = obs

    m.keyframes[:] = keyframes
    m.frames_by_id = frames
    m.keyframes_by_id = {f.frame_id: f for f in keyframes if
                         f.is_keyframe}
    m.points[:] = [points[ps["point_id"]] for ps in state["points"]]
    m.points_by_id = {p.point_id: p for p in m.points}
    m.tracking_quality_history = dict(state["quality_history"])
    m.consecutive_low_quality_frames = state[
        "consecutive_low_quality_frames"]
    m.covisibility_graph = defaultdict(lambda: defaultdict(int))
    for k, v in state["covisibility_graph"].items():
        inner = m.covisibility_graph[int(k)]
        for kk, vv in v.items():
            inner[int(kk)] = vv
    m._covisibility_score_cache = {
        tuple(int(x) for x in k.split("_")) if isinstance(k, str)
        and "_" in k else k: v
        for k, v in state["covisibility_cache"].items()}
    m._registry_rows = dict(state["registry_rows"])
    m.point_coords = set(state["point_coords"])
    m.extractor_name = state["extractor_name"]
    m.video_fps = state["video_fps"]
    m.archives = []
    for a in state["archives"]:
        am = ArchivedMap([], [])
        _restore_archived(am, a, K, k1)
        m.archives.append(am)
    if state.get("g2o_ema") is not None:
        m.g2o_optimizer._pnp_median_ema = state["g2o_ema"]


def _restore_archived(am: ArchivedMap, state: dict, K, k1) -> None:
    """The archives are read-only collections — the same rebuild.
    The archived observations wire too: the archive-recovery
    candidate filter reads point.is_observed_by(frame_id)."""
    frames = {}
    for ks in state["keyframes"]:
        f = _restore_keyframe(ks, K, k1, None)
        frames[f.frame_id] = f
    am.keyframes[:] = [frames.get(ks["frame_id"]) or
                       _restore_keyframe(ks, K, k1, None)
                       for ks in state["keyframes"]]
    pid_to_point = {}
    for ps in state["points"]:
        p = Point(np.asarray(ps["pt_3d"], dtype=float).copy(),
                  None,
                  (ps["descriptor"].copy()
                   if ps["descriptor"] is not None else None))
        p.point_id = ps["point_id"]
        p.average_reprojection_error = float(ps["error"])
        pid_to_point[p.point_id] = p
    am.points[:] = list(pid_to_point.values())
    for ks in state["keyframes"]:
        f = frames.get(ks["frame_id"])
        if f is None:
            continue
        for pid, kp_idx in ks["observed"].items():
            p = pid_to_point.get(pid)
            if p is None:
                continue
            obs = Observation(f, p, int(kp_idx))
            f.observed_points[int(pid)] = obs
            p.observations[f.frame_id] = obs


# ------------------------------------------------------------ tracker
TRACKER_FIELDS = ("velocity", "step", "frame_count", "index",
                  "consecutive_successful_relocalizations",
                  "consecutive_relocalization_failures",
                  "iterations_since_last_keyframe",
                  "_rot_model_ema", "_rot_model_prev_t",
                  "_rot_model_engaged", "_rot_model_bootstrap_count")


def _snapshot_tracker(t) -> dict:
    state = {}
    for name in TRACKER_FIELDS:
        if hasattr(t, name):
            state[name] = getattr(t, name)
    if state.get("velocity") is not None:
        state["velocity"] = np.asarray(state["velocity"]).copy()
    return state


def _restore_tracker(t, state: dict) -> None:
    for name, value in state.items():
        setattr(t, name, value)


# ------------------------------------------------------------- intake
def _snapshot_intake(gate) -> Optional[dict]:
    if gate is None:
        return None
    state = {}
    for name, v in vars(gate).items():
        if name == "feature_extractor" or name == "matcher":
            # The extractor/matcher are stateless holders rebuilt at
            # construction.
            continue
        if isinstance(v, (int, float, bool, str)):
            state[name] = v
        elif isinstance(v, np.ndarray):
            # The underscored displacement reference rides too: the
            # gate's feed decision compares every decoded frame
            # against it - a stale reference diverges the intake.
            state[name] = np.asarray(v).copy()
        elif name == "last_extraction" and v is not None:
            kps, desc = v
            state["last_extraction"] = (
                [(k.pt[0], k.pt[1], k.size) for k in kps],
                np.asarray(desc) if desc is not None else None)
    return state or None


def _restore_intake(gate, state: Optional[dict]) -> None:
    if gate is None or state is None:
        return
    for name, v in state.items():
        if name == "last_extraction":
            if v is not None:
                pts, desc = v
                gate.last_extraction = (
                    [cv.KeyPoint(float(x), float(y), float(s))
                     for x, y, s in pts], desc)
            continue
        if hasattr(gate, name) and not callable(getattr(gate, name)):
            setattr(gate, name, v)


# --------------------------------------------------------------- api
def count_continue_map(m: Map, extra_frame_ids=()) -> None:
    """Continue the frame/point counters past the restored map's
    maxima (called post-restore, per map). The archives fold in too:
    their ids are live references inside the restored session."""
    arch = getattr(m, "archives", None) or []
    max_f = max([f.frame_id for f in m.frames_by_id.values()] +
                [f.frame_id for a in arch for f in a.keyframes] +
                list(extra_frame_ids),
                default=0)
    max_kf = max((f.frame_id for f in m.keyframes), default=0)
    frame_mod._next_frame_id = itertools.count(max(max_f, max_kf) + 1)
    max_p = max([p.point_id for p in m.points_by_id.values()] +
                [p.point_id for a in arch for p in a.points],
                default=0)
    max_pl = max((p.point_id for p in m.points), default=0)
    point_mod._next_point_id = itertools.count(max(max_p, max_pl) + 1)


def capture(session) -> Checkpoint:
    """The checkpoint at the session's current step boundary."""
    m = session.global_map
    t = session.tracker
    live_pids = {p.point_id for p in m.points}
    tracker_frame = (None if getattr(t, "cur_frame", None) is None
                     else _snapshot_keyframe(t.cur_frame, live_pids))
    return Checkpoint(
        frame_index=session.frame_index,
        step_count=session.step_count,
        map_state=_snapshot_map(m),
        tracker_state=_snapshot_tracker(t),
        tracker_frame=tracker_frame,
        intake_state=_snapshot_intake(session.intake_gate),
        branch_seed=42,
        label=f"step {session.step_count}",
    )


def restore(session, cp: Checkpoint) -> None:
    """In-place restore: the map/tracker/intake return to the
    checkpoint's state; the source seeks to its frame index. The cv
    RNG is NOT re-seeded here: Session.step seeds each step from its
    own ordinal (the step-deterministic contract), so a resume at
    step N re-enters exactly the stream the original run used at its
    step N."""
    _restore_map(session.global_map, cp.map_state)
    _restore_tracker(session.tracker, cp.tracker_state)
    _restore_intake(session.intake_gate, cp.intake_state)
    m = session.global_map
    K = m.intrinsics.K
    k1 = m.intrinsics.k1
    # The tracker's LAST PROCESSED frame is the next step's match
    # reference (_get_next_pose_estimate rides its pose): mount it
    # back or the first resumed step matches against a stale frame.
    extra_ids = ()
    if cp.tracker_frame is not None:
        pid_to_point = {p.point_id: p for p in m.points}
        for sps in cp.tracker_frame.get("stale_points", ()):
            pid = sps["point_id"]
            if pid not in pid_to_point:
                pid_to_point[pid] = _restore_point(sps)
        f = _restore_keyframe(cp.tracker_frame, K, k1, pid_to_point)
        if f.frame_id not in m.frames_by_id:
            m.frames_by_id[f.frame_id] = f
        session.tracker.cur_frame = f
        session.tracker.prev_frame = None
        extra_ids = (f.frame_id,)
    count_continue_map(m, extra_frame_ids=extra_ids)
    # The g2o persistent graph references pre-restore map data (the
    # external-write precedent from the archive re-anchor: it resets
    # before the writes matter).
    if getattr(m, "g2o_optimizer", None) is not None:
        m.g2o_optimizer.reset_persistent_graph()
    # The loop-closure DB at checkpoint time holds exactly the live
    # keyframes' features (added on insert, purged on cull): rebuild
    # in temporal order and give the detector a fresh self.
    m.feature_db = KeyframeFeatureDatabase()
    for kf in sorted(m.keyframes, key=lambda k: k.frame_id):
        if kf.descriptors is not None:
            m.feature_db.add_keyframe_features(
                kf.frame_id, kf.descriptors)
    m.loop_detector = type(m.loop_detector)(m, m.feature_db)
    # The native registries rebuild from the CHECKPOINT's captured
    # state (the registry is the point geometry's home
    # (Point.pt_3d reads THROUGH it) - the views must be the
    # checkpoint's, not whatever the live engine holds at the
    # restore moment). The obs registry is the fold's row-order
    # contract - cleared and re-added in the restored frames'
    # temporal order (the tracker frames too: the fold includes
    # never-promoted frames' observations).
    reg_state = cp.map_state.get("point_registry")
    reg = getattr(m, "_point_registry", None)
    if reg is not None:
        reg.clear()
        if reg_state is not None:
            reg.register_points(
                list(reg_state["ids"]), reg_state["pos"],
                reg_state["desc"],
                np.array(reg_state["has"], dtype=np.uint8))
            m._refresh_registry_index()
    if m._obs_registry is not None:
        m._obs_registry.clear()
        for f in sorted(m.frames_by_id.values(),
                        key=lambda k: k.frame_id):
            if f.observed_points:
                m._add_observation_events(
                    list(f.observed_points.values()))
    # The stream position: the sequence/camera seek (the M0 spike:
    # EXACT at ~1 ms); the extraction worker (file sources) is
    # re-launched at the offset (frame_index + 1: the next read is
    # the checkpoint frame's successor) by the session's rewind path.
    seek = getattr(session.capture, "seek", None)
    if seek is not None and cp.frame_index >= 0:
        seek(cp.frame_index + 1)
    info_log(LOG_TAG, f"restored to frame {cp.frame_index} "
             f"(step {cp.step_count}, branch {cp.branch_seed})")
