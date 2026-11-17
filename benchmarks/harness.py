"""Headless SLAM benchmark runner.

Drives Map + Tracker directly over a benchmark sequence - no renderer, no GUI.
Records per-frame poses (the system's estimate after each `Tracker.update`),
keyframe events, map state, and resets, so metrics can compare the estimate
against ground truth.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

import slam.utils.logger as slam_logger
from benchmarks.health import heading_deg
from slam.calibration.refit import IntrinsicsEstimate
from slam.core.frame import Frame
from slam.core.map import Map
from slam.core.tracker import Tracker
from slam.utils import constants as const
from slam.utils import telemetry


@dataclass
class RunResult:
    frames_processed: int
    poses: np.ndarray            # (N, 4, 4) per-frame Twc estimates
    keyframe_indices: List[int]  # frame indices promoted to keyframe
    map_points: np.ndarray        # (M, 3) final map points
    num_keyframes: int
    resets: int
    seconds: float
    telemetry: dict = field(default_factory=dict)
    # telemetry snapshot: stage_seconds / stage_calls / events
    final_intrinsics: dict = field(default_factory=dict)
    # the final intrinsics estimate (f, k1, frozen,
    # source) - the recovery experiments' readout
    keyframe_poses: np.ndarray = field(default_factory=lambda:
                                       np.empty((0, 4, 4)))
    # the FINAL map keyframe poses (post-correction,
    # post-culling - loop closure rewrites the keyframe trajectory
    # after the per-frame stream is history), paired with
    keyframe_gt_indices: List[int] = field(default_factory=list)
    # the frame indices of those same keyframes (the KF-trajectory
    # ATE's pairing; the harness sets _bench_frame_index at insertion
    # so culled keyframes drop out cleanly)
    # ---- observation-only additions (additive; no SLAM state touched) ----
    health_rows: List[dict] = field(default_factory=list)
    # the per-frame health stream (the shared benchmarks/health.py
    # schema) — lets GT'd ladder runs attribute ACCURACY to health
    # mechanisms (windowed ATE x quality/flags) the same way the
    # video census attributes failure modes without GT.
    health_reset_indices: List[int] = field(default_factory=list)
    # WHERE the resets happened in the fed stream (health needs
    # positions; the int `resets` above keeps the battery contract).
    loop_corrections: int = 0
    # how many loop-closure corrections fired during the
    # run (the tracker flag the harness observes per update).
    pre_correction_keyframe_poses: np.ndarray = field(
        default_factory=lambda: np.empty((0, 4, 4)))
    pre_correction_keyframe_gt_indices: List[int] = field(
        default_factory=list)
    # the keyframe-poses snapshot taken immediately BEFORE
    # the FIRST loop-closure correction rewrote them — the
    # loop-benefit measurement's "before" side (KF-ATE pre vs post).

    @property
    def fps(self) -> float:
        return self.frames_processed / self.seconds if self.seconds > 0 else 0.0


def load_calibration(dataset_dir: Path):
    """Dataset calibration -> (K, k1). The calibration file
    may carry a 5th value, the simple-radial k1 (the generator writes
    it only when nonzero, so existing sequences' files are unchanged).
    The FOCAL_SCALE diagnostic perturbs the focal on demand;
    k1 lives in normalized units and is untouched by it."""
    dataset_dir = Path(dataset_dir)
    vals = np.atleast_1d(np.loadtxt(dataset_dir / "calibration.txt"))
    fx, fy, cx, cy = vals[:4]
    k1 = float(vals[4]) if len(vals) > 4 else 0.0

    # Diagnostic: perturb the focal length on demand
    # (--override FOCAL_SCALE=0.8). Literal no-op at the default 1.0
    # (the multiply is skipped) so default runs are bit-identical.
    if const.FOCAL_SCALE != 1.0:
        fx, fy = fx * const.FOCAL_SCALE, fy * const.FOCAL_SCALE

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K, k1


def load_trajectory(dataset_dir: Path) -> np.ndarray:
    """TUM-format file -> (N, 4, 4) Twc ground-truth poses."""
    dataset_dir = Path(dataset_dir)
    data = np.loadtxt(dataset_dir / "trajectory.txt")
    poses = np.zeros((len(data), 4, 4))
    for i, row in enumerate(data):
        R = Rotation.from_quat(row[4:8]).as_matrix()
        poses[i] = np.eye(4)
        poses[i, :3, :3] = R
        poses[i, :3, 3] = row[1:4]
    return poses


def load_scene(dataset_dir: Path):
    dataset_dir = Path(dataset_dir)
    points = np.loadtxt(dataset_dir / "scene_points.txt")
    planes = np.loadtxt(dataset_dir / "scene_planes.txt")
    return points, planes


def run_slam(dataset_dir: Path,
             seed: int = 42,
             extractor: str = "ORB",
             multiscale: bool = False,
             enable_ba: bool = True,
             debug: bool = False,
             rotmodel: bool = False,
             frames_cap: int = None) -> RunResult:
    """Run the SLAM pipeline over a sequence headlessly and deterministically.

    frames_cap: screening-only prefix leg (the first N frames, capped
    GT aligned by evaluate_sequence). Default None runs the sequence
    unchanged. The frame loop consumes the identical randomness and
    map dynamics as the full run, frame for frame, so a prefix run is
    the full run's prefix.
    """
    slam_logger.DEBUG = debug

    dataset_dir = Path(dataset_dir)
    K, k1 = load_calibration(dataset_dir)
    frame_files = sorted((dataset_dir / "frames").glob("*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {dataset_dir / 'frames'}")
    if frames_cap is not None:
        frame_files = frame_files[:frames_cap]

    # The one real randomness source in the pipeline: OpenCV's RANSAC.
    cv.setRNGSeed(seed)

    world_map = Map()
    # The live intrinsics estimate. Constructed from the
    # (possibly FOCAL_SCALE-perturbed) dataset K; the frame loop reads
    # it every frame so mid-run refits reach subsequent frames. The
    # refit activates only via CALIB_REFIT_ENABLED (default 0 - the
    # estimate is then static and the run is bit-identical).
    world_map.intrinsics = IntrinsicsEstimate.from_K(
        K, k1 + const.CALIB_K1_INIT_OFFSET)
    tracker = Tracker(world_map, extractor, multiscale, enable_ba)
    # The production wiring (the video path enables the rotation
    # model through the intake gate; the census mirrors slam.py).
    # Default False keeps the battery/ladder legacy path
    # bit-identical by construction; --rotmodel opts a leg into the
    # production-representative machine.
    tracker.rotation_model_enabled = bool(rotmodel)

    poses: List[np.ndarray] = []
    keyframe_indices: List[int] = []
    steps: List[int] = []
    # the per-frame health stream + the loop-benefit
    # "before" snapshot (pure observation; no pipeline interaction).
    health_rows: List[dict] = []
    loop_corrections = 0
    pre_correction = None   # (kf poses, kf frame indices) before the
    #                        FIRST correction fires

    # Attach the telemetry recorder for this run only. It is a pure
    # observer: stages time call sites, events count pipeline decisions;
    # nothing here can perturb the pipeline's numerics.
    recorder = telemetry.TelemetryRecorder()
    telemetry.set_recorder(recorder)

    t_start = time.perf_counter()
    try:
        for i, frame_file in enumerate(frame_files):
            img = cv.imread(str(frame_file))
            if img is None:
                raise IOError(f"Failed to read frame: {frame_file}")
            frame = Frame(img, world_map.intrinsics.K,
                          world_map.intrinsics.k1)
            prev_pose = (tracker.cur_frame.pose.copy()
                         if tracker.cur_frame is not None else None)
            before_ev = dict(recorder.events)
            # Snapshot the keyframe state while no
            # correction has fired yet. At the moment the FIRST
            # correction rewrites poses (inside this update), this
            # holds the pre-correction trajectory.
            if loop_corrections == 0 and world_map.keyframes:
                pre_correction = (
                    [kf.pose.copy() for kf in world_map.keyframes],
                    [kf._bench_frame_index
                     for kf in world_map.keyframes
                     if hasattr(kf, "_bench_frame_index")])
            is_new_keyframe = tracker.update(frame)

            # The per-frame event delta FIRST: loop-correction
            # detection keys off the 'loop_corrected' telemetry event.
            # (The tracker's _loop_correction_applied flag is set AND
            # cleared WITHIN the same update() — the velocity-reset
            # block clears it before update() returns — so a
            # post-update flag check sees nothing; measured: fr3
            # applied corrections visibly while the flag check
            # counted zero.)
            ev = {k: v - before_ev.get(k, 0)
                  for k, v in recorder.events.items()
                  if v != before_ev.get(k, 0)}
            if ev.get("loop_corrected", 0):
                loop_corrections += ev["loop_corrected"]

            poses.append(tracker.cur_frame.pose)
            steps.append(tracker.step)
            # the health row (the shared
            # benchmarks/health.py schema). The harness path never
            # enables the rotation model, so 'rm' is always
            # False here — the auto motion-signal is a census tool;
            # ladder entries carry their known axis labels instead.
            pose = tracker.cur_frame.pose
            R = pose[:3, :3]
            d_rot = (float(np.degrees(np.arccos(np.clip(
                (np.trace(prev_pose[:3, :3].T @ R) - 1.) / 2., -1., 1.))))
                if prev_pose is not None else 0.)
            health_rows.append({
                "i": i,
                "kf": bool(is_new_keyframe),
                "feat": len(frame.keypoints)
                        if frame.keypoints is not None else 0,
                "obs": len(frame.observed_points),
                "q": float(world_map.avg_tracking_quality),
                "low_q": world_map.consecutive_low_quality_frames,
                "kfs": len(world_map.keyframes),
                "pts": len(world_map.points),
                "archives": len(world_map.archives),
                "head": heading_deg(pose),
                "t": float(np.linalg.norm(pose[:3, 3])),
                "d_rot": d_rot,
                "d_t": float(np.linalg.norm(
                    pose[:3, 3] - prev_pose[:3, 3]))
                    if prev_pose is not None else 0.,
                "v_rot": float(np.degrees(np.arccos(np.clip(
                    (np.trace(tracker.velocity[:3, :3]) - 1.) / 2.,
                    -1., 1.)))),
                "v_t": float(np.linalg.norm(tracker.velocity[:3, 3])),
                "rm": bool(tracker._rot_model_engaged),
                "ev": ev,
            })
            if is_new_keyframe:
                keyframe_indices.append(i)
                # The KF-trajectory ATE's pairing needs each
                # FINAL map keyframe's frame index - keyframe_indices
                # (above) is insertion-ordered and keeps culled
                # entries, so the index rides on the frame itself and
                # the post-loop gather drops culled keyframes cleanly.
                tracker.cur_frame._bench_frame_index = i

        # The one-shot basin search at STREAM END.
        # The hook only gate-checks during the run; the search runs
        # once here (inside the try so its stage timing is recorded),
        # and freezes the estimate.
        if tracker.calib_refiner is not None:
            with telemetry.stage("calib"):
                tracker.calib_refiner.final_search()
    finally:
        elapsed = time.perf_counter() - t_start
        # Always detach, even on crashes, so no recorder ever leaks into
        # subsequent runs/tests in this process.
        telemetry.set_recorder(None)

    resets = int(sum(1 for a, b in zip(steps, steps[1:]) if b < a))
    if world_map.points:
        map_points = np.array([p.pt_3d for p in world_map.points], dtype=np.float64)
    else:
        map_points = np.empty((0, 3))

    # reset positions in the fed stream (health
    # classifiers need WHERE, RunResult.resets keeps the battery's
    # int contract) + the loop-benefit "before" attachment (only
    # meaningful when a correction actually fired).
    health_reset_indices = [j + 1 for j, (a, b)
                             in enumerate(zip(steps, steps[1:]))
                             if b < a]
    if loop_corrections and pre_correction is not None:
        pre_poses = np.array(pre_correction[0])
        pre_indices = list(pre_correction[1])
    else:
        pre_poses = np.empty((0, 4, 4))
        pre_indices = []

    return RunResult(
        frames_processed=len(frame_files),
        poses=np.array(poses),
        keyframe_indices=keyframe_indices,
        map_points=map_points,
        num_keyframes=len(world_map.keyframes),
        resets=resets,
        seconds=elapsed,
        telemetry=recorder.snapshot(),
        final_intrinsics=(world_map.intrinsics.as_dict()
                          if world_map.intrinsics is not None else {}),
        keyframe_poses=np.array([
            kf.pose for kf in world_map.keyframes
            if hasattr(kf, "_bench_frame_index")]),
        keyframe_gt_indices=[
            kf._bench_frame_index
            for kf in world_map.keyframes
            if hasattr(kf, "_bench_frame_index")],
        health_rows=health_rows,
        health_reset_indices=health_reset_indices,
        loop_corrections=loop_corrections,
        pre_correction_keyframe_poses=pre_poses,
        pre_correction_keyframe_gt_indices=pre_indices,
    )
