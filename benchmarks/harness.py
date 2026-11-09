"""Headless SLAM benchmark runner.

Drives Map + Tracker directly over a benchmark sequence - no renderer, no GUI.
Records per-frame poses (the system's estimate after each `Tracker.update`),
keyframe events, map state, and resets, so metrics can compare the estimate
against ground truth.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

import slam.utils.logger as slam_logger
from slam.core.frame import Frame
from slam.core.map import Map
from slam.core.tracker import Tracker
from slam.utils import constants as const
from slam.utils import telemetry


@dataclass
class RunResult:
    frames_processed: int
    poses: np.ndarray            # (N, 4, 4) per-frame Twc estimates
    keyframe_indices: List[int]   # frame indices promoted to keyframe
    map_points: np.ndarray        # (M, 3) final map points
    num_keyframes: int
    resets: int
    seconds: float
    telemetry: dict = field(default_factory=dict)
    # telemetry snapshot: stage_seconds / stage_calls / events

    @property
    def fps(self) -> float:
        return self.frames_processed / self.seconds if self.seconds > 0 else 0.0


def load_calibration(dataset_dir: Path):
    """Dataset calibration -> (K, k1). Phase 2j-3: the calibration file
    may carry a 5th value, the simple-radial k1 (the generator writes
    it only when nonzero, so existing sequences' files are unchanged).
    The phase-2j-0 FOCAL_SCALE diagnostic perturbs the focal on demand;
    k1 lives in normalized units and is untouched by it."""
    dataset_dir = Path(dataset_dir)
    vals = np.atleast_1d(np.loadtxt(dataset_dir / "calibration.txt"))
    fx, fy, cx, cy = vals[:4]
    k1 = float(vals[4]) if len(vals) > 4 else 0.0

    # Literal no-op at the default 1.0 (the multiply is skipped) so
    # default runs are bit-identical.
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
             debug: bool = False) -> RunResult:
    """Run the SLAM pipeline over a sequence headlessly and deterministically."""
    slam_logger.DEBUG = debug

    dataset_dir = Path(dataset_dir)
    K, k1 = load_calibration(dataset_dir)
    frame_files = sorted((dataset_dir / "frames").glob("*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {dataset_dir / 'frames'}")

    # The one real randomness source in the pipeline: OpenCV's RANSAC.
    cv.setRNGSeed(seed)

    world_map = Map()
    tracker = Tracker(world_map, extractor, multiscale, enable_ba)

    poses: List[np.ndarray] = []
    keyframe_indices: List[int] = []
    steps: List[int] = []

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
            frame = Frame(img, K, k1)
            is_new_keyframe = tracker.update(frame)

            poses.append(tracker.cur_frame.pose)
            steps.append(tracker.step)
            if is_new_keyframe:
                keyframe_indices.append(i)
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

    return RunResult(
        frames_processed=len(frame_files),
        poses=np.array(poses),
        keyframe_indices=keyframe_indices,
        map_points=map_points,
        num_keyframes=len(world_map.keyframes),
        resets=resets,
        seconds=elapsed,
        telemetry=recorder.snapshot(),
    )
