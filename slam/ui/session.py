"""The editor's session controller (M2).

A Session owns ONE SLAM run: the resolved source (via the M1a resolver
+ the M1b intrinsics cache), the pipeline (map, tracker, intake gate,
the pre-extraction worker on file sources), and the transport state.
slam.py's loop body lives here as `step()` so the editor's transport
buttons and the legacy loop share one definition of a frame step.

Threading model: exactly ONE thread (the editor's step worker) touches
the pipeline; the UI thread reads it BETWEEN steps (on a stepDone
signal) and mutates only via the controller's methods while the
worker is idle. The timeline rewind (M5) reuses that rule.

Recording parity: `record_frame` mirrors the legacy renderer's
capture semantics ([capture | render] side-by-side mp4 at 30 fps in
output/sessions/<timestamp>/) and `save_session_data` mirrors the
JSON-metadata + PLY point-cloud save.
"""
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np

from slam.calibration import IntrinsicsEstimate
from slam.core.frame import Frame
from slam.core import intake as intake_mod
from slam.core.intake import FrameIntakeGate
from slam.core.map import Map
from slam.core.source import (CAMERA_KIND, FILE_KIND, SEQUENCE_KIND,
                              SequenceCapture, SourceSpec, open_source,
                              resolve_intrinsics)
from slam.core import intrinsics_cache
from slam.core.tracker import Tracker
from slam.utils import logger as slam_logger_mod
from slam.utils.logger import info_log, warning_log

LOG_TAG = "Editor"


@dataclass
class SlamOptions:
    """The editor's tuning options (the config's global properties);
    the constructor-time ones force a session rebuild; the live ones
    flip mid-session."""
    feature_extractor: str = "ALIKED"
    enable_multiscale_features: bool = False
    enable_ba: bool = True
    record_session: bool = False
    display_capture: bool = True
    enable_debug: bool = True
    enable_profiling: bool = False


@dataclass
class StepResult:
    step: int                       # the fed-frame counter (1-based)
    frame_index: int                # the decoded-stream index
    is_keyframe: bool
    keyframes: int
    points: int
    avg_quality: float
    eos: bool = False               # the source is exhausted
    capture: Optional[np.ndarray] = None  # the last raw frame
    pose: Optional[np.ndarray] = None     # the current camera pose


class Session:
    def __init__(self, spec: SourceSpec, options: SlamOptions):
        self.spec = spec
        self.options = options

        slam_logger_mod.DEBUG = bool(options.enable_debug)

        # --- the intrinsics provenance flow (M1b): resolver + cache
        resolved = open_source(spec)
        if resolved is None:
            raise FileNotFoundError(
                f"Could not open the source: {spec.location}")
        self.capture, self.K, self.k1, meta = resolved
        self.cache_hit = intrinsics_cache.lookup(spec.location)
        if self.cache_hit is not None:
            self.K = intrinsics_cache.cached_K(self.cache_hit)
            self.k1 = float(self.cache_hit.get("k1", 0.0))
        self.CAP_W = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.CAP_H = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))

        self.init_frames = None
        self.intrinsics_prior = None
        if self.K is None:
            from slam.calibration import estimate_video_intrinsics
            self.intrinsics_prior, self.init_frames = \
                estimate_video_intrinsics(self.capture)
            self.K = self.intrinsics_prior.K
            k1_prior = float(getattr(self.intrinsics_prior, "k1", 0.0))
            if intrinsics_cache.lookup(spec.location) is None:
                intrinsics_cache.store(
                    spec.location, "geocalib", fx=float(self.K[0, 0]),
                    fy=float(self.K[1, 1]), cx=float(self.K[0, 2]),
                    cy=float(self.K[1, 2]), k1=k1_prior)
            self.k1 = 0.0
        elif self.cache_hit is None:
            intr = resolve_intrinsics(spec, self.CAP_W, self.CAP_H)[2]
            intrinsics_cache.store(spec.location, "config", **intr)

        # --- the pipeline (slam.py's setup, verbatim)
        self.global_map = Map()
        if self.intrinsics_prior is not None:
            self.global_map.intrinsics = \
                IntrinsicsEstimate.from_prior(self.intrinsics_prior)
        else:
            self.global_map.intrinsics = IntrinsicsEstimate.from_K(
                self.K, k1=self.k1, source="config")
        self.container_fps = self.capture.get(cv.CAP_PROP_FPS) or 30.0
        self.global_map.video_fps = self.container_fps

        self.intake_gate = None
        if spec.kind in (FILE_KIND, CAMERA_KIND) or True:
            # The displacement-gated intake is the video path's health
            # gate; sequence dirs feed the same gate (the direct read
            # path carries no pre-extraction).
            self.intake_gate = FrameIntakeGate(
                self.options.feature_extractor,
                self.options.enable_multiscale_features,
                self.container_fps)

        self.extract_pipeline = None
        if (self.intake_gate is not None
                and spec.kind == FILE_KIND):
            from slam.core.intake import ExtractionProcess
            self.extract_pipeline = ExtractionProcess(
                spec.location, self.CAP_W, self.CAP_H,
                self.options.feature_extractor,
                self.options.enable_multiscale_features,
                worker_threads=intake_mod.INTAKE_EXTRACT_WORKER_THREADS)

        self.tracker = Tracker(
            self.global_map, self.options.feature_extractor,
            self.options.enable_multiscale_features,
            self.options.enable_ba)
        # The rotation gate is tied to the intake cadence (the video
        # path's measured contract).
        self.tracker.rotation_model_enabled = self.intake_gate is not None

        # The profiling recorder (observer-only by design — it cannot
        # perturb the pipeline's numerics, the harness precedent).
        self.telemetry_recorder = None
        if self.options.enable_profiling:
            self.enable_profiling()

        self.frame_index = -1
        self.step_count = 0
        self._done = False

        # --- the recording state (the parity block)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = f"output/sessions/{self.timestamp}"
        os.makedirs(self.output_path, exist_ok=True)
        self.recording = False
        self.video_writer = None
        self.video_filename = None
        self.recorded_frames = 0
        self.session_data = {"keyframe_poses": [], "per_frame": []}

    # ------------------------------------------------------------------
    # the step (slam.py's loop body)
    def step(self) -> Optional[StepResult]:
        if self._done:
            return None
        prev_img = getattr(self, "_prev_img", None)
        img = preextracted = None
        if self.extract_pipeline is not None:
            item = self.extract_pipeline.next_frame()
            if item is None:
                self._on_eos()
                return None
            img, preextracted = item[1], (item[2], item[3])
            if self.init_frames:
                init_frames = self.init_frames
                init_frames.pop(0)
        elif self.init_frames:
            img = self.init_frames.pop(0)
            preextracted = None
        else:
            ret, img = self.capture.read()
            if not ret:
                self._on_eos()
                return None
            preextracted = None
        self.frame_index += 1

        if self.intake_gate is not None and not self.intake_gate.should_feed(
                img, preextracted):
            # The gate skipped: no SLAM step for this decoded frame.
            return self._as_step(keyframe=False, capture=img,
                                 skipped=True)

        frame = Frame(img, self.global_map.intrinsics.K,
                      self.global_map.intrinsics.k1)
        if self.intake_gate is not None and \
                intake_mod.INTAKE_CARRY_EXTRACTION:
            frame.set_features(*self.intake_gate.last_extraction)
        is_kf = self.tracker.update(frame)

        result = self._as_step(keyframe=is_kf, capture=img)
        if self.recording:
            self.record_frame(prev_img, img)
        return result

    def _on_eos(self):
        self._done = True
        info_log(LOG_TAG,
                 f"SLAM processing finished with "
                 f"{len(self.global_map.keyframes)} keyframes and "
                 f"{len(self.global_map.points)} points in the map!")

    def _as_step(self, keyframe, capture, skipped=False):
        self.step_count += 1
        return StepResult(
            step=self.step_count, frame_index=self.frame_index,
            is_keyframe=keyframe,
            keyframes=len(self.global_map.keyframes),
            points=len(self.global_map.points),
            avg_quality=float(self.global_map.avg_tracking_quality),
            capture=capture,
            pose=(self.tracker.cur_frame.pose.copy()
                  if getattr(self.tracker, "cur_frame", None) else None),
        )

    # ------------------------------------------------------------------
    # the reset (a fresh pipeline over the same source)
    def reset(self) -> None:
        self.close_pipeline()
        self.capture = None
        # Re-resolve from the same spec (the cache now serves K)
        resolved = open_source(self.spec)
        if resolved is None:
            raise FileNotFoundError("source failed to open on reset")
        self.capture, K_from_meta, self.k1, meta = resolved
        if self.cache_hit is not None:
            self.K = intrinsics_cache.cached_K(self.cache_hit)
        else:
            self.K = K_from_meta
        self.init_frames = self.intrinsics_prior and None
        self.global_map = Map()
        if self.intrinsics_prior is not None:
            self.global_map.intrinsics = \
                IntrinsicsEstimate.from_prior(self.intrinsics_prior)
        else:
            self.global_map.intrinsics = IntrinsicsEstimate.from_K(
                self.K, k1=self.k1, source="config")
        self.global_map.video_fps = self.container_fps
        self.intake_gate = FrameIntakeGate(
            self.options.feature_extractor,
            self.options.enable_multiscale_features,
            self.container_fps) if self.intake_gate else None
        self.extract_pipeline = None
        if (self.intake_gate is not None
                and self.spec.kind == FILE_KIND):
            from slam.core.intake import ExtractionProcess
            self.extract_pipeline = ExtractionProcess(
                self.spec.location, self.CAP_W, self.CAP_H,
                self.options.feature_extractor,
                self.options.enable_multiscale_features,
                worker_threads=intake_mod.INTAKE_EXTRACT_WORKER_THREADS)
        old_tracker = self.tracker
        del old_tracker
        self.tracker = Tracker(
            self.global_map, self.options.feature_extractor,
            self.options.enable_multiscale_features,
            self.options.enable_ba)
        self.tracker.rotation_model_enabled = self.intake_gate is not None
        self.frame_index = -1
        self.step_count = 0
        self._done = False
        info_log(LOG_TAG, "Session reset")

    # ------------------------------------------------------------------
    # the live-option machinery (M3): the hot toggles
    def enable_profiling(self) -> None:
        from slam.utils import telemetry
        self.telemetry_recorder = telemetry.TelemetryRecorder()
        telemetry.set_recorder(self.telemetry_recorder)

    def disable_profiling(self) -> None:
        from slam.utils import telemetry
        telemetry.set_recorder(None)

    def profile_summary(self) -> Optional[dict]:
        """The per-call ms of the top pipeline stages since the
        recorder's attachment."""
        if self.telemetry_recorder is None:
            return None
        snap = self.telemetry_recorder.snapshot()
        sec = snap.get("stage_seconds", {})
        calls = snap.get("stage_calls", {})
        rows = {}
        for name, secs in sec.items():
            n = max(calls.get(name, 0), 1)
            rows[name] = round(1000 * secs / n, 2)
        top = sorted(rows.items(), key=lambda x: -x[1])[:5]
        return dict(top)

    def apply_live_options(self, opts: SlamOptions) -> dict:
        """The hot flips; returns the branch-required option names
        that still need a session rebuild (the constructor-time
        options)."""
        old = self.options
        self.options = opts
        slam_logger_mod.DEBUG = bool(opts.enable_debug)
        self.tracker.enable_ba = bool(opts.enable_ba)
        if opts.enable_profiling and self.telemetry_recorder is None:
            self.enable_profiling()
        elif not opts.enable_profiling and \
                self.telemetry_recorder is not None:
            self.disable_profiling()
        if opts.record_session and not self.recording:
            self.start_recording()
        elif not opts.record_session and self.recording:
            self.stop_recording()
        branch_required = []
        if opts.feature_extractor != old.feature_extractor:
            branch_required.append("feature_extractor")
        if opts.enable_multiscale_features != \
                old.enable_multiscale_features:
            branch_required.append("enable_multiscale_features")
        return branch_required

    # ------------------------------------------------------------------
    # the timeline (M5a): the checkpoint/restore over the live session
    CHECKPOINT_EVERY = 25

    def capture_checkpoint(self, label="") -> Optional[dict]:
        """The checkpoint at the current step boundary (the step worker
        is idle between ticks)."""
        from slam.ui import checkpoint as cp_mod
        if self._done:
            return None
        cp = cp_mod.capture(self)
        cp.label = label or f"step {self.step_count}"
        info_log(cp_mod.LOG_TAG, f"checkpoint @ frame {cp.frame_index}")
        return cp

    def restore_checkpoint(self, cp) -> None:
        """The in-place restore + the source reposition (the rewind's
        engine). File sources re-launch the extraction worker at the
        offset; sequence/camera captures reposition directly."""
        from slam.ui import checkpoint as cp_mod
        self.close_pipeline()
        cp_mod.restore(self, cp)
        if self.extract_pipeline is not None and \
                self.spec.kind == FILE_KIND:
            self.extract_pipeline = None
            from slam.core.intake import ExtractionProcess
            self.extract_pipeline = ExtractionProcess(
                self.spec.location, self.CAP_W, self.CAP_H,
                self.options.feature_extractor,
                self.options.enable_multiscale_features,
                worker_threads=intake_mod.INTAKE_EXTRACT_WORKER_THREADS,
                start_index=cp.frame_index + 1)

    # ------------------------------------------------------------------
    # the recording parity
    def start_recording(self, filename=None):
        if filename:
            self.video_filename = filename
            if not filename.endswith(".mp4"):
                self.video_filename += ".mp4"
        else:
            self.video_filename = os.path.join(
                self.output_path, f"slam_session_{self.timestamp}.mp4")
        self.recording = True
        self.recorded_frames = 0
        info_log(LOG_TAG, f"Started recording to {self.video_filename}")

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        info_log(LOG_TAG, f"Stopped recording. Saved "
                 f"{self.recorded_frames} frames to {self.video_filename}")
        self.recorded_frames = 0

    def record_frame(self, prev_img, capture_img):
        """[capture | render] side-by-side, the parity of the legacy
        capture_frame(): the editor's OFFSCREEN render buffer is the
        renderer frame; the raw capture is the complement."""
        renderer_frame = self.render_bgr  # set by the editor per tick
        if renderer_frame is None:
            return
        if capture_img is not None:
            h_r, w_r = renderer_frame.shape[:2]
            h_c, w_c = capture_img.shape[:2]
            if h_c == 0 or w_c == 0:
                return
            if h_r / h_c >= 1:
                scale = h_r / h_c
                new_w = max(1, int(w_c * scale))
                resized = cv.resize(capture_img, (new_w, h_r))
                pad_l = max(0, (w_r - new_w) // 2)
                pad_r = max(0, w_r - new_w - pad_l)
                if pad_l < 0 or new_w > w_r:
                    scale = w_r / w_c
                    new_h = int(h_c * scale)
                    resized = cv.resize(capture_img, (w_r, new_h))
                    resized = cv.copyMakeBorder(
                        resized, 0, h_r - new_h, 0, 0,
                        cv.BORDER_CONSTANT, value=(0, 0, 0))
                else:
                    resized = cv.copyMakeBorder(
                        resized, 0, 0, pad_l, pad_r,
                        cv.BORDER_CONSTANT, value=(0, 0, 0))
                save_img = np.concatenate([resized, renderer_frame],
                                          axis=1)[:, :w_r * 2]
            else:
                save_img = np.concatenate(
                    [cv.resize(capture_img, (w_r // 2, h_r // 2)),
                     renderer_frame], axis=1)
        else:
            save_img = renderer_frame
        h_o, w_o = save_img.shape[:2]
        if self.video_writer is None:
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv.VideoWriter(
                self.video_filename, fourcc, 30.0, (w_o, h_o))
        self.video_writer.write(save_img)
        self.recorded_frames += 1

    def save_session_data(self) -> None:
        self.session_data["timestamp"] = self.timestamp
        self.session_data["camera_params"] = {
            "K": self.K.tolist(), "width": self.CAP_W,
            "height": self.CAP_H}
        filename = f"slam_session_{self.timestamp}"
        json_filename = os.path.join(self.output_path,
                                     f"{filename}.json")
        with open(json_filename, "w") as f:
            json.dump(self.session_data, f, indent=2)
        ply_filename = os.path.join(self.output_path, f"{filename}.ply")
        if len(self.global_map.points) > 0:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(
                [p._pt_3d for p in self.global_map.points]))
            o3d.io.write_point_cloud(ply_filename, pcd)
            info_log(LOG_TAG, f"Saved point cloud to {ply_filename}")

    # ------------------------------------------------------------------
    def close_pipeline(self):
        if self.extract_pipeline is not None:
            self.extract_pipeline.close()
            self.extract_pipeline = None

    def close(self):
        if self.recording:
            self.stop_recording()
        self.close_pipeline()

    def save(self):
        self.save_session_data()
