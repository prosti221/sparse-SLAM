"""The slam editor (M2-M5): a dark PySide6 window whose central
viewport is Open3D's OFFSCREEN render of the live map, with the
pipeline stepped on the MAIN thread by a QTimer and the checkpoint
timeline driving scrubs and rewinds.

Threading model (measured, the M2 pivot): the pipeline is
main-thread-bound on darwin (torch/Metal + the OpenCV/g2o paths die
on a worker thread — both extractor kinds) — the legacy-proven
main-thread loop survives as the step timer; the UI's callbacks
interleave between ticks and the 30 Hz paint timer keeps the render
smooth at the step cadence. The M5 rewind/scrub mutate between ticks
on this same thread — no locking anywhere by construction.

Teardown hazard (measured in M0): destroy the OffscreenRenderer before
the Qt session exits — the exit path tears the session down first and
ends with an explicit process exit.
M5 semantics (the honest rule): a rewind is branch exploration, not
bit-replay — the restore re-seeds the RNG with the branch seed, so
every branch resuming from a checkpoint shares the same post-rewind
stream; the tweak is the only delta.
"""
import os
import gc
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

import open3d as o3d
from PySide6.QtCore import (QEvent, QEventLoop, QObject,
                             Qt, QTimer, Signal)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDockWidget,
    QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QSlider, QSpinBox, QStyle, QTabWidget,
    QVBoxLayout, QWidget)

from slam.calibration import IntrinsicsEstimate
from slam.core import intrinsics_cache
from slam.core.source import (CAMERA_KIND, FILE_KIND, SEQUENCE_KIND,
                              SourceSpec)
from slam.ui.advanced_tab import UiState
from slam.ui.intrinsics_dialog import ask_manual_intrinsics
from slam.ui.row_layout import tight_row
from slam.ui.session import Session, SlamOptions
from slam.utils import logger as slam_logger_mod
from slam.utils.logger import info_log

LOG_TAG = "Editor"


class _FlyKeyFilter(QObject):
    """The editor's fly/hotkey routing: KeyPress/KeyRelease handled
    BEFORE the focused widget sees them, so the legacy viewer's keys
    are global (no click into the scene needed first). Text widgets
    keep their keys (typing wins); foreign windows (dialogs) and
    open popups are untouched."""

    TEXT_WIDGETS = (QLineEdit, QDoubleSpinBox, QAbstractSpinBox,
                    QComboBox)

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def eventFilter(self, obj, ev):
        kind = ev.type()
        if kind not in (QEvent.KeyPress, QEvent.KeyRelease):
            return False
        if not isinstance(obj, QWidget):
            # Native-window-level key events (a QWindow has no
            # .window()); the widget tree delivers the keys we route.
            return False
        if QApplication.activePopupWidget() is not None:
            return False
        if obj.window() is not self.editor:
            return False
        if isinstance(obj, self.TEXT_WIDGETS):
            return False
        pressed = kind == QEvent.KeyPress
        if pressed and ev.isAutoRepeat():
            return False
        return self.editor.fly_route(ev.key(), pressed)




DARK_QSS = """
QMainWindow { background: #1e1e20; }
QWidget#CentralHost, QDockWidget {
    background: #1e1e20; color: #9a9aa2; }
QTabWidget::pane { border: 1px solid #3a3a3e; background: #262629; }
QTabBar::tab { background: #1e1e20; color: #9a9aa2; padding: 8px 18px;
    border: 1px solid #3a3a3e; }
QTabBar::tab:selected { background: #3a3a3e; color: #e8e8ea; }
QPushButton { background: #2e2e31; color: #d8d8dc; border: 1px solid
    #454549; border-radius: 4px; padding: 6px 14px; font-size: 16px; }
QPushButton:hover { background: #454549; }
QLabel { color: #ccccce; }
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit { background: #262629;
    color: #d8d8dc; border: 1px solid #454549; padding: 4px; }
QCheckBox { color: #9a9aa2; }
QSlider::groove:horizontal { border: 1px solid #3a3a3e; height: 10px;
    background: #262629; border-radius: 5px; }
QSlider::handle:horizontal { background: #b0b0b4; width: 16px;
    margin: -4px 0; border-radius: 8px; }
"""


class Viewport(QLabel):
    """The offscreen Open3D render. Geometry refreshes only on step
    boundaries; the 30 Hz paint tick is camera-only. The camera carries
    the legacy viewer's fly contract (viz/renderer.py, the same
    constants): held W/A/S/D move with Q/E down/up (Shift boost,
    -/= speed) applied per paint tick against a wall-clock dt; the
    mouse drag orbits turntable-style around the lookat; the wheel
    dollies; Space toggles pause (window level). A SCRUB renders a
    checkpoint's snapshot geometry (the pipeline untouched); the
    camera is free in every state."""

    # The legacy fly constants (slam/viz/renderer.py, verbatim).
    FLY_SPEED_SCENE_FRACTION = 0.3      # scene diagonal per second @1x
    FLY_SPEED_BOOST = 3.0
    FLY_SPEED_MULT_MIN = 1.0 / 32.0
    FLY_SPEED_MULT_MAX = 32.0
    FLY_SPEED_STEP = 2.0
    FLY_MAX_DT_S = 0.05                 # per-tick dt cap (BA stalls)
    FLY_DEFAULT_SCENE_DIAGONAL = 100.0  # before the first points
    ORBIT_SENSITIVITY = 0.4             # degrees per pixel
    ZOOM_PER_NOTCH = 0.87               # per-notch dolly factor

    flyStatus = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #000000;")
        self.setFocusPolicy(Qt.ClickFocus)
        self.renderer = None
        self.last_render_bgr = None
        self.scrubbing = False
        self._cloud = None   # the geometry currently mounted
        self._frustums = {}  # keyframe id -> (quality, pose)
        self._cam_state = None   # (K, W, H) for the frustum math
        self._render_size = None
        self._rebuild_timer = QTimer()
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.timeout.connect(self._rebuild_render_target)
        self._up = np.array([0.0, 1.0, 0.0])
        self._eye = None
        self._lookat = None
        self._placed = False
        self._fly_keys = set()
        self._fly_boost = False
        self._fly_speed_multiplier = 1.0
        self._fly_diag = self.FLY_DEFAULT_SCENE_DIAGONAL
        self._fly_last_tick = None
        self._drag_last = None

    def _render_target(self):
        """The widget's size in device pixels - the sharp render
        (a fixed 960x540 target upscaled ~2x with smoothing is the
        M2-era blur)."""
        dpr = max(self.devicePixelRatioF(), 1.0)
        return (max(int(self.width() * dpr), 64),
                max(int(self.height() * dpr), 64))

    def attach(self, w=None, h=None):
        if self.renderer is not None:
            return
        if w is None or h is None:
            w, h = self._render_target()
        self._render_size = (int(w), int(h))
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(
            self._render_size[0], self._render_size[1])
        self.mat = o3d.visualization.rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 2.0
        # The keyframe frustums render as 2px unlit lines, white base
        # so the per-line quality colors show.
        self.mat_line = o3d.visualization.rendering.MaterialRecord()
        self.mat_line.shader = "unlitLine"
        self.mat_line.line_width = 2.0
        self.mat_line.base_color = [1.0, 1.0, 1.0, 1.0]
        self.renderer.scene.set_background(
            [0.0, 0.0, 0.0, 1.0])

    def detach(self):
        self.renderer = None
        self.scrubbing = False
        # The camera state resets with the session: the next source
        # places anew at its own pipeline pose.
        self._placed = False
        self._eye = None
        self._lookat = None
        self._fly_diag = self.FLY_DEFAULT_SCENE_DIAGONAL
        self._fly_last_tick = None
        self._frustums = {}

    @staticmethod
    def _cloud_from(points):
        """The legacy pt_obj_to_array contract: each point's color is
        the RGB sampled from the camera at its keypoint (p.color,
        uint8), normalized to [0, 1]; colorless points fall back to
        neutral gray."""
        pcd = o3d.geometry.PointCloud()
        fallback = np.full(3, 200, dtype=np.uint8)
        if points:
            pts = np.array([p._pt_3d for p in points], dtype=float)
            cols = np.array([
                (p.color if p.color is not None else fallback)
                for p in points], dtype=float)
        else:
            pts = np.zeros((1, 3))
            cols = np.full((1, 3), float(fallback[0]))
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols / 255.0)
        return pcd

    @staticmethod
    def _frustum_lines(K, W, H, pose, quality):
        """The legacy camera wireframe (viz/renderer.py's
        _draw_camera_object): a unit-depth pyramid sized by the
        intrinsics, at the pose, colored by tracking quality
        (red .. green on the HSV scale)."""
        fx = max(float(K[0, 0]), 1e-6)
        w, h = W / fx, H / fx
        cx, cy = float(K[0, 2]) / fx, float(K[1, 2]) / fx
        pts = np.array([[0.0, 0.0, 0.0],
                        [cx - w / 2.0, cy - h / 2.0, 1.0],
                        [-w / 2.0, -h / 2.0, 1.0],
                        [w / 2.0, -h / 2.0, 1.0],
                        [w / 2.0, h / 2.0, 1.0],
                        [-w / 2.0, h / 2.0, 1.0],
                        [-w / 2.0, -h / 2.0, 1.0]])
        lines = np.array([[0, 1], [2, 3], [3, 4], [4, 5], [5, 6],
                          [0, 2], [0, 3], [0, 4], [0, 5], [2, 4],
                          [3, 5]], dtype=np.int32)
        R, tt = pose[:3, :3], pose[:3, 3]
        pts = (R @ pts.T).T + tt
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        hue = int(60.0 * float(np.clip(quality, 0.0, 1.0)))
        rgb = cv.cvtColor(
            np.uint8([[[hue, 230, 230]]]),
            cv.COLOR_HSV2RGB)[0][0]
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(rgb.astype(float) / 255.0, (len(lines), 1)))
        return ls

    def _mount_frustum(self, frame_id, K, W, H, pose, quality):
        name = f"kf_{frame_id}"
        self.renderer.scene.remove_geometry(name)
        self.renderer.scene.add_geometry(
            name, Viewport._frustum_lines(K, W, H, pose, quality),
            self.mat_line)
        self._frustums[frame_id] = (quality, np.array(pose[:4, :4]))

    def update_frustums(self, session):
        """The keyframe pose wireframes (the legacy update_poses):
        rebuild on a quality change, drop the culled."""
        if self.renderer is None or self.mat_line is None:
            return
        self._cam_state = (session.K, session.CAP_W, session.CAP_H)
        seen = set()
        for kf in session.global_map.keyframes:
            q = float(kf.tracking_quality)
            cur = self._frustums.get(kf.frame_id)
            if cur is None or cur[0] != q:
                self._mount_frustum(kf.frame_id, session.K,
                                    session.CAP_W, session.CAP_H,
                                    kf.pose, q)
            seen.add(kf.frame_id)
        for fid in [f for f in self._frustums if f not in seen]:
            self.renderer.scene.remove_geometry(f"kf_{fid}")
            del self._frustums[fid]

    def update_scene_snapshot(self, summary):
        """On the step boundary (same thread, between ticks); the
        geometry rides the step worker's summary."""
        if self.renderer is None:
            self.attach()
        pos = np.asarray(summary.get("pos"), dtype=np.float32)
        cols = np.asarray(summary.get("colors"), dtype=np.uint8)
        pcd = o3d.geometry.PointCloud()
        if len(pos):
            pcd.points = o3d.utility.Vector3dVector(
                np.asarray(pos, dtype=np.float64))
            if len(cols) == len(pos):
                pcd.colors = o3d.utility.Vector3dVector(
                    cols.astype(np.float64) / 255.0)
            else:
                pcd.paint_uniform_color([0.78, 0.78, 0.78])
        else:
            pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
            pcd.paint_uniform_color([0.78, 0.78, 0.78])
        self.renderer.scene.remove_geometry("cloud")
        self.renderer.scene.add_geometry("cloud", pcd, self.mat)
        self._cloud = pcd
        if len(pos):
            extent = pos.astype(np.float64).max(axis=0) - \
                pos.astype(np.float64).min(axis=0)
            diag = float(np.linalg.norm(extent))
            if diag > 0.0:
                self._fly_diag = diag
        pose = summary.get("cur_pose")
        if pose is not None:
            self.place_view_from_pose(np.asarray(pose))
        self.update_frustums_snapshot(summary)

    def update_frustums_snapshot(self, summary):
        """The keyframe pose wireframes (the legacy update_poses):
        rebuild on a quality change, drop the culled - fed from the
        step worker's summary."""
        if self.renderer is None or self.mat_line is None:
            return
        kfr_poses = np.asarray(summary.get("frustum_poses"),
                               dtype=np.float64)
        if kfr_poses.size == 0:
            return
        K = np.asarray(summary.get("K"), dtype=np.float64)
        W = int(summary.get("W", 640))
        H = int(summary.get("H", 480))
        self._cam_state = (K, W, H)
        ids = summary.get("frustum_ids", [])
        quals = summary.get("frustum_quality", [])
        seen = set()
        for i, fid in enumerate(ids):
            if i >= len(kfr_poses):
                break
            q = float(quals[i]) if i < len(quals) else 0.0
            pose = np.asarray(kfr_poses[i], dtype=np.float64)
            cur = self._frustums.get(fid)
            if cur is None or cur[0] != q:
                self._mount_frustum(fid, K, W, H, pose, q)
            seen.add(fid)
        for fid in [f for f in self._frustums if f not in seen]:
            self.renderer.scene.remove_geometry(f"kf_{fid}")
            del self._frustums[fid]

    def scrub_render(self, checkpoint):
        """The VIEW-only render of a checkpoint's snapshot geometry
        (the pipeline is not touched), with the snapshot's colors."""
        if self.renderer is None:
            self.attach()
        self.scrubbing = True
        states = checkpoint.map_state["points"]
        fallback = np.full(3, 200, dtype=np.uint8)
        pts = [ps["pt_3d"] for ps in states]
        cols = [(ps["color"] if ps["color"] is not None else fallback)
                for ps in states] or [fallback]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.array(pts, dtype=float) if pts else np.zeros((1, 3)))
        pcd.colors = o3d.utility.Vector3dVector(
            np.array(cols, dtype=float) / 255.0)
        self.renderer.scene.remove_geometry("cloud")
        self.renderer.scene.add_geometry("cloud", pcd, self.mat)
        self._cloud = pcd
        # The scrub's own frustums (the checkpoint's snapshot), until
        # unscrub returns to the live set.
        if self._cam_state is not None:
            K, W, H = self._cam_state
            for f in list(self._frustums):
                self.renderer.scene.remove_geometry(f"kf_{f}")
            self._frustums = {}
            for ks in checkpoint.map_state["keyframes"]:
                self._mount_frustum(ks["frame_id"], K, W, H,
                                    ks["pose"],
                                    float(ks["tracking_quality"]))
        self.paint()

    def unscrub(self):
        self.scrubbing = False

    # ------------------------------------------------ legacy fly input
    def place_view_from_pose(self, pose):
        """One-time placement AT the pipeline's current camera, facing
        its view direction (the legacy _place_initial_camera: the CV
        frame X right / Y down / Z forward)."""
        if self._placed:
            return
        self._placed = True
        R = np.asarray(pose[:3, :3], dtype=float)
        self._eye = np.asarray(pose[:3, 3], dtype=float)
        fwd = R @ np.array([0.0, 0.0, 1.0])
        up = -(R @ np.array([0.0, 1.0, 0.0]))
        if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-6:
            up = np.array([0.0, 1.0, 0.0])
        self._up = up / np.linalg.norm(up)
        fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
        self._lookat = self._eye + fwd * max(
            1.0, 0.05 * self._fly_diag)
        self._commit_camera()

    def set_held(self, key, pressed):
        (self._fly_keys.add if pressed else self._fly_keys.discard)(key)

    def set_boost(self, pressed):
        self._fly_boost = pressed

    def fly_speed(self, direction):
        mult = (self._fly_speed_multiplier * self.FLY_SPEED_STEP
                if direction > 0
                else self._fly_speed_multiplier / self.FLY_SPEED_STEP)
        mult = min(self.FLY_SPEED_MULT_MAX,
                   max(self.FLY_SPEED_MULT_MIN, mult))
        self._fly_speed_multiplier = mult
        self.flyStatus.emit(
            f"fly speed {mult:.2f}x ("
            f"{self.FLY_SPEED_SCENE_FRACTION * self._fly_diag * mult:.2f}"
            f" u/s)")

    def _camera_axes(self):
        """(forward, right, up) of the current camera pose."""
        fwd = self._lookat - self._eye
        n = np.linalg.norm(fwd)
        fwd = (fwd / n if n > 1e-9
               else np.array([0.0, 0.0, -1.0]))
        right = np.cross(fwd, self._up)
        nr = np.linalg.norm(right)
        right = (right / nr if nr > 1e-9
                 else np.array([1.0, 0.0, 0.0]))
        return fwd, right, np.cross(right, fwd)

    @staticmethod
    def _rotate(v, axis, angle):
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        c, s = np.cos(angle), np.sin(angle)
        return (v * c + np.cross(axis, v) * s +
                axis * (axis @ v) * (1.0 - c))

    def _apply_fly(self):
        """One fly-movement step, run from the 30 Hz paint tick in
        every state (playing, paused, scrubbing): the legacy contract
        of process_fly_input. dt is wall-clock based and capped."""
        now = time.monotonic()
        dt = 0.0
        if self._fly_last_tick is not None:
            dt = min(now - self._fly_last_tick, self.FLY_MAX_DT_S)
        self._fly_last_tick = now
        if dt <= 0.0 or self._eye is None:
            return
        held = self._fly_keys
        forward = float(Qt.Key_W in held) - float(Qt.Key_S in held)
        right = float(Qt.Key_D in held) - float(Qt.Key_A in held)
        up = float(Qt.Key_E in held) - float(Qt.Key_Q in held)
        norm = np.sqrt(forward ** 2 + right ** 2 + up ** 2)
        if norm == 0.0:
            return
        speed = (self.FLY_SPEED_SCENE_FRACTION * self._fly_diag *
                 self._fly_speed_multiplier)
        if self._fly_boost:
            speed *= self.FLY_SPEED_BOOST
        step = speed * dt / norm
        f, r, u = self._camera_axes()
        move = (f * (forward * step) + r * (right * step) +
                u * (up * step))
        self._eye = self._eye + move
        self._lookat = self._lookat + move
        self._commit_camera()

    def _commit_camera(self):
        if self.renderer is None or self._eye is None:
            return
        self.renderer.scene.camera.look_at(
            self._lookat.astype(np.float32),
            self._eye.astype(np.float32),
            self._up.astype(np.float32))
    def set_point_size(self, size: float):
        """The point radius (pixels). Filament copies the material
        at geometry-add time: the change only lands after the cloud
        re-mounts."""
        if self.mat is None:
            return
        self.mat.point_size = float(size)
        if self.renderer is not None and self._cloud is not None:
            self.renderer.scene.remove_geometry("cloud")
            self.renderer.scene.add_geometry(
                "cloud", self._cloud, self.mat)
            self.paint()

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, ev):
        self.setFocus(Qt.MouseFocusReason)
        if ev.button() == Qt.LeftButton:
            self._drag_last = ev.position()

    def mouseMoveEvent(self, ev):
        if self._drag_last is None or not (ev.buttons() & Qt.LeftButton):
            return
        dx = ev.position().x() - self._drag_last.x()
        dy = ev.position().y() - self._drag_last.y()
        self._drag_last = ev.position()
        self.orbit(dx, dy)
        self.paint()

    def mouseReleaseEvent(self, ev):
        self._drag_last = None

    def wheelEvent(self, ev):
        steps = ev.angleDelta().y() / 120.0
        if steps == 0.0:
            return
        self.dolly(steps)
        self.paint()

    def orbit(self, dx, dy):
        """Turntable drag: yaw around the world up, pitch around the
        camera's right, the polar angle clamped short of the poles.
        The scene follows the cursor: dragging left/right moves the
        camera right/left around the lookat (and dragging down moves
        it up), so the geometry tracks the pointer."""
        if self._eye is None:
            return
        offset = self._eye - self._lookat
        n = np.linalg.norm(offset)
        if n < 1e-9:
            return
        offset = offset / n
        pol = np.degrees(np.arccos(np.clip(
            float(offset @ self._up), -1.0, 1.0)))
        target_pol = np.clip(pol - dy * self.ORBIT_SENSITIVITY,
                             6.0, 174.0)
        yaw = np.radians(-dx * self.ORBIT_SENSITIVITY)
        _, right, _ = self._camera_axes()
        offset = self._rotate(offset, self._up, yaw)
        offset = self._rotate(offset, right,
                              np.radians(target_pol - pol))
        self._eye = self._lookat + offset
        self._commit_camera()

    def dolly(self, steps):
        if self._eye is None:
            return
        offset = (self._eye - self._lookat) * (
            self.ZOOM_PER_NOTCH ** steps)
        self._eye = self._lookat + offset
        self._commit_camera()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rebuild_timer.start(150)

    def _rebuild_render_target(self):
        """The render engine's target size is fixed at construction
        (no Filament resize API): the widget outgrowing it is the
        M2-era blur. The re-create is SEQUENTIAL - the old engine is
        fully released before the new one is built (two live engines
        abort at teardown; measured SIGABRT) - and the pipeline state
        is the viewer's own, so the remount is cheap."""
        if self.renderer is None:
            return
        w, h = self._render_target()
        if (w, h) == self._render_size:
            return
        self.renderer = None
        gc.collect()
        self.attach()
        if self._cloud is not None:
            self.renderer.scene.add_geometry(
                "cloud", self._cloud, self.mat)
        if self._cam_state is not None:
            K, W, H = self._cam_state
            for fid, (q, pose) in self._frustums.items():
                self.renderer.scene.add_geometry(
                    f"kf_{fid}",
                    Viewport._frustum_lines(
                        K, W, H, pose, q),
                    self.mat_line)
        if self._eye is not None:
            self._commit_camera()
        self.paint()

    def paint(self):
        if self.renderer is None:
            return
        self._apply_fly()
        img = self.renderer.render_to_image()
        arr = np.ascontiguousarray(np.asarray(img))
        h, w, ch = arr.shape
        qimg = QImage(arr.data, w, h, ch * w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        dpr = self.devicePixelRatioF()
        if (abs(w - self.width() * dpr) <= 2 and
                abs(h - self.height() * dpr) <= 2):
            # The render matches the widget in device pixels: 1:1, no
            # resample - the sharp path.
            pm.setDevicePixelRatio(dpr)
            self.setPixmap(pm)
        else:
            # A drifted size (between a resize and the debounce): the
            # smooth-fit fallback.
            self.setPixmap(pm.scaled(
                self.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        self.last_render_bgr = cv.cvtColor(arr, cv.COLOR_RGB2BGR)


class TimelineStrip(QFrame):
    """The floating transport pill: a compact translucent panel
    OVER the viewer's bottom-center (the resize handler places it).
    A play/pause toggle + a reset on top, the checkpoint slider
    below with the rewind next to it. The toggle shows the CURRENT
    state's symbol; the slider's drag scrubs view-only, the rewind
    commits the position; the fork markers paint as orange ticks on
    the groove."""

    toggleRequested = Signal()
    resetRequested = Signal()
    rewindRequested = Signal(int)
    scrubRequested = Signal(int)

    PILL_QSS = """
        QFrame#TimelinePill {
            background: rgba(26, 26, 28, 186);
            border: 1px solid rgba(90, 90, 94, 110);
            border-radius: 12px; }
        QFrame#TimelinePill QPushButton {
            background: rgba(62, 62, 66, 150); color: #d0d0d4;
            border: none; border-radius: 4px; padding: 0;
            font-size: 14px; }
        QFrame#TimelinePill QPushButton:hover {
            background: rgba(96, 96, 100, 170); }
        QFrame#TimelinePill QLabel { color: #8e8e94; font-size: 11px; }
        QFrame#TimelinePill QSlider::groove:horizontal {
            background: rgba(74, 74, 78, 160); height: 6px;
            border-radius: 3px; border: none; }
        QFrame#TimelinePill QSlider::handle:horizontal {
            background: #c2c2c6; width: 13px; height: 13px;
            margin: -4px 0; border-radius: 6px; }
    """

    def __init__(self):
        super().__init__()
        self.setObjectName("TimelinePill")
        self.setStyleSheet(self.PILL_QSS)
        # The two rows plus the bottom padding: 6 + 26 + 3 + 22 + 14
        # = 71 in a 74px pill (the slider row breathes off the edge).
        self.setFixedHeight(74)
        self.fork_positions = set()

        top = QHBoxLayout()
        top.setContentsMargins(0, 6, 0, 0)
        top.setSpacing(8)
        self.btn_toggle = QPushButton("▶")
        self.btn_toggle.setToolTip("play / pause")
        self.btn_toggle.setFixedSize(40, 26)
        self.btn_reset = QPushButton("⏮")
        self.btn_reset.setToolTip("reset - a fresh session over the "
                                  "same source")
        self.btn_reset.setFixedSize(40, 26)
        top.addStretch(1)
        top.addWidget(self.btn_toggle)
        top.addWidget(self.btn_reset)
        top.addStretch(1)
        self.btn_toggle.clicked.connect(self.toggleRequested.emit)
        self.btn_reset.clicked.connect(self.resetRequested.emit)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 3, 0, 14)
        bottom.setSpacing(8)
        self.btn_rewind = QPushButton("⇤")
        self.btn_rewind.setToolTip("rewind the pipeline to the "
                                   "slider's checkpoint (branch)")
        self.btn_rewind.setFixedSize(28, 22)
        self.btn_rewind.clicked.connect(
            lambda: self.rewindRequested.emit(self.slider.value()))
        self.slider = CheckpointSlider()
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(
            lambda v: self.scrubRequested.emit(v))
        self.slider.sliderReleased.connect(
            lambda: self.scrubRequested.emit(self.slider.value()))
        bottom.addWidget(self.btn_rewind)
        bottom.addWidget(self.slider, stretch=1)
        self.frame_label = QLabel("frame -")
        bottom.addWidget(self.frame_label)

        vlay = QVBoxLayout(self)
        vlay.setContentsMargins(10, 0, 10, 0)
        vlay.addLayout(top)
        vlay.addLayout(bottom)

    def set_state(self, playing: bool):
        """The toggle's symbol reflects the CURRENT state."""
        self.btn_toggle.setText("⏸" if playing else "▶")
        self.status = "playing" if playing else "paused"

    def set_forks(self, positions):
        self.fork_positions = set(positions)
        self.slider.fork_positions = self.fork_positions
        self.slider.update()

    def sync(self, n_checkpoints, current_index):
        self.slider.setRange(0, max(n_checkpoints - 1, 0))
        self.slider.setEnabled(n_checkpoints > 0)
        self.slider.blockSignals(True)
        self.slider.setValue(current_index)
        self.slider.blockSignals(False)
        self.frame_label.setText(
            f"cp {current_index + 1}/{n_checkpoints}"
            + (f" · {len(self.fork_positions)} fork(s)"
               if self.fork_positions else ""))


class CheckpointSlider(QSlider):
    """The checkpoint slider (horizontal) with the fork markers
    painted on the groove."""

    fork_positions = set()

    def __init__(self):
        super().__init__(Qt.Horizontal)
        self.setRange(0, 10_000)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.fork_positions or self.maximum() <= 0:
            return
        from PySide6.QtGui import QPainter, QColor
        painter = QPainter(self)
        painter.setPen(QColor(255, 160, 60))
        groove_x, groove_w = 4, self.width() - 8
        for pos in sorted(self.fork_positions):
            frac = pos / max(self.maximum(), 1)
            x = int(groove_x + frac * groove_w)
            painter.drawLine(x, 4, x, 14)
        painter.end()
        del groove_x, groove_w


class EditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("sparse-slam editor")
        self.resize(1560, 900)
        self.session = None
        self.summary = None
        self.playing = False
        # The last manual intrinsics entry (the apply-manually form);
        # also the pre-load spec's values when set before a load.
        self._manual_intrinsics = None
        self.status_label = QLabel("no session")
        self.statusBar().addPermanentWidget(self.status_label)

        host = QWidget()
        host.setObjectName("CentralHost")
        vlay = QVBoxLayout(host)
        vlay.setContentsMargins(0, 0, 0, 0)
        self.viewport = Viewport()
        vlay.addWidget(self.viewport, stretch=1)
        self.strip = TimelineStrip()
        self.strip.toggleRequested.connect(self._on_toggle)
        self.strip.resetRequested.connect(self._on_reset)
        self.strip.rewindRequested.connect(self._on_rewind)
        self.strip.scrubRequested.connect(self._on_scrub)
        self.setCentralWidget(host)
        # The transport floats OVER the viewer's bottom-center (the
        # overlay consumes no layout height).
        self.strip.setParent(self.centralWidget())
        self.strip.raise_()

        self.panel = QDockWidget("controls")
        self.panel.setWidget(self._build_panel())
        # No dock title row: an empty title-bar widget collapses it
        # to zero height (the panel's tabs are the header).
        self.panel.setTitleBarWidget(QWidget(self.panel))
        # Flex-shrink: the dock sizes to its content's width once the
        # layout knows the hints (the user can drag it afterwards).
        QTimer.singleShot(120, self._fit_dock_width)
        # The contract holds across tab switches: each page gets a
        # sidebar as wide as that page needs.
        self.tabs.currentChanged.connect(
            lambda _i: QTimer.singleShot(0, self._fit_dock_width))
        self.panel.setFeatures(QDockWidget.DockWidgetFeature.
                               DockWidgetMovable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.panel)

        # The sidebar's collapse toggle: pinned at the viewer's
        # top-left, always reachable while collapsed.
        self.dock_toggle = QPushButton("<", self.centralWidget())
        self.dock_toggle.setToolTip("collapse / expand the sidebar")
        self.dock_toggle.setFixedSize(22, 40)
        # The global QSS pads buttons 6x14 - at 22 px wide the content
        # rect goes negative and the glyph clips to nothing; a compact
        # override keeps the text visible.
        self.dock_toggle.setStyleSheet(
            "padding: 0; font-size: 15px;")
        self.dock_toggle.clicked.connect(self._toggle_sidebar)
        self.dock_toggle.setFocusPolicy(Qt.NoFocus)
        self.dock_toggle.raise_()

        self.capture_label = QLabel(host)
        self.capture_label.setStyleSheet(
            "background: #121214; border: 1px solid #454549;")
        self.capture_label.setFixedSize(300, 169)
        self.capture_label.setAlignment(Qt.AlignCenter)
        self.capture_label.raise_()

        self.paint_timer = QTimer()
        self.paint_timer.timeout.connect(self.viewport.paint)
        self.paint_timer.start(1000 // 30)
        self.viewport.flyStatus.connect(
            lambda msg: self.status_label.setText(f"{msg}"))
        for b in self.findChildren(QPushButton):
            b.setFocusPolicy(Qt.NoFocus)
        # The fly keys route before the focused widget (an app-level
        # filter): clicking load leaves focus on the path field, and
        # a line edit would consume every letter - the filter keeps
        # the legacy keys global.
        self._fly_filter = _FlyKeyFilter(self)
        QApplication.instance().installEventFilter(self._fly_filter)
        self.setFocusPolicy(Qt.StrongFocus)
        self.step_timer = QTimer()
        self.step_timer.timeout.connect(self._tick)
        self.step_timer.start(1000 // 30)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_overlays()

    def _reposition_overlays(self):
        c = self.centralWidget()
        if c is None:
            return
        if hasattr(self, "capture_label"):
            w, h = self.capture_label.width(), \
                self.capture_label.height()
            self.capture_label.move(c.width() - w - 12, 12)
            self.capture_label.raise_()
        if hasattr(self, "dock_toggle"):
            self.dock_toggle.move(4, 12)
            self.dock_toggle.raise_()
        if hasattr(self, "strip"):
            w = min(480, c.width() - 24)
            h = self.strip.height()
            self.strip.setFixedSize(w, self.strip.height())
            self.strip.move((c.width() - w) // 2,
                            c.height() - h - 10)
            self.strip.raise_()

    def _fit_dock_width(self):
        """The sidebar is as wide as its CURRENT page needs (the
        flex-shrink contract: no slack; the rows' trailing stretches
        absorb whatever width the dock keeps)."""
        if not self.panel.isVisible():
            return
        page = self.tabs.currentWidget()
        if page is None:
            return
        # The page hint includes its content (the Advanced page's
        # scroll area reports its inner widget's width, not the
        # scrollbar that its own height forces).
        extra = 24  # tab frame + panel margins
        if page is self.advanced:
            extra += self.style().pixelMetric(
                QStyle.PixelMetric.PM_ScrollBarExtent)
        target = page.sizeHint().width() + extra
        self.resizeDocks([self.panel], [target], Qt.Horizontal)

    def _toggle_sidebar(self):
        collapsed = self.panel.isVisible()
        self.panel.setVisible(not collapsed)
        self.dock_toggle.setText(">" if collapsed else "<")
        # The dock swap resolves the central resize asynchronously;
        # re-place the floating overlays after it lands.
        QTimer.singleShot(0, self._reposition_overlays)
        if not collapsed:
            QTimer.singleShot(0, self._fit_dock_width)

    # ------------------------------------------------------- panel
    def _row_builder(self, svlay):
        """The sidebar rows: each label beside its field (the
        tight_row geometry), appended to the tab's vertical layout."""
        def add_row(label_text, widget):
            svlay.addLayout(tight_row(label_text, widget))
        return add_row

    def _build_panel(self):
        panel = QWidget()
        vlay = QVBoxLayout(panel)
        vlay.setContentsMargins(10, 4, 6, 6)
        vlay.setSpacing(2)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        UiState.apply_everything(lambda msg: info_log(LOG_TAG, msg))

        src = QWidget()
        svlay = QVBoxLayout(src)
        svlay.setContentsMargins(10, 8, 8, 8)
        svlay.setSpacing(6)
        row = self._row_builder(svlay)

        self.kind_combo = QComboBox()
        self.kind_combo.addItems(["video file", "frame sequence",
                                  "camera"])
        self.path_edit = QLineEdit()
        self.camera_row = QSpinBox()
        self.camera_row.setRange(0, 16)

        row("source", self.kind_combo)
        path_row = QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.setSpacing(8)
        path_row.addWidget(self.path_edit, stretch=1)
        browse = QPushButton("browse…")
        browse.clicked.connect(self._browse)
        path_row.addWidget(browse)
        row(None, path_row)
        load_btn = QPushButton("load session")
        load_btn.clicked.connect(self._load_session)
        row(None, load_btn)
        row("camera index", self.camera_row)
        # The point radius (the cloud's point size, pixels), live
        # and granular (quarter-pixel steps).
        self.point_size_row = QDoubleSpinBox()
        self.point_size_row.setRange(0.5, 20.0)
        self.point_size_row.setSingleStep(0.25)
        self.point_size_row.setDecimals(2)
        self.point_size_row.setValue(2.0)
        self.point_size_row.valueChanged.connect(
            self.viewport.set_point_size)
        row("point radius", self.point_size_row)
        self.cache_label = QLabel("")
        row("cache", self.cache_label)
        manual = QVBoxLayout()
        manual.setContentsMargins(0, 0, 0, 0)
        manual.setSpacing(4)
        apply_btn = QPushButton("apply manually (live + cached)")
        apply_btn.clicked.connect(self._open_manual_intrinsics)
        forget_btn = QPushButton("forget this source")
        forget_btn.clicked.connect(self._forget_source)
        manual.addWidget(apply_btn)
        manual.addWidget(forget_btn)
        row("manual intrinsics", manual)
        svlay.addStretch(1)
        self.tabs.addTab(src, "Source")

        opts = QWidget()
        ovlay = QVBoxLayout(opts)
        ovlay.setContentsMargins(10, 8, 8, 8)
        ovlay.setSpacing(6)
        orow = self._row_builder(ovlay)
        self.opt_widgets = {}
        for name, default, marker in (
                ("feature_extractor", "ALIKED", "(rebuild)"),
                ("enable_multiscale_features", False, "(rebuild)"),
                ("enable_ba", True, "(live)"),
                ("record_session", False, "(live)"),
                ("display_capture", True, "(live)"),
                ("enable_debug", True, "(live)"),
                ("enable_profiling", False, "(live)")):
            if isinstance(default, str):
                w = QComboBox()
                w.addItems(["ALIKED", "ORB"])
            else:
                w = QCheckBox()
                w.setChecked(bool(default))
            orow(f"{name} {marker}", w)
            self.opt_widgets[name] = w
        apply_opts = QPushButton("apply options")
        apply_opts.clicked.connect(self._apply_options)
        orow(None, apply_opts)
        ovlay.addStretch(1)
        self.tabs.addTab(opts, "Options")

        from slam.ui.advanced_tab import AdvancedTab
        self.advanced = AdvancedTab(self.status_label.setText)
        # The pipeline's constants live in the step worker: a pin
        # applies UI-side (the tab's own apply_override) AND ships to
        # the worker's modules.
        self.advanced.on_pin_applied = (
            lambda n, v: (self.session.set_pins([(n, v)])
                          if self.session is not None else None))
        self.tabs.addTab(self.advanced, "Advanced")

        vlay.addWidget(self.tabs)
        return panel

    # -------------------------------------------------- source flow
    def _browse(self):
        if self.kind_combo.currentIndex() == 2:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "open source", "", "Videos (*.mp4 *.avi *.mov)")
        if path:
            self.path_edit.setText(path)

    def _current_spec(self) -> SourceSpec:
        index = self.kind_combo.currentIndex()
        location = (str(self.camera_row.value()) if index == 2
                    else self.path_edit.text().strip())
        kind = [FILE_KIND, SEQUENCE_KIND, CAMERA_KIND][index]
        if self._manual_intrinsics is None:
            return SourceSpec(kind, location)
        fx, fy, cx, cy, k1 = self._manual_intrinsics
        return SourceSpec(kind, location, fx=fx,
                          fy=fy or None, cx=cx or None, cy=cy or None,
                          k1=k1)

    def _load_session(self):
        self._start_session(self._current_spec())

    def _start_session(self, spec: SourceSpec):
        self._stop_session()
        try:
            from slam.ui.session import ProcessSession
            session = ProcessSession(spec, self._options())
        except Exception as e:
            # The optional extras (the GeoCalib branch needs 'calib';
            # the learned extractors need 'learned') fail with an
            # actionable message - surfaced LOUDLY, then the user can
            # supply intrinsics manually or install the extra.
            from PySide6.QtWidgets import QMessageBox
            msg = str(e)
            hint = ""
            if "optional 'calib' extra" in msg:
                hint = ("\n\nFix A: run `uv sync --extra calib` in "
                        "this checkout.\nFix B: untick 'fx auto' and "
                        "enter fx/fy/cx/cy/k1 manually, then load "
                        "again.")
            elif "No module named" in msg and "lightglue" in msg:
                hint = ("\n\nFix: run `uv sync --extra learned` in "
                        "this checkout.")
            self.status_label.setText(f"load failed: {msg[:140]}")
            if os.environ.get("EDITOR_SMOKE") != "1":
                QMessageBox.critical(self, "session load failed",
                                     msg + hint)
            return
        self.session = session
        self.viewport.attach()
        self.viewport.unscrub()
        if self.session.options.record_session:
            self.session.start_recording()
        cached = self.session.cache_hit
        fx_txt = (f"{self.session.K[0, 0]:.1f}"
                  if self.session.K is not None else "auto")
        self.cache_label.setText(
            f"{'hit: ' + str(cached.get('provenance')) if cached else 'miss'}"
            f" — fx {fx_txt}")
        self.status_label.setText("loaded")
        self.playing = True

    def _options(self) -> SlamOptions:
        return SlamOptions(
            feature_extractor=self.opt_widgets[
                "feature_extractor"].currentText(),
            enable_multiscale_features=self.opt_widgets[
                "enable_multiscale_features"].isChecked(),
            enable_ba=self.opt_widgets["enable_ba"].isChecked(),
            record_session=self.opt_widgets["record_session"].isChecked(),
            display_capture=self.opt_widgets[
                "display_capture"].isChecked(),
            enable_debug=self.opt_widgets["enable_debug"].isChecked(),
            enable_profiling=self.opt_widgets[
                "enable_profiling"].isChecked())

    def _on_toggle(self):
        if self.session is None:
            if self.path_edit.text().strip():
                self._load_session()
            return
        self._on_pause() if self.playing else self._on_play()

    # The legacy viewer's keys (viz/renderer.py): space pauses, held
    # W/A/S/D/Q/E fly (shift boost), -/= speed. The routing sits at
    # the WINDOW level so no click is needed first; text fields
    # consume their own keys before this sees them.
    def keyPressEvent(self, ev):
        if ev.isAutoRepeat() or not self.fly_route(ev.key(), True):
            super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.isAutoRepeat() or not self.fly_route(ev.key(), False):
            super().keyReleaseEvent(ev)

    def fly_route(self, key, pressed):
        if key == Qt.Key_Space:
            if pressed:
                self._on_toggle()
            return True
        if key in (Qt.Key_Minus, Qt.Key_Equal):
            if pressed:
                self.viewport.fly_speed(
                    1 if key == Qt.Key_Equal else -1)
            return True
        if key == Qt.Key_Shift:
            self.viewport.set_boost(pressed)
            return True
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D,
                   Qt.Key_Q, Qt.Key_E):
            self.viewport.set_held(key, pressed)
            return True
        return False

    def _tick(self) -> None:
        if self.session is None:
            return
        if self.playing:
            self.session.request_step()
        for summary, slot in self.session.drain():
            if summary is None:
                self.playing = False
                self.strip.set_state(False)
                self._on_step_done(None)
                break
            self.summary = summary
            self._on_step_done(
                self.session.result_from_summary(summary, slot))

    # --------------------------------------------------- transport
    def _on_play(self):
        if self.session is None:
            return
        self.viewport.unscrub()
        self.playing = True
        self.strip.set_state(True)
        self.status_label.setText("playing")

    def _on_pause(self):
        self.playing = False
        self.strip.set_state(False)

    def _on_step(self):
        if self.session is None:
            return
        self.playing = False
        self.strip.set_state(False)
        summary, slot = self.session.step()
        if summary is None:
            self.playing = False
            self.strip.set_state(False)
            self._on_step_done(None)
            return
        self.summary = summary
        self._on_step_done(
            self.session.result_from_summary(summary, slot))

    def _on_scrub(self, index):
        if self.session is None or not self.session.checkpoints:
            return
        index = max(0, min(index, len(self.session.checkpoints) - 1))
        cp = self.session.checkpoints[index]
        self.playing = False
        self.viewport.scrub_render(cp)
        fork = (" · forked" if (self.session.checkpoints[index].forked)
                else "")
        self.status_label.setText(
            f"scrub: {cp.label}{fork} — the pipeline untouched "
            f"(rewind to make this state live)")

    def _on_rewind(self, index):
        if self.session is None or not self.session.checkpoints:
            return
        index = max(0, min(index, len(self.session.checkpoints) - 1))
        cp = self.session.checkpoints[index]
        record = self.session.restore_checkpoint(cp)
        self._refresh_forks()
        self.status_label.setText(
            f"rewound to {cp.label} — branch B{record['branch']} "
            f"forked (seed {cp.branch_seed}; "
            f"{len(self.session.branch_log)} branch(es) this session); "
            f"the pipeline resumes from the checkpoint's state "
            f"with the current options/pins")

    def _on_reset(self):
        if self.session is None:
            return
        spec = self.session.spec
        self._stop_session()
        self._start_session(spec)

    def _stop_session(self):
        self.playing = False
        if self.session is not None:
            self.session.save()
            self.session.close()
        self.session = None
        self.viewport.detach()

    def _refresh_forks(self):
        cps = self.session.checkpoints if self.session else []
        self.strip.set_forks([i for i, c in enumerate(cps) if c.forked])

    def _on_step_done(self, result):
        if result is None:
            self.playing = False
            self.session.save()
            self.status_label.setText(
                f"end of stream — {self.session.step_count} step(s)")
            self.strip.sync(len(self.session.checkpoints),
                            max(len(self.session.checkpoints) - 1, 0))
            return
        self._refresh_forks()
        self.strip.sync(len(self.session.checkpoints),
                        max(len(self.session.checkpoints) - 1, 0))
        self.viewport.update_scene_snapshot(self.summary)
        if self.session.options.display_capture and \
                result.capture is not None:
            arr = np.ascontiguousarray(result.capture)
            h, w, ch = arr.shape
            qimg = QImage(arr.data, w, h, ch * w,
                          QImage.Format_RGB888)
            self.capture_label.setPixmap(
                QPixmap.fromImage(qimg).scaled(
                    self.capture_label.size(), Qt.KeepAspectRatio,
                    Qt.SmoothTransformation))
        self.frame_label = self.strip.frame_label
        self.frame_label.setText(f"frame {result.frame_index}")
        status_txt = (f"step {result.step} | kfs {result.keyframes} | "
                      f"points {result.points} | "
                      f"q {result.avg_quality:.2f}")
        profile = self.summary.get("profile") or {}
        if profile:
            top = " ".join(f"{k} {v:.1f}ms" for k, v in
                           list(profile.items())[:3])
            status_txt += f" | {top}"
        self.status_label.setText(status_txt)

    # ------------------------------------- the manual intrinsics JIT
    def _manual_defaults(self):
        """The dialog's prefill: the last entry, else the live
        session's K, else the source's cached row."""
        if self._manual_intrinsics is not None:
            return self._manual_intrinsics
        if self.session is not None and self.session.K is not None:
            K = self.session.K
            k1 = float((self.session.cache_hit or {}).get("k1", 0.0))
            return (float(K[0, 0]), float(K[1, 1]),
                    float(K[0, 2]), float(K[1, 2]), k1)
        cached = intrinsics_cache.lookup(self._current_spec().location)
        if cached:
            return (float(cached.get("fx") or 0.0),
                    float(cached.get("fy") or 0.0),
                    float(cached.get("cx") or 0.0),
                    float(cached.get("cy") or 0.0),
                    float(cached.get("k1") or 0.0))
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    def _open_manual_intrinsics(self):
        values = ask_manual_intrinsics(self, self._manual_defaults())
        if values is None:
            return
        self._manual_intrinsics = values
        if self.session is None:
            self.status_label.setText(
                "manual intrinsics stored; the next load uses them")
            return
        self._apply_manual_intrinsics(*values)

    def _apply_manual_intrinsics(self, fx, fy, cx, cy, k1):
        self._manual_intrinsics = (fx, fy, cx, cy, k1)
        if self.session is None:
            return
        fy = fy or fx
        cx = cx or self.session.CAP_W // 2
        cy = cy or self.session.CAP_H // 2
        # The write-through is automatic: a manual apply PERSISTS (the
        # manual precedence wins on the next load, no button needed) -
        # the pipeline lives in the step worker, so this is a command.
        fx_applied = self.session.set_intrinsics(fx, fy, cx, cy, k1)
        self.session.cache_hit = {"provenance": "manual",
                                  "fx": fx_applied}
        self.cache_label.setText(
            f"hit: manual — fx {float(fx_applied or 0):.1f}")
        self.status_label.setText(
            f"intrinsics applied + cached (manual, fx "
            f"{float(fx_applied or 0):.1f})")

    def _forget_source(self):
        if self.session is None:
            return
        intrinsics_cache.forget(self.session.spec.location)
        self.session.cache_hit = None
        self.cache_label.setText("cached row dropped")

    def _apply_options(self):
        if self.session is None:
            return
        opts = self._options()
        branch = self.session.apply_live_options(opts)
        self.capture_label.setVisible(opts.display_capture)
        if branch:
            self.status_label.setText(
                f"applied live — rebuild required for "
                f"{', '.join(branch)} (the next reset/rewind applies "
                f"them)")
        elif self.session.telemetry_recorder is not None:
            prof = self.session.profile_summary()
            self.status_label.setText(
                f"applied live — profile {prof}")
        else:
            self.status_label.setText("applied live")

    # ------------------------------------------------------ teardown
    def closeEvent(self, event):
        self.paint_timer.stop()
        self.step_timer.stop()
        self._stop_session()
        event.accept()


def run_editor():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = EditorWindow()
    win.show()
    app.exec()
    os._exit(0)  # the M0-measured OffscreenRenderer teardown wedge


def main():
    # The GUI smoke: load, play to the checkpoint, scrub, rewind,
    # then close — on the REAL platform.
    out_dir = Path(__file__).resolve().parents[2] / "output" / "ui_spikes"
    out_dir.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = EditorWindow()
    win.show()

    import faulthandler
    faulthandler.enable()

    seq_dir = Path(__file__).resolve().parents[2] / "output" / \
        "ui_spikes" / "m2_seq3"   # 50 frames: the rewind->EOF trap
    if not seq_dir.exists():
        seq_dir.mkdir(parents=True, exist_ok=True)
        import cv2 as cv
        rng = np.random.default_rng(11)
        base = np.full((480, 640, 3), 90, dtype=np.uint8)
        dots = rng.integers(6, 474, size=(200, 2))
        for i in range(50):
            img = base.copy()
            shift = int(round(-3.0 * i))
            for (x0, y0) in dots:
                x = x0 + shift
                if 2 <= x < 638:
                    cv.circle(img, (int(x), int(y0)), 2,
                              (220, 220, 220), -1)
            cv.imwrite(str(seq_dir / f"{i:06d}.png"), img)
    seq = seq_dir
    win.kind_combo.setCurrentIndex(1)
    win.path_edit.setText(str(seq))
    # The pre-load manual values (the source panel carries no
    # intrinsics fields; the apply-manually form owns them).
    win._manual_intrinsics = (500.0, 500.0, None, None, 0.0)
    win._load_session()

    def spin(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    spin(6000)
    n1 = win.session.step_count
    assert n1 >= 1, f"no steps ran (got {n1})"
    # The drag direction: the scene follows the cursor — a leftward
    # drag moves the camera along +right, a downward one along +up.
    # Restore the camera afterwards so the capture keeps the pose.
    vp = win.viewport
    assert vp._eye is not None, "the camera was never placed"
    before = vp._eye.copy()
    _, right, _ = vp._camera_axes()
    vp.orbit(-20.0, 0.0)
    assert float((vp._eye - before) @ right) > 0.0, \
        "the orbit yaw is inverted (the scene moves against the drag)"
    vp._eye = before.copy()
    vp._commit_camera()
    before = vp._eye.copy()
    vp.orbit(0.0, 20.0)
    assert float((vp._eye - before) @ vp._up) > 0.0, \
        "the orbit pitch is inverted (the scene moves against the drag)"
    vp._eye = before.copy()
    vp._commit_camera()
    print("[smoke] orbit follows the drag (yaw + pitch)", flush=True)
    # The first checkpoint lands at step 25 (CHECKPOINT_EVERY).
    deadline = time.time() + 25
    while len(win.session.checkpoints) < 1 and time.time() < deadline:
        spin(250)
    assert win.session.checkpoints, "no checkpoints captured"
    cps = win.session.checkpoints
    print(f"[smoke] played: steps {win.session.step_count}, "
          f"checkpoints {len(cps)}", flush=True)

    win._on_pause()
    # The scrub: the LAST checkpoint renders view-only.
    win._on_scrub(len(cps) - 1)
    assert win.viewport.scrubbing, "the scrub did not enter"
    print(f"[smoke] scrubbed to {cps[-1].label} "
          f"(the pipeline untouched: step_count "
          f"{win.session.step_count})", flush=True)
    win._on_play()
    spin(400)

    # The rewind: the FIRST checkpoint, then resume stepping.
    win._on_rewind(0)
    assert win.session.frame_index == cps[0].frame_index, \
        "the rewind did not reposition"
    print(f"[smoke] rewound to {cps[0].label} (frame "
          f"{win.session.frame_index})", flush=True)
    spin(1200)
    n2 = win.session.step_count
    print(f"[smoke] post-rewind steps {n2}", flush=True)

    win._on_pause()
    win.session.save()
    forks = [i for i, c in enumerate(win.session.checkpoints)
             if c.forked]
    print(f"[smoke] branch(es): {len(win.session.branch_log)}, "
          f"forked cps: {forks}, "
          f"current branch: B{win.session.current_branch}",
          flush=True)
    assert len(win.session.branch_log) >= 1, \
        "the rewind did not register the fork"
    assert len(forks) >= 1, "no forked checkpoint in the registry"
    if os.environ.get("EDITOR_CAPTURE") == "1":
        win.grab().save(str(out_dir / "editor_m6.png"))
        print("[smoke] window capture saved", flush=True)
    # The point radius field + the automatic cache write-through.
    win.point_size_row.setValue(3.5)
    assert win.viewport.mat.point_size == 3.5, \
        "the radius field did not drive the material"
    win.point_size_row.setValue(2.0)
    # The JIT form builds with the live values prefilled (a modal exec
    # is untestable headless; the prefill + the apply are the wiring).
    from slam.ui.intrinsics_dialog import ManualIntrinsicsDialog
    dialog = ManualIntrinsicsDialog(win, win._manual_defaults())
    assert dialog.fx.value() > 0.0, "the intrinsics form did not prefill"
    dialog.deleteLater()
    win._apply_manual_intrinsics(123.456, None, None, None, 0.0)
    assert win.cache_label.text().startswith("hit: manual"), \
        "the manual apply did not write the cache"
    print("[smoke] point radius live (granular field), manual "
          "intrinsics auto-cached", flush=True)
    # The keyframe frustums (the legacy pose visualization).
    assert len(win.viewport._frustums) >= 1, \
        "no keyframe frustums were drawn"
    print(f"[smoke] frustums: "
          f"{len(win.viewport._frustums)} keyframe wireframes",
          flush=True)
    # The sharp render: the target matches the widget in device
    # pixels, and a real resize rebuilds the engine sequentially.
    vp = win.viewport
    assert vp._render_size == vp._render_target(), \
        f"the render is not device-sized ({vp._render_size} vs " \
        f"{vp._render_target()})"
    win.resize(1100, 760)
    spin(450)
    assert vp._render_size == vp._render_target(), \
        "the resize did not rebuild the render target"
    print(f"[smoke] sharp render: {vp._render_size[0]}x"
          f"{vp._render_size[1]} device px, rebuild-on-resize ok",
          flush=True)
    # The sidebar collapse toggle.
    win._toggle_sidebar()
    spin(100)
    assert not win.panel.isVisible(), "the sidebar did not collapse"
    win._toggle_sidebar()
    spin(100)
    assert win.panel.isVisible(), "the sidebar did not expand"
    print("[smoke] sidebar toggle: collapse/expand ok", flush=True)

    # The fly camera (the legacy movement contract): real key events
    # through the window routing + the pointer manipulations.
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtGui import QKeyEvent

    vp = win.viewport

    def key(key, pressed):
        kind = QEvent.KeyPress if pressed else QEvent.KeyRelease
        ev = QKeyEvent(kind, key, Qt.KeyboardModifier.NoModifier)
        QCoreApplication.sendEvent(win, ev)

    win._on_pause()
    spin(300)
    eye0 = vp._eye.copy()
    assert eye0 is not None, "the camera never placed"
    key(Qt.Key_W, True)
    spin(700)
    key(Qt.Key_W, False)
    spin(100)
    assert not np.allclose(eye0, vp._eye), "W did not move the camera"
    print(f"[smoke] fly: the eye moved "
          f"{np.linalg.norm(vp._eye - eye0):.3f} units on held W",
          flush=True)

    fwd, _, _ = vp._camera_axes()
    delta = vp._eye - eye0
    along = abs(float(delta @ fwd)) / max(np.linalg.norm(delta), 1e-9)
    assert along > 0.99, \
        f"the fly move left the view axis (forward share {along:.2f})"

    diag0 = np.linalg.norm(vp._eye - vp._lookat)
    vp.dolly(2)
    diag1 = np.linalg.norm(vp._eye - vp._lookat)
    assert diag1 < diag0 * 0.8, "the wheel dolly did not zoom in"
    vp.orbit(60, 0)
    lookat = vp._lookat.copy()
    assert np.allclose(vp._lookat, lookat), "the orbit moved the lookat"
    print(f"[smoke] orbit+dolly: the camera distance "
          f"{diag0:.2f} -> {diag1:.2f}, the lookat pinned",
          flush=True)

    playing_before = win.playing
    assert not playing_before, "expected paused for the space toggle"
    key(Qt.Key_Space, True)
    assert win.playing, "space did not resume"
    key(Qt.Key_Space, True)
    assert not win.playing, "space did not pause"
    print("[smoke] space toggle: pause/resume ok", flush=True)

    # The real-dispatch path: a TEXT field keeps its keys, focus on
    # the scene routes to the fly (the user-visible contract).
    win.path_edit.setFocus()
    wev = QKeyEvent(QEvent.KeyPress, Qt.Key_W,
                    Qt.KeyboardModifier.NoModifier, "w")
    QCoreApplication.sendEvent(win.path_edit, wev)
    assert win.path_edit.text().endswith("w"), "typing lost"
    assert np.allclose(vp._eye, vp._eye), "typed while the scene flew"
    moved_before = vp._eye.copy()
    vp.setFocus()
    kp = QKeyEvent(QEvent.KeyPress, Qt.Key_W,
                   Qt.KeyboardModifier.NoModifier)
    QCoreApplication.sendEvent(vp, kp)
    spin(300)
    kr = QKeyEvent(QEvent.KeyRelease, Qt.Key_W,
                   Qt.KeyboardModifier.NoModifier)
    QCoreApplication.sendEvent(vp, kr)
    spin(150)
    assert not np.allclose(vp._eye, moved_before), \
        "W did not fly from scene focus"
    win.path_edit.setText(str(seq))
    print("[smoke] dispatch: typing keeps its keys, scene focus flies",
          flush=True)
    spin(200)
    print("[smoke] PASS", flush=True)
    win.close()
    spin(200)
    app.processEvents()
    os._exit(0)


def run_editor():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = EditorWindow()
    win.show()
    app.exec()
    os._exit(0)  # the M0-measured OffscreenRenderer teardown wedge


if __name__ == "__main__":
    if os.environ.get("EDITOR_SMOKE") == "1":
        main()
    else:
        run_editor()
