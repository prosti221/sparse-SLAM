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
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

import open3d as o3d
from PySide6.QtCore import QEventLoop, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QSlider, QSpinBox, QTabWidget,
    QVBoxLayout, QWidget)

from slam.calibration import IntrinsicsEstimate
from slam.core import intrinsics_cache
from slam.core.source import (CAMERA_KIND, SEQUENCE_KIND, SourceSpec)
from slam.ui.advanced_tab import UiState
from slam.ui.session import Session, SlamOptions
from slam.utils import logger as slam_logger_mod
from slam.utils.logger import info_log

LOG_TAG = "Editor"

DARK_QSS = """
QMainWindow { background: #1b1e24; }
QWidget#CentralHost, QDockWidget, QDockWidget::title {
    background: #1b1e24; color: #9aa4b2; }
QTabWidget::pane { border: 1px solid #2a2f3a; background: #232730; }
QTabBar::tab { background: #1b1e24; color: #9aa4b2; padding: 8px 18px;
    border: 1px solid #2a2f3a; }
QTabBar::tab:selected { background: #2a2f3a; color: #e6ebf2; }
QPushButton { background: #2a2f3a; color: #d5dbe4; border: 1px solid
    #3a4254; border-radius: 4px; padding: 6px 14px; }
QPushButton:hover { background: #3a4254; }
QLabel { color: #c9d1dc; }
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit { background: #232730;
    color: #d5dbe4; border: 1px solid #3a4254; padding: 4px; }
QCheckBox { color: #9aa4b2; }
QSlider::groove:horizontal { border: 1px solid #2a2f3a; height: 8px;
    background: #232730; border-radius: 4px; }
QSlider::handle:horizontal { background: #5b9cf5; width: 14px;
    margin: -4px 0; border-radius: 7px; }
"""


class Viewport(QLabel):
    """The offscreen Open3D render. Geometry refreshes only on step
    boundaries; the 30 Hz paint tick is camera-only. A SCRUB renders
    a checkpoint's snapshot geometry (the pipeline untouched)."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #0d0f13;")
        self.renderer = None
        self._angle = 0.0
        self._last_center = None
        self.last_render_bgr = None
        self.scrubbing = False

    def attach(self, w=960, h=540):
        if self.renderer is not None:
            return
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(
            w, h)
        self.mat = o3d.visualization.rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 2.0
        self.renderer.scene.set_background([0.04, 0.05, 0.07, 1.0])

    def detach(self):
        self.renderer = None
        self.scrubbing = False

    @staticmethod
    def _cloud_from(points):
        pcd = o3d.geometry.PointCloud()
        if points:
            pcd.points = o3d.utility.Vector3dVector(np.array(
                [p._pt_3d for p in points]))
        else:
            pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        pcd.paint_uniform_color([0.55, 0.75, 1.0])
        return pcd

    def update_scene(self, session: Session):
        """On the step boundary (same thread, between ticks)."""
        if self.renderer is None:
            self.attach()
        self.renderer.scene.remove_geometry("cloud")
        self.renderer.scene.add_geometry(
            "cloud", Viewport._cloud_from(session.global_map.points),
            self.mat)
        pose = (session.tracker.cur_frame.pose
                if getattr(session.tracker, "cur_frame", None) else None)
        if pose is not None:
            self.note_center(pose[:3, 3])

    def scrub_render(self, checkpoint):
        """The VIEW-only render of a checkpoint's snapshot geometry
        (the pipeline is not touched)."""
        if self.renderer is None:
            self.attach()
        self.scrubbing = True
        pts = [ps["pt_3d"] for ps in checkpoint.map_state["points"]]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.array(pts) if pts else np.zeros((1, 3)))
        pcd.paint_uniform_color([0.8, 0.65, 0.4])
        self.renderer.scene.remove_geometry("cloud")
        self.renderer.scene.add_geometry("cloud", pcd, self.mat)
        self.paint()

    def unscrub(self):
        self.scrubbing = False

    def note_center(self, center):
        if center is not None:
            self._last_center = np.asarray(center, dtype=np.float32)

    def follow_center(self):
        if self._last_center is None:
            return
        c = self._last_center
        self._angle += np.radians(1.2)
        eye = c + np.array([2.5 * np.cos(self._angle), 2.0,
                            2.5 * np.sin(self._angle)],
                           dtype=np.float32)
        self.renderer.scene.camera.look_at(c, eye,
                                           np.array([0, 1, 0],
                                                    dtype=np.float32))

    def paint(self):
        if self.renderer is None:
            return
        if not self.scrubbing:
            self.follow_center()
        img = self.renderer.render_to_image()
        arr = np.ascontiguousarray(np.asarray(img))
        h, w, ch = arr.shape
        qimg = QImage(arr.data, w, h, ch * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation))
        self.last_render_bgr = cv.cvtColor(arr, cv.COLOR_RGB2BGR)


class TimelineStrip(QFrame):
    """The bottom strip: the transport + the checkpoint timeline.

    While PLAYING the slider tracks the newest checkpoint; while
    PAUSED selecting a checkpoint SCRUBS (the viewport renders that
    checkpoint's snapshot geometry, the pipeline untouched) and
    REWIND restores the pipeline to it (the branch point)."""

    playRequested = Signal()
    pauseRequested = Signal()
    stepRequested = Signal()
    resetRequested = Signal()
    rewindRequested = Signal(int)
    scrubRequested = Signal(int)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(64)
        self.setStyleSheet("background: #15181e;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        self.btn_rewind = QPushButton("rewind")
        self.btn_play = QPushButton("play")
        self.btn_pause = QPushButton("pause")
        self.btn_step = QPushButton("step")
        self.btn_reset = QPushButton("reset")
        self.btn_rewind.clicked.connect(self._emit_rewind)
        self.btn_play.clicked.connect(self.playRequested.emit)
        self.btn_pause.clicked.connect(self.pauseRequested.emit)
        self.btn_step.clicked.connect(self.stepRequested.emit)
        self.btn_reset.clicked.connect(self.resetRequested.emit)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(self._on_scrub)
        self.checkpoint_ticks: list = []
        for b in (self.btn_rewind, self.btn_play, self.btn_pause,
                  self.btn_step, self.btn_reset):
            lay.addWidget(b)
        lay.addWidget(self.slider, stretch=1)
        self.frame_label = QLabel("frame —")
        lay.addWidget(self.frame_label)

    def _emit_rewind(self):
        self.rewindRequested.emit(self.slider.value())

    def _on_scrub(self):
        if self.slider.isEnabled():
            self.scrubRequested.emit(self.slider.value())

    def sync(self, n_checkpoints, current_index):
        self.slider.setRange(0, max(n_checkpoints - 1, 0))
        self.slider.setEnabled(n_checkpoints > 0)
        self.slider.blockSignals(True)
        self.slider.setValue(current_index)
        self.slider.blockSignals(False)
        self.frame_label.setText(
            f"cp {current_index + 1}/{n_checkpoints}" if
            n_checkpoints else "frame —")


class EditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("sparse-slam editor")
        self.resize(1560, 900)
        self.session: Session = None
        self.playing = False
        self.status_label = QLabel("no session")
        self.statusBar().addPermanentWidget(self.status_label)

        host = QWidget()
        host.setObjectName("CentralHost")
        vlay = QVBoxLayout(host)
        vlay.setContentsMargins(0, 0, 0, 0)
        self.viewport = Viewport()
        vlay.addWidget(self.viewport, stretch=1)
        self.strip = TimelineStrip()
        self.strip.playRequested.connect(self._on_play)
        self.strip.pauseRequested.connect(self._on_pause)
        self.strip.stepRequested.connect(self._on_step)
        self.strip.resetRequested.connect(self._on_reset)
        self.strip.rewindRequested.connect(self._on_rewind)
        self.strip.scrubRequested.connect(self._on_scrub)
        vlay.addWidget(self.strip)
        self.setCentralWidget(host)

        panel = QDockWidget("controls")
        panel.setWidget(self._build_panel())
        panel.setFeatures(QDockWidget.DockWidgetFeature.
                          DockWidgetMovable)
        self.addDockWidget(Qt.LeftDockWidgetArea, panel)

        self.capture_label = QLabel()
        self.capture_label.setStyleSheet("background: #0d0f13;")
        self.capture_label.setFixedHeight(150)
        self.capture_label.setAlignment(Qt.AlignCenter)

        self.capture_label.setVisible(True)
        vlay.addWidget(self.capture_label)

        self.paint_timer = QTimer()
        self.paint_timer.timeout.connect(self.viewport.paint)
        self.paint_timer.start(1000 // 30)
        self.step_timer = QTimer()
        self.step_timer.timeout.connect(self._tick)
        self.step_timer.start(1000 // 30)

    # ------------------------------------------------------- panel
    def _build_panel(self):
        panel = QWidget()
        vlay = QVBoxLayout(panel)
        vlay.setContentsMargins(6, 6, 6, 6)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        UiState.apply_everything(lambda msg: info_log(LOG_TAG, msg))

        src = QWidget()
        form = QFormLayout(src)
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(["video file", "frame sequence",
                                  "camera"])
        self.path_edit = QLineEdit()
        browse = QPushButton("browse…")
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(browse)
        self.camera_row = QSpinBox()
        self.camera_row.setRange(0, 16)
        self.fx_spin = QDoubleSpinBox()
        self.fx_spin.setRange(0.0, 100000.0)
        self.fx_spin.setDecimals(3)
        self.fy_spin = QDoubleSpinBox()
        self.fy_spin.setRange(0.0, 100000.0)
        self.fy_spin.setDecimals(3)
        self.cx_spin = QDoubleSpinBox()
        self.cx_spin.setMaximum(100000.0)
        self.cy_spin = QDoubleSpinBox()
        self.cy_spin.setMaximum(100000.0)
        self.k1_spin = QDoubleSpinBox()
        self.k1_spin.setRange(-1.0, 1.0)
        self.k1_spin.setDecimals(6)
        self.fx_auto = QCheckBox()
        form.addRow("source", self.kind_combo)
        form.addRow("", path_row)
        form.addRow("camera index", self.camera_row)
        form.addRow("fx", self.fx_spin)
        form.addRow("fy", self.fy_spin)
        form.addRow("cx", self.cx_spin)
        form.addRow("cy", self.cy_spin)
        form.addRow("k1", self.k1_spin)
        form.addRow("fx auto (GeoCalib)", self.fx_auto)
        self.cache_label = QLabel("")
        form.addRow("cache", self.cache_label)
        load_btn = QPushButton("load session")
        load_btn.clicked.connect(self._load_session)
        form.addRow("", load_btn)
        manual = QFormLayout()
        apply_btn = QPushButton("apply manually (live)")
        apply_btn.clicked.connect(self._apply_manual_intrinsics)
        save_btn = QPushButton("save to cache")
        save_btn.clicked.connect(self._save_intrinsics_to_cache)
        forget_btn = QPushButton("forget this source")
        forget_btn.clicked.connect(self._forget_source)
        manual.addRow(apply_btn)
        manual.addRow(save_btn)
        manual.addRow(forget_btn)
        form.addRow("manual intrinsics", manual)
        self.tabs.addTab(src, "Source")

        opts = QWidget()
        oform = QFormLayout(opts)
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
            oform.addRow(f"{name} {marker}", w)
            self.opt_widgets[name] = w
        apply_opts = QPushButton("apply options")
        apply_opts.clicked.connect(self._apply_options)
        oform.addRow("", apply_opts)
        self.tabs.addTab(opts, "Options")

        from slam.ui.advanced_tab import AdvancedTab
        self.advanced = AdvancedTab(self.status_label.setText)
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
        fx = "auto" if self.fx_auto.isChecked() else (
            self.fx_spin.value() or "auto")
        return SourceSpec(
            ["file", "sequence", "camera"][self.kind_combo.currentIndex()],
            self.path_edit.text().strip(), fx=fx,
            fy=(self.fy_spin.value() or None),
            cx=(self.cx_spin.value() or None),
            cy=(self.cy_spin.value() or None),
            k1=self.k1_spin.value())

    def _load_session(self):
        if self.kind_combo.currentIndex() == 2:
            spec = SourceSpec(CAMERA_KIND, str(self.camera_row.value()),
                              fx="auto")
        else:
            spec = self._current_spec()
        self._start_session(spec)

    def _start_session(self, spec: SourceSpec):
        self._stop_session()
        try:
            session = Session(spec, self._options())
        except FileNotFoundError as e:
            self.status_label.setText(f"open failed: {e}")
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

    def _tick(self) -> None:
        if not self.playing or self.session is None:
            return
        self._on_step_done(self.session.step())

    # --------------------------------------------------- transport
    def _on_play(self):
        if self.session is None:
            return
        self.viewport.unscrub()
        self.playing = True
        self.status_label.setText("playing")

    def _on_pause(self):
        self.playing = False

    def _on_step(self):
        if self.session is None:
            return
        self.playing = False
        self._on_step_done(self.session.step())

    def _on_scrub(self, index):
        if self.session is None or not self.session.checkpoints:
            return
        index = max(0, min(index, len(self.session.checkpoints) - 1))
        cp = self.session.checkpoints[index]
        self.playing = False
        self.viewport.scrub_render(cp)
        self.status_label.setText(
            f"scrub: {cp.label} — the pipeline untouched "
            f"(rewind to make this state live)")

    def _on_rewind(self, index):
        if self.session is None or not self.session.checkpoints:
            return
        index = max(0, min(index, len(self.session.checkpoints) - 1))
        cp = self.session.checkpoints[index]
        opts = self._options()
        self.session.restore_checkpoint(cp)
        self.session.options = opts
        self.session.tracker.enable_ba = opts.enable_ba
        self.viewport.unscrub()
        self.playing = True
        self.status_label.setText(
            f"rewound to {cp.label} — the branch seed "
            f"{cp.branch_seed}; the pipeline resumes from the "
            f"checkpoint's state")

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

    def _on_step_done(self, result):
        if result is None:
            self.playing = False
            self.session.save()
            self.status_label.setText(
                f"end of stream — {self.session.step_count} step(s)")
            self.strip.sync(len(self.session.checkpoints),
                            max(len(self.session.checkpoints) - 1, 0))
            return
        self.strip.sync(len(self.session.checkpoints),
                        max(len(self.session.checkpoints) - 1, 0))
        self.viewport.update_scene(self.session)
        if result.pose is not None:
            self.viewport.note_center(result.pose[:3, 3])
        self.session.render_bgr = self.viewport.last_render_bgr
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
        profile = self.session.profile_summary() or {}
        if profile:
            top = " ".join(f"{k} {v:.1f}ms" for k, v in
                           list(profile.items())[:3])
            status_txt += f" | {top}"
        self.status_label.setText(status_txt)

    # ------------------------------------- M1b's manual entry surface
    def _apply_manual_intrinsics(self):
        if self.session is None:
            return
        fx = self.fx_spin.value()
        fy = self.fy_spin.value() or fx
        cx = self.cx_spin.value() or self.session.CAP_W // 2
        cy = self.cy_spin.value() or self.session.CAP_H // 2
        self.session.global_map.intrinsics = IntrinsicsEstimate.from_K(
            np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
            k1=self.k1_spin.value(), source="manual")
        self.status_label.setText("intrinsics applied (manual)")

    def _save_intrinsics_to_cache(self):
        if self.session is None:
            return
        K = self.session.global_map.intrinsics.K
        intrinsics_cache.store(
            self.session.spec.location, "manual",
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
            k1=float(self.k1_spin.value()))
        self.session.cache_hit = intrinsics_cache.lookup(
            self.session.spec.location)
        self.cache_label.setText(
            f"hit: manual — fx {float(K[0, 0]):.1f}")

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
        self.capture_label.setFixedHeight(150 if opts.display_capture
                                          else 0)
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
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = EditorWindow()
    win.show()

    seq = Path(os.environ.get(
        "SMOKE_SEQ",
        str(Path(__file__).resolve().parents[2] / "output"
            / "ui_spikes" / "m2_seq2")))
    win.kind_combo.setCurrentIndex(1)
    win.path_edit.setText(str(seq))
    win.fx_spin.setValue(500.0)
    win.fy_spin.setValue(500.0)
    win._load_session()

    def spin(ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    spin(6000)
    n1 = win.session.step_count
    assert n1 >= 1, f"no steps ran (got {n1})"
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
