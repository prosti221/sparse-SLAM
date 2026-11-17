"""The slam editor (M2): a dark PySide6 window whose central viewport
is Open3D's OFFSCREEN render of the live map, with the pipeline
stepped on the MAIN thread by a QTimer.

Threading model (measured, the M2 pivot): the pipeline is
main-thread-bound on darwin (torch/Metal + the OpenCV/g2o paths die
on a worker thread — both extractor kinds) — the legacy-proven
main-thread loop survives as the step timer; the UI's callbacks
interleave between ticks and the 30 Hz paint timer keeps the render
smooth at the step cadence. The M5 rewind (later) mutates between
ticks on this same thread — no locking anywhere by construction.

Teardown hazard (measured in M0): destroy the OffscreenRenderer before
the Qt session exits — the exit path tears the session down first and
ends with an explicit process exit.
"""
import os
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

import open3d as o3d
from PySide6.QtCore import QEventLoop, Qt, QTimer
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
from slam.utils.logger import DEBUG as slam_debug_flag, info_log

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
    boundaries (the tick boundary — no pipeline reads from the paint);
    the 30 Hz paint tick advances the camera and re-renders."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #0d0f13;")
        self.renderer = None
        self._angle = 0.0
        self._last_center = None
        self.last_render_bgr = None

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
        if self.renderer is not None:
            self.renderer = None

    def update_scene(self, session: Session):
        """Call on the step boundary (same thread, between ticks)."""
        if self.renderer is None:
            self.attach()
        pcd = o3d.geometry.PointCloud()
        n = len(session.global_map.points)
        if n:
            pts = np.array([p._pt_3d
                            for p in session.global_map.points])
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.paint_uniform_color([0.55, 0.75, 1.0])
        else:
            pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
            pcd.paint_uniform_color([0.55, 0.75, 1.0])
        self.renderer.scene.remove_geometry("cloud")
        self.renderer.scene.add_geometry("cloud", pcd, self.mat)
        pose = (session.tracker.cur_frame.pose
                if getattr(session.tracker, "cur_frame", None) else None)
        if pose is not None:
            self.note_center(pose[:3, 3])

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
        """The 30 Hz tick: camera-only advance + re-render (no map
        reads)."""
        if self.renderer is None:
            return
        self.follow_center()
        img = self.renderer.render_to_image()
        arr = np.ascontiguousarray(np.asarray(img))
        h, w, ch = arr.shape
        qimg = QImage(arr.data, w, h, ch * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation))
        self.last_render_bgr = cv.cvtColor(arr, cv.COLOR_RGB2BGR)


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
        vlay.addWidget(self._build_strip())
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
        vlay.addWidget(self.capture_label)

        self.status_label = QLabel("no session")
        self.statusBar().addPermanentWidget(self.status_label)

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

        # The launch-time pin application (the ui_state's persistence
        # precedes any session build).
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
        apply_opts = QPushButton("apply options (new-build marked)")
        apply_opts.clicked.connect(self._apply_options)
        oform.addRow("", apply_opts)
        self.tabs.addTab(opts, "Options")

        from slam.ui.advanced_tab import AdvancedTab
        self.advanced = AdvancedTab(self.status_label.setText)
        self.tabs.addTab(self.advanced, "Advanced")

        vlay.addWidget(self.tabs)
        return panel

    def _build_strip(self):
        strip = QFrame()
        strip.setFixedHeight(64)
        strip.setStyleSheet("background: #15181e;")
        lay = QHBoxLayout(strip)
        lay.setContentsMargins(10, 6, 10, 6)
        self.btn_prev = QPushButton("prev")
        self.btn_play = QPushButton("play")
        self.btn_pause = QPushButton("pause")
        self.btn_step = QPushButton("step")
        self.btn_reset = QPushButton("reset")
        self.btn_play.clicked.connect(self._on_play)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_step.clicked.connect(self._on_step)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_prev.clicked.connect(self._on_prev)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 10_000)
        self.timeline.setEnabled(False)
        for b in (self.btn_prev, self.btn_play, self.btn_pause,
                  self.btn_step, self.btn_reset):
            lay.addWidget(b)
        lay.addWidget(self.timeline, stretch=1)
        self.frame_label = QLabel("frame —")
        lay.addWidget(self.frame_label)
        return strip

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
            print(f"[editor] open failed: {e}", flush=True)
            self.status_label.setText(f"open failed: {e}")
            return
        self.session = session
        self.viewport.attach()
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
        self.playing = True

    def _on_pause(self):
        self.playing = False

    def _on_step(self):
        if self.session is None:
            return
        self.playing = False
        self._on_step_done(self.session.step())

    def _on_prev(self):
        # The M5 rewind machinery: the checkpoint walk-back placeholder.
        self.status_label.setText("prev = the checkpoint walk-back (M5)")

    def _on_reset(self):
        if self.session is None:
            return
        spec = self.session.spec
        self._stop_session()
        self._start_session(spec)

    def _stop_session(self):
        self.playing = False
        if self.session is not None:
            self.session.close()
        self.session = None
        self.viewport.detach()

    def _on_step_done(self, result):
        if result is None:
            self.playing = False
            self.session.save()
            self.status_label.setText(
                f"end of stream — {self.session.step_count} step(s)")
            return
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
        self.timeline.setValue(
            int(10000 * result.frame_index /
                max(result.frame_index + 1, 1)))

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


def main():
    # The GUI smoke: load the Source tab's current spec, play to a
    # step count, pause/step, reset, then close — on the REAL
    # platform (the pipeline is main-thread-bound per the M2 pivot).
    frames = int(os.environ.get("SMOKE_FRAMES", "8"))
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

    spin(frames * 500)
    n1 = win.session.step_count
    assert n1 >= 1, f"no steps ran (got {n1})"
    print(f"[smoke] played: steps {n1}, pts "
          f"{len(win.session.global_map.points)}", flush=True)

    win._on_pause()
    held = win.session.step_count
    spin(400)
    assert win.session.step_count == held, "a paused session stepped"
    print(f"[smoke] paused at {held}", flush=True)

    # M3's hot toggles: the debug flag, the capture view, the
    # profiling recorder.
    win.opt_widgets["enable_debug"].setChecked(False)
    win._apply_options()
    assert slam_logger_mod.DEBUG is False, "the debug hot flip failed"
    win.opt_widgets["display_capture"].setChecked(False)
    win._apply_options()
    assert win.capture_label.isHidden(), "the capture view stayed up"
    win.opt_widgets["display_capture"].setChecked(True)
    win._apply_options()
    assert not win.capture_label.isHidden(), "the capture view hid"
    print("[smoke] capture view toggled", flush=True)
    win.opt_widgets["enable_profiling"].setChecked(True)
    win._apply_options()
    assert win.session.telemetry_recorder is not None, \
        "the profiling recorder did not attach"
    assert win.session.profile_summary() is not None, \
        "no profile summary after the attach"
    print(f"[smoke] profiling on: {win.session.profile_summary()}",
          flush=True)
    win.opt_widgets["enable_debug"].setChecked(True)
    win._apply_options()
    assert slam_logger_mod.DEBUG is True, "the debug flag did not restore"

    win.opt_widgets["enable_ba"].setChecked(False)
    win._apply_options()
    assert win.session.tracker.enable_ba is False, \
        "the enable_ba hot flip failed"
    win.opt_widgets["enable_ba"].setChecked(True)
    win._apply_options()
    assert win.session.tracker.enable_ba is True, \
        "the enable_ba flag did not restore"

    # M4: the constants pinning (live through the shared resolver).
    base_row = win.advanced.rows["RELOCALIZATION_QUALITY_THRESHOLD"][0]
    base_row.setValue(0.5)
    from slam.utils.overrides import current_value
    assert current_value("RELOCALIZATION_QUALITY_THRESHOLD") == 0.5, \
        "the constant pin did not reach the module"
    assert Path("data/ui/ui_state.json").exists(), \
        "the pin was not persisted"
    print("[smoke] constant pinned live + persisted", flush=True)
    win.advanced.reset_all()
    assert abs(current_value("RELOCALIZATION_QUALITY_THRESHOLD")
               - 0.3) < 1e-9, "the reset did not restore the default"
    print("[smoke] reset to defaults", flush=True)

    win._on_step()
    spin(600)
    print(f"[smoke] stepped once: {win.session.step_count}", flush=True)

    win._on_reset()
    spin(2000)
    assert win.session is not None, "reset left no session"
    n2 = win.session.step_count
    print(f"[smoke] reset: new session stepping (steps {n2})", flush=True)
    assert n2 >= 1, "the reset session did not step"

    win._on_pause()
    win.session.save()
    print("[smoke] PASS", flush=True)
    win.close()
    spin(200)
    app.processEvents()
    print("0", flush=True)


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
