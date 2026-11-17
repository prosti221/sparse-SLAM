"""The Advanced tab: the impactful constants, live-pinned through the
shared override resolver.

The shortlist comes from the measured programs (the recovery/storm
machinery, the frame-intake cadence, the rotation-model machinery, the
PnP acceptance clamps, the map/keyframe/BA core, the loop gates) —
each group a header, each row: a spin/flag widget (the live module
value), the default captured at launch, and a per-row reset. Pins
apply LIVE (the consumers read the constants through the module alias
at call time — the same mechanism the harness's --override sweeps
use) and persist to ui_state.json so the tuning survives restarts.
"""
import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QFrame, QLabel, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from slam.utils.overrides import (apply_override, current_value,
                                  resolve_module)

LOG_TAG = "Advanced"

UI_STATE_PATH = Path("data/ui/ui_state.json")

# (group, name, kind) — kind: f = a float spin, i = an int spin,
# flag = a 0/1 checkbox.
SPECS = [
    ("recovery / reset storm machinery",
     [("RELOCALIZATION_QUALITY_THRESHOLD", "f"),
      ("RELOCALIZATION_FRAME_COUNT", "i"),
      ("MAX_CONSECUTIVE_SUCCESSFUL_RELOCALIZATIONS", "i"),
      ("MAX_CONSECUTIVE_RELOCALIZATION_FAILURES", "i"),
      ("MIN_RELOCALIZATION_INLIERS", "i"),
      ("MIN_AGGRESSIVE_RELOCALIZATION_INLIERS", "i")]),
    ("frame intake cadence",
     [("FRAME_INTAKE_MIN_DISPLACEMENT_PX", "f"),
      ("FRAME_INTAKE_MAX_GAP_S", "f"),
      ("FRAME_INTAKE_MIN_MATCHES", "i")]),
    ("rotation model + bootstrap yield",
     [("ROTMODEL_BOOTSTRAP_YIELD_EVERY", "i"),
      ("ROTMODEL_BOOTSTRAP_MIN_POINTS", "i"),
      ("ROTATION_MODEL_T_COHERENCE_REARM", "f"),
      ("MATCHER_PRIOR_DECISIVE_FLIP_RATIO", "f")]),
    ("PnP acceptance",
     [("PNP_MAX_MEDIAN_REPROJECTION_ERROR", "f"),
      ("PNP_GUARD_TAU_K", "f"),
      ("PNP_GUARD_EMA_ALPHA", "f"),
      ("PNP_MAX_CENTER_STEP_RATIO", "f"),
      ("PNP_MAX_ROTATION_STEP_DEG", "f"),
      ("PNP_MINIMUM_INLIERS", "i")]),
    ("map / keyframe / BA core",
     [("MINIMUM_BASELINE_THRESHOLD", "f"),
      ("KEYFRAME_MINIMUM_ROTATION_DEG", "f"),
      ("MAX_NUMBER_OF_FRAMES_BETWEEN_KEYFRAMES", "i"),
      ("LOCAL_MAP_WINDOW_SIZE", "i"),
      ("BA_TERMINATE_GAIN_THRESHOLD", "f"),
      ("MIN_PARALLAX_DEG", "f"),
      ("MAX_TRI_REPROJECTION_PX", "f")]),
    ("loop closure",
     [("LOOP_CLOSURE_ENABLED", "flag"),
      ("LOOP_CORRECTION_ENABLED", "flag"),
      ("LOOP_MIN_DRIFT_M", "f"),
      ("LOOP_MAX_DRIFT_M", "f"),
      ("LOOP_CORRECTION_COOLDOWN_KF", "i")]),
]


class AdvancedTab(QWidget):
    """The grouped shortlist with live pinning + the session state."""

    def __init__(self, status_cb=lambda *_: None):
        super().__init__()
        self.status_cb = status_cb
        self.rows = {}           # name -> (widget, default, kind)
        inner = QWidget()
        vlay = QVBoxLayout(inner)
        vlay.setContentsMargins(8, 8, 8, 8)
        for group, rows in SPECS:
            header = QLabel(group)
            header.setStyleSheet("color: #6fa8dc; font-weight: bold;")
            vlay.addWidget(header)
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignRight)
            for name, kind in rows:
                w = self._make_row(name, kind)
                if w is None:  # unresolvable name: skipped with a note
                    form.addRow(f"{name}", QLabel("unresolvable"))
                    continue
                form.addRow(name, w)
            holder = QFrame()
            holder.setLayout(form)
            holder.setStyleSheet("color: #c9d1dc;")
            vlay.addWidget(holder)
        vlay.addStretch(1)

        buttons = QFrame()
        blay = QVBoxLayout(buttons)
        refresh = QPushButton("refresh from the live modules")
        refresh.clicked.connect(self.refresh_from_modules)
        reset = QPushButton("reset ALL to defaults")
        reset.clicked.connect(self.reset_all)
        blay.addWidget(refresh)
        blay.addWidget(reset)
        vlay.addWidget(buttons)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ------------------------------------------------ the row widgets
    def _make_row(self, name, kind):
        current = current_value(name)
        if current is None:
            return None
        default = current
        if kind == "flag":
            w = QCheckBox()
            w.setChecked(bool(current and current != 0))
            w.toggled.connect(lambda checked, n=name:
                              self._pin_bool(n, checked))
        elif kind == "i":
            w = QSpinBox()
            w.setRange(-10_000_000, 10_000_000)
            w.setValue(int(current))
            w.valueChanged.connect(lambda v, n=name:
                                   self._pin_int(n, v))
        else:
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setRange(-10_000.0, 100_000.0)
            w.setValue(float(current))
            w.valueChanged.connect(lambda v, n=name:
                                   self._pin_float(n, v))
        self.rows[name] = (w, default, kind)
        return w

    # ------------------------------------------------------ the pins
    def _pin_bool(self, name, checked):
        self._apply(name, "1" if checked else "0")

    def _pin_int(self, name, value):
        self._apply(name, str(int(value)))

    def _pin_float(self, name, value):
        self._apply(name, repr(float(value)))

    def _apply(self, name, value):
        try:
            holder, parsed = apply_override(name, value)
        except (KeyError, ValueError) as e:
            self.status_cb(f"{name}: {e}")
            return
        self.status_cb(f"{name} pinned to {parsed} (live)")
        UiState.save({name: parsed})

    # ------------------------------------------------------ refresh
    def refresh_from_modules(self):
        for name, (w, _default, _kind) in self.rows.items():
            holder = resolve_module(name)
            if holder is None:
                continue
            current = getattr(holder, name)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(current and current != 0))
            elif isinstance(w, QSpinBox):
                w.setValue(int(current))
            else:
                w.setValue(float(current))
        self.status_cb("refreshed from the live modules")

    # ------------------------------------------------------ resets
    def reset_all(self):
        for name, (w, default, kind) in self.rows.items():
            holder = resolve_module(name)
            if holder is None:
                continue
            try:
                apply_override(name, repr(default))
            except (KeyError, ValueError):
                continue
            if kind == "flag":
                w.setChecked(bool(default and default != 0))
            else:
                w.setValue(default)
        UiState.clear()
        self.status_cb("all pins reset to the launch defaults")


class UiState:
    """The pinned-constant persistence (survives restarts; never
    touches constants.py)."""

    @staticmethod
    def load() -> dict:
        if not UI_STATE_PATH.exists():
            return {}
        return json.loads(UI_STATE_PATH.read_text())

    @staticmethod
    def save(pins: dict):
        UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = json.loads(UI_STATE_PATH.read_text()) \
            if UI_STATE_PATH.exists() else {}
        state.update({str(k): v for k, v in pins.items()})
        UI_STATE_PATH.write_text(json.dumps(state, indent=1) + "\n")

    @staticmethod
    def clear():
        if UI_STATE_PATH.exists():
            UI_STATE_PATH.unlink()

    @staticmethod
    def apply_everything(status_cb=lambda *_: None) -> int:
        """The launch-time application (before any session builds)."""
        state = UiState.load()
        applied = 0
        for name, value in state.items():
            try:
                apply_override(name, repr(value))
                applied += 1
            except (KeyError, ValueError):
                pass
        if applied:
            status_cb(f"ui_state: {applied} pin(s) applied on launch")
        return applied


def build_advanced_tab(status_cb):
    AdvancedTab.ui_state = UiState  # the editor's save hook rides it
    return AdvancedTab(status_cb)
