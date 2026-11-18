"""The apply-manually form: the source panel keeps only the flow
buttons, and a manual intrinsics apply asks for the values on demand.
"""
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QLabel, QVBoxLayout)

from slam.ui.row_layout import tight_row


class ManualIntrinsicsDialog(QDialog):
    """fx/fy/cx/cy/k1 at their natural widths; 0 means auto."""

    def __init__(self, owner, values):
        super().__init__(owner)
        self.setWindowTitle("manual intrinsics")
        self.setModal(True)
        fx, fy, cx, cy, k1 = values
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 8)
        lay.setSpacing(6)
        note = QLabel("fx is required; 0 means auto for fy/cx/cy\n"
                      "(fy follows fx; cx/cy the frame center)")
        note.setStyleSheet("color: #6e6e74; font-size: 11px;")
        lay.addWidget(note)
        self.fx = self._spin(0.0, 100000.0, fx, "auto")
        self.fy = self._spin(0.0, 100000.0, fy, "auto")
        self.cx = self._spin(0.0, 100000.0, cx, "auto")
        self.cy = self._spin(0.0, 100000.0, cy, "auto")
        self.k1 = self._spin(-1.0, 1.0, k1, None, decimals=6)
        for label, field in (("fx", self.fx), ("fy", self.fy),
                             ("cx", self.cx), ("cy", self.cy),
                             ("k1", self.k1)):
            lay.addLayout(tight_row(label, field))
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.fx.valueChanged.connect(self._sync_ok)
        self._sync_ok()

    @staticmethod
    def _spin(minimum, maximum, value, auto_text, decimals=3):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(float(value or 0.0))
        if auto_text:
            spin.setSpecialValueText(auto_text)
        return spin

    def _sync_ok(self):
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(self.fx.value() > 0.0)

    def values(self):
        return (self.fx.value(), self.fy.value(), self.cx.value(),
                self.cy.value(), self.k1.value())


def ask_manual_intrinsics(owner, values):
    """The dialog's result: the five values, or None on cancel."""
    dialog = ManualIntrinsicsDialog(owner, values)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.values()
    return None
