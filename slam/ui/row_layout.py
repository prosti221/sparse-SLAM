"""The sidebar's row geometry: a label, its field at the field's
natural width, and the row's leftover width trailing.

Two default behaviors are avoided here: QFormLayout's shared label
column indents every field by the widest label, and a QLabel in a
horizontal box is the item that grows when the field cannot, which
pulls the field away from its label.  A fixed label plus a trailing
stretch keeps the pair adjacent and parks the slack at the right.
"""
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLayout,
                               QSizePolicy)

LABEL_STYLE = "color: #a8a8ae;"
ROW_SPACING = 8


def tight_row(label_text, widget):
    """A [label][field][stretch] row (the label optional)."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(ROW_SPACING)
    if label_text:
        label = QLabel(label_text)
        label.setStyleSheet(LABEL_STYLE)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        row.addWidget(label)
    if isinstance(widget, QLayout):
        row.addLayout(widget)
    else:
        policy = widget.sizePolicy()
        widget.setSizePolicy(QSizePolicy.Maximum, policy.verticalPolicy())
        row.addWidget(widget)
    row.addStretch(1)
    return row
