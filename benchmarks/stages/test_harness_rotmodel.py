"""The L1b pin: the harness's rotation-model wiring.

The battery's semantics are the legacy no-rotation-model machine
(bit-identical by construction - run_slam's rotmodel parameter
defaults False and the tracker attribute is only set via it);
--rotmodel opts a ladder leg into the production-representative
wiring (the video path enables the model through the intake gate's
cadence; the census mirrors slam.py).
"""
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from benchmarks import harness as harness_mod


def _micro_dataset(tmp_path: Path, n: int = 4) -> Path:
    """Four synthetic dot-pattern frames with a lateral-translation
    GT (ORB-friendly, non-degenerate: the pattern shifts 25 px per
    frame at fx=500, Z=2)."""
    d = tmp_path / "rotmodel_micro"
    (d / "frames").mkdir(parents=True)
    rng = np.random.default_rng(11)
    base = np.full((480, 640, 3), 90, dtype=np.uint8)
    dots = rng.integers(6, 474, size=(200, 2))
    for i in range(n):
        img = base.copy()
        shift = int(round(-25.0 * i))
        for (x0, y0) in dots:
            x = x0 + shift
            if 2 <= x < 638:
                cv.circle(img, (int(x), int(y0)), 2, (220, 220, 220), -1)
        cv.imwrite(str(d / "frames" / f"{i:06d}.png"), img)
        t = 0.1 * i
        row = f"{i * 0.1:.2f} {t:.4f} 0.0 0.0 0.0 0.0 0.0 1.0"
        with (d / "trajectory.txt").open("a") as f:
            f.write(row + "\n")
    (d / "calibration.txt").write_text("500.0 500.0 320.0 240.0\n")
    return d


def test_run_slam_rotmodel_wiring(tmp_path, monkeypatch):
    d = _micro_dataset(tmp_path)
    made = []

    real = harness_mod.Tracker

    class Probe(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(harness_mod, "Tracker", Probe)

    import inspect
    sig = inspect.signature(harness_mod.run_slam)
    assert sig.parameters["rotmodel"].default is False

    harness_mod.run_slam(d, seed=7)
    harness_mod.run_slam(d, seed=7, rotmodel=True)
    # The harness assigns rotation_model_enabled right after the
    # constructor returns - the same instance carries the call's flag.
    assert not made[0].rotation_model_enabled
    assert made[1].rotation_model_enabled
