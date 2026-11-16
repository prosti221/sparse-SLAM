"""The L4 pins: the rotation model's bootstrap-yield duty.

An engaged-but-unestablished map (fewer than
ROTMODEL_BOOTSTRAP_MIN_POINTS points) has no PnP anchoring, so the
[R | 0] commits would freeze the camera centre and starve the
first triangulation - the gate yields to the legacy pair pose on a
fixed cadence until the map carries the minimum points. Once
established the yield never fires and the L1b behavior holds by
construction. Flag 0 keeps the never-yield behavior exactly.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from slam.core import tracker as tmod


def _tracker(n_points: int, engaged: bool = True):
    t = tmod.Tracker.__new__(tmod.Tracker)
    t.map = SimpleNamespace(points=list(range(n_points)))
    t._rot_model_ema = 0.5
    t._rot_model_engaged = engaged
    t._rot_model_prev_t = None
    t.rotation_model_enabled = True
    t.cur_frame = None
    t.prev_frame = None
    t.feature_extraction_method = "ORB"
    return t


def test_unestablished_yields_every_fifth_pair(monkeypatch):
    t = _tracker(10)  # unestablished

    monkeypatch.setattr(tmod, "rotation_model_pose_between_frames",
                        lambda *a, **k: np.eye(4))
    legacy = np.eye(4)
    legacy[:3, 3] = [0.1, 0.0, 0.0]

    committed = []
    for _ in range(10):
        committed.append(t._update_rotation_model_gate(legacy))

    # the yield fires at the 5th and 10th committed pair: those
    # committed results are the legacy OBJECT itself (identity); the
    # non-yield pairs commit the consensus-derived hybrid (a fresh
    # array with the legacy's unit-t translation and the consensus
    # rotation).
    assert [committed[4] is legacy, committed[9] is legacy] == [True,
                                                                True]
    assert all(committed[i] is not legacy for i in (0, 1, 2, 3, 5, 6, 7,
                                                    8))


def test_established_never_yields(monkeypatch):
    t = _tracker(60)  # established

    consensus = np.eye(4)
    monkeypatch.setattr(tmod, "rotation_model_pose_between_frames",
                        lambda *a, **k: consensus)
    legacy = np.eye(4)
    legacy[:3, 3] = [0.1, 0.0, 0.0]

    for _ in range(12):
        committed = t._update_rotation_model_gate(legacy)
        assert committed is consensus


def test_flag_off_keeps_never_yield_behavior(monkeypatch):
    t = _tracker(10)  # unestablished

    monkeypatch.setattr(tmod, "ROTMODEL_BOOTSTRAP_YIELD_ENABLED", 0)
    monkeypatch.setattr(tmod, "rotation_model_pose_between_frames",
                        lambda *a, **k: np.eye(4))
    legacy = np.eye(4)
    legacy[:3, 3] = [0.1, 0.0, 0.0]

    for _ in range(10):
        assert t._update_rotation_model_gate(legacy) is not legacy
