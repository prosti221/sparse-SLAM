"""The source-resolver pins: slam/core/source.py's three kinds.

The editor's source tab resolves video files, frame-sequence
directories, and cameras; sequence dirs duck-type the cv.VideoCapture
surface the session consumes. The `fx: auto` sentinel semantics are the
GeoCalib branch's trigger and pinned here.
"""
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from slam.config.config_parser import ConfigParser
from slam.core.source import (CAMERA_KIND, FILE_KIND, SEQUENCE_KIND,
                              SequenceCapture, SourceSpec, detect_kind,
                              open_source, resolve_intrinsics)


def _sequence_dir(tmp_path: Path, n=12, w=640, h=480):
    d = tmp_path / "seq_frames"
    d.mkdir()
    for i in range(n):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        x = min(5 * i, w - 20)
        cv.rectangle(img, (x, 100), (x + 10, 120), (0, 200, 0), -1)
        cv.imwrite(str(d / f"{i:06d}.png"), img)
    return d


def test_sequence_capture_serves_the_videocapture_surface(tmp_path):
    d = _sequence_dir(tmp_path)
    cap = SequenceCapture(d)
    assert cap.get(cv.CAP_PROP_FRAME_COUNT) == 12
    assert cap.get(cv.CAP_PROP_FRAME_HEIGHT) == 480
    assert cap.get(cv.CAP_PROP_FRAME_WIDTH) == 640
    assert cap.get(cv.CAP_PROP_FPS) == 30.0

    ok, img = cap.read()
    assert ok and img.shape == (480, 640, 3)
    assert cap.get(cv.CAP_PROP_POS_FRAMES) == 1

    while cap.read()[0]:
        pass  # consume to end
    ok, img = cap.read()  # EOR
    assert not ok and img is None

    # The rewind support: exact reposition, then a clean resume.
    assert cap.seek(4) is True
    ok, img = cap.read()
    assert ok and cap.get(cv.CAP_PROP_POS_FRAMES) == 5
    assert int(np.where(img[:, :, 1] > 100)[1][0]) // 5 == 4


def test_detect_classifies_the_three_kinds(tmp_path):
    d = _sequence_dir(tmp_path)
    assert detect_kind(str(d)) == SEQUENCE_KIND
    assert detect_kind("123") == CAMERA_KIND
    assert detect_kind("/videos/kyoto_original.mp4") == FILE_KIND


def test_resolve_intrinsics_semantics(tmp_path):
    d = _sequence_dir(tmp_path)
    # auto sentinel -> (None, 0.0): the GeoCalib branch trigger.
    assert resolve_intrinsics(
        SourceSpec(SEQUENCE_KIND, str(d), fx="auto"), 640, 480) \
        == (None, 0.0, {"fx": "auto"})
    # absent fx behaves as auto.
    assert resolve_intrinsics(
        SourceSpec(SEQUENCE_KIND, str(d)), 640, 480)[0] is None
    # explicit -> construct K with the cx/cy/k1 defaults (principal
    # point at the frame center).
    K, k1, provenance = resolve_intrinsics(
        SourceSpec(SEQUENCE_KIND, str(d), fx=500.0, k1=-0.02,
                   cx=318.0, cy=241.0), 640, 480)
    assert K.tolist() == [[500.0, 0.0, 318.0],
                          [0.0, 500.0, 241.0],
                          [0.0, 0.0, 1.0]]
    assert k1 == -0.02 and provenance["fy"] == 500.0
    # The principal point defaults to the frame center.
    K2, _, _ = resolve_intrinsics(
        SourceSpec(SEQUENCE_KIND, str(d), fx=500.0), 640, 480)
    assert K2[0, 2] == 320.0 and K2[1, 2] == 240.0


def test_open_source_runs_a_sequence_through_the_harness_loop(tmp_path):
    d = _sequence_dir(tmp_path)
    resolved = open_source(SourceSpec(SEQUENCE_KIND, str(d), fx=500.0))
    assert resolved is not None
    cap, K, k1, meta = resolved
    assert meta["kind"] == SEQUENCE_KIND
    assert K[0, 0] == 500.0 and k1 == 0.0
    # The duck-typed capture feeds slam.py's loop shape untouched.
    ret, img = cap.read()
    assert ret and isinstance(img, np.ndarray)


def test_camera_kind_detection_and_open_contract():
    assert detect_kind("1") == CAMERA_KIND
    # Opening an actual device is environment-gated; the resolver's
    # contract is the None-on-failure path.
    resolved = open_source(SourceSpec(CAMERA_KIND, "99999999"))
    assert resolved is None
