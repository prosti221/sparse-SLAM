"""Stage tests: the extraction process (worker decode+extract, one
ahead, behind INTAKE_EXTRACT_WORKER_THREADS).

The real worker PROCESS runs in these tests (spawn is the macOS
default start method; the worker target is module-level and
spawn-safe). The fixtures use a small deterministic mp4 written once;
the extracted features are compared against the main process's inline
extraction of the same decoded frames. Open3D is not involved; the
frozen battery is untouched by construction (frame files, no gate).
"""
import hashlib
import sys
import threading
import types

import cv2 as cv
import numpy as np
import pytest

from slam.core import intake as intake_mod
from slam.core.intake import (  # noqa: E402
    ExtractionProcess, _extract_process_main, FrameIntakeGate)
from slam.features.feature_extractor import FeatureExtractor  # noqa: E402
from slam.utils import telemetry  # noqa: E402
from multiprocessing import shared_memory  # noqa: E402


def _write_fixture_video(path, n_frames=24, w=64, h=48):
    """A deterministic mp4: a moving bright rectangle on noise-free
    gray - ORB-stable content, identical on every decode."""
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"),
                            30.0, (w, h))
    assert writer.isOpened()
    for i in range(n_frames):
        img = np.full((h, w, 3), 60, dtype=np.uint8)
        x = 4 + (i * 2) % (w - 20)
        y = 6 + (i * 1) % (h - 16)
        img[y:y + 8, x:x + 10] = 230
        writer.write(img)
    writer.release()


@pytest.fixture(scope="module")
def fixture_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("extract_proc") / "fixture.mp4"
    _write_fixture_video(path)
    return path


def _decode_frames(path):
    """The fixture's decoded BGR frames, decoded in the main process
    (the inline-extraction ground truth)."""
    cap = cv.VideoCapture(str(path))
    frames = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frames.append(img)
    cap.release()
    return frames


def _kp_fields(kps):
    return [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response,
             kp.octave, kp.class_id) for kp in kps]


def test_worker_extraction_matches_inline(fixture_video):
    """The worker process's extraction is FIELD-identical to the main
    process's inline extraction of the same decoded frames, the
    transported frame bytes are the decoded bytes, and the worker
    thread bound (k=1) does not move a bit (thread-count-invariant
    numerics, measured on both gate machines)."""
    inline_frames = _decode_frames(fixture_video)
    inline_extractor = FeatureExtractor("ORB")
    inline_extractor.set_multiscale_enabled(False)  # the pipeline's config

    for worker_threads in (0, 1):
        pipeline = ExtractionProcess(fixture_video, 64, 48, "ORB",
                                     worker_threads=worker_threads)
        try:
            items = []
            while True:
                item = pipeline.next_frame()
                if item is None:
                    break
                items.append(item)
        finally:
            pipeline.close()

        assert len(items) == len(inline_frames) > 0
        for idx, img, kps, desc in items:
            ref_img = inline_frames[idx]
            assert hashlib.sha256(
                np.ascontiguousarray(img)).digest() == \
                hashlib.sha256(np.ascontiguousarray(ref_img)).digest()
            ref_kps, ref_desc = inline_extractor.extract(
                intake_mod._frame_gray(ref_img))
            assert _kp_fields(kps) == _kp_fields(ref_kps)
            if ref_desc is None:
                assert desc is None
            else:
                np.testing.assert_array_equal(desc, ref_desc)


def test_eos_sentinel_and_close_semantics(fixture_video):
    """EOS ends the stream; the worker exits on its own; close() is
    idempotent; an early close (the cap break) terminates a worker
    that is mid-stream."""
    pipeline = ExtractionProcess(fixture_video, 64, 48, "ORB")
    consumed = 0
    while pipeline.next_frame() is not None:
        consumed += 1
    assert consumed > 0
    pipeline.close()
    assert not pipeline._proc.is_alive()
    pipeline.close()          # idempotent
    pipeline.close()
    assert pipeline.next_frame() is None   # nothing after close

    # Early close (the consumer's cap break): the worker is mid-stream
    pipeline = ExtractionProcess(fixture_video, 64, 48, "ORB")
    assert pipeline.next_frame() is not None
    pipeline.close()
    assert not pipeline._proc.is_alive()


def test_worker_error_surfaces_on_the_consumer(tmp_path):
    """A worker failure (unopenable source) raises on next_frame
    instead of hanging or silently truncating."""
    pipeline = ExtractionProcess(tmp_path / "missing.mp4", 64, 48, "ORB")
    with pytest.raises(RuntimeError):
        pipeline.next_frame()
    pipeline.close()


def test_preextracted_gate_path_matches_inline(fixture_video):
    """should_feed with preextracted makes the SAME decision, emits
    the SAME telemetry events and leaves the SAME gate state as the
    inline path on the same frame - only the extraction source
    differs."""
    frames = _decode_frames(fixture_video)
    extractor = FeatureExtractor("ORB")
    extractor.set_multiscale_enabled(False)

    def fresh_gate():
        return FrameIntakeGate("ORB", False, 30.0)

    pre = [extractor.extract(intake_mod._frame_gray(img))
           for img in frames]

    inline_events, pre_events = [], []
    gate_a, gate_b = fresh_gate(), fresh_gate()
    rec = telemetry.TelemetryRecorder()
    telemetry.set_recorder(rec)
    try:
        for img in frames:
            before = dict(rec.events)
            ok_a = gate_a.should_feed(img)
            inline_events.append(
                (ok_a, tuple(sorted((k, v - before.get(k, 0))
                                    for k, v in rec.events.items()
                                    if v != before.get(k, 0)))))
        rec.events.clear()
        for img, (kps, desc) in zip(frames, pre):
            before = dict(rec.events)
            ok_b = gate_b.should_feed(img, preextracted=(kps, desc))
            pre_events.append(
                (ok_b, tuple(sorted((k, v - before.get(k, 0))
                                    for k, v in rec.events.items()
                                    if v != before.get(k, 0)))))
    finally:
        telemetry.set_recorder(None)

    assert inline_events == pre_events
    assert gate_a.decoded_frames == gate_b.decoded_frames
    assert gate_a.fed_frames == gate_b.fed_frames
    assert gate_a._last_fed_decoded_index == gate_b._last_fed_decoded_index
    assert gate_a._reference_desc is None or np.array_equal(
        gate_a._reference_desc, gate_b._reference_desc)


def test_pipeline_emits_no_telemetry_and_leaves_cv_rng_alone(fixture_video):
    """The pipeline's existence consumes no telemetry events and does
    not disturb the consumer process's cv RNG stream (the C5b audit
    methodology: same seed, identical post-run draws)."""
    cv.setRNGSeed(42)
    before = np.array([cv.randu(np.zeros((4, 4), np.float32), 0, 1)
                       for _ in range(3)])

    rec = telemetry.TelemetryRecorder()
    telemetry.set_recorder(rec)
    try:
        pipeline = ExtractionProcess(fixture_video, 64, 48, "ORB")
        count = 0
        while pipeline.next_frame() is not None:
            count += 1
        pipeline.close()
        assert count > 0
    finally:
        telemetry.set_recorder(None)
    assert dict(rec.events) == {}

    cv.setRNGSeed(42)
    after = np.array([cv.randu(np.zeros((4, 4), np.float32), 0, 1)
                      for _ in range(3)])
    np.testing.assert_array_equal(before, after)


def test_worker_thread_bound_routing(monkeypatch, fixture_video):
    """worker_threads > 0 bounds the torch pool for the torch-backed
    stack ONLY; an ORB worker never imports torch (no import cost, no
    thread state for an extractor that has none)."""
    calls = []
    fake_torch = types.SimpleNamespace(
        set_num_threads=lambda k: calls.append(("set", int(k))),
        get_num_threads=lambda: 6)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class StubExtractor:
        def __init__(self, name):
            calls.append(("ctor", name))

        def set_multiscale_enabled(self, enabled):
            calls.append(("multiscale", enabled))

        def extract(self, img):
            return [], None

    monkeypatch.setattr(intake_mod, "FeatureExtractor", StubExtractor)

    def run_worker(name, k):
        shms = [shared_memory.SharedMemory(create=True, size=48 * 64 * 3)
                for _ in range(2)]
        try:
            msgs = intake_mod.mp.Queue(maxsize=1)
            tokens = intake_mod.mp.BoundedSemaphore(2)
            stop = intake_mod.mp.Event()
            t = threading.Thread(
                target=_extract_process_main,
                args=(str(fixture_video), name, 0, False, 64, 48,
                      [s.name for s in shms], msgs, tokens, stop, k),
                daemon=True)
            t.start()
            got = msgs.get(timeout=30.0)   # the first frame's message
            stop.set()
            t.join(timeout=30.0)
            assert not t.is_alive()
            return got
        finally:
            for s in shms:
                s.close()
                s.unlink()

    got = run_worker("ALIKED", 3)
    assert got[0] == 0                      # the first decoded idx
    assert ("set", 3) in calls              # the bound applied
    assert ("ctor", "ALIKED") in calls

    calls.clear()
    got = run_worker("ORB", 3)
    assert got[0] == 0
    assert not any(c[0] == "set" for c in calls)   # no torch import
    assert ("ctor", "ORB") in calls


def test_flag_contract(monkeypatch):
    """INTAKE_EXTRACT_WORKER_THREADS is a module-level INT with the
    ADOPTED default (the worker bound at k=1) and the --override
    machinery reaches it (including 0 = no bound). The process gate
    flag itself is gone: the extraction process is unconditional with
    the gate (the C7b adoption's user-directed flag removal)."""
    assert isinstance(intake_mod.INTAKE_EXTRACT_WORKER_THREADS, int)
    assert intake_mod.INTAKE_EXTRACT_WORKER_THREADS == 1
    assert not hasattr(intake_mod, "INTAKE_EXTRACT_PROCESS")

    from benchmarks.run import apply_overrides
    apply_overrides(["INTAKE_EXTRACT_WORKER_THREADS=0"])
    try:
        assert intake_mod.INTAKE_EXTRACT_WORKER_THREADS == 0
    finally:
        intake_mod.INTAKE_EXTRACT_WORKER_THREADS = 1
    assert intake_mod.INTAKE_EXTRACT_WORKER_THREADS == 1
