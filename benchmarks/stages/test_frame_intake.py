"""Stage tests: displacement-gated frame intake.

Pure gate tests - no SLAM run. The gate (slam/core/intake.py) decides
which decoded VIDEO frames reach the tracker: feed when the median
matched-feature displacement vs the LAST FED frame clears the
threshold, with the max-gap time valve and the measurement-failure
feeds as safety nets. The harness feeds frame files and never
constructs a gate - the frozen battery is untouched by construction;
these pins document and lock the gate's decision boundaries.
"""
import cv2 as cv
import numpy as np
import pytest

from slam.core.intake import FrameIntakeGate
from slam.utils import constants as const
from slam.utils import telemetry
from slam.utils.telemetry import TelemetryRecorder


@pytest.fixture
def recorder():
    rec = TelemetryRecorder()
    telemetry.set_recorder(rec)
    yield rec
    telemetry.set_recorder(None)


def textured(seed=7):
    """Blurred random noise: dense, reproducible ORB texture."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    return cv.GaussianBlur(img, (0, 0), 1.2)


def shifted(img, dx):
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv.warpAffine(img, M, (img.shape[1], img.shape[0]))


def test_constants_pin_the_measured_band():
    """8-5: the valve bounds full-stop staleness at 1 s of container
    time; the match floor is the minimum sample for a displacement
    MEDIAN (below it the gate cannot measure and must not skip).
    The displacement threshold is USER-DECIDED by measurement, and
    the pin tracks the configured value:
    2.0 = the adopted healthy edge (kyoto_original 0.0 px
    median at 60 fps -> 6 resets in 30 s; 2.4 px -> 0 resets), 8.0 =
    the user's 2-fps-cadence adjustment, 4.0 = the user's conclusion
    of the 8->2->4 sweep (the kyoto_2_5 pan
    sweep is the mid-point evidence)."""
    assert const.FRAME_INTAKE_MIN_DISPLACEMENT_PX == 4.0
    assert const.FRAME_INTAKE_MAX_GAP_S == 1.0
    assert const.FRAME_INTAKE_MIN_MATCHES == 10


def test_valve_in_decoded_frames_pins_representative_containers():
    """8-5: the 1 s valve converts to DECODED frames at the capture's
    fps, with a 3-frame floor so slow containers keep a heartbeat
    (2 fps -> 3 frames = 1.5 s; a 0/absent fps falls back to the
    30 fps anchor, mirroring slam.py's wiring)."""
    assert FrameIntakeGate('ORB', False, 59.94).max_gap_frames == 60
    assert FrameIntakeGate('ORB', False, 60.0).max_gap_frames == 60
    assert FrameIntakeGate('ORB', False, 30.0).max_gap_frames == 30
    assert FrameIntakeGate('ORB', False, 15.0).max_gap_frames == 15
    assert FrameIntakeGate('ORB', False, 2.0).max_gap_frames == 3
    assert FrameIntakeGate('ORB', False, 1.0).max_gap_frames == 3
    assert FrameIntakeGate('ORB', False, 0.0).max_gap_frames == 30


def test_first_frame_feeds_and_becomes_the_reference(recorder):
    gate = FrameIntakeGate('ORB', False, 30.0)
    assert gate.should_feed(textured())
    assert gate.fed_frames == 1
    assert gate.decoded_frames == 1
    assert gate.effective_fps == 30.0
    assert recorder.events['intake_first_frame'] == 1


def test_static_frames_skip_until_the_valve_forces_a_feed(recorder):
    """No displacement -> skip; the stream must not go stale forever:
    the valve forces a feed at max_gap decoded frames (10 at a 10 fps
    container), keeping appearance updates flowing at a full stop."""
    gate = FrameIntakeGate('ORB', False, 10.0)
    img = textured()
    assert gate.should_feed(img)                     # decoded 1: first frame
    for _ in range(9):
        assert not gate.should_feed(img.copy())      # decoded 2..10: static
    assert gate.should_feed(img.copy())              # decoded 11: 11-1 >= 10
    assert (gate.fed_frames, gate.decoded_frames) == (2, 11)
    assert recorder.events['intake_skip'] == 9
    assert recorder.events['intake_forced_max_gap'] == 1


def test_displacement_threshold_and_reference_update(recorder):
    """A frame that moved feeds; the reference is the LAST FED frame
    (the next candidate is measured against it, not against the run
    start) - the cumulative displacement is what the gate measures."""
    gate = FrameIntakeGate('ORB', False, 30.0)
    a = textured()
    b = shifted(a, 12)
    assert gate.should_feed(a)                       # first frame
    assert not gate.should_feed(a.copy())           # 0 px -> skip
    assert gate.should_feed(b)                       # 12 px -> feed
    # b itself re-offered: identical to the (updated) reference -> skip.
    assert not gate.should_feed(b.copy())
    assert (gate.fed_frames, gate.decoded_frames) == (2, 4)
    assert recorder.events['intake_feed_displacement'] == 1
    assert recorder.events['intake_skip'] == 2


def test_featureless_frame_feeds_without_clobbering_the_reference(recorder):
    """No features -> no basis to measure -> feed (the tracker must
    see it); but a featureless frame cannot serve as a displacement
    reference - the next measurable frame still compares against the
    last FEATURED fed frame."""
    gate = FrameIntakeGate('ORB', False, 30.0)
    a = textured()
    assert gate.should_feed(a)
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    assert gate.should_feed(blank)                   # intake_few_features
    assert not gate.should_feed(a.copy())            # reference is still a
    assert (gate.fed_frames, gate.decoded_frames) == (2, 3)
    assert recorder.events['intake_few_features'] == 1


def test_few_matches_feed_conservatively(recorder):
    """Features exist but do not correspond to the reference (scene
    cut, occlusion): feed - and keep the old reference, since the
    unmatchable frame provides no displacement measurement."""
    gate = FrameIntakeGate('ORB', False, 30.0)
    a = textured()

    class _NoMatchMatcher:
        def knnMatch(self, *args, **kwargs):
            return []

    assert gate.should_feed(a)
    real_matcher = gate.matcher
    gate.matcher = _NoMatchMatcher()
    assert gate.should_feed(shifted(a, 12))          # intake_few_matches
    gate.matcher = real_matcher                      # measurement restored
    # the reference was NOT updated by the unmatchable frame:
    assert not gate.should_feed(a.copy())            # 0 px vs reference a
    assert (gate.fed_frames, gate.decoded_frames) == (2, 3)
    assert recorder.events['intake_few_matches'] == 1


def test_low_fps_container_keeps_a_three_frame_heartbeat(recorder):
    """2 fps container: the valve floors at 3 decoded frames (1.5 s) -
    never a collapsed single-frame window, and a static scene still
    gets its appearance heartbeat."""
    gate = FrameIntakeGate('ORB', False, 2.0)
    img = textured()
    assert gate.should_feed(img)                      # decoded 1
    assert not gate.should_feed(img.copy())          # 2-1 < 3: skip
    assert not gate.should_feed(img.copy())          # 3-1 < 3: skip
    assert gate.should_feed(img.copy())              # 4-1 >= 3: forced
    assert (gate.fed_frames, gate.decoded_frames) == (2, 4)
    assert recorder.events['intake_forced_max_gap'] == 1


def test_effective_fps_tracks_the_fed_ratio():
    """DIAGNOSTIC ONLY (the A/B): the fed/decoded ratio is
    recorded for measurement, but the cadence constants
    deliberately keep the CONTAINER fps - declaring the fed cadence
    collapses the recovery window (10 -> 5 consecutive quality-
    failing fed frames) and re-arms salem's reset spiral (measured:
    7 vs 2 resets over the same 34 s segment). The ratio itself:
    identity with the container while everything is fed, scaled by
    fed/decoded as frames are skipped."""
    gate = FrameIntakeGate('ORB', False, 30.0)
    a = textured()
    b = shifted(a, 12)
    gate.should_feed(a)
    assert gate.effective_fps == 30.0                 # 1/1
    gate.should_feed(a.copy())                        # skip
    assert gate.effective_fps == 15.0                 # 1/2
    gate.should_feed(b)                               # feed
    assert gate.effective_fps == 20.0                 # 2/3


def test_gate_carries_the_extraction(monkeypatch):
    """[26c] the dedup: should_feed caches the decoded frame's
    extraction in last_extraction, bit-identical to what the tracker's
    extractor computes from the same gray image (the gate's gray
    conversion mirrors Frame.gray_image exactly)."""
    from slam.core.intake import _frame_gray, INTAKE_CARRY_EXTRACTION
    from slam.features.feature_extractor import FeatureExtractor

    assert INTAKE_CARRY_EXTRACTION == 1
    gate = FrameIntakeGate('ORB', False, 30.0)
    img = textured()
    assert gate.should_feed(img)
    direct_kps, direct_desc = gate.feature_extractor.extract(
        _frame_gray(img))
    carried_kps, carried_desc = gate.last_extraction
    assert len(carried_kps) == len(direct_kps)
    np.testing.assert_array_equal(carried_desc, direct_desc)
    assert [kp.pt for kp in carried_kps] == [kp.pt for kp in direct_kps]
