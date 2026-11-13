"""Displacement-gated frame intake (the VIDEO path).

When consecutive fed
frames carry sub-pixel matched-feature displacement, the two-view
geometry between them is pure noise - the pose chain noise-fits, the
keyframe baseline criterion is fed by a noise random walk, and map
quality collapses into reset spirals. kyoto_original at 60 fps has
a 0.0 px MEDIAN displacement (measured); the same content fed at
15 fps (2.4 px median) runs healthy with the SAME constants.

The gate sits on the INPUT side (slam.py's video loop): a decoded
frame is FED to the tracker only when it carries measurable motion
signal - median matched-feature displacement vs the LAST FED frame
>= FRAME_INTAKE_MIN_DISPLACEMENT_PX - with two unconditional feeds:
the TIME valve (FRAME_INTAKE_MAX_GAP_S: appearance drifts even at a
full stop) and the measurement-failure path (too few features or
matches: the tracker must see state it may need to recover from).
It reuses the PRODUCTION FeatureExtractor (same construction as
Tracker.__init__) and the shared matcher cache, and mirrors
match_features' Lowe + mutual-best filter - so the gate's notion of
"a matched feature" is exactly the tracker's, without the RANSAC/
essential-matrix machinery (this is a 2D displacement measurement,
not a pose estimate).

The benchmark harness feeds frame FILES and never constructs a
gate: the frozen battery is untouched by construction. The fed
stream's EFFECTIVE fps (container_fps * fed / decoded) is exposed
as a DIAGNOSTIC ONLY: the cadence constants keep the
CONTAINER's time base. That is a measured decision, not a taste -
A/B on walking_salem's hard 34 s segment (same gate, same skips,
only the declaration differs): container-declared = 2 resets /
26 relocations, fed-cadence-declared = 7 / 42. Declaring the fed
cadence collapses the recovery window back to the anchor count
(10 -> 5 consecutive quality-failing fed frames) and re-arms the
reset spiral; the container declaration preserves the
health and can only ever LENGTHEN windows under gating (the
safe direction).
"""
import queue
import threading

import cv2 as cv
import numpy as np

from slam.features.feature_extractor import FeatureExtractor
from slam.features.matcher import _get_or_create_matcher, lowe_ratio_filter
from slam.utils import telemetry
from slam.utils.constants import (FRAME_INTAKE_MAX_GAP_S,
                                  FRAME_INTAKE_MIN_DISPLACEMENT_PX,
                                  FRAME_INTAKE_MIN_MATCHES,
                                  KNN_K_VALUE,
                                  ORB_EXTRACTOR_NAME)
from slam.utils.logger import debug_log

LOG_TAG = 'Intake'

# The gate's extraction carry (the [26c] dedup): the live loop
# attaches should_feed's cached extraction to the fed frame, so the
# tracker's update() skips its own re-extraction. Flag 0 = the
# pre-dedup behavior (double extraction); flag 1 = the shipped
# default (the carried features are bit-identical by construction -
# the gate extracts from the same gray image). INT flag (the
# --override trap); read at call time via the module alias.
INTAKE_CARRY_EXTRACTION = 1

# The decode+extract-ahead pipeline ([26c] step 2): ONE worker thread
# owns the capture and the extractor - it reads frames in order and
# extracts each into a bounded queue; the consumer pops on the main
# thread. Frame order is unchanged and every audited call is
# cv-RNG-clean ([26a]), so the pipeline is numerically silent; it
# only overlaps the GPU inference behind the consumer's CPU stages.
# Flag 0 = the sequential read path (the pre-pipeline behavior); the
# consumer must also hold the carry (a gate without the carry would
# re-extract what the worker already extracted). INT flag (the
# --override trap); read at call time via the module alias.
INTAKE_EXTRACT_AHEAD = 1

_EOS = object()  # the producer's end-of-stream sentinel


class ExtractionPipeline:
    """Decode+extract ahead of the consumer thread.

    ONE worker thread owns the capture and the extractor exclusively:
    it reads frames in order, extracts each from the same gray
    conversion the tracker's frame would use, and pushes into a
    bounded queue (blocking puts = natural backpressure). The main
    thread pops and never extracts while the pipeline runs - the
    model access stays single-threaded by construction.

    pop() never blocks long: a short queue wait returns 'pending' so
    a renderer-paced loop stays live between SLAM frames.
    """

    def __init__(self, cap, extractor, depth: int = 2):
        self._cap = cap
        self._extractor = extractor
        self._q = queue.Queue(maxsize=depth)
        self._eos_seen = False
        self._stopped = False
        self._thread = threading.Thread(
            target=self._produce, daemon=True, name="intake-extract")
        self._thread.start()

    def _produce(self):
        while not self._stopped:
            ret, img = self._cap.read()
            if not ret:
                self._q.put(_EOS)
                return
            kps, desc = self._extractor.extract(_frame_gray(img))
            self._q.put((img, kps, desc))

    def pop(self, timeout_s: float = 0.01):
        """('ready', (img, kps, desc)) | ('pending', None) | ('eos', None)."""
        if self._eos_seen:
            return "eos", None
        try:
            item = self._q.get(timeout=timeout_s)
        except queue.Empty:
            return "pending", None
        if item is _EOS:
            self._eos_seen = True
            return "eos", None
        return "ready", item

    def stop(self):
        """Ask the producer to exit at its next read boundary (it may
        still be blocked inside cap.read(); the daemon thread dies
        with the process either way)."""
        self._stopped = True

    def join(self, timeout_s: float = None):
        """Wait for the producer to exit (after stop(), or naturally
        at end-of-stream) - the caller must not release the capture
        while the producer still reads it."""
        self._thread.join(timeout=timeout_s)


def _frame_gray(img: np.ndarray) -> np.ndarray:
    """The same grayscale conversion Frame.gray_image applies, so the
    gate's features are computed from the same image the tracker's
    extractor will see (including the historical BGR-array-through-
    RGB2GRAY convention of the capture loop)."""
    if img.ndim == 3:
        if img.shape[2] == 4:    # RGBA
            return cv.cvtColor(img.copy(), cv.COLOR_RGBA2GRAY)
        elif img.shape[2] == 3:  # RGB/BGR (3-channel capture frames)
            return cv.cvtColor(img.copy(), cv.COLOR_RGB2GRAY)
    return img.copy()


class FrameIntakeGate:
    """Displacement-gated video frame intake."""

    def __init__(self, feature_extraction_method: str = ORB_EXTRACTOR_NAME,
                 enable_multiscale: bool = False,
                 container_fps: float = 30.0):
        # The same extractor construction as Tracker.__init__ - the
        # gate's features ARE the tracker's features (identical gray
        # conversion; the [26a] probe measured the double extraction
        # on fed frames at ~23% of the live-path wall on ALIKED,
        # which is why should_feed now carries them).
        self.feature_extractor = FeatureExtractor(feature_extraction_method)
        self.feature_extractor.set_multiscale_enabled(enable_multiscale)
        # The shared matcher cache (binary/L2 norm choice included).
        self.matcher = _get_or_create_matcher(feature_extraction_method)
        # The current decoded frame's extraction, carried for the
        # caller (see should_feed's docstring).
        self.last_extraction = None
        self.container_fps = float(container_fps) or 30.0
        # The valve in DECODED frames: FRAME_INTAKE_MAX_GAP_S of
        # container time, floored at 3 frames so slow containers
        # still get a heartbeat (2 fps -> 3 frames = 1.5 s).
        self.max_gap_frames = max(
            3, int(np.ceil(FRAME_INTAKE_MAX_GAP_S * self.container_fps)))
        # Counters (the probe telemetry reads these directly).
        self.decoded_frames = 0
        self.fed_frames = 0
        # The displacement reference: the last FED frame that
        # produced features. A featureless frame is fed through but
        # cannot serve as a reference - the next measurable frame
        # still compares against the last measurable one.
        self._reference_kp_pts = None
        self._reference_desc = None
        self._last_fed_decoded_index = 0

    def should_feed(self, img: np.ndarray, pre=None) -> bool:
        """One decoded frame: feed it to the tracker, or skip it?

        Deterministic, input-side only - no tracker/map state is
        touched. Every call counts one decoded frame.

        `pre` is an already-extracted (kps, desc) pair for THIS image
        (the [26c] pipeline's worker output): the decision consumes it
        instead of re-extracting, and last_extraction carries it for
        the fed frame. The pair must come from the same gray
        conversion the tracker's extractor would use (_frame_gray
        mirrors Frame.gray_image exactly), so both sources are
        bit-identical.
        """
        self.decoded_frames += 1
        if pre is not None:
            kps, desc = pre
        else:
            kps, desc = self.feature_extractor.extract(_frame_gray(img))
        self.last_extraction = (kps, desc)
        kp_pts = cv.KeyPoint.convert(kps) if len(kps) else None
        has_features = (kp_pts is not None and desc is not None
                       and len(kps) >= FRAME_INTAKE_MIN_MATCHES)

        if self._reference_desc is None:
            # Run start: the tracker needs its first frame.
            return self._feed('intake_first_frame',
                              kp_pts if has_features else None,
                              desc if has_features else None)

        if (self.decoded_frames - self._last_fed_decoded_index
                >= self.max_gap_frames):
            # Time valve: bounded staleness of the fed stream even at
            # a full stop (appearance drifts; features age).
            debug_log(LOG_TAG,
                      f"forced feed: {FRAME_INTAKE_MAX_GAP_S}s valve "
                      f"({self.max_gap_frames} decoded frames)")
            return self._feed('intake_forced_max_gap',
                              kp_pts if has_features else None,
                              desc if has_features else None)

        if not has_features:
            # Extraction failed / too sparse: no basis to measure, no
            # basis to skip.
            return self._feed('intake_few_features')

        # Median matched-feature displacement vs the last fed frame:
        # knn + Lowe + mutual-best via the matcher module's shared
        # filter, mirroring match_features exactly (the RANSAC/E step
        # is deliberately absent - displacement is a 2D measurement,
        # and geometric filtering here would gate on the very signal
        # we are trying to measure).
        good = lowe_ratio_filter(self.matcher.knnMatch(
            desc, self._reference_desc, k=KNN_K_VALUE))

        if len(good) < FRAME_INTAKE_MIN_MATCHES:
            # Features exist but do not correspond to the reference
            # (scene cut, occlusion): feed - the tracker must see it.
            return self._feed('intake_few_matches')

        pts_cur = kp_pts[[m.queryIdx for m in good]]
        pts_ref = self._reference_kp_pts[[m.trainIdx for m in good]]
        displacement = float(np.median(
            np.linalg.norm(pts_cur - pts_ref, axis=1)))

        if displacement >= FRAME_INTAKE_MIN_DISPLACEMENT_PX:
            return self._feed('intake_feed_displacement', kp_pts, desc)

        telemetry.event('intake_skip')
        debug_log(LOG_TAG,
                  f"skipped: {displacement:.2f} px < "
                  f"{FRAME_INTAKE_MIN_DISPLACEMENT_PX} px")
        return False

    def _feed(self, reason_event: str, kp_pts=None, desc=None) -> bool:
        """Accept a frame, count it, and (when it has features) make
        it the displacement reference for the next candidates."""
        telemetry.event(reason_event)
        self._last_fed_decoded_index = self.decoded_frames
        self.fed_frames += 1
        if kp_pts is not None:
            self._reference_kp_pts = kp_pts
            self._reference_desc = desc
        return True

    @property
    def effective_fps(self) -> float:
        """The FED stream's fps estimate: the
        container's fps scaled by the fed/decoded ratio. Identical to
        the container fps while every frame is fed; converges to the
        gate's actual intake cadence as frames are skipped (the start
        transient - the first few frames before the first skip - is
        bounded and precedes the first keyframes)."""
        if self.decoded_frames == 0:
            return self.container_fps
        return self.container_fps * self.fed_frames / self.decoded_frames
