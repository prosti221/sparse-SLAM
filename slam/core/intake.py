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
import threading

import multiprocessing as mp
import queue as queue_mod
import time
from multiprocessing import shared_memory

import cv2 as cv
import numpy as np

from slam.features.feature_extractor import FeatureExtractor
from slam.features.matcher import _get_or_create_matcher, lowe_ratio_filter
from slam.utils import telemetry
from slam.utils.constants import (EXTRACTOR_DESCRIPTOR_KINDS,
                                  FRAME_INTAKE_MAX_GAP_S,
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

# The extraction process (the dominant `extract` bucket off the
# critical path): the decode+extract producer ONE AHEAD in a worker
# PROCESS. The thread shape was measured and deleted three times - the
# extractor's GIL behavior (torch op dispatch needs the GIL the
# consumer holds; cv2 releases it) is the measured boundary, and a
# process sidesteps it. ADOPTED by C7b (2026-11-15) and made
# UNCONDITIONAL after it (user-directed flag removal: verified
# bit-silent on both machines; the inline path lives in git history
# and the C7/C7b records).

# The worker's torch thread bound: the torch-default intra-op pool is
# ~2x SLOWER than the bounded variants on the gate machine AND its
# threads contend with the consumer's native stages across processes
# (the measured C7 deficit). k = 1 is the C7b-adopted value (the best
# measured wall); 0 = no bound. The ORB worker never imports torch
# (the bound routes to the float-descriptor stack only). INT flag;
# read at call time via the module alias.
INTAKE_EXTRACT_WORKER_THREADS = 1

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

    def should_feed(self, img: np.ndarray, preextracted=None) -> bool:
        """One decoded frame: feed it to the tracker, or skip it?

        Deterministic, input-side only - no tracker/map state is
        touched. Every call counts one decoded frame.

        preextracted (the extraction-process seam): a (kps, desc) pair
        produced for THIS frame by the ExtractionProcess worker. When
        given, the gate consumes it instead of extracting inline -
        every other line identical (last_extraction is set either way,
        so the caller's carry keeps working).

        The extraction is CACHED in last_extraction: the caller (the
        live loop) attaches it to the fed frame, so the tracker's
        update() skips its own re-extraction - the gate's features
        come from the SAME gray image Frame.gray_image produces
        (_frame_gray mirrors that conversion exactly), so the carried
        features are bit-identical to what the tracker would have
        computed ([26a] measured every fed frame paying the
        extraction twice).
        """
        self.decoded_frames += 1
        if preextracted is None:
            kps, desc = self.feature_extractor.extract(_frame_gray(img))
        else:
            kps, desc = preextracted
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


# ===== The extraction process (worker decode+extract, one ahead) =====

def _kps_to_rows(kps) -> np.ndarray:
    """cv.KeyPoint does not pickle - the message carries the fields
    the consumer rebuilds into a FIELD-identical list (x, y, size,
    angle, response, octave, class_id)."""
    if len(kps) == 0:
        return np.empty((0, 7), dtype=np.float64)
    return np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response,
                      kp.octave, kp.class_id] for kp in kps],
                    dtype=np.float64)


def _rows_to_kps(rows: np.ndarray) -> list:
    return [cv.KeyPoint(x=r[0], y=r[1], size=r[2], angle=r[3],
                        response=r[4], octave=int(r[5]),
                        class_id=int(r[6])) for r in rows]


def _extract_process_main(video_path, extractor_name, start_index, enable_multiscale,
                          frame_w, frame_h, shm_names, msg_queue,
                          free_slots, stop_event, worker_threads):
    """The worker PROCESS target (module level: spawn-safe, picklable
    args). Opens its own capture on the same path, decodes + extracts
    ONE AHEAD, writes each decoded BGR frame into the shared slot
    whose token it holds, and ships (idx, slot, kps_rows, desc) on the
    message queue. Message convention: idx None is terminal - desc
    carrying a BaseException means a worker error; otherwise (idx,
    slot, kps rows, desc). No telemetry; no cv RNG draws (the
    extraction is deterministic given the frame).

    worker_threads > 0 bounds the torch intra-op pool (the torch
    stack only - an ORB worker never imports torch): the torch
    default pool is ~2x slower than the bounded variants on the gate
    machine and its threads contend with the consumer's native stages
    across processes (the measured C7 deficit). Extraction numerics
    are thread-count-invariant (measured on both gate machines), so
    the bound does not move a bit."""
    try:
        if worker_threads > 0 and EXTRACTOR_DESCRIPTOR_KINDS.get(
                extractor_name) == 'float':
            import torch
            torch.set_num_threads(worker_threads)

        cap = cv.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(
                f"extract worker cannot open {video_path}")
        if start_index > 0:
            cap.set(cv.CAP_PROP_POS_FRAMES, start_index)
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        if (w, h) != (frame_w, frame_h):
            raise RuntimeError(
                f"extract worker capture is {w}x{h}, expected "
                f"{frame_w}x{frame_h}")
        shms = [shared_memory.SharedMemory(name=n) for n in shm_names]
        slots = [np.ndarray((frame_h, frame_w, 3), dtype=np.uint8,
                            buffer=s.buf) for s in shms]
        extractor = FeatureExtractor(extractor_name)
        extractor.set_multiscale_enabled(enable_multiscale)

        idx = start_index - 1
        slot = 0
        while True:
            ret, img = cap.read()
            if not ret:
                msg_queue.put((None, None, None, None))
                return
            idx += 1
            # The free-slot token: timeout-bounded so a stopped
            # consumer cannot hang the worker.
            while not free_slots.acquire(timeout=0.1):
                if stop_event.is_set():
                    return
            kps, desc = extractor.extract(_frame_gray(img))
            slots[slot][:] = img
            rows = _kps_to_rows(kps)
            while True:
                try:
                    msg_queue.put((idx, slot, rows, desc), timeout=0.1)
                    break
                except queue_mod.Full:
                    if stop_event.is_set():
                        return
            slot = 1 - slot
    except BaseException as e:
        try:
            msg_queue.put((None, None, e, None))
        except Exception:
            pass
        raise


class ExtractionProcess:
    """The decode+extract producer ONE AHEAD in a worker PROCESS.

    next_frame() yields (idx, img, kps, desc) per decoded frame (the
    consumer copies the frame out of the shared slot before the slot's
    token is released); None at end-of-stream; the worker's exception
    re-raises here. close() is idempotent: it stops and joins the
    worker (bounded, loud on a stuck worker, terminate as the last
    resort), then releases the shared blocks. The pacing keeps the
    worker at most ~2 frames ahead of the consumer."""

    def __init__(self, video_path, frame_w, frame_h, extractor_name,
                 enable_multiscale=False, worker_threads=0,
                 start_index=0):
        self._frame_w = int(frame_w)
        self._frame_h = int(frame_h)
        frame_bytes = self._frame_h * self._frame_w * 3
        self._shm = [shared_memory.SharedMemory(create=True,
                                                size=frame_bytes)
                     for _ in range(2)]
        self._slots = [np.ndarray((self._frame_h, self._frame_w, 3),
                                  dtype=np.uint8, buffer=s.buf)
                       for s in self._shm]
        self._free_slots = mp.BoundedSemaphore(2)
        self._msgs = mp.Queue(maxsize=1)
        self._stop = mp.Event()
        self._proc = mp.Process(
            target=_extract_process_main,
            args=(str(video_path), extractor_name, start_index,
                  enable_multiscale,
                  self._frame_w, self._frame_h,
                  [s.name for s in self._shm], self._msgs,
                  self._free_slots, self._stop, int(worker_threads)),
            daemon=True)
        self._proc.start()
        self._closed = False

    def next_frame(self):
        while True:
            try:
                idx, slot, rows, desc = self._msgs.get(timeout=0.1)
                break
            except queue_mod.Empty:
                if self._stop.is_set():
                    return None
                if not self._proc.is_alive():
                    raise RuntimeError(
                        "extract worker died without a terminal "
                        "message")
        if idx is None:
            if isinstance(rows, BaseException):
                raise rows
            return None
        img = np.array(self._slots[slot])
        self._free_slots.release()
        return idx, img, _rows_to_kps(rows), desc

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        # Drain so a worker blocked on the bounded put can observe the
        # stop and exit; then a bounded join.
        while True:
            try:
                self._msgs.get(timeout=0.05)
            except queue_mod.Empty:
                break
        self._proc.join(timeout=5.0)
        if self._proc.is_alive():
            debug_log(LOG_TAG, "extract worker stuck; terminating")
            self._proc.terminate()
            self._proc.join(timeout=2.0)
        for s in self._shm:
            s.close()
            s.unlink()
