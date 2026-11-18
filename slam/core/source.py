"""The session source resolver: video files, frame sequences, cameras.

The generalization of slam.py's original loader (`(cap, K|None, k1)`, `fx: auto`
yields K=None and triggers the GeoCalib branch in slam.py) over the
three source kinds the editor supports. Sequence dirs resolve to a
duck-typed capture (read/get match cv.VideoCapture's surface, so
slam.py's loop runs unmodified); camera sources open by device index.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

from slam.utils.logger import info_log

LOG_TAG = "Source"

CAMERA_KIND = "camera"
FILE_KIND = "file"
SEQUENCE_KIND = "sequence"


@dataclass(frozen=True)
class SourceSpec:
    kind: str                    # FILE_KIND | SEQUENCE_KIND | CAMERA_KIND
    location: str                # path for files/dirs, str(index) for cameras
    fx: object = None            # "auto", a number, or None (absent)
    fy: object = None
    cx: object = None
    cy: object = None
    k1: float = 0.0
    sequence_fps: float = 30.0   # the identity anchor for sequence dirs


class SequenceCapture:
    """Duck-typed cv.VideoCapture over a frame-sequence directory.

    Serves .read() / .get() on the cv.VideoCapture surface slam.py's
    loop consumes; the underlying sorted file list also carries the
    rewind support (seek(frame_index) is exact by construction).
    """

    def __init__(self, directory: Path, fps: float = 30.0):
        self.directory = Path(directory)
        self.files = sorted(self.directory.glob("*.png"))
        if not self.files:
            raise FileNotFoundError(
                f"No frames found in {self.directory}")
        self.index = -1
        self._fps = fps if fps and fps > 0 else 30.0
        first = cv.imread(str(self.files[0]))
        if first is None:
            raise FileNotFoundError(
                f"First frame unreadable: {self.files[0]}")
        self._h, self._w = first.shape[:2]

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.index + 1 >= len(self.files):
            return False, None
        self.index += 1
        img = cv.imread(str(self.files[self.index]))
        return img is not None, img

    def get(self, prop: int) -> float:
        if prop == cv.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == cv.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop == cv.CAP_PROP_FPS:
            return float(self._fps)
        if prop == cv.CAP_PROP_FRAME_COUNT:
            return float(len(self.files))
        if prop == cv.CAP_PROP_POS_FRAMES:
            return float(self.index + 1)
        return 0.0

    def seek(self, frame_index: int) -> bool:
        """Exact source reposition for the rewind (frame_index is
        zero-based)."""
        if not 0 <= frame_index < len(self.files):
            return False
        self.index = frame_index - 1
        return True

    def release(self) -> None:
        self.index = -1


def detect_kind(location: str) -> str:
    """Directory with frame images -> a sequence; an int-able string
    -> a camera; everything else -> a video file."""
    path = Path(location)
    if path.is_dir():
        return SEQUENCE_KIND
    try:
        int(location)
        return CAMERA_KIND
    except ValueError:
        return FILE_KIND


def open_capture(spec: SourceSpec):
    if spec.kind in (FILE_KIND, CAMERA_KIND):
        cap = cv.VideoCapture(spec.location)
        if not cap.isOpened():
            return None
        return cap
    if spec.kind == SEQUENCE_KIND:
        return SequenceCapture(Path(spec.location), spec.sequence_fps)
    raise ValueError(f"Unknown source kind: {spec.kind}")


def resolve_intrinsics(spec: SourceSpec, width: int, height: int
                       ) -> Tuple[Optional[np.ndarray], float, dict]:
    """The original loader's semantics on a spec: `fx: auto` (or absent/zero
    -> (None, 0.0), the GeoCalib sentinel; explicit intrinsics ->
    (K, k1). Returns the third element as the resolution's provenance
    dict (the M1b cache writer's seed)."""
    fx = spec.fx
    fy = spec.fy if spec.fy not in (None, "auto", 0) else fx
    cx = spec.cx if spec.cx not in (None, "auto", 0) else width // 2
    cy = spec.cy if spec.cy not in (None, "auto", 0) else height // 2
    k1 = float(spec.k1) if spec.k1 not in (None, "auto") else 0.0
    if fx in ("auto", 0, None):
        return None, 0.0, {"fx": "auto"}
    fx = float(fx)
    fy = float(fy) if fy not in ("auto", None) else fx
    K = np.array([[fx, 0, float(cx)],
                  [0, fy, float(cy)],
                  [0, 0, 1]], dtype=float)
    return K, k1, {"fx": fx, "fy": fy, "cx": float(cx),
                   "cy": float(cy), "k1": k1}


def open_source(spec: SourceSpec):
    """The editor-facing entry: (cap, K|None, k1, meta) with meta
    carrying the capture's dimensions + the intrinsics provenance, or
    None when the source cannot open. The LOCATION decides the kind
    (spec.kind is the UI's intent label only)."""
    kind = detect_kind(spec.location)
    cap = open_capture(SourceSpec(kind, spec.location, spec.fx,
                                  spec.fy, spec.cx, spec.cy,
                                  spec.k1, spec.sequence_fps))
    if cap is None:
        return None
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    K, k1, intr = resolve_intrinsics(spec, W, H)
    info_log(LOG_TAG, f"Loaded source: {spec.location} ({kind}, "
             f"{W}x{H}) intrinsics {intr}")
    return cap, K, k1, {"kind": kind, "width": W, "height": H,
                        "intrinsics": intr, "location": spec.location}


def spec_from_config(config) -> SourceSpec:
    """The config adapter: the videos-map entry -> a spec (the
    the original loader's semantics, verbatim)."""
    tag = config.get_global_config_property("load_video")
    return SourceSpec(
        FILE_KIND, config.get_video_property(tag, "path"),
        fx=config.get_video_property(tag, "fx", default=None),
        fy=config.get_video_property(tag, "fy", default=None),
        cx=config.get_video_property(tag, "cx", default=None),
        cy=config.get_video_property(tag, "cy", default=None),
        k1=config.get_video_property(tag, "k1", default=0.0))


