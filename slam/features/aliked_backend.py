"""ALIKED + LightGlue backend (phase 24).

The learned feature frontend: ALIKED (IEEE TIM 2023, BSD-3-Clause)
detects sub-pixel keypoints with 128-dim L2-normalized float
descriptors, LightGlue (ICCV 2023, Apache-2.0) matches them with
per-pair confidences. Both live in the lightglue package
(cvg/LightGlue, installed from a pinned git rev) behind the OPTIONAL
'learned' extra (torch/torchvision/lightglue) - the geocalib/'calib'
pattern: the import is lazy and every torch import sits INSIDE a
function, so an ORB-only run never imports torch, and the suite runs
green without the extra. Without the extra the failure is a clear
install instruction, never a silent default.

Device + determinism (the [24a] probe adjudicates; the pins are the
probe's own): fp32 only (LightGlue's fp16 pruning path has a known
upstream bug, cvg/LightGlue issue #186); CUDA when available with
torch.use_deterministic_algorithms(True) and
CUBLAS_WORKSPACE_CONFIG=:4096:8 (set BEFORE the torch import - the
first caller does it). No cv-RNG is drawn anywhere here (the
Finding-39 invariant: new mid-run code never touches the cv stream).

Weights: download-on-demand with a skip-if-exists guard into the
installed lightglue package's weights/ dir (where LightGlue looks)
and TORCH_HOME (where ALIKED's torch.hub loads from) - never
committed (the ~100 MB learned-weights rule mirrors the
videos/datasets license rule: they live on the user's own machines).

Keypoint coordinates: the backend resizes the longest side to
ALIKED_RESIZE (LightGlue's preprocess default, multiple of 8) before
extraction and scales the detected keypoints back to ORIGINAL image
pixels - the Frame contract (set_features -> kp_pts -> kp_pts_norm)
is unchanged. Matching runs on original pixel coordinates with each
frame's own (W, H) - LightGlue normalizes internally per image, so
the pair is self-consistent even though the descriptors were
computed on the resized image.
"""
import os
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from slam.utils import constants as const
from slam.utils.logger import debug_log, warning_log

LOG_TAG = 'ALIKED'

_install_hint = ("the learned feature stack is the OPTIONAL 'learned' extra: "
                 "uv sync --extra learned (torch, torchvision, lightglue)")

# Models are heavy and singletons per process - the tracker and the
# intake gate share one extractor instance through this cache (the
# _matcher_cache pattern).
_backend_cache = {}


def _determinism_env() -> None:
    """Pre-import pins. Must run BEFORE the first torch import (the
    cuBLAS workspace env var is read at cuBLAS handle creation)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _import_torch():
    _determinism_env()
    try:
        import torch
    except ImportError as e:
        raise ImportError(_install_hint) from e
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_grad_enabled(False)
    return torch


def _device():
    """Device priority: CUDA (the measurement/gate device - the [24a]
    adjudicator passed it bit-identical double-runs) > MPS (the
    production laptop: the [24a] MPS addendum probe MEASURED it
    viable - deform_conv2d's Metal kernel exists and works, extract
    + match bit-stable double-runs, steady-state extraction
    138 ms/frame vs 639 ms CPU, ~4.6x; evidence
    p24_mps probe JSON in the session temp, cited in the phase-24
    record) > CPU. ALL devices run the same fp32 + deterministic-
    algorithms pins; measured legs stay on the desktop regardless
    (cross-device bit-identity is not a thing, phase 16)."""
    torch = _import_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _weights_note() -> None:
    """Weights live in torch.hub's cache (~/.cache/torch/hub by
    default): BOTH checkpoints - the ALIKED detector weights
    (ALIKED.__init__ loads via torch.hub.load_state_dict_from_url)
    and the LightGlue matcher checkpoint (its loader falls back to
    its release URL, also into the hub cache) - download on first
    use and skip-if-exists thereafter (the hub cache IS the guard;
    measured on the desktop probe). Never committed (the ~100 MB
    learned-weights rule mirrors the videos/datasets license rule)."""


_MATCHER_URL = ("https://github.com/cvg/LightGlue/releases/download/"
                "v0.1_arxiv/aliked_lightglue.pth")
_ALIKED_URL = ("https://github.com/Shiaoming/ALIKED/raw/main/models/"
               "{model}.pth")


def _get_models():
    """The cached (ALIKED, LightGlue) pair on the adjudicated device."""
    if "models" in _backend_cache:
        return _backend_cache["models"]
    torch = _import_torch()
    try:
        from lightglue import ALIKED, LightGlue
    except ImportError as e:
        raise ImportError(_install_hint) from e

    _weights_note()
    device = _device()
    extractor = ALIKED(
        model_name=const.ALIKED_MODEL_NAME,
        max_num_keypoints=const.ALIKED_MAX_NUM_KEYPOINTS,
        detection_threshold=const.ALIKED_DETECTION_THRESHOLD,
        nms_radius=const.ALIKED_NMS_RADIUS,
    ).eval().to(device)
    matcher = LightGlue(
        features="aliked",
        depth_confidence=const.LIGHTGLUE_DEPTH_CONFIDENCE,
        width_confidence=const.LIGHTGLUE_WIDTH_CONFIDENCE,
        filter_threshold=const.LIGHTGLUE_FILTER_THRESHOLD,
    ).eval().to(device)
    _backend_cache["models"] = (torch, extractor, matcher, device)
    return _backend_cache["models"]


def _to_tensor(img_bgr: np.ndarray):
    """BGR uint8 -> (1, 3, H, W) RGB float in [0, 1]. NO manual
    resize: extractor.extract runs its own ImagePreprocessor
    (ALIKED_RESIZE longest side) and scales the returned keypoints
    back to ORIGINAL image coordinates itself - the canonical
    upstream path (verified against the installed source)."""
    torch, _, _, _ = _get_models()
    rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def extract_aliked(img_bgr: np.ndarray
                   ) -> Tuple[List[cv.KeyPoint], Optional[np.ndarray]]:
    """ALIKED detection + description on one BGR frame.

    Returns the Frame contract: cv.KeyPoint list in ORIGINAL pixel
    coordinates (response = the detector score) + (N, 128) float32
    L2-normalized descriptors, or ([], None) when nothing is
    detected (the ORB path's empty convention - the finding-35
    featureless handling applies unchanged).
    """
    torch, extractor, _, device = _get_models()
    feats = extractor.extract(_to_tensor(img_bgr).to(device))
    kps = feats["keypoints"][0].cpu().numpy()
    desc = feats["descriptors"][0].cpu().numpy()
    scores = feats.get("keypoint_scores")
    if scores is not None:
        scores = scores[0].cpu().numpy()
    if len(kps) == 0:
        return [], None

    h, w = img_bgr.shape[:2]
    keypoints = []
    for i in range(len(kps)):
        x = float(np.clip(kps[i][0], 0, w - 1))
        y = float(np.clip(kps[i][1], 0, h - 1))
        response = float(scores[i]) if scores is not None else 1.0
        keypoints.append(cv.KeyPoint(x=x, y=y, size=10,
                                     angle=-1, response=response,
                                     octave=0, class_id=-1))
    return keypoints, np.asarray(desc, dtype=np.float32)


def match_aliked(frame1, frame2) -> List[cv.DMatch]:
    """LightGlue matching for one pair of ALIKED frames.

    API (verified against the installed source): ONE dict keyed
    image0/image1, each holding a per-image feature dict with
    'keypoints' (1,N,2 pixel coords), 'descriptors' (1,N,128) and
    'image_size' (1,2 = (W,H) - LightGlue normalizes per image; the
    sub-dict fallback (min/max of kpts) is worse, always pass it).
    Output: the DMatch contract (queryIdx/trainIdx into the frames'
    keypoint lists; distance = 1 - confidence, monotone with
    LightGlue's confidence - nothing downstream reads distance
    except the dispatch this replaces for learned pairs).
    """
    torch, _, matcher, device = _get_models()
    f1_kp, f1_desc = frame1.get_keypoints_descriptors()
    f2_kp, f2_desc = frame2.get_keypoints_descriptors()
    if (f1_desc is None or f2_desc is None
            or len(f1_desc) == 0 or len(f2_desc) == 0):
        return []

    def _image_dict(kps, desc, frame):
        h, w = frame.image.shape[:2]
        pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps],
                       dtype=np.float32)
        return {
            "keypoints": torch.from_numpy(pts).unsqueeze(0).to(device),
            "descriptors": torch.from_numpy(
                np.asarray(desc)).unsqueeze(0).to(device),
            "image_size": torch.tensor(
                [[w, h]], dtype=torch.float32, device=device),
        }

    pred = matcher({"image0": _image_dict(f1_kp, f1_desc, frame1),
                    "image1": _image_dict(f2_kp, f2_desc, frame2)})
    matches = pred["matches"][0].cpu().numpy()
    scores = pred["scores"][0].cpu().numpy()
    return [cv.DMatch(queryIdx=int(m[0]), trainIdx=int(m[1]),
                      distance=float(1.0 - s))
            for m, s in zip(matches, scores)]
