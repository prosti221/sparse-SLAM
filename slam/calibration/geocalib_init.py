"""GeoCalib intrinsics initializer (phase 2j-2).

Estimates shared camera intrinsics from a batch of frames of ONE
camera, using GeoCalib's multi-image mode (cvg/GeoCalib, ECCV 2024):

    model.calibrate(batch, shared_intrinsics=True,
                    camera_model="simple_radial")

GeoCalib predicts focal + one simple-radial k1 (its "distorted"
weights) and - critically for the 2j-4 refit's prior weighting - its
uncertainty-aware LM reports a focal uncertainty. The principal point
is ASSUMED AT THE IMAGE CENTER and not optimized (GeoCalib's own
convention), which matches the 2j design (c pinned at center).

ViPE-style usage for video feeds: buffer the first ~CALIB_INIT_SECONDS
of frames and calibrate them together (estimate_video_intrinsics) -
the caller must then replay the buffered frames into the SLAM loop
(they have been consumed from the capture).

The geocalib dependency is the OPTIONAL 'calib' extra (it declares
torch itself since phase 23 moved torch out of the core deps); the
import is lazy and failure gives a clear install instruction. The
2j-5 recovery tests on
synthetic data deliberately do NOT use this initializer (GeoCalib's
learned priors are natural-image statistics; on the generator's
speckle textures it is +113% off on f while reporting sigma ~5% -
out-of-domain overconfidence, recorded in the 2j-2 record).
Real-image accuracy is measured on TUM in 2j-5c.
"""
import cv2 as cv
import numpy as np

from slam.utils import constants as const
from slam.utils import telemetry
from slam.utils.logger import error_log, info_log, warning_log

LOG_TAG = "CalibInit"


class IntrinsicsPrior:
    """Pinhole intrinsics + one simple-radial k1 with the initializer's
    own uncertainty (absolute px / k1 units, 1-sigma). The principal
    point is pinned at the image center - GeoCalib's assumption, and
    the 2j-4 refit's design. sigma defaults to inf (no prior)."""

    def __init__(self, fx, fy, cx, cy, k1=0.0,
                 sigma_f=float("inf"), sigma_k1=float("inf"),
                 source="unknown"):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.k1 = float(k1)
        self.sigma_f = float(sigma_f)
        self.sigma_k1 = float(sigma_k1)
        self.source = source

    @property
    def f(self):
        return 0.5 * (self.fx + self.fy)

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    def __repr__(self):
        return (f"IntrinsicsPrior(f=({self.fx:.1f},{self.fy:.1f}) "
                f"c=({self.cx:.1f},{self.cy:.1f}) k1={self.k1:.5f} "
                f"sigma_f={self.sigma_f:.1f} sigma_k1={self.sigma_k1:.4f} "
                f"source={self.source})")


def _to_geocalib_batch(frames_bgr):
    """BGR uint8 frames -> (B, C, H, W) float tensor in [0, 1], RGB -
    GeoCalib's expected input (range checked in its calibrate())."""
    import torch

    arr = np.stack([cv.cvtColor(f, cv.COLOR_BGR2RGB) for f in frames_bgr])
    batch = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    return batch


def _extract_prior(result, img_w, img_h):
    """GeoCalib result -> IntrinsicsPrior (pure; kept separate for the
    mocked stage pins). shared_intrinsics=True makes every per-image
    entry identical - the last entry represents the batch.

    NOTE on the API (validated against the installed package): the
    uncertainties live at the TOP LEVEL of the calibrate() result
    (extractor.py spreads every internal key containing "uncertainty");
    result["covariance"] is the raw covariance MATRIX and is NOT
    key-addressable. focal_uncertainty is GeoCalib's RELATIVE 1-sigma
    of f (measured: 0.0465 on a batch whose f it estimated at 958 px).
    If the key is absent, GeoCalib declined to report it - the prior
    degrades to sigma=inf (anchor only, no trust)."""
    cam = result["camera"]
    f = np.asarray(cam.f[0].detach().cpu().numpy(), dtype=np.float64)
    k1 = float(np.asarray(cam.k1[0].detach().cpu().numpy()).reshape(-1)[0])

    sigma_f = float("inf")
    if "focal_uncertainty" in result:
        rel = result["focal_uncertainty"]
        rel = rel.detach().cpu().numpy() if hasattr(rel, "detach") \
            else np.asarray(rel)
        rel_sigma_f = float(np.asarray(rel).reshape(-1)[-1])
        sigma_f = abs(rel_sigma_f) * 0.5 * (f[0] + f[1])

    return IntrinsicsPrior(
        fx=f[0], fy=f[1],
        cx=img_w / 2.0, cy=img_h / 2.0,
        k1=k1,
        sigma_f=sigma_f,
        # GeoCalib reports no k1 uncertainty - use the loose policy
        # constant (the refit treats it as a soft prior only).
        sigma_k1=const.CALIB_PRIOR_SIGMA_K1,
        source="geocalib")


def estimate_intrinsics_prior(frames_bgr,
                              camera_model="simple_radial",
                              weights="distorted"):
    """Estimate shared intrinsics from a batch of frames of ONE camera.

    frames_bgr: list of BGR uint8 images (the ~2s video-init batch).
    Returns an IntrinsicsPrior with GeoCalib's self-reported focal
    uncertainty. NOTE: the reported sigma is calibrated for GeoCalib's
    TRAINED domain (natural imagery); on out-of-domain input it is
    overconfident - the 2j-4 refit's prior-weight policy accounts for
    this (see the 2j-2 record).
    """
    if not frames_bgr:
        raise ValueError("estimate_intrinsics_prior: no frames given")

    try:
        import torch
        from geocalib import GeoCalib
    except ImportError as e:
        raise ImportError(
            "camera intrinsics auto-calibration (fx: auto) requires the "
            "optional 'calib' extra: uv sync --extra calib "
            "(geocalib + its inference dependencies). "
            f"Original import error: {e}") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCalib(weights=weights).to(device).eval()
    h, w = frames_bgr[0].shape[:2]
    batch = _to_geocalib_batch(frames_bgr).to(device)

    with torch.no_grad():
        result = model.calibrate(batch, shared_intrinsics=True,
                                 camera_model=camera_model)

    prior = _extract_prior(result, w, h)

    telemetry.event("calib_init_frames", len(frames_bgr))
    telemetry.event("calib_init_f", round(prior.f, 2))
    telemetry.event("calib_init_k1", round(prior.k1, 5))
    telemetry.event("calib_init_sigma_f", round(prior.sigma_f, 2))
    info_log(LOG_TAG,
             f"GeoCalib init from {len(frames_bgr)} frames: {prior}")
    return prior


def estimate_video_intrinsics(cap):
    """Buffer the first ~CALIB_INIT_SECONDS of video from a capture
    and estimate shared intrinsics (ViPE-style init). Returns
    (prior, frames): the frames have been CONSUMED from the capture
    and must be replayed into the SLAM loop by the caller."""
    fps = cap.get(cv.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
        warning_log(LOG_TAG, "capture reports no FPS; assuming 30 "
                            f"for the {const.CALIB_INIT_SECONDS}s init batch")
    n = int(np.clip(round(const.CALIB_INIT_SECONDS * fps),
                    const.CALIB_INIT_MIN_FRAMES,
                    const.CALIB_INIT_MAX_FRAMES))

    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    if len(frames) < const.CALIB_INIT_MIN_FRAMES:
        error_log(LOG_TAG,
                  f"fx: auto needs >= {const.CALIB_INIT_MIN_FRAMES} frames "
                  f"to calibrate; got {len(frames)}")
        raise RuntimeError(
            f"auto intrinsics calibration: too few frames in video "
            f"({len(frames)} < {const.CALIB_INIT_MIN_FRAMES})")

    prior = estimate_intrinsics_prior(frames)
    return prior, frames
