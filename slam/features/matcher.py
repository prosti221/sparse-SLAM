import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional, Dict
from slam.utils.utils import *
from slam.utils.constants import *
from slam.utils.logger import debug_log, error_log, warning_log
from slam.utils import telemetry
from slam.core.frame import Frame

LOG_TAG = 'Matcher'

# Matcher cache to avoid recreating expensive matcher objects
_matcher_cache: Dict[str, cv.DescriptorMatcher] = {}

# ===== Phase 13: the rotation-dominant pair pose (finding 37) =====
# Module-level flags + constants beside the machinery (the phase 9/10
# convention - the user's constants.py is their live experiment file).
# INT for the enable flag: the --override parser's type(current)(value)
# trap (bool("0") is True). Read at CALL time via this module's
# namespace so probes can pin them in memory for A/B.
#
# The finding-37 defect: on rotation-only motion the E-chain pair pose
# is degenerate (with true t~0 the epipolar constraint holds for ANY
# translation - a one-parameter exact-solution family), so every pair
# commits unit-normalized |t|=1.0 in noise-driven INCOHERENT directions
# plus a ~12% shrunken rotation. Five per-pair gate candidates died in
# measurement (sv-spread: the far-cluster homography is near-rotational
# on translation content; consensus fraction: inverted - sub-noise
# translation pairs are rotation-consistent by residual size;
# directionality: the ray-Kabsch refit absorbs a uniform parallax field
# as rotation; flip-decisiveness: the E's fabricated t is
# cheirality-confirmed on 80% of pan pairs). The surviving gate is
# TEMPORAL and lives in the tracker (see Tracker._update_rotation_
# model_gate): an EMA of |cos| between consecutive frame-chain pair
# E-t directions. These constants parameterize the ROTATION MODEL the
# engaged path commits (the 13a2-validated instrument: 3-point
# orthogonal Procrustes on unit rays, numpy-RNG only).
ROTATION_MODEL_ENABLED = 1
# The rotation-only consensus residual, pixel-space at unit depth
# (MATCHER_RANSAC_THRESHOLD's units): ~2.8 px at fx 940. Measured on
# the pan: consensus fraction p50 0.91 at this threshold with
# inlier counts in the hundreds-to-thousands.
ROTATION_MODEL_RESIDUAL = 3e-3
ROTATION_MODEL_MIN_INLIERS = 20
ROTATION_MODEL_RANSAC_ITERS = 256
# Fixed per call: deterministic, and numpy's Generator is NOT the cv
# global RNG stream - the engaged path changes no cv-RNG consumption
# order anywhere (the phase-3b RNG-transparency convention).
ROTATION_MODEL_RNG_SEED = 1305
# ===== the t-coherence gate (13c, phase13c_t_coherence.json) =====
# EMA over |cos(t_k, t_{k-1})| of consecutive frame-chain pair E-t
# directions. ENGAGE the rotation model below ENGAGE, re-arm legacy
# above REARM (sticky both ways). Pinned from the measured traces
# with margin: the kyoto_2_5 pan engages at the first sustained dip
# and stays engaged 281/281 pairs (its EMA p90 is 0.93, below
# re-arm); the kyoto_2_5 WALKING segment never engages (EMA min
# 0.958 - above engage with margin); walking_salem never engages
# (min 0.996). The synthetic harness sequences at their stride-1
# battery cadence read incoherent (their per-pair parallax sits at
# the ORB quantization noise - the pan's regime), which is why the
# gate is VIDEO-PATH scoped (Tracker.rotation_model_enabled,
# default False; slam.py enables it where it wires the intake
# gate). These bounds are measured pins, not tuning targets.
ROTATION_MODEL_EMA_ALPHA = 0.9
ROTATION_MODEL_T_COHERENCE_ENGAGE = 0.80
ROTATION_MODEL_T_COHERENCE_REARM = 0.95


def descriptor_norm(feature_extraction_method: str) -> int:
    """The cv norm for an extractor's descriptor kind - the phase-23
    seam: constants.EXTRACTOR_DESCRIPTOR_KINDS holds ONE entry per
    extractor name ('binary' = Hamming, 'float' = L2); a learned
    float-descriptor extractor (ALIKED) adds its own line there.
    Unknown names raise KeyError - the registry declares, never
    silently defaults."""
    kind = EXTRACTOR_DESCRIPTOR_KINDS[feature_extraction_method]
    return cv.NORM_HAMMING if kind == 'binary' else cv.NORM_L2


def _get_or_create_matcher(feature_extraction_method: str) -> cv.DescriptorMatcher:
    cache_key = feature_extraction_method
    if cache_key in _matcher_cache:
        return _matcher_cache[cache_key]

    # Phase 23: the FLANN branch died with SuperPoint. Every
    # extractor now matches with brute-force knn - deterministic, and
    # the triangulation-partner tool depends on that property.
    matcher = cv.BFMatcher(
        descriptor_norm(feature_extraction_method), crossCheck=False)

    _matcher_cache[cache_key] = matcher
    debug_log(
        LOG_TAG, f"Created and cached matcher for {feature_extraction_method}")
    return matcher


def estimate_essential_matrix(pts_f1_norm: np.ndarray, pts_f2_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    E, mask = cv.findEssentialMat(
        pts_f1_norm, pts_f2_norm,
        np.eye(3),
        method=cv.RANSAC,
        prob=MATCHER_RANSAC_PROBABILITY,
        threshold=MATCHER_RANSAC_THRESHOLD
    )
    inliers = np.sum(mask) if mask is not None else 0
    return E, mask, inliers


def evaluate_model_quality(mask: np.ndarray) -> float:
    """Evaluate the quality of a geometric model"""
    if mask is None or len(mask) == 0:
        return 0.0

    inlier_ratio = np.sum(mask) / len(mask)
    inlier_count = np.sum(mask)

    # Weight by both inlier ratio and absolute count
    quality_score = 0.7 * inlier_ratio + 0.3 * min(1.0, inlier_count / 100.0)

    return quality_score


def select_best_model(
    pts_f1_norm: np.ndarray,
    pts_f2_norm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Essential-matrix estimation with RANSAC + the quality score.

    Phase 23: the homography alternative is REMOVED. It was disabled
    in practice since its introduction (the early return made the
    entire selection branch dead code - the "disabling model
    selection until I figure out why the Homography is so shit for
    planar scenes" question was never answered), and no measured
    tier ever exercised it. The E-only policy is now the stated
    contract instead of an accident of statement ordering.
    """
    E, E_mask, E_inliers = estimate_essential_matrix(pts_f1_norm, pts_f2_norm)
    E_quality = evaluate_model_quality(E_mask)

    debug_log(
        LOG_TAG, f"Essential Matrix - Inliers: {E_inliers}, Quality: {E_quality:.3f}")
    return E, E_mask, "essential"


def lowe_ratio_filter(matches) -> List[cv.DMatch]:
    """Lowe's ratio test + one-to-one enforcement over knnMatch(k=2)
    results. Phase 23: extracted from its two inline copies
    (match_features and rotation_model_pose_between_frames) so the
    planned ALIKED+LightGlue seam has ONE named insertion point to
    replace.

    One deliberate semantic touch-up, pre-declared in the phase-23
    plan: the len(pair) < 2 guard (from the rotation-model copy) now
    protects BOTH callers - a knnMatch query with fewer than 2
    neighbors (tiny descriptor sets) previously crashed
    match_features' unpacking. Unexercised on every measured tier
    (the finding-35 featureless path returns earlier).
    """
    good_matches = []
    indices_f1_set, indices_f2_set = set(), set()
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if (m.distance < LOWE_RATIO * n.distance
                and m.queryIdx not in indices_f1_set
                and m.trainIdx not in indices_f2_set):
            indices_f1_set.add(m.queryIdx)
            indices_f2_set.add(m.trainIdx)
            good_matches.append(m)
    return good_matches


def match_features(frame1: Frame, frame2: Frame, feature_extraction_method: str,
                   motion_prior: Optional[np.ndarray] = None) -> Tuple[List[cv.DMatch], Optional[np.ndarray]]:
    matcher = _get_or_create_matcher(feature_extraction_method)

    f1_kp, f1_desc = frame1.get_keypoints_descriptors()
    f2_kp, f2_desc = frame2.get_keypoints_descriptors()

    # Finding 35 (phase 10a forced-reset probe): a featureless frame
    # yields descriptors=None, and BFMatcher.knnMatch is ASYMMETRIC
    # about it - knnMatch(None, X) silently returns no matches, but
    # knnMatch(X, None) raises the batchDistance assertion. Matching
    # against a featureless PREVIOUS or REFERENCE frame therefore
    # crashed the whole pipeline (the probe's post-reset bootstrap:
    # textured replay frames matched against the featureless reset
    # origin; a dark-scene reset on real video takes the same path).
    # Zero features on either side means zero matches: take the
    # finding-10 path (callers fall back to the kinematic estimate /
    # the next candidate). The event fires only on this previously-
    # unreachable-except-by-crash path, so battery event comparisons
    # stay exact (no benchmark frame pair is featureless).
    if (f1_desc is None or f2_desc is None
            or len(f1_desc) == 0 or len(f2_desc) == 0):
        telemetry.event("match_no_features")
        debug_log(LOG_TAG, "Featureless frame in match pair - "
                  "no matches possible")
        return [], None

    matches = matcher.knnMatch(f1_desc, f2_desc, k=2)
    # The planned ALIKED+LightGlue seam: a learned matcher would
    # replace matcher.knnMatch + lowe_ratio_filter with per-pair
    # confidences; everything downstream (the RANSAC E, extractRtFromE,
    # set_match_data contract) is unchanged.
    good_matches = lowe_ratio_filter(matches)

    debug_log(
        LOG_TAG, f"Found {len(good_matches)} good matches from ratio test")

    # Geometric filter: RANSAC on the essential matrix
    if len(good_matches) > MATCHER_RANSAC_MINIMUM_INLIERS:
        indices_f1 = [m.queryIdx for m in good_matches]
        indices_f2 = [m.trainIdx for m in good_matches]

        pts_f1_norm = frame1.kp_pts_norm[indices_f1]
        pts_f2_norm = frame2.kp_pts_norm[indices_f2]

        E, mask, selected_model = select_best_model(
            pts_f1_norm, pts_f2_norm)

        if E is None or mask is None or int(np.sum(mask)) < 5:
            # RANSAC could not estimate a usable model; no geometric pose.
            # Callers fall back to the kinematic estimate (finding 11: the
            # old code indexed into a None mask and crashed).
            telemetry.event("match_no_model")
            return good_matches, None

        inlier_mask = mask.ravel() == 1
        filtered_matches = [m for i, m in enumerate(
            good_matches) if inlier_mask[i]]

        # Recover the relative pose from the essential matrix. The four-fold
        # ambiguity is resolved by explicit cheirality scoring of all
        # decompositions, with the constant-velocity motion prior (the
        # previous frame-to-frame relative pose) breaking the ties that a
        # cheirality vote alone coin-flips under lateral motion (phase 1a).
        # All good matches participate, matching recoverPose's input set
        # (A/B measured: the E-inlier subset scores the same branches).
        # Convention: Rt is T_{f2<-f1}, the pose of frame1's camera in
        # frame2's camera frame, so Twc_f1 = Twc_f2 @ Rt. Translation is
        # unit-normalized (monocular scale).
        Rt = extractRtFromE(
            pts_f1_norm, pts_f2_norm,
            E, np.eye(3), motion_prior=motion_prior)

        debug_log(
            LOG_TAG,
            f"Found {len(filtered_matches)} filtered matches using {selected_model} model, "
            f"{len(filtered_matches) / len(good_matches) * 100:.2f}% inliers"
        )
        return filtered_matches, Rt
    else:
        telemetry.event("match_no_ransac")
        warning_log(LOG_TAG, "Not enough matches found for RANSAC")

    return good_matches, None


def match_features_between_frames(frame1: Frame, frame2: Frame, feature_extraction_method: str,
                                  motion_prior: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    matches, Rt = match_features(frame1, frame2, feature_extraction_method,
                                 motion_prior=motion_prior)

    # The list belongs to THIS pair, even when empty. Otherwise a failed
    # current-to-keyframe match leaves previous-frame trainIdx values
    # attached, which dedup/triangulation interpret against the wrong KF.
    frame1.set_match_data(matches)

    if len(matches) == 0:
        warning_log(
            LOG_TAG, f"No matches found between frames: {frame1.frame_id} and {frame2.frame_id}")
        telemetry.event("match_zero")
        # Finding 10 (see benchmarks/IMPROVEMENT.md): this used to return
        # `None, None` - a truthy 2-tuple that sailed through callers'
        # `Rt is not None` check and crashed the pose composition with a
        # TypeError. Zero matches means no pose estimate.
        return None

    debug_log(
        LOG_TAG, f"Found {len(matches)} matches between frames: {frame1.frame_id} and {frame2.frame_id}")

    return Rt


def rotation_model_pose_between_frames(frame1: Frame, frame2: Frame,
                                       feature_extraction_method: str
                                       ) -> Optional[np.ndarray]:
    """Phase 13: the rotation-dominant pair pose - the model the
    tracker commits when the t-coherence gate (see
    Tracker._update_rotation_model_gate) says the frame chain is in a
    rotation-only regime (finding 37: on such motion the E-chain's
    translation is a noise-driven unit vector - the pair cannot
    observe it, so the honest pair pose carries NO translation).

    The estimator is the 13a2-validated instrument: a numpy-RNG
    3-point orthogonal-Procrustes (Kabsch) RANSAC on unit rays, with
    residuals in pixel-space at unit depth - the same units as
    MATCHER_RANSAC_THRESHOLD. For a camera rotating in place this
    recovers the true pair rotation (the census pseudo-GT class);
    translation is exactly the unobservable component, so the
    committed pose is [R | 0].

    Mirrors match_features_between_frames' contract: the consensus
    matches are attached to frame1 (the pair's own list, even when
    empty - the finding-22 replacement rule), and the return is the
    T_{f2<-f1} pair pose or None on any degeneracy (caller keeps the
    legacy estimate - the no-basis principle).

    RNG discipline: numpy's Generator only (fixed seed per call) -
    NO cv-RNG draws, so the engaged path changes no cv-stream
    consumption order anywhere.
    """
    if not ROTATION_MODEL_ENABLED:
        return None

    f1_kp, f1_desc = frame1.get_keypoints_descriptors()
    f2_kp, f2_desc = frame2.get_keypoints_descriptors()
    if (f1_desc is None or f2_desc is None
            or len(f1_desc) == 0 or len(f2_desc) == 0):
        # The legacy E path (which always runs first) has already
        # taken the finding-35 no-matches path for this pair; the
        # rotation model has no basis either.
        return None

    matcher = _get_or_create_matcher(feature_extraction_method)
    matches = matcher.knnMatch(f1_desc, f2_desc, k=2)
    good_matches = lowe_ratio_filter(matches)
    if len(good_matches) < ROTATION_MODEL_MIN_INLIERS:
        return None

    q_idx = np.array([m.queryIdx for m in good_matches])
    t_idx = np.array([m.trainIdx for m in good_matches])
    pts_f1 = frame1.kp_pts_norm[q_idx]
    pts_f2 = frame2.kp_pts_norm[t_idx]
    n = len(pts_f1)
    x1h = np.hstack([pts_f1, np.ones((n, 1))])
    x2h = np.hstack([pts_f2, np.ones((n, 1))])
    r1 = x1h / np.linalg.norm(x1h, axis=1, keepdims=True)
    r2 = x2h / np.linalg.norm(x2h, axis=1, keepdims=True)

    def _residual(R):
        p = x1h @ R.T
        z = p[:, 2]
        ok = z > 1e-9
        res = np.full((n, 2), np.inf)
        res[ok] = p[ok, :2] / z[ok, None] - pts_f2[ok]
        return res

    def _kabsch(a, b):
        U, _, Vt = np.linalg.svd(a.T @ b)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            R = Vt.T @ np.diag([1.0, 1.0, -1.0]) @ U.T
        return R

    rng = np.random.default_rng(ROTATION_MODEL_RNG_SEED)
    best_cnt, best_mask = -1, None
    for _ in range(ROTATION_MODEL_RANSAC_ITERS):
        idx = rng.choice(n, 3, replace=False)
        R = _kabsch(r1[idx], r2[idx])
        if not np.isfinite(R).all():
            continue
        mask = np.linalg.norm(_residual(R), axis=1) < ROTATION_MODEL_RESIDUAL
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt, best_mask = cnt, mask
    if best_cnt < ROTATION_MODEL_MIN_INLIERS:
        debug_log(LOG_TAG, "Rotation model: no consensus "
                  f"({best_cnt} inliers) - keeping the legacy estimate")
        return None

    R = _kabsch(r1[best_mask], r2[best_mask])
    if not np.isfinite(R).all():
        return None
    res = _residual(R)
    mask = np.linalg.norm(res, axis=1) < ROTATION_MODEL_RESIDUAL
    if int(mask.sum()) < ROTATION_MODEL_MIN_INLIERS:
        return None

    frame1.set_match_data([m for i, m in enumerate(good_matches)
                           if mask[i]])
    Rt = np.eye(4)
    Rt[:3, :3] = R
    # t = 0 BY CONSTRUCTION: translation is exactly the component a
    # rotation-only pair cannot observe. The committed chain step
    # therefore carries no fabricated translation.
    telemetry.event("pair_rotation_model")
    debug_log(LOG_TAG, f"Rotation model: {int(mask.sum())}/{n} "
              f"consensus matches, pair rotation "
              f"{np.degrees(np.arccos(np.clip((np.trace(R) - 1.) / 2., -1., 1.))):.3f} deg")
    return Rt
