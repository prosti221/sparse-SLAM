import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional
from slam.utils.utils import *
from slam.utils.constants import *
from slam.utils.logger import debug_log, error_log, warning_log
from slam.core.frame import Frame

LOG_TAG = 'Matcher'


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


def estimate_homography(pts_f1_norm: np.ndarray, pts_f2_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    H, mask = cv.findHomography(
        pts_f1_norm, pts_f2_norm,
        method=cv.RANSAC,
        ransacReprojThreshold=MATCHER_RANSAC_THRESHOLD,
        confidence=MATCHER_RANSAC_PROBABILITY
    )
    inliers = np.sum(mask) if mask is not None else 0
    return H, mask, inliers


def homography_to_essential(H_norm: np.ndarray) -> np.ndarray:
    # Ensure E has the proper essential matrix constraints
    U, S, Vt = np.linalg.svd(H_norm)
    # Essential matrix should have two equal singular values
    S_corrected = np.array([1, 1, 0])
    E_corrected = U @ np.diag(S_corrected) @ Vt

    return E_corrected


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
    # Estimate Essential Matrix
    E, E_mask, E_inliers = estimate_essential_matrix(pts_f1_norm, pts_f2_norm)
    E_quality = evaluate_model_quality(E_mask)

    # Disabling model selection until I figure out why the Homogrophy is so shit for planar scenes    return E, E_mask, "essential"
    return E, E_mask, "essential"

    # Estimate Homography
    H, H_mask, H_inliers = estimate_homography(pts_f1_norm, pts_f2_norm)
    H_quality = evaluate_model_quality(H_mask)

    debug_log(
        LOG_TAG, f"Essential Matrix - Inliers: {E_inliers}, Quality: {E_quality:.3f}")
    debug_log(
        LOG_TAG, f"Homography - Inliers: {H_inliers}, Quality: {H_quality:.3f}")

    # Additional heuristics for model selection
    # If homography quality is much higher, scene might be planar

    if H_quality > E_quality * 1.3 and H_inliers > E_inliers:
        # Homography fits better - likely planar scene
        debug_log(LOG_TAG, "Selected Homography model (planar scene detected)")

        # Convert homography to essential matrix for pose estimation
        E_from_H = homography_to_essential(H)
        return E_from_H, H_mask, "homography"
    else:
        # Essential matrix fits better - general 3D scene
        debug_log(LOG_TAG, "Selected Essential Matrix model (3D scene)")
        return E, E_mask, "essential"


def match_features(frame1: Frame, frame2: Frame, feature_extraction_method: str) -> Tuple[List[cv.DMatch], Optional[np.ndarray]]:
    if feature_extraction_method == DNN_EXTRACTOR_NAME:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    else:
        is_binary_desc = feature_extraction_method in BINARY_DESCRIPTION_METHODS
        matcher = cv.BFMatcher(
            cv.NORM_HAMMING if is_binary_desc else cv.NORM_L2, crossCheck=False)

    f1_kp, f1_desc = frame1.get_keypoints_descriptors()
    f2_kp, f2_desc = frame2.get_keypoints_descriptors()

    matches = matcher.knnMatch(f1_desc, f2_desc, k=2)

    good_matches = []
    indices_f1_set, indices_f2_set = set(), set()
    for m, n in matches:
        if m.distance < LOWE_RATIO * n.distance and m.queryIdx not in indices_f1_set and m.trainIdx not in indices_f2_set:
            indices_f1_set.add(m.queryIdx)
            indices_f2_set.add(m.trainIdx)
            good_matches.append(m)

    debug_log(
        LOG_TAG, f"Found {len(good_matches)} good matches from ratio test")

    # Filter using RANSAC with model selection
    if len(good_matches) > MATCHER_RANSAC_MINIMUM_INLIERS:
        pts_f1 = np.array([f1_kp[m.queryIdx].pt for m in good_matches])
        pts_f2 = np.array([f2_kp[m.trainIdx].pt for m in good_matches])

        pts_f1_norm = normalize(pts_f1, frame1.Kinv)
        pts_f2_norm = normalize(pts_f2, frame2.Kinv)

        # Select best model between Essential Matrix and Homography
        E, mask, selected_model = select_best_model(
            pts_f1_norm, pts_f2_norm)

        filtered_matches = [m for i, m in enumerate(
            good_matches) if mask[i] == 1]

        Rt = extractRt(E)

        debug_log(
            LOG_TAG,
            f"Found {len(filtered_matches)} filtered matches using {selected_model} model, "
            f"{len(filtered_matches) / len(matches) * 100:.2f}% inliers"
        )
        return filtered_matches, Rt
    else:
        warning_log(LOG_TAG, "Not enough matches found for RANSAC")

    return good_matches, None


def match_features_between_frames(frame1: Frame, frame2: Frame, feature_extraction_method: str) -> Optional[np.ndarray]:
    matches, Rt = match_features(frame1, frame2, feature_extraction_method)

    if len(matches) == 0:
        warning_log(
            LOG_TAG, f"No matches found between frames: {frame1.frame_id} and {frame2.frame_id}")
        return None, None

    debug_log(
        LOG_TAG, f"Found {len(matches)} matches between frames: {frame1.frame_id} and {frame2.frame_id}")

    frame1.set_match_data(matches)

    return Rt
