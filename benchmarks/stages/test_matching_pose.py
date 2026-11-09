"""Stage test: feature matching and relative-pose recovery vs ground truth.

Regression for the extractRt replacement: match_features must return the
relative pose in the T_{frame2<-frame1} convention, composable with right
multiplication onto frame2's pose (Twc_1 = Twc_2 @ Rt).
"""
import cv2 as cv
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.harness import load_calibration, load_trajectory
from benchmarks.synthetic.generator import ensure_sequence
from benchmarks.stages.helpers import se3
from slam.core.frame import Frame
from slam.features.feature_extractor import FeatureExtractor
from slam.features.matcher import match_features_between_frames
from slam.utils.utils import extractRtFromE


def _rot_error(R_a, R_b):
    return np.degrees(np.arccos(np.clip(
        (np.trace(R_a.T @ R_b) - 1.0) / 2.0, -1.0, 1.0)))


@pytest.fixture(scope="module")
def translation_seq():
    return ensure_sequence("translation")


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor("ORB")


def _frame_pair(seq_dir, K, extractor, i):
    """(prev, cur) frames with features extracted, tracker-style."""
    frames = []
    for idx in (i, i + 1):
        f = Frame(cv.imread(f"{seq_dir}/frames/frame_{idx:06d}.png"), K)
        kps, desc = extractor.extract(f.gray_image)
        f.set_features(kps, desc)
        frames.append(f)
    return frames


def test_relative_pose_convention(translation_seq, extractor):
    K, _k1 = load_calibration(translation_seq)
    gt = load_trajectory(translation_seq)
    cv.setRNGSeed(42)

    # Frame pair 21->22 is on the clean side of the lateral-motion ambiguity
    # (deterministic under the fixed seed; see PLAN.md known limitation).
    prev_f, cur_f = _frame_pair(translation_seq, K, extractor, 21)
    Rt = match_features_between_frames(cur_f, prev_f, "ORB")

    # Expected: pose of cur expressed in prev's camera frame.
    Rt_gt = np.linalg.inv(gt[21]) @ gt[22]
    assert _rot_error(Rt[:3, :3], Rt_gt[:3, :3]) < 1.0

    t_est = Rt[:3, 3] / np.linalg.norm(Rt[:3, 3])
    t_gt = Rt_gt[:3, 3] / np.linalg.norm(Rt_gt[:3, 3])
    assert float(t_est @ t_gt) > 0.95


def test_composition_matches_ground_truth(translation_seq, extractor):
    """The tracker's exact usage: Twc_cur = Twc_prev @ Rt must reproduce the
    GT motion (rotation exactly; translation direction, monocular scale)."""
    K, _k1 = load_calibration(translation_seq)
    gt = load_trajectory(translation_seq)
    cv.setRNGSeed(42)

    prev_f, cur_f = _frame_pair(translation_seq, K, extractor, 21)
    Rt = match_features_between_frames(cur_f, prev_f, "ORB")

    composed = gt[21] @ Rt  # right-multiplication composition

    assert _rot_error(composed[:3, :3], gt[22][:3, :3]) < 1.0
    t_comp = composed[:3, 3] / np.linalg.norm(composed[:3, 3])
    t_gt = gt[22][:3, 3] / np.linalg.norm(gt[22][:3, 3])
    assert float(t_comp @ t_gt) > 0.95


def test_matching_is_deterministic(translation_seq, extractor):
    """OpenCV's RANSAC must be seedable: same seed, same result."""
    K, _k1 = load_calibration(translation_seq)
    cv.setRNGSeed(123)
    prev_a, cur_a = _frame_pair(translation_seq, K, extractor, 30)
    Rt_a = match_features_between_frames(cur_a, prev_a, "ORB")

    cv.setRNGSeed(123)
    prev_b, cur_b = _frame_pair(translation_seq, K, extractor, 30)
    Rt_b = match_features_between_frames(cur_b, prev_b, "ORB")

    assert np.allclose(Rt_a, Rt_b, atol=1e-12)


# --- Phase 1a: essential-matrix disambiguation (IMPROVEMENT.md) ---

def _skew(t):
    return np.array([[0.0, -t[2], t[1]],
                     [t[2], 0.0, -t[0]],
                     [-t[1], t[0], 0.0]])


def _analytic_scene():
    """Noise-free non-planar scene with known relative motion.

    Points live in camera 1's frame; camera 2 satisfies X_2 = R X_1 + t
    (the extractRtFromE convention). E = [t]_x R is built analytically, so
    the true decomposition is exact and the tests are seed-independent.
    """
    rng = np.random.default_rng(7)
    P = np.hstack([rng.uniform(-2, 2, (60, 2)),
                   rng.uniform(6, 10, (60, 1))])
    R_true = Rotation.from_euler("y", np.radians(3.0)).as_matrix()
    t_true = np.array([0.4, 0.05, 0.02])
    X2 = (R_true @ P.T).T + t_true
    x1 = P[:, :2] / P[:, 2:3]
    x2 = X2[:, :2] / X2[:, 2:3]
    E = _skew(t_true) @ R_true
    return x1, x2, E, R_true, t_true


def test_extract_rt_recovers_exact_motion():
    """With or without a prior, an unambiguous E must decompose exactly."""
    x1, x2, E, R_true, t_true = _analytic_scene()

    for prior in (None, se3(R_true, t_true)):
        Rt = extractRtFromE(x1, x2, E, np.eye(3), motion_prior=prior)
        assert _rot_error(Rt[:3, :3], R_true) < 1e-6
        assert float(Rt[:3, 3] @ t_true) / np.linalg.norm(t_true) > 0.999


def test_prior_cannot_override_clear_geometry():
    """The prior only breaks residual ties: when cheirality support is
    unambiguous, even a prior pointing the WRONG way must not flip the
    answer. (A bad prior must never corrupt a good estimate.)"""
    x1, x2, E, R_true, t_true = _analytic_scene()

    wrong_prior = se3(R_true, -t_true)  # translation reversed
    Rt = extractRtFromE(x1, x2, E, np.eye(3), motion_prior=wrong_prior)

    assert _rot_error(Rt[:3, :3], R_true) < 1e-6
    assert float(Rt[:3, 3] @ t_true) > 0.0


def test_prior_leaves_coinflips_alone():
    """Phase 1a gate semantics: the motion prior arbitrates only DECISIVE
    cheirality verdicts (winner out-supports its opposite-translation twin
    by MATCHER_PRIOR_DECISIVE_FLIP_RATIO). Points between the cameras
    (behind cam2 for the true forward motion, since cam2 sits at -t) make
    the +-t support split a coin flip (107/93 by construction); a prior
    overriding such noise would correlate it into drift (measured:
    translation ATE 1.81 -> 3.31 when it did). Even a prior OPPOSING the
    cheirality winner must leave the coin-flip result unchanged."""
    rng = np.random.default_rng(3)
    close = np.hstack([rng.uniform(-0.4, 0.4, (200, 2)),
                       rng.uniform(0.15, 0.85, (200, 1))])
    far = np.hstack([rng.uniform(-0.6, 0.6, (6, 2)),
                     rng.uniform(3.0, 5.0, (6, 1))])
    P = np.vstack([close, far])

    R_true = np.eye(3)
    t_true = np.array([0.0, 0.0, -1.0])  # cam2 ahead of cam1 at z=+1
    X2 = (R_true @ P.T).T + t_true
    x1 = P[:, :2] / P[:, 2:3]
    x2 = X2[:, :2] / X2[:, 2:3]
    E = _skew(t_true) @ R_true

    no_prior = extractRtFromE(x1, x2, E, np.eye(3), motion_prior=None)
    winner_t = no_prior[:3, 3] / np.linalg.norm(no_prior[:3, 3])

    # A prior opposing the winner's direction must not flip a coin flip.
    opposing = np.eye(4)
    opposing[:3, :3] = R_true
    opposing[:3, 3] = -winner_t
    arbitrated = extractRtFromE(x1, x2, E, np.eye(3), motion_prior=opposing)

    assert np.allclose(arbitrated[:3, :3], no_prior[:3, :3])
    assert float(arbitrated[:3, 3] @ winner_t) > 0.99, \
        "prior arbitrated a coin-flip cheirality verdict"
