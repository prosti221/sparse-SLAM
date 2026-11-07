"""Stage test: feature matching and relative-pose recovery vs ground truth.

Regression for the extractRt replacement: match_features must return the
relative pose in the T_{frame2<-frame1} convention, composable with right
multiplication onto frame2's pose (Twc_1 = Twc_2 @ Rt).
"""
import cv2 as cv
import numpy as np
import pytest

from benchmarks.harness import load_calibration, load_trajectory
from benchmarks.synthetic.generator import ensure_sequence
from slam.core.frame import Frame
from slam.features.feature_extractor import FeatureExtractor
from slam.features.matcher import match_features_between_frames


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
    K = load_calibration(translation_seq)
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
    K = load_calibration(translation_seq)
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
    K = load_calibration(translation_seq)
    cv.setRNGSeed(123)
    prev_a, cur_a = _frame_pair(translation_seq, K, extractor, 30)
    Rt_a = match_features_between_frames(cur_a, prev_a, "ORB")

    cv.setRNGSeed(123)
    prev_b, cur_b = _frame_pair(translation_seq, K, extractor, 30)
    Rt_b = match_features_between_frames(cur_b, prev_b, "ORB")

    assert np.allclose(Rt_a, Rt_b, atol=1e-12)
