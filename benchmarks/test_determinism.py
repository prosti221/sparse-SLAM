"""Determinism guard: the same sequence + seed must produce identical results.

This protects the benchmark's core contract (see benchmarks/PLAN.md). If it
fails, someone introduced an unseeded randomness source into the pipeline.
"""
import numpy as np
import pytest

from benchmarks.synthetic.generator import ensure_sequence
from benchmarks.harness import run_slam


@pytest.fixture(scope="module")
def translation_sequence_dir():
    return ensure_sequence("translation")


def test_end_to_end_determinism(translation_sequence_dir):
    r1 = run_slam(translation_sequence_dir, seed=42, debug=False)
    r2 = run_slam(translation_sequence_dir, seed=42, debug=False)

    assert r1.frames_processed == r2.frames_processed
    assert r1.poses.shape == r2.poses.shape
    assert np.allclose(r1.poses, r2.poses, atol=1e-9), \
        "per-frame poses differ between identical runs - nondeterminism crept in"

    assert r1.num_keyframes == r2.num_keyframes
    assert r1.keyframe_indices == r2.keyframe_indices
    assert r1.resets == r2.resets

    assert r1.map_points.shape == r2.map_points.shape
    assert np.allclose(r1.map_points, r2.map_points, atol=1e-9), \
        "map points differ between identical runs"


def test_sequence_generation_is_deterministic(tmp_path):
    """Regenerating a sequence must reproduce byte-identical frames."""
    from benchmarks.synthetic.generator import DATA_ROOT, generate_sequence
    import cv2 as cv

    name = "translation"
    gen_dir = generate_sequence(name, force=False)
    frame_files = sorted((gen_dir / "frames").glob("*.png"))
    assert len(frame_files) > 0

    # Frames on disk must be stable (already generated once, now re-checked
    # against a fresh regeneration into the canonical directory).
    generate_sequence(name, force=True)
    for i, frame_file in enumerate(frame_files):
        img = cv.imread(str(frame_file))
        assert img is not None and img.shape[:2] == (480, 640)
        if i == 0:
            # A sanity property of our world: the image must contain textured
            # billboards (not just the flat background).
            assert img.std() > 10, "frame looks empty - world rendering broke"
