"""Phase 24: the ALIKED + LightGlue seam pins.

The learned stack (torch/lightglue) is OPTIONAL ('learned' extra);
every pin here runs WITHOUT it except the last one, which skips
unless the weights are already in the torch.hub cache (the desktop
probe downloads them; the laptop stays skip-until-provisioned - the
suite must never download weights). The seam contract:

- the registry declares ALIKED; the descriptor-kind map routes its
  128-dim float descriptors to cv.NORM_L2 (intake gate matcher, the
  partner tool, and the tracker's reloc norm all read it);
- match_pair_descriptors dispatches learned pairs to the backend
  and everything else to knn+Lowe - ONE insertion point, both
  call sites (match_features + rotation_model_pose_between_frames);
- the reloc descriptor-accept threshold is per KIND (the old
  literal 20 was an ORB-Hamming pin);
- without torch the failure is a clear install instruction.
"""
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

import slam.features.aliked_backend as backend
import slam.features.matcher as matcher_mod
from benchmarks.stages.helpers import make_frame, set_keypoints
from slam.features.feature_extractor import FeatureExtractor
from slam.utils import constants as const


def _textured(w=120, h=90, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, (h, w, 3))).astype(np.uint8)


def test_registry_declares_aliked_and_the_kind_map_routes_l2():
    ex = FeatureExtractor(const.ALIKED_EXTRACTOR_NAME)
    assert ex.feature_extraction_method == "ALIKED"
    # ALIKED is natively multi-scale: both handlers are the same
    # closure and the flag is a documented no-op.
    assert ex._extract_multiscale is ex._extract_single
    ex.set_multiscale_enabled(False)
    assert ex.extract_handler is ex._extract_single
    ex.set_multiscale_enabled(True)
    assert ex.extract_handler is ex._extract_multiscale
    # the kind seam: float -> L2 (intake matcher + tracker reloc norm)
    assert const.EXTRACTOR_DESCRIPTOR_KINDS["ALIKED"] == "float"
    assert matcher_mod.descriptor_norm("ALIKED") == cv.NORM_L2
    assert matcher_mod.descriptor_norm("ORB") == cv.NORM_HAMMING


def test_kind_map_rejects_unknown_names():
    """The seam declares, never silently defaults: an unregistered
    name raises at the lookup (the phase-23 contract)."""
    with pytest.raises(KeyError):
        matcher_mod.descriptor_norm("SIFT")


def test_reloc_threshold_map_pins_the_kinds():
    """tracker.py's old literal 20 was an ORB-Hamming pin; the map
    carries it for 'binary'. The 'float' value is the [24a]
    reloc-threshold probe's MEASURED pin: the aggregate p99 of
    geometrically-verified inlier ALIKED descriptor L2 distances on
    the 3 TUM stems (0.8639; evidence
    output/benchmarks/phase24_probe_reloc_threshold.json)."""
    assert const.EXTRACTOR_RELOC_DESC_THRESHOLDS["binary"] == 20.0
    assert const.EXTRACTOR_RELOC_DESC_THRESHOLDS["float"] == 0.8639


def test_match_dispatch_routes_learned_pairs_to_the_backend(monkeypatch):
    f1 = set_keypoints(make_frame(), [(10.0, 10.0), (50.0, 20.0)])
    f2 = set_keypoints(make_frame(), [(11.0, 10.0), (51.0, 20.0)])
    sentinel = [cv.DMatch(0, 1, 0.0)]
    monkeypatch.setattr(backend, "match_aliked",
                        lambda a, b: sentinel)
    assert matcher_mod.match_pair_descriptors(
        f1, f2, "ALIKED") is sentinel
    # the ORB path is the REAL knn+Lowe route (never the backend)
    out = matcher_mod.match_pair_descriptors(f1, f2, "ORB")
    assert isinstance(out, list) and out is not sentinel


def test_match_dispatch_declares_unknown_names():
    f1 = set_keypoints(make_frame(), [(10.0, 10.0)])
    f2 = set_keypoints(make_frame(), [(11.0, 10.0)])
    with pytest.raises(KeyError):
        matcher_mod.match_pair_descriptors(f1, f2, "SIFT")


def test_backend_without_torch_gives_the_install_hint(monkeypatch):
    """The geocalib pattern: absence of the optional extra is a
    clear instruction, never a silent default. sys.modules['torch']
    = None makes `import torch` raise ImportError."""
    monkeypatch.setitem(sys.modules, "torch", None)
    backend._backend_cache.pop("models", None)
    with pytest.raises(ImportError, match="learned"):
        backend._import_torch()


def _learned_stack_ready():
    """The real-inference pin runs ONLY where the stack is actually
    provisioned: lightglue importable AND both weight files in the
    torch.hub cache (the suite never downloads weights)."""
    try:
        import lightglue  # noqa: F401
    except ImportError:
        return False
    hub = Path(os.environ.get("TORCH_HOME",
                              "~/.cache/torch")).expanduser() / "hub"
    cache = hub / "checkpoints"
    needed = ["aliked-n16.pth", "aliked_lightglue_v0-1_arxiv.pth"]
    return all((cache / n).exists() for n in needed)


@pytest.mark.skipif(not _learned_stack_ready(),
                    reason="learned stack not provisioned on this "
                           "machine (suite never downloads weights)")
def test_real_aliked_extraction_contract():
    """The REAL frontend, where provisioned: extract on a textured
    image and check the Frame contract end-to-end."""
    ex = FeatureExtractor("ALIKED")
    kps, desc = ex.extract(_textured())
    assert len(kps) > 0
    assert desc is not None and desc.dtype == np.float32
    assert desc.shape[1] == 128
    h, w = 90, 120
    for kp in kps:
        assert 0 <= kp.pt[0] < w and 0 <= kp.pt[1] < h
        assert kp.response > 0
