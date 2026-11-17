"""The ALIKED + LightGlue seam pins.

The learned stack (torch/lightglue) is OPTIONAL ('learned' extra);
every pin here runs WITHOUT it except the last two, which skip
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
- without torch the failure is a clear install instruction;
- the MPS-only compile gates (ALIKED_MPS_COMPILE_ENABLED /
  LIGHTGLUE_MPS_COMPILE_ENABLED, read at call time) default ON and
  are reachable through benchmarks.run --override; the eager path is
  exact, and the compiled path is bit-stable with a measured parity
  bound vs eager (the inductor reassociation).
"""
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2 as cv
import numpy as np
import pytest

import slam.features.aliked_backend as backend
import slam.features.matcher as matcher_mod
from benchmarks.stages.helpers import make_K, make_frame, set_keypoints
from slam.core.frame import Frame
from slam.features.feature_extractor import FeatureExtractor
from slam.features.matcher import match_features
from slam.utils import constants as const


def _textured(w=120, h=90, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, (h, w, 3))).astype(np.uint8)


def _textured_shifted(dx=6, w=120, h=90, seed=3):
    """The same texture shifted horizontally (the same content, a
    different camera center) for the matching pin."""
    return np.roll(_textured(w=w, h=h, seed=seed), dx, axis=1)


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
    name raises at the lookup (the declared contract)."""
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


def test_mps_compile_flags_default_on_and_override_reachable(monkeypatch):
    """The compile gates ship ON (user-directed default) and resolve
    through benchmarks.run --override (the flag_modules seam: the
    override lands on THIS module object, which the call-time reads
    see)."""
    assert backend.ALIKED_MPS_COMPILE_ENABLED == 1
    assert backend.LIGHTGLUE_MPS_COMPILE_ENABLED == 1
    from benchmarks.run import apply_overrides
    monkeypatch.setattr(backend, "ALIKED_MPS_COMPILE_ENABLED", 1)
    monkeypatch.setattr(backend, "LIGHTGLUE_MPS_COMPILE_ENABLED", 1)
    apply_overrides(["ALIKED_MPS_COMPILE_ENABLED=0",
                     "LIGHTGLUE_MPS_COMPILE_ENABLED=0"])
    assert backend.ALIKED_MPS_COMPILE_ENABLED == 0
    assert backend.LIGHTGLUE_MPS_COMPILE_ENABLED == 0


class _RecordingExtractor:
    """Stub for the eager-path gate pins: records the call, never
    touches torch.compile."""

    def __init__(self):
        self.calls = []

    def extract(self, tensor):
        self.calls.append(("extract", tensor))
        return "feats"


class _FakeDevice:
    def __init__(self, type_):
        self.type = type_


class _NoCompileTorch:
    """A torch stand-in whose compile() fails the test if reached."""

    def compile(self, *a, **k):
        raise AssertionError("torch.compile reached on the eager path")


def test_extract_gate_is_device_and_flag_conditional():
    """_extract_feats reroutes ONLY on (mps device AND flag on); every
    other combination is the exact upstream extractor.extract call -
    the CUDA/CPU desktops and the flag-off A/B control keep the eager
    path byte-identical by construction."""
    ex = _RecordingExtractor()
    tensor = np.zeros((1, 3, 8, 8), dtype=np.float32)
    torch_stub = _NoCompileTorch()
    feats = backend._extract_feats(torch_stub, ex, _FakeDevice("cpu"),
                                   tensor)
    assert feats == "feats" and ex.calls == [("extract", tensor)]
    ex.calls.clear()
    monkey_flag = backend.ALIKED_MPS_COMPILE_ENABLED
    try:
        backend.ALIKED_MPS_COMPILE_ENABLED = 0
        feats = backend._extract_feats(torch_stub, ex, _FakeDevice("mps"),
                                       tensor)
        assert feats == "feats" and ex.calls == [("extract", tensor)]
    finally:
        backend.ALIKED_MPS_COMPILE_ENABLED = monkey_flag


class _StubMatcher:
    def __init__(self):
        self.calls = []

    def __call__(self, inputs):
        self.calls.append(inputs)
        return "pred"


def test_matcher_gate_is_device_and_flag_conditional():
    """The matcher call reroutes ONLY on (mps device AND flag on) - the
    symmetric gate to the extract side."""
    matcher = _StubMatcher()
    inputs = {"image0": {}, "image1": {}}
    assert backend._run_matcher(_NoCompileTorch(), matcher,
                                _FakeDevice("cpu"), inputs) == "pred"
    assert matcher.calls == [inputs]
    matcher.calls.clear()
    flag = backend.LIGHTGLUE_MPS_COMPILE_ENABLED
    try:
        backend.LIGHTGLUE_MPS_COMPILE_ENABLED = 0
        assert backend._run_matcher(_NoCompileTorch(), matcher,
                                    _FakeDevice("mps"),
                                    inputs) == "pred"
        assert matcher.calls == [inputs]
    finally:
        backend.LIGHTGLUE_MPS_COMPILE_ENABLED = flag


def test_compiled_or_eager_falls_back_and_marks_dead():
    """A first-call compile failure falls back to eager PERMANENTLY for
    the process (the dead marker short-circuits the build) - a broken
    compile must never fail extraction or matching."""
    builds, target_calls = [], []

    class _RaisingTorch:
        def compile(self, target, dynamic=None):
            builds.append(1)
            raise RuntimeError("probe-forced compile failure")

    def target(x):
        target_calls.append(x)
        return f"eager:{x}"

    key = "compiled_pin"
    backend._backend_cache.pop(key, None)
    backend._backend_cache.pop(key + "_dead", None)
    try:
        assert backend._compiled_or_eager(_RaisingTorch(), target, key,
                                          dynamic=False,
                                          args=("in",)) == "eager:in"
        assert builds == [1] and target_calls == ["in"]
        assert backend._backend_cache.get(key + "_dead") is True
        # the dead marker short-circuits: no second build attempt
        assert backend._compiled_or_eager(_RaisingTorch(), target, key,
                                          dynamic=False,
                                          args=("in2",)) == "eager:in2"
        assert builds == [1] and target_calls == ["in", "in2"]
    finally:
        backend._backend_cache.pop(key, None)
        backend._backend_cache.pop(key + "_dead", None)


def test_compiled_or_eager_builds_once_and_caches():
    """The wrapper is built lazily on first use and reused - one build,
    every call through it."""
    builds, calls = [], []

    class _Torch:
        def compile(self, target, dynamic=None):
            builds.append(dynamic)

            def wrapper(*args):
                calls.append(args)
                return f"compiled:{args[0]}"
            return wrapper

    def target(x):
        return f"eager:{x}"

    key = "compiled_pin"
    backend._backend_cache.pop(key, None)
    backend._backend_cache.pop(key + "_dead", None)
    try:
        assert backend._compiled_or_eager(_Torch(), target, key,
                                          dynamic=False,
                                          args=("in",)) == "compiled:in"
        assert backend._compiled_or_eager(_Torch(), target, key,
                                          dynamic=False,
                                          args=("in2",)) == "compiled:in2"
        assert builds == [False] and calls == [("in",), ("in2",)]
    finally:
        backend._backend_cache.pop(key, None)
        backend._backend_cache.pop(key + "_dead", None)


def test_model_rebuild_invalidates_compiled_state(monkeypatch):
    """The compiled wrappers close over the model instances: a rebuild
    pops them (and the dead markers) so the next call recompiles
    against the new models."""
    class _StubModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def requires_grad_(self, flag):
            return self

    lightglue_stub = ModuleType("lightglue")
    lightglue_stub.ALIKED = lambda **conf: _StubModel()
    lightglue_stub.LightGlue = lambda **conf: _StubModel()
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=None,
        device=lambda spec: SimpleNamespace(type="cpu"),
        use_deterministic_algorithms=lambda *a, **k: None,
        set_grad_enabled=lambda flag: None)
    monkeypatch.setitem(sys.modules, "lightglue", lightglue_stub)
    monkeypatch.setattr(backend, "_import_torch", lambda: torch_stub)
    backend._backend_cache["compiled_dense"] = "stale"
    backend._backend_cache["compiled_dense_dead"] = True
    try:
        backend._get_models()
        assert "compiled_dense" not in backend._backend_cache
        assert "compiled_dense_dead" not in backend._backend_cache
    finally:
        backend._backend_cache.pop("models", None)


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


@pytest.mark.skipif(not _learned_stack_ready(),
                    reason="learned stack not provisioned on this "
                           "machine (suite never downloads weights)")
def test_real_aliked_matching_contract():
    """The REAL matcher, where provisioned: matching two shifted
    views of the same texture produces cv.DMatch objects (positional
    construction - the DMatch ctor is positional-only in the OpenCV
    Python bindings), and the consensus lands in frame1's match list."""
    f1 = Frame(_textured(), make_K())
    f2 = Frame(_textured_shifted(dx=6), make_K())
    f1.set_features(*FeatureExtractor("ALIKED").extract(_textured()))
    f2.set_features(*FeatureExtractor("ALIKED").extract(_textured_shifted(dx=6)))
    matches, _ = match_features(f1, f2, "ALIKED")
    assert isinstance(matches, list)
    assert len(matches) > 0
    for m in matches:
        assert isinstance(m, cv.DMatch)
        assert 0 <= m.queryIdx < len(f1.get_keypoints_descriptors()[0])
        assert 0 <= m.trainIdx < len(f2.get_keypoints_descriptors()[0])
        assert 0.0 <= m.distance <= 1.0


def test_keypoint_clip_vectorization_is_bit_identical():
    """The extract-time keypoint construction clips per COLUMN (one
    vectorized clip per axis); the per-element clip it replaced
    produced the same floats."""
    rng = np.random.default_rng(7)
    kps = rng.uniform(-5, 700, (64, 2)).astype(np.float32)
    w, h = 640, 480
    scalar = np.array([[float(np.clip(x, 0, w - 1)),
                        float(np.clip(y, 0, h - 1))] for x, y in kps])
    column = np.stack([np.clip(kps[:, 0], 0, w - 1),
                       np.clip(kps[:, 1], 0, h - 1)], axis=1)
    assert np.array_equal(scalar, column)


def _mps_ready():
    """The compiled-path pins run only where the REAL MPS stack is
    provisioned (weights cached; the suite never downloads) - the CUDA
    desktop keeps its byte-identical eager path and skips these."""
    if not _learned_stack_ready():
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


@pytest.mark.skipif(not _mps_ready(),
                    reason="real MPS stack not provisioned on this "
                           "machine (suite never downloads weights)")
def test_real_compiled_assembly_is_op_for_op_bit_exact(monkeypatch):
    """THE assembly contract: with the compiled wrapper bypassed (the
    eager dense map substituted), the flag-1 assembly path is
    BIT-IDENTICAL to the flag-0 extractor.extract path - every
    flag0/flag1 difference is the compiled map's inductor reassociation
    and nothing else."""
    flag = backend.ALIKED_MPS_COMPILE_ENABLED
    try:
        backend.ALIKED_MPS_COMPILE_ENABLED = 0
        kp0, d0 = FeatureExtractor("ALIKED").extract(_textured())
        backend.ALIKED_MPS_COMPILE_ENABLED = 1
        monkeypatch.setattr(
            backend, "_compiled_or_eager",
            lambda torch, target, cache_key, dynamic, args: target(*args))
        kp1, d1 = FeatureExtractor("ALIKED").extract(_textured())
    finally:
        backend.ALIKED_MPS_COMPILE_ENABLED = flag
    assert len(kp0) == len(kp1)
    assert np.array_equal(np.array([k.pt for k in kp0]),
                          np.array([k.pt for k in kp1]))
    assert np.array_equal(np.array([k.response for k in kp0]),
                          np.array([k.response for k in kp1]))
    assert np.array_equal(d0, d1)


@pytest.mark.skipif(not _mps_ready(),
                    reason="real MPS stack not provisioned on this "
                           "machine (suite never downloads weights)")
def test_real_compiled_double_run_and_parity_bounds(monkeypatch):
    """The compiled path's own contracts, at the shipped constants:
    bit-identical double-runs (extraction and matching), and the
    measured parity vs eager - the inductor reassociation (~1e-6 on
    the dense features) leaves the top-k keypoint set, its order, and
    the LightGlue match set unchanged within the pinned bounds."""
    monkeypatch.setattr(backend, "ALIKED_MPS_COMPILE_ENABLED", 1)
    monkeypatch.setattr(backend, "LIGHTGLUE_MPS_COMPILE_ENABLED", 1)
    img = _textured()
    img2 = _textured_shifted(dx=6)
    kp1, d1 = FeatureExtractor("ALIKED").extract(img)
    kp2, d2 = FeatureExtractor("ALIKED").extract(img)
    P1 = np.array([k.pt for k in kp1])
    assert np.array_equal(P1, np.array([k.pt for k in kp2]))
    assert np.array_equal(d1, d2)

    # eager reference for both frames (the matcher parity needs
    # IDENTICAL inputs so the extractor's micro-drift cannot confound)
    flag = backend.ALIKED_MPS_COMPILE_ENABLED
    try:
        backend.ALIKED_MPS_COMPILE_ENABLED = 0
        kp0, d0 = FeatureExtractor("ALIKED").extract(img)
        kp0b, d0b = FeatureExtractor("ALIKED").extract(img2)
    finally:
        backend.ALIKED_MPS_COMPILE_ENABLED = flag
    P0 = np.array([k.pt for k in kp0])
    assert len(kp0) == len(kp1)
    S0 = P0[np.lexsort((P0[:, 0], P0[:, 1]))]
    S1 = P1[np.lexsort((P1[:, 0], P1[:, 1]))]
    assert np.abs(S0 - S1).max() <= 2e-4
    # nearest-neighbor descriptor alignment (chunked: the full pairwise
    # matrix does not fit MPS at the top-k budget)
    nn = np.empty(len(P0), dtype=np.int64)
    for i in range(0, len(P0), 256):
        block = ((P0[i:i + 256, None, :] - P1[None, :, :]) ** 2).sum(-1)
        nn[i:i + 256] = block.argmin(axis=1)
    drift = np.abs(d0.astype(np.float64)
                   - d1[nn].astype(np.float64)).max(axis=1)
    assert (drift <= 1e-3).mean() >= 0.99
    assert drift.max() <= 0.05

    # matcher: identical inputs, gate on vs off -> the same index pairs
    f1 = Frame(img, make_K())
    f2 = Frame(img2, make_K())
    f1.set_features(kp0, d0)
    f2.set_features(kp0b, d0b)
    backend.LIGHTGLUE_MPS_COMPILE_ENABLED = 0
    m0 = backend.match_aliked(f1, f2)
    backend.LIGHTGLUE_MPS_COMPILE_ENABLED = 1
    m1 = backend.match_aliked(f1, f2)
    m1b = backend.match_aliked(f1, f2)
    pairs = lambda m: sorted((x.queryIdx, x.trainIdx) for x in m)  # noqa: E731
    assert pairs(m1) == pairs(m1b)
    common = len(set(pairs(m0)) & set(pairs(m1)))
    assert common / max(len(m0), 1) >= 0.99

    # end-to-end (the production path): flag-1 features -> flag-1
    # matcher, bit-identical double-run
    f1c = Frame(img, make_K())
    f2c = Frame(img2, make_K())
    f1c.set_features(kp1, d1)
    f2c.set_features(*FeatureExtractor("ALIKED").extract(img2))
    mc = backend.match_aliked(f1c, f2c)
    assert pairs(mc) == pairs(backend.match_aliked(f1c, f2c))
    assert len(mc) > 0
