"""6e contracts: measured sizes, source sampling, expected NON-invariances."""
import io
import json
from types import SimpleNamespace
import tarfile

import cv2 as cv
import numpy as np
import pytest

from benchmarks.harness import load_calibration, load_trajectory, run_slam
from benchmarks.keyframe_audit import KeyframeAudit, capture_signals
from benchmarks.keyframe_cost import KeyframeCostAudit
from benchmarks.keyframe_sensitivity import (
    HELDOUT_DATASET, associate_rgb_gt, extract_heldout, matched_sample_comparison,
    prepare_view, resize_intrinsics,
)
from benchmarks.stages.helpers import make_frame, project_gt, set_keypoints, yaw_pose
from benchmarks.stages.test_keyframe_audit import _geometry_pair
from benchmarks.stages.test_keyframe_policy import _policy_frame
from benchmarks.stages.test_keyframe_policy import _criterion_map
from benchmarks.synthetic.generator import ensure_sequence
from slam.ba.bundle_adjustment_g2o import G2OBundleAdjustment
from slam.core.map import Map
from slam.core.tracker import Tracker
from slam.utils import constants as const
from slam.utils.utils import is_valid_triangulated_point, should_insert_keyframe


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "frames").mkdir()
    trajectory = []
    for i in range(5):
        image = np.random.default_rng(i).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        cv.imwrite(str(root / "frames" / f"frame_{i:06d}.png"), image)
        trajectory.append([i, i * .1, .1, .2, 0, 0, 0, 1])
    np.savetxt(root / "trajectory.txt", trajectory, fmt="%.17g")
    (root / "calibration.txt").write_text("45 46 32 24 -0.01\n")
    (root / "source_manifest.json").write_text(json.dumps({"timestamps_seconds": [10 + i * .03 for i in range(5)]}))
    return root


@pytest.mark.parametrize("mode,count", [("native", 5), ("half_resolution", 5), ("stride2", 3)])
def test_views_preserve_source_mapping_gt_and_timestamps(source, tmp_path, mode, count):
    before = {p.name: p.read_bytes() for p in (source / "frames").glob("*.png")}
    destination = tmp_path / mode
    record = prepare_view(source, destination, mode)
    indices = record["source_indices"]
    assert len(indices) == count
    np.testing.assert_array_equal(load_trajectory(destination), load_trajectory(source)[indices])
    assert record["timestamps_seconds"] == [10 + i * .03 for i in indices]
    assert {p.name: p.read_bytes() for p in (source / "frames").glob("*.png")} == before
    K, k1 = load_calibration(destination)
    assert k1 == -.01
    if mode == "half_resolution":
        np.testing.assert_array_equal(K, resize_intrinsics(load_calibration(source)[0], .5, .5))
        assert cv.imread(str(destination / "frames" / "frame_000000.png")).shape == (24, 32, 3)
    else:
        for new, original in enumerate(indices):
            assert (destination / "frames" / f"frame_{new:06d}.png").read_bytes() == (source / "frames" / f"frame_{original:06d}.png").read_bytes()


def test_native_calibration_and_gt_bytes_are_exact(source, tmp_path):
    destination = tmp_path / "native"
    prepare_view(source, destination, "native")
    for filename in ("calibration.txt", "trajectory.txt"):
        assert (destination / filename).read_bytes() == (source / filename).read_bytes()
    with pytest.raises(FileExistsError):
        prepare_view(source, destination, "native")


def test_synthetic_frame_indices_are_not_called_acquisition_seconds(source, tmp_path):
    (source / "source_manifest.json").unlink()
    record = prepare_view(source, tmp_path / "native", "native")
    assert record["timestamps_seconds"] is None and record["timestamp_units"] == "unavailable"


def test_resize_intrinsics_matches_opencv_pixel_center_geometry():
    frame = make_frame()
    uv, _ = project_gt(frame.pose, np.array([.2, -.3, 5.]), frame.K)
    resized = make_frame(K=resize_intrinsics(frame.K, .5, .75))
    actual, _ = project_gt(resized.pose, np.array([.2, -.3, 5.]), resized.K)
    np.testing.assert_allclose(actual, (uv + .5) * [.5, .75] - .5, atol=1e-12)


@pytest.mark.parametrize("scale", [.1, 10.])
def test_dimensionless_diagnostics_remain_scale_invariant_but_not_raw_selector(scale):
    world, current, reference, _ = _geometry_pair()
    base = capture_signals(current, reference, world)
    world, current, reference, _ = _geometry_pair(scale)
    changed = capture_signals(current, reference, world)
    assert changed["last_reference"]["baseline_over_depth"] == pytest.approx(base["last_reference"]["baseline_over_depth"])
    assert changed["last_reference"]["parallax_deg"]["median"] == pytest.approx(base["last_reference"]["parallax_deg"]["median"], abs=1e-6)
    reference = _policy_frame(yaw_pose(0.), .5)
    current = _policy_frame(yaw_pose(0., (scale, 0, 0)), .5)
    assert should_insert_keyframe(current, reference, 3)[0] is (scale >= 2.)


def test_resolution_changes_fixed_pixel_acceptance_without_relaxing_the_gate():
    truth = np.array([.2, .1, 5.])
    results = []
    for scale in (1., .5):
        K = resize_intrinsics(make_frame().K, scale, scale)
        previous = make_frame(yaw_pose(0.), K)
        current = make_frame(yaw_pose(0., (1., 0, 0)), K)
        set_keypoints(previous, [project_gt(previous.pose, truth, K)[0]])
        # Equivalent observed image perturbation shrinks with the image.
        set_keypoints(current, [project_gt(current.pose, truth, K)[0] + [1.5 * scale, 0.]])
        current.matches = [cv.DMatch(0, 0, 0.)]
        results.append(is_valid_triangulated_point(0, current, previous, np.append(truth, 1.)))
    code = const.TRIANGULATION_VALIDATION_CODE["MAX_REPROJECTION_ERROR_VIOLATION"]
    assert code in results[0]["failed_codes"] and results[1]["failed_codes"] == []
    assert const.MAX_TRI_REPROJECTION_PX == 1.


def test_frame_gap_is_not_a_physical_time_backstop():
    reference = _policy_frame(yaw_pose(0.), .5)
    decisions = []
    for stride in (1, 2):
        for input_gap in range(1, 5):
            current = _policy_frame(yaw_pose(np.radians(input_gap * stride)), .5)
            if should_insert_keyframe(current, reference, input_gap)[0]:
                decisions.append(input_gap * stride)
                break
    assert decisions == [3, 6]  # same gap cap, different source-frame motion span


def test_rpe_endpoints_and_ate_populations_are_identical_after_subsampling():
    poses = np.array([yaw_pose(.01 * i, (.1 * i, 0, .01 * i)) for i in range(41)])
    native = {"poses": poses.copy(), "keyframe_poses": poses[::5], "keyframe_gt_indices": np.arange(0, 41, 5)}
    indices = np.arange(0, 41, 2)
    candidate = SimpleNamespace(poses=poses[indices].copy(), keyframe_poses=poses[::10],
                                keyframe_gt_indices=np.arange(0, len(indices), 5))
    comparison = matched_sample_comparison(candidate, native, indices, poses, indices * .03)
    assert comparison["rpe_pair_source_indices"] == [[0, 10], [10, 20], [20, 30], [30, 40]]
    np.testing.assert_allclose(comparison["rpe_pair_acquisition_seconds"], .3)
    assert comparison["control_ate"] == comparison["candidate_ate"]
    assert comparison["rpe"]["control"] == comparison["rpe"]["candidate"]
    assert comparison["common_final_kf_ate"]["count"] == 5


def test_no_rpe_pairs_is_explicit_not_nan():
    poses = np.array([yaw_pose(0., (.1 * i, 0, 0)) for i in range(4)])
    native = {"poses": poses, "keyframe_poses": poses, "keyframe_gt_indices": np.arange(4)}
    candidate = SimpleNamespace(poses=poses, keyframe_poses=poses, keyframe_gt_indices=list(range(4)))
    result = matched_sample_comparison(candidate, native, np.arange(4), poses)
    assert result["rpe"]["control"] == {"pairs": 0, "translation_rmse": None, "rotation_rmse_deg": None}


def test_heldout_association_preserves_later_ties_rejects_unmatched_and_keeps_rgb_time(tmp_path):
    (tmp_path / "rgb").mkdir()
    for name in ("a", "b", "c"):
        (tmp_path / "rgb" / (name + ".png")).write_bytes(b"image")
    (tmp_path / "rgb.txt").write_text("1.01 rgb/a.png\n1.04 rgb/b.png\n2.0 rgb/c.png\n")
    (tmp_path / "groundtruth.txt").write_text("1.0 0 0 0 0 0 0 1\n1.02 1 0 0 0 0 0 1\n1.04 2 0 0 0 0 0 1\n")
    pairs = associate_rgb_gt(tmp_path)
    assert len(pairs) == 2
    assert pairs[0][0:2] == (1.01, 1.02) and pairs[0][3][0] == "1"
    assert pairs[1][0:2] == (1.04, 1.04)


@pytest.mark.parametrize("bad", ["../escape", "/escape", "different_root/rgb/a.png"])
def test_heldout_tar_rejects_path_escape_and_wrong_dataset(tmp_path, bad):
    archive = tmp_path / "bad.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(bad)
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        extract_heldout(archive, tmp_path / "out")


def test_heldout_tar_extracts_only_rgb_and_metadata_not_depth_or_links(tmp_path):
    archive = tmp_path / "source.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("rgb/a.png", "depth/a.png", "groundtruth.txt"):
            info = tarfile.TarInfo(HELDOUT_DATASET + "/" + name)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    root = tmp_path / "out"
    extract_heldout(archive, root)
    assert (root / HELDOUT_DATASET / "rgb/a.png").read_bytes() == b"x"
    assert not (root / HELDOUT_DATASET / "depth/a.png").exists()
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(HELDOUT_DATASET + "/rgb/link.png")
        info.type = tarfile.SYMTYPE
        info.linkname = "/escape"
        tar.addfile(info)
    with pytest.raises(ValueError, match="links"):
        extract_heldout(archive, root)


def test_cost_sampler_reads_sizes_without_rng_quality_or_registry_writes():
    world, current, _, points = _geometry_pair()
    audit = KeyframeCostAudit(np.arange(10))
    before = [set(f.__dict__) for f in world.keyframes + [current]]
    observations = [dict(p.observations) for p in points]
    cv.setRNGSeed(1741)
    expected = np.empty((8, 2))
    cv.randu(expected, 0., 1.)
    cv.setRNGSeed(1741)
    row = audit._map(world, "fixture")
    actual = np.empty_like(expected)
    cv.randu(actual, 0., 1.)
    np.testing.assert_array_equal(actual, expected)
    assert row["points"] == 6 and row["point_side_observations"] == 18
    assert [set(f.__dict__) for f in world.keyframes + [current]] == before
    assert [dict(p.observations) for p in points] == observations


def test_graph_peaks_are_not_lost_when_the_local_cache_is_invalidated(monkeypatch):
    from slam.ba import bundle_adjustment_g2o as _ba_mod
    monkeypatch.setattr(_ba_mod, "BA_LOCAL_NATIVE", 0)  # persistent-machinery pin
    world, _, _ = _criterion_map(5, [list(range(5)) for _ in range(32)], make_frame().K)
    assert world.g2o_optimizer.local_bundle_adjustment(world.cur_keyframe.frame_id)
    audit = KeyframeCostAudit(np.arange(10))
    audit._graph(world.g2o_optimizer._local_opt, world.g2o_optimizer._kf_vertex, "local_ba")
    edges = audit.graphs[-1]["edges"]
    world.g2o_optimizer.reset_persistent_graph()
    assert audit.summary({"stage_calls": {}, "events": {}})["graph_peaks"]["local_ba"]["peak_edges"] == edges > 0


def test_cost_context_restores_bindings_on_exception_and_rejects_overlap():
    bindings = [(Tracker, "update"), (Map, "add_points"), (G2OBundleAdjustment, "_create_keyframe_vertices")]
    originals = [getattr(owner, name) for owner, name in bindings]
    with pytest.raises(ValueError, match="injected"):
        with KeyframeCostAudit(np.arange(10)):
            with pytest.raises(RuntimeError, match="overlap"):
                with KeyframeCostAudit(np.arange(10)):
                    pass
            raise ValueError("injected")
    assert [getattr(owner, name) for owner, name in bindings] == originals


def test_fresh_and_pose_graph_handles_are_measured_then_released():
    world, frames, _ = _criterion_map(5, [list(range(5)) for _ in range(32)], make_frame().K)
    reference = frames[-1]
    audit = KeyframeCostAudit(np.arange(10))
    with audit:
        assert world.g2o_optimizer.global_bundle_adjustment()
        # Zero-drift constraint, same convention as production odometry.
        older = world.keyframes[0]
        relative = older.pose @ np.linalg.inv(reference.pose)
        assert world.g2o_optimizer.pose_graph_optimization([(older.frame_id, reference.frame_id, relative)])
    assert {row["kind"] for row in audit.graphs} == {"fresh_ba", "pose_graph"}
    assert all(row["edges"] > 0 for row in audit.graphs)
    assert not audit._fresh
    assert all(not any(key in row for key in ("opt", "poses")) for row in audit.graphs)


def test_window_uses_original_source_indices_and_real_times_not_input_fps():
    audit = KeyframeCostAudit([0, 2, 4], [10., 10.06, 10.12])
    frames = [make_frame(yaw_pose(0., (i, 0, 0))) for i in range(3)]
    audit._indices = {f.frame_id: i for i, f in enumerate(frames)}
    row = audit.window(frames)
    assert row["source_frame_span"] == 4
    assert row["acquisition_seconds_span"] == pytest.approx(.12)
    assert row["estimated_path_map_units"] == 2.


def test_actual_native_cost_observer_preserves_arrays_events_calls_frames_and_cohorts(tmp_path):
    sequence = ensure_sequence("translation")
    short = tmp_path / "short"
    short.mkdir()
    (short / "calibration.txt").symlink_to(sequence / "calibration.txt")
    (short / "frames").mkdir()
    for path in sorted((sequence / "frames").glob("*.png"))[:24]:
        (short / "frames" / path.name).symlink_to(path)
    with KeyframeAudit() as original_audit:
        original = run_slam(short)
    with KeyframeCostAudit(np.arange(24)) as costs:
        with KeyframeAudit() as audit:
            actual = run_slam(short)
    for name in ("poses", "map_points", "keyframe_poses"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(original, name))
    assert actual.telemetry["events"] == original.telemetry["events"]
    assert actual.telemetry["stage_calls"] == original.telemetry["stage_calls"]
    assert original_audit.frames == audit.frames and original_audit.insertions == audit.insertions
    assert len(costs.frames) == 24 and costs.graphs
    assert any(g["pose_vertices"] > const.LOCAL_MAP_WINDOW_SIZE for g in costs.graphs)
