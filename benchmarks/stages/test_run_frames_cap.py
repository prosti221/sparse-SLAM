"""The frames-cap pins: run.py's prefix-leg screening path.

`--frames N` (and run_slam's frames_cap) truncate the frame list to
the first N files; the default None runs the sequence unchanged with
an untouched path. The prefix run consumes the pipeline's one
randomness source and the map dynamics in the identical frame-ordered
stream, so a prefix run is the full run's prefix over its shared
frames - pinned here at micro scale, verified remotely per the
protocol's prefix control.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.stages.test_harness_rotmodel import _micro_dataset
from benchmarks import harness as harness_mod
from benchmarks import run as run_mod


def test_run_slam_frames_cap_truncates_and_is_a_prefix(tmp_path):
    d = _micro_dataset(tmp_path, n=6)
    full = harness_mod.run_slam(d, seed=7)
    assert len(full.poses) == 6
    capped = harness_mod.run_slam(d, seed=7, frames_cap=3)
    assert len(capped.poses) == 3
    # The prefix property: the capped run's poses ARE the full run's
    # first three (identical frame-ordered randomness consumption).
    assert all(np.array_equal(a, b) for a, b in
               zip(capped.poses, full.poses[:3]))
    # None (the default) runs the sequence unchanged.
    default = harness_mod.run_slam(d, seed=7, frames_cap=None)
    assert len(default.poses) == 6


def test_run_slam_frames_cap_larger_than_files_runs_all(tmp_path):
    d = _micro_dataset(tmp_path, n=4)
    capped = harness_mod.run_slam(d, seed=7, frames_cap=99)
    assert len(capped.poses) == 4


def test_evaluate_sequence_aligns_the_capped_gt(monkeypatch, tmp_path):
    """The GT slice must ride the same switch: evaluate_sequence caps
    gt_poses with the frames the pipeline actually ran."""
    (tmp_path / "trajectory.txt").write_text("\n".join(
        f"{0.1 * i:.2f} 0.0 0.0 0.0 0.0 0.0 0.0 1.0"
        for i in range(6)) + "\n")
    captured = {}

    def fake_run_slam(seq_dir, **kwargs):
        captured["frames_cap"] = kwargs.get("frames_cap")
        return SimpleNamespace(poses=[], final_intrinsics=None)

    def fake_evaluate(result, gt_poses, scene_points, planes):
        captured["gt_len"] = len(gt_poses)
        return {}

    monkeypatch.setattr(run_mod, "generate_sequence",
                        lambda name: tmp_path)
    monkeypatch.setattr(run_mod, "run_slam", fake_run_slam)
    monkeypatch.setattr(run_mod.metrics, "evaluate", fake_evaluate)
    run_mod.evaluate_sequence("translation", 42, frames_cap=3)
    assert captured["frames_cap"] == 3 and captured["gt_len"] == 3
    run_mod.evaluate_sequence("translation", 42, frames_cap=None)
    assert captured["frames_cap"] is None and captured["gt_len"] == 6


class _Bail(Exception):
    pass


def test_cli_frames_flag_reaches_evaluate(tmp_path, monkeypatch):
    """End to end at the CLI level: --frames N in sys.argv lands in
    evaluate_sequence's frames_cap; the default stays None."""
    import sys as _sys
    monkeypatch.setattr(run_mod, "generate_sequence",
                        lambda name: tmp_path)
    captured = {}

    def fake_eval(name, seed, feature_extractor="ORB", rotmodel=False,
                  frames_cap=None):
        captured["frames_cap"] = frames_cap
        raise _Bail()

    monkeypatch.setattr(run_mod, "evaluate_sequence", fake_eval)

    monkeypatch.setattr(
        _sys, "argv",
        ["run.py", "--frames", "2", "--no-baseline", "--sequences",
         "translation"])
    with pytest.raises(_Bail):
        run_mod.main()
    assert captured["frames_cap"] == 2

    captured.clear()
    monkeypatch.setattr(_sys, "argv",
                        ["run.py", "--no-baseline", "--sequences",
                         "translation"])
    with pytest.raises(_Bail):
        run_mod.main()
    assert captured["frames_cap"] is None


import inspect


def test_main_passes_frames_into_evaluate():
    """The CLI flag's last link: main's loop passes frames_cap into
    evaluate_sequence (the call site carries the kwarg)."""
    src = inspect.getsource(run_mod.main)
    assert "frames_cap=args.frames" in src
