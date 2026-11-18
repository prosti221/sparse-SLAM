# Sparse SLAM

A monocular visual SLAM system: a feature-based pipeline that tracks a
single camera and builds a sparse 3D map online, with a desktop editor
for inspecting and steering the run.

## Overview

The pipeline covers the full monocular stack: ORB (multiscale) or the
optional ALIKED + LightGlue frontend, feature matching and pose
estimation, keyframe-based sparse mapping, bundle adjustment on a
bundled C++ (g2o) core exposed as the `slam_native` extension, loop
closure with drift gating, relocalization, and map culling.

The editor (PySide6 hosting an offscreen Open3D viewport) is the
application:

- live 3D map, keyframe frustums, and a capture inset
- a transport pill with play/pause, reset, checkpoint scrubbing, and
  rewind — a rewind forks a branch, and the SLAM step runs in a spawned
  worker process, so the UI stays responsive
- Source / Options / Advanced tabs: source selection, live pipeline
  options, and live constant pins; per-source intrinsics are cached and
  can be applied live through a small form
- session recording (a side-by-side capture + render video and the map
  point cloud)

Benchmarking accompanies the pipeline: a frozen synthetic + TUM
battery, a video health census, a ground-truth ladder with
per-mechanism attribution, and a remote runner for the gate machine.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.11 is pinned in
`.python-version`). On macOS, Open3D needs libusb at runtime:

```bash
brew install libusb
```

Install the environment — this also builds the native extension:

```bash
uv sync --extra ui
```

Optional extras: `--extra learned` (ALIKED + LightGlue; weights
download on first use) and `--extra calib` (GeoCalib intrinsics
priors). The benchmark tiers exercise the optional stacks, so the gate
machine provisions with `--all-extras`. After changing C++ under
`native/`, rebuild the extension:

```bash
uv sync --reinstall-package sparse-slam
```

## Run the editor

```bash
uv run --extra ui python slam.py
```

In the Source tab choose a video file, a frame-sequence directory, or a
camera, then press **load session**. The transport pill at the bottom
plays/pauses, resets, scrubs checkpoints view-only, and rewinds. In the
viewport, W/A/S/D fly with Q/E down/up (Shift boosts, -/= changes
speed), the wheel dollies, and dragging orbits. The Options tab applies
live pipeline flips; the Advanced tab pins constants live (persisted in
`data/ui/ui_state.json`).

## Benchmarks

The frozen battery is the adoption gate (synthetic + `tum_xyz`):

```bash
uv run python -m benchmarks.run
```

A ground-truth ladder with per-mechanism attribution (the sequences
download on demand into `data/`):

```bash
uv run python -m benchmarks.run --sequences tum_fr2_xyz tum_fr1_desk \
    tum_fr3_long_office_household euroc_v1_01_easy --no-baseline
uv run python -m benchmarks.attribution --all
```

The 14-video health census (`videos/` holds the corpus):

```bash
uv run python -m benchmarks.video_census
```

Dataset inspection and preparation:

```bash
uv run python -m benchmarks.datasets info
uv run python -m benchmarks.datasets prepare --all
```

Offload a run to the remote desktop (sync, run, fetch the artifacts):

```bash
uv run python -m benchmarks.remote sync
uv run python -m benchmarks.remote run -- python -m benchmarks.run
uv run python -m benchmarks.remote fetch
```

Run the test suite:

```bash
uv run python -m pytest benchmarks -q
```
