"""The video health census (phase 14): run the production video path
over the WHOLE videos/ corpus — first N decoded frames each — and
emit the per-video health signature, so the corpus stops being the
unmeasured tier of the benchmark suite.

Production fidelity (mirrors slam.py's video loop exactly):
- intake gate per the CONFIG POLICY resolved for each video
  (per-video `enable_adaptive_frame_intake`, fallback global,
  default on — NOTE: production slam.py currently reads the
  property GLOBAL-ONLY, so the user's per-video `false` under
  `walking_woods_2` is inert on the production path; the census
  honors the per-video intent and records the discrepancy);
- phase 13: `tracker.rotation_model_enabled = intake gate present`;
- ORB / no multiscale / BA on from the config's GLOBAL section;
- `IntrinsicsEstimate.from_K`, container-fps time base, seed 42.

Calibration resolution per video (recorded in every output):
1. config entry matched by PATH (config names are not filenames:
   `kyoto` → kyoto_original.mp4, `kyoto_rotation` → kyoto_2_5.mp4,
   `desk` → vid3.mp4, `walking_woods_2` → walking_woods_2_15.mp4);
2. same-content aliases for the decimation family (kyoto.mp4 shares
   kyoto_original's camera; kyoto_2.mp4 shares kyoto_2_5's content;
   walking_woods_2.mp4 shares the 2_15 config entry's);
3. carried: walking_salem uses the 1449.6 the prior probes measured;
4. nominal: fx = 0.6 x width (median of the config-measured fx/W
   ratios 0.49-0.82), cx W/2, cy H/2, k1 0 — the finding-37 caveat
   applies to absolute angles; health signatures are first-order
   K-insensitive.

Evidence discipline: per-video outputs are skip-if-exists (resume
after an interruption; --overwrite re-runs explicitly, never
silently). The combined summary table is DERIVED from the guarded
per-video files and may be regenerated freely.

Usage:
    uv run python -m benchmarks.video_census                 # all videos
    uv run python -m benchmarks.video_census --videos drone.mp4 kyoto_2_5.mp4
    uv run python -m benchmarks.video_census --frames 300 --overwrite
"""
import argparse
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np

import slam.utils.logger as slam_logger
import slam.core.intake as intake_mod
from benchmarks.health import heading_deg, health_table, summarize_run
from slam.calibration.refit import IntrinsicsEstimate
from slam.config.config_parser import ConfigParser
from slam.core.frame import Frame
from slam.core.intake import FrameIntakeGate
from slam.core.map import Map
from slam.core.tracker import Tracker
from slam.utils import telemetry

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = REPO_ROOT / "videos"
OUT_DIR = REPO_ROOT / "output" / "benchmarks" / "video_census"
CONFIG_PATH = REPO_ROOT / "slam" / "config" / "config.yaml"

# Same-content aliases: decimation family members share their
# sibling's config entry camera (rationale recorded per video).
CONTENT_ALIASES = {
    "kyoto.mp4": ("kyoto", "2fps decimation of kyoto_original.mp4"),
    "kyoto_2.mp4": ("kyoto_rotation",
                    "same 783 s content as kyoto_2_5.mp4 (20fps sibling)"),
    "walking_woods_2.mp4": (
        "walking_woods_2",
        "same 485 s content as walking_woods_2_15.mp4 (2fps sibling)"),
}
# Carried intrinsics measured in a prior phase (probe continuity).
CARRIED = {
    "walking_salem.mp4": (1449.6, 0.0,
                          "prior-phase salem estimate (probe continuity)"),
}
NOMINAL_FX_WIDTH_RATIO = 0.6   # median of config-measured fx/W (0.49-0.82)


def resolve_intrinsics(config, filename, W, H):
    """(K, k1, source, intake_override) for one video file."""
    videos = config.config["videos"]
    path_to_entry = {Path(v["path"]).name: name
                     for name, v in videos.items()
                     if isinstance(v, dict) and "path" in v}

    entry_name = path_to_entry.get(filename)
    alias_rationale = None
    if entry_name is None and filename in CONTENT_ALIASES:
        entry_name, alias_rationale = CONTENT_ALIASES[filename]
    if entry_name is not None:
        entry = videos[entry_name]
        fx = float(entry.get("fx", 0))
        if fx:
            fy = float(entry.get("fy", fx))
            cx = float(entry.get("cx", W / 2))
            cy = float(entry.get("cy", H / 2))
            k1 = float(entry.get("k1", 0.0))
            src = f"config:{entry_name}"
            if alias_rationale:
                src += f" (alias: {alias_rationale})"
            intake = entry.get("enable_adaptive_frame_intake", None)
            return (np.array([[fx, 0.0, cx], [0.0, fy, cy],
                              [0.0, 0.0, 1.0]]), k1, src, intake)
    if filename in CARRIED:
        fx, k1, rationale = CARRIED[filename]
        cx, cy = W / 2, H / 2
        return (np.array([[fx, 0.0, cx], [0.0, fx, cy],
                          [0.0, 0.0, 1.0]]), k1,
                f"carried ({rationale})", None)
    # nominal
    fx = NOMINAL_FX_WIDTH_RATIO * W
    return (np.array([[fx, 0.0, W / 2], [0.0, fx, H / 2],
                      [0.0, 0.0, 1.0]]), 0.0,
            f"nominal (fx={NOMINAL_FX_WIDTH_RATIO}x{int(W)})", None)


def intake_policy(config, override):
    """The effective intake-gate decision for one video, honoring the
    per-video entry first (the user's documented intent), then the
    global property, then the production default (on)."""
    if override is not None:
        return bool(override)
    return bool(config.get_global_config_property(
        "enable_adaptive_frame_intake", True))


def run_one(config, video_path, frames_cap, seed,
            feature_extractor_override=None):
    """One video through the production path; returns the result dict
    (the caller owns writing it)."""
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    K, k1, k_source, intake_override = resolve_intrinsics(
        config, video_path.name, W, H)
    gate_on = intake_policy(config, intake_override)

    feature_extractor = (feature_extractor_override
                         or config.get_global_config_property(
                             "feature_extractor", "ORB"))
    multiscale = bool(config.get_global_config_property(
        "enable_multiscale_features", False))
    enable_ba = bool(config.get_global_config_property("enable_ba", True))

    world_map = Map()
    world_map.intrinsics = IntrinsicsEstimate.from_K(
        K, k1=k1, source="census")
    world_map.video_fps = fps
    tracker = Tracker(world_map, feature_extractor, multiscale, enable_ba)
    gate = (FrameIntakeGate(feature_extractor, multiscale, fps)
            if gate_on else None)
    # Phase 13 production semantics: the rotation-model gate rides
    # the intake gate's cadence (slam.py's exact wiring).
    tracker.rotation_model_enabled = gate is not None

    recorder = telemetry.TelemetryRecorder()
    telemetry.set_recorder(recorder)
    cv.setRNGSeed(seed)
    slam_logger.DEBUG = False

    frames = []
    gate_rows = []
    resets = []
    idx = -1
    t0 = time.perf_counter()
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                break
            idx += 1
            if idx >= frames_cap:
                break

            fed = gate.should_feed(img) if gate is not None else True
            gate_rows.append({"i": idx, "fed": bool(fed)})
            if not fed:
                continue

            before_ev = dict(recorder.events)
            frame = Frame(img, world_map.intrinsics.K,
                          world_map.intrinsics.k1)
            if gate is not None and intake_mod.INTAKE_CARRY_EXTRACTION:
                # The live path's [26c] dedup, mirrored here: the
                # census measures the production video loop's wall.
                frame.set_features(*gate.last_extraction)
            prev_pose = (tracker.cur_frame.pose.copy()
                         if tracker.cur_frame is not None else None)
            prev_step = tracker.step
            is_kf = tracker.update(frame)

            pose = tracker.cur_frame.pose
            ev = {k: v - before_ev.get(k, 0)
                  for k, v in recorder.events.items()
                  if v != before_ev.get(k, 0)}
            if tracker.step < prev_step:
                resets.append(idx)
            R = pose[:3, :3]
            d_rot = (float(np.degrees(np.arccos(np.clip(
                (np.trace(prev_pose[:3, :3].T @ R) - 1.) / 2., -1., 1.))))
                if prev_pose is not None else 0.)
            frames.append({
                "i": idx,
                "kf": bool(is_kf),
                "feat": len(frame.keypoints)
                        if frame.keypoints is not None else 0,
                "obs": len(frame.observed_points),
                "q": float(world_map.avg_tracking_quality),
                "low_q": world_map.consecutive_low_quality_frames,
                "kfs": len(world_map.keyframes),
                "pts": len(world_map.points),
                "archives": len(world_map.archives),
                "head": heading_deg(pose),
                "t": float(np.linalg.norm(pose[:3, 3])),
                "d_rot": d_rot,
                "d_t": float(np.linalg.norm(
                    pose[:3, 3] - prev_pose[:3, 3]))
                    if prev_pose is not None else 0.,
                "rm": bool(tracker._rot_model_engaged),
                "ev": ev,
            })
    finally:
        elapsed = time.perf_counter() - t0
        telemetry.set_recorder(None)
        cap.release()

    summary = summarize_run(frames, resets, dict(recorder.events),
                            idx + 1, len(frames), elapsed)
    return {
        "scope": "phase-14 video health census: production video path "
                 "(intake gate per config policy, phase-13 rotation "
                 "model wired as slam.py), first-N-decoded-frames cap",
        "video": video_path.name,
        "fps": fps, "W": W, "H": H,
        "frames_cap": frames_cap, "seed": seed,
        "K_used": K.tolist(), "k1_used": k1, "k_source": k_source,
        "intake_gate": gate_on,
        "intake_threshold_px": intake_mod.FRAME_INTAKE_MIN_DISPLACEMENT_PX,
        "rotation_model_enabled": bool(tracker.rotation_model_enabled),
        "production_note": "slam.py reads enable_adaptive_frame_intake "
                           "GLOBAL-ONLY; the census honors the per-video "
                           "entry first (walking_woods_2_15 feeds "
                           "stride-1 per the config intent)",
        "decoded": idx + 1,
        "resets": resets,
        "keyframe_indices": [f["i"] for f in frames if f["kf"]],
        "telemetry_totals": dict(recorder.events),
        "stage_calls": dict(recorder.stage_calls),
        "summary": summary,
        "frames": frames,
        "gate_rows": gate_rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=600,
                    help="decode cap per video (default 600)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--videos", nargs="*", default=None,
                    metavar="FILE.mp4",
                    help="subset of videos/ filenames (default: all)")
    ap.add_argument("--feature-extractor", default=None,
                    help="override the config's feature_extractor for "
                         "these legs (phase 24 A/B: 'ALIKED' runs the "
                         "learned stack via the 'learned' extra)")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo legs whose outputs exist")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output filenames "
                         "(<stem>_<tag>.json) - keeps verification/A-B "
                         "legs from colliding with existing evidence "
                         "(the skip-if-exists guard stays intact)")
    ap.add_argument("--override", action="append", default=[],
                    help="pin a sweepable flag for this run "
                         "(repeatable NAME=VALUE; same resolution "
                         "as benchmarks.run's --override, so A/B legs "
                         "can hold machinery flags)")
    args = ap.parse_args()

    from benchmarks.run import apply_overrides
    apply_overrides(args.override)

    config = ConfigParser(str(CONFIG_PATH))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(p.name for p in VIDEO_DIR.glob("*.mp4"))
    if args.videos:
        missing = [v for v in args.videos if v not in videos]
        if missing:
            raise SystemExit(f"[census] unknown videos: {missing}")
        videos = args.videos

    for name in videos:
        out = OUT_DIR / (Path(name).stem + (f"_{args.tag}" if args.tag
                                            else "") + ".json")
        if out.exists() and not args.overwrite:
            print(f"[census] {name}: output exists, skipping "
                  f"(resume; --overwrite to redo)", flush=True)
            continue
        print(f"[census] running {name} (cap {args.frames}) ...",
              flush=True)
        result = run_one(config, VIDEO_DIR / name, args.frames, args.seed,
                         args.feature_extractor)
        if args.overwrite and out.exists():
            out.unlink()
        out.write_text(json.dumps(result) + "\n")
        s = result["summary"]
        print(f"[census] {name}: fed {s['fed']}/{s['decoded']} "
              f"({100 * s['fed_rate']:.0f}%) q_p50 "
              f"{s['quality']['p50']:.2f} pts_med "
              f"{s['map']['pts_median']:.0f} resets {len(s['resets'])} "
              f"| labels: {','.join(s['labels']) or 'HEALTHY'} "
              f"({s['motion']['label']})", flush=True)

    # The DERIVED combined table (guarded per-video files are the
    # evidence; this is a re-projection of them). The summary file
    # itself is excluded (different shape; regenerated every run).
    # Old-schema artifacts (pre-rotation_model summaries) are skipped:
    # the derived table is a projection of the current schema.
    summaries = {}
    for p in sorted(OUT_DIR.glob("*.json")):
        if p.name == "video_census_summary.json":
            continue
        d = json.loads(p.read_text())
        s = d.get("summary")
        if not isinstance(s, dict) or "rotation_model" not in s:
            continue
        summaries[d["video"]] = s
    print(health_table(summaries))
    (OUT_DIR / "video_census_summary.json").write_text(
        json.dumps({"videos": summaries}, indent=1) + "\n")
    print(f"\n[census] per-video evidence + derived summary in {OUT_DIR}")


if __name__ == "__main__":
    main()
