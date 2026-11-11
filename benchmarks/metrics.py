"""SLAM evaluation metrics.

Implements the standard monocular-SLAM evaluation stack with no dependencies
beyond numpy/scipy: Umeyama Sim(3) alignment, ATE, RPE, scale consistency,
and map-vs-scene geometry checks.

Conventions: poses are Twc (camera-to-world). Monocular SLAM has arbitrary
scale, so ATE aligns with scale estimation and reports the recovered scale.
"""
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def umeyama_alignment(est_xyz: np.ndarray, gt_xyz: np.ndarray,
                      with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """Umeyama: find s, R, t minimizing ||s*R@x_i + t - y_i||^2.

    est_xyz, gt_xyz: (N, 3). Returns (s, R, t) with gt ~ s*R@est + t.
    """
    assert est_xyz.shape == gt_xyz.shape and est_xyz.shape[1] == 3
    n = est_xyz.shape[0]
    mu_e = est_xyz.mean(axis=0)
    mu_g = gt_xyz.mean(axis=0)
    e = (est_xyz - mu_e).T          # (3, N)
    g = (gt_xyz - mu_g).T

    cov = g @ e.T / n              # (3, 3)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    var_e = e.var(axis=1).sum()
    s = float(np.trace(np.diag(D) @ S) / var_e) if with_scale and var_e > 0 else 1.0
    t = mu_g - s * R @ mu_e
    return s, R, t


def ate(est_poses: np.ndarray, gt_poses: np.ndarray,
        with_scale: bool = True) -> Dict[str, object]:
    """Absolute Trajectory Error on camera centers (RMSE), after alignment."""
    est = est_poses[:, :3, 3]
    gt = gt_poses[:, :3, 3]
    s, R, t = umeyama_alignment(est, gt, with_scale)
    aligned = (s * (R @ est.T)).T + t
    errors = np.linalg.norm(aligned - gt, axis=1)
    return {
        "ate_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "ate_max": float(errors.max()),
        "scale": s,
        "alignment": (s, R, t),
    }


def _scaled_poses(poses: np.ndarray, s: float) -> np.ndarray:
    """Copy of poses with translation scaled by s (metric units)."""
    out = poses.copy()
    out[:, :3, 3] *= s
    return out


def rpe(est_poses: np.ndarray, gt_poses: np.ndarray, s: float,
        delta: int = 10) -> Dict[str, float]:
    """Relative Pose Error over non-overlapping pairs (i, i+delta).

    For each pair: err = inv(d_gt) @ d_est, decomposed into translation
    magnitude (scene units, after scale normalization) and rotation angle (deg).
    """
    est = _scaled_poses(est_poses, s)
    rot_errs, trans_errs = [], []
    n = min(len(est), len(gt_poses))
    for i in range(0, n - delta, delta):
        d_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[i + delta]
        d_est = np.linalg.inv(est[i]) @ est[i + delta]
        err = np.linalg.inv(d_gt) @ d_est
        trans_errs.append(np.linalg.norm(err[:3, 3]))
        rot = err[:3, :3]
        angle = np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))
        rot_errs.append(np.degrees(angle))
    return {
        "rpe_trans_rmse": float(np.sqrt(np.mean(np.square(trans_errs)))),
        "rpe_rot_rmse": float(np.sqrt(np.mean(np.square(rot_errs)))),
        "rpe_pairs": len(trans_errs),
    }


def scale_consistency(est_poses: np.ndarray, gt_poses: np.ndarray,
                      window: int = 15) -> float:
    """Ratio of end-of-run alignment scale vs start-of-run scale.

    1.0 == scale held steady throughout; deviation means scale drift.
    """
    n = len(est_poses)
    window = max(3, min(window, n // 3))
    s_start, _, _ = umeyama_alignment(
        est_poses[:window, :3, 3], gt_poses[:window, :3, 3])
    s_end, _, _ = umeyama_alignment(
        est_poses[-window:, :3, 3], gt_poses[-window:, :3, 3])
    return float(s_end / s_start) if s_start != 0 else float("inf")


def _transform_points(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray):
    return (s * (R @ points.T)).T + t


def map_on_surface_pct(map_points: np.ndarray, planes: np.ndarray,
                       s: float, R: np.ndarray, t: np.ndarray,
                       tol: float = 0.3) -> float:
    """% of map points within `tol` of any GT wall plane (after alignment)."""
    if len(map_points) == 0:
        return 0.0
    pts = _transform_points(map_points, s, R, t)
    normals = planes[:, :3]
    offsets = planes[:, 3]
    dists = np.abs(pts @ normals.T - offsets)   # (M, P)
    min_dists = dists.min(axis=1)
    return float(np.mean(min_dists <= tol) * 100.0)


def map_near_scene_pct(map_points: np.ndarray, scene_points: np.ndarray,
                       s: float, R: np.ndarray, t: np.ndarray,
                       tol: float = 0.5) -> float:
    """% of map points within `tol` of a GT billboard center (after alignment)."""
    if len(map_points) == 0 or len(scene_points) == 0:
        return 0.0
    pts = _transform_points(map_points, s, R, t)
    tree = cKDTree(scene_points)
    dists, _ = tree.query(pts, k=1)
    return float(np.mean(dists <= tol) * 100.0)


def evaluate(result, gt_poses: np.ndarray, scene_points: Optional[np.ndarray],
             planes: Optional[np.ndarray]) -> Dict[str, object]:
    """Full metric suite for one run. `result` is a harness.RunResult.

    scene_points/planes may be None (real-world datasets with no analytic
    scene geometry); map-vs-scene metrics are skipped in that case.
    """
    metrics = ate(result.poses, gt_poses)
    s, R, t = metrics["alignment"]
    metrics.pop("alignment")

    # Phase 3c: the KEYFRAME-trajectory ATE - the FINAL map keyframe
    # poses vs GT at their frame indices (the ORB-SLAM-standard
    # loop-closure evaluation: an end-of-run closure rewrites the
    # keyframe trajectory AFTER the per-frame stream is history - a
    # corrected map is invisible to the per-frame ATE by construction).
    # Same alignment convention as the per-frame ATE (comparable
    # numbers); the culled keyframes dropped out of the pairing at
    # capture time.
    if (len(result.keyframe_gt_indices) >= 2
            and len(result.keyframe_poses) == len(result.keyframe_gt_indices)):
        kf_metrics = ate(
            result.keyframe_poses,
            gt_poses[np.asarray(result.keyframe_gt_indices)])
        metrics["ate_kf_rmse"] = kf_metrics["ate_rmse"]
        metrics["ate_kf_max"] = kf_metrics["ate_max"]
    else:
        metrics["ate_kf_rmse"] = None
        metrics["ate_kf_max"] = None

    metrics.update(rpe(result.poses, gt_poses, s))
    metrics["scale_consistency"] = scale_consistency(result.poses, gt_poses)
    if scene_points is not None and len(scene_points) > 0:
        metrics["map_on_surface_pct"] = map_on_surface_pct(
            result.map_points, planes, s, R, t)
        metrics["map_near_scene_pct"] = map_near_scene_pct(
            result.map_points, scene_points, s, R, t)
    else:
        metrics["map_on_surface_pct"] = None
        metrics["map_near_scene_pct"] = None
    metrics["keyframes"] = result.num_keyframes
    metrics["map_points"] = len(result.map_points)
    metrics["resets"] = result.resets
    metrics["fps"] = result.fps

    # ---- phase 14/15 additions (additive observables only) ----
    # The shared health summary (same module the video census uses):
    # on ladder entries this joins with GT accuracy in the analysis;
    # on battery entries it is harmless extra context. The harness
    # feeds every frame, so decoded == fed and the fed rate is 1.
    if getattr(result, "health_rows", None):
        from benchmarks.health import summarize_run
        metrics["health"] = summarize_run(
            result.health_rows,
            getattr(result, "health_reset_indices", []),
            result.telemetry.get("events", {}),
            result.frames_processed, result.frames_processed,
            result.seconds)

    # The loop-closure BENEFIT (fr3_long_office is the reason this
    # exists): KF-ATE of the keyframe trajectory as it stood
    # IMMEDIATELY BEFORE the first correction vs the FINAL (post-
    # correction, post-culling) trajectory. Positive = the
    # correction improved the keyframe trajectory against GT.
    metrics["loop_corrections"] = getattr(result, "loop_corrections", 0)
    if (metrics["loop_corrections"]
            and len(result.pre_correction_keyframe_poses) >= 2
            and metrics.get("ate_kf_rmse") is not None):
        idx = np.asarray(result.pre_correction_keyframe_gt_indices)
        if len(idx) == len(result.pre_correction_keyframe_poses) \
                and idx.max() < len(gt_poses):
            pre = ate(result.pre_correction_keyframe_poses,
                      gt_poses[idx])
            metrics["ate_kf_pre_rmse"] = pre["ate_rmse"]
            metrics["loop_correction_benefit"] = (
                pre["ate_rmse"] - metrics["ate_kf_rmse"])

    # Collapse tripwire: the largest single-step camera-center displacement.
    # A finding-8-style blowup (pose chain stepping backward, PnP diverging
    # to |t| ~ 1e4) flags itself here instead of hiding behind an ATE number.
    centers = result.poses[:, :3, 3]
    if len(centers) > 1:
        jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        metrics["max_pose_jump"] = float(jumps.max())
    else:
        metrics["max_pose_jump"] = 0.0
    metrics["max_cam_dist"] = float(np.linalg.norm(centers, axis=1).max()) \
        if len(centers) else 0.0

    # Per-stage timing, ms per processed frame ("other" = measured wall
    # time minus the stage totals). Diagnostic only - never a gate.
    n = max(result.frames_processed, 1)
    stage_seconds = result.telemetry.get("stage_seconds", {})
    timing = {name: seconds / n * 1000.0
              for name, seconds in stage_seconds.items()}
    timing["other"] = max(0.0, result.seconds - sum(stage_seconds.values())) \
        / n * 1000.0
    metrics["timing_ms_per_frame"] = timing
    metrics["events"] = dict(result.telemetry.get("events", {}))
    return metrics
