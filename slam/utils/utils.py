import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from slam.utils import constants as const
from slam.utils.constants import *
from slam.utils.logger import *

LOG_TAG = 'Utils'


def load_video(config):
    video_tag = config.get_global_config_property("load_video")
    VIDEO_PATH = config.get_video_property(video_tag, 'path')

    cap = cv.VideoCapture(VIDEO_PATH)
    W = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    Fx = config.get_video_property(video_tag, 'fx', default=0)
    Fy = config.get_video_property(video_tag, 'fy', default=Fx)
    Cx = config.get_video_property(video_tag, 'cx', default=W // 2)
    Cy = config.get_video_property(video_tag, 'cy', default=H // 2)
    # The lens's simple-radial k1 in the normalized coordinate.
    # Absent -> 0.0, the LITERAL identity path everywhere
    # k1 is read, so existing configs are bit-identical by construction.
    # The explicit-intrinsics path returns it alongside K (slam.py flows
    # it through IntrinsicsEstimate.from_K into every Frame, engaging
    # the undistort-at-normalization + forward-model projection
    # boundaries). The 'fx: auto' path returns 0.0 and IGNORES any
    # explicit k1 by design: GeoCalib supplies both the focal and its
    # own k1 estimate, consumed through the prior.
    K1 = config.get_video_property(video_tag, 'k1', default=0.0)

    # 'fx: auto' (or an absent/zero fx) switches the video
    # path to GeoCalib auto-calibration - slam.py buffers the first
    # ~CALIB_INIT_SECONDS from this capture and estimates shared
    # intrinsics; the returned K is None until then.
    if Fx in ("auto", 0, None):
        info_log(LOG_TAG,
                 f"Loaded video: {VIDEO_PATH} ({int(W)}x{int(H)}) - "
                 "fx: auto -> GeoCalib intrinsics estimation")
        return cap, None, 0.0

    K = construct_K(Fx, Fy, Cx, Cy)

    info_log(LOG_TAG, f"Loaded video: {VIDEO_PATH} with properties:\n"
             f"  Width: {W}, Height: {H}\n"
             f"  Focal Lengths: Fx={Fx}, Fy={Fy}\n"
             f"  Principal Point: Cx={Cx}, Cy={Cy}\n"
             f"  Radial distortion: k1={K1}")

    return cap, K, float(K1)


def pt_obj_to_array(pts):
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    num_pts = len(pts)
    points_3d = np.empty((num_pts, 3), dtype=np.float32)
    color = np.empty((num_pts, 3), dtype=np.float32)

    for i, pt in enumerate(pts):
        points_3d[i] = pt.pt_3d
        color[i] = pt.color

    color /= 255.0
    return points_3d, color


def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def normalize(pts, Kinv):
    return np.dot(Kinv, add_ones(pts).T).T[:, :2]


def undistort_points_norm(pts, K, k1):
    """Pixel (N,2) -> UNDISTORTED normalized (N,2) under the
    simple-radial model. cv.undistortPoints implements the
    exact inverse of the generator's forward model
    (x_d = x_u * (1 + k1 * r_u^2)) - locked by the round-trip pin in
    benchmarks/stages/test_calibration.py. The k1 == 0 case never
    reaches this helper: the identity path stays the literal
    `normalize` (the default pipeline is bit-identical)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    und = cv.undistortPoints(pts, K, np.array([k1, 0.0, 0.0, 0.0, 0.0]))
    return und.reshape(-1, 2)


def _prior_t_flip_penalty(t, prior):
    """Penalty (0 or 180 degrees) for a candidate translation opposing the
    motion prior's direction.

    The prior arbitrates ONLY the +-t choice, not the R1/R2 twist: the +-t
    flip is where cheirality measurably anti-correlates with truth under
    degenerate baselines (measured: true branch 140/3071 vs backward branch
    2931/3071 support), while the R1/R2 twist is well-resolved by cheirality
    except on exact planar ties - and there, the true family is almost
    always R1 (the twist is a 180 deg rotation), which the fixed candidate
    order already prefers. Arbitrating R1/R2 with the prior's own rotation
    locks a drifting estimate into the twisted branch (measured: orbit RPE
    rot 33 -> 43 deg).

    A prior with (near-)zero translation carries no direction information
    (rotation-dominant motion); the penalty is then skipped.
    """
    t_p = np.asarray(prior[:3, 3], dtype=float)
    t_p_norm = np.linalg.norm(t_p)
    if t_p_norm > 1e-12 and float(np.dot(t, t_p)) < 0.0:
        return 180.0
    return 0.0


def extractRtFromE(pts1, pts2, E, K, motion_prior=None):
    """Recover the relative pose T_{f2<-f1} from an essential matrix.

    cv.recoverPose resolves the four-fold ambiguity (R1, R2, +/-t) with a
    cheirality vote alone. Under lateral motion over a dominant plane,
    noisy features let a wrong branch win that vote - the pose chain then
    steps backward and diverges (the IMPROVEMENT.md record holds the
    measurement).

    Instead, all four decompositions are scored explicitly:
    1. geometric viability: a candidate must triangulate a meaningful
       number of points in front of both cameras (>= max(5, 2% of the
       winner's support)),
    2. the cheirality verdict on the +-t flip is DECISIVE when the winning
       branch out-supports its opposite-translation twin by
       MATCHER_PRIOR_DECISIVE_FLIP_RATIO. Only then may the
       constant-velocity motion prior (the previous frame-to-frame
       relative pose) veto it: a decisive verdict opposing the recent
       motion direction is the backward-chain collapse signature.
       Coin-flip verdicts are left alone - their errors are
       uncorrelated and partially cancel, while prior arbitration would
       correlate them into drift (measured: translation ATE 1.81 -> 3.31,
       orbit RPE rot 33 -> 43 deg),
    3. support count and fixed candidate order break ties
       deterministically, matching recoverPose's empirically correct
       first-match behavior. Reprojection error is deliberately NOT a
       tiebreak: over a dominant plane the twisted pair earns exactly
       equal cheirality support AND near-equal reprojection, so it
       decides such ties by noise (measured: orbit RPE rot 25 -> 43).

    A prior cannot override clear geometry: with a single viable
    candidate the selection is purely geometric (a wrong prior can never
    corrupt a good estimate).

    Returns T_{f2<-f1} as a 4x4 matrix, translation unit-normalized
    (monocular scale). Convention: X_f2 = R @ X_f1 + t, so camera-to-world
    poses compose as Twc_f1 = Twc_f2 @ Rt.
    """
    pts1 = np.asarray(pts1, dtype=float).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=float).reshape(-1, 2)
    K = np.asarray(K, dtype=float)

    R1, R2, t = cv.decomposeEssentialMat(E)
    t = np.asarray(t, dtype=float).flatten()
    candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    P1 = K @ np.eye(3, 4)
    scores = []  # (support, reproj_error, R, t) per candidate
    for R_c, t_c in candidates:
        P2 = K @ np.hstack([R_c, t_c.reshape(3, 1)])
        points_4d = triangulate(P1, P2, pts1, pts2)
        w = points_4d[:, 3]
        finite = np.abs(w) > 1e-9
        X1 = np.full((len(points_4d), 3), np.nan)
        X1[finite] = points_4d[finite, :3] / w[finite, None]
        X2 = (R_c @ X1[finite].T).T + t_c

        in_front = (X1[finite, 2] > 0) & (X2[:, 2] > 0)
        support = int(in_front.sum())
        if support > 0:
            proj1 = X1[finite][in_front, :2] / X1[finite][in_front, 2:3]
            proj2 = X2[in_front, :2] / X2[in_front, 2:3]
            errs = np.concatenate([
                np.linalg.norm(proj1 - pts1[finite][in_front], axis=1),
                np.linalg.norm(proj2 - pts2[finite][in_front], axis=1)])
            reproj = float(np.mean(errs))
        else:
            reproj = float("inf")
        scores.append((support, reproj, R_c, t_c))

    best_support = max(s[0] for s in scores)
    if best_support == 0:
        # Degenerate E: no candidate has cheirality support. Match the
        # pre-1a behavior (recoverPose's answer) and let the caller's
        # guards handle the rest.
        _, R, t_rec, _ = cv.recoverPose(E, pts1, pts2, K)
        ret = np.eye(4)
        ret[:3, :3] = R
        ret[:3, 3] = t_rec.flatten()
        return ret

    # Geometric viability floor - deliberately tiny relative to the winner
    # (see docstring: under degenerate baselines the cheirality vote is
    # noise-driven and can anti-correlate with the true branch).
    support_floor = max(5.0, 0.02 * best_support)
    viable = [i for i, s in enumerate(scores) if s[0] >= support_floor]
    if not viable:
        # No candidate reaches the floor (best support < 5 points): fall
        # back to the geometric winner rather than nothing.
        viable = [max(range(len(scores)), key=lambda i: scores[i][0])]

    # Decisiveness of the cheirality verdict on the +-t flip: ratio of the
    # winner's support to its opposite-translation twin's. Only decisive
    # verdicts may be vetoed by the motion prior (see docstring + the
    # MATCHER_PRIOR_DECISIVE_FLIP_RATIO constant).
    winner_idx = max(range(len(scores)), key=lambda i: scores[i][0])
    winner_sign = 1 if winner_idx in (0, 2) else -1
    opposite_support = max(
        (scores[i][0] for i in range(len(scores))
         if (1 if i in (0, 2) else -1) != winner_sign), default=0)
    decisive = scores[winner_idx][0] >= \
        MATCHER_PRIOR_DECISIVE_FLIP_RATIO * max(opposite_support, 1)

    if motion_prior is not None and len(viable) > 1 and decisive:
        chosen = min(viable, key=lambda i: (
            _prior_t_flip_penalty(scores[i][3], motion_prior),
            -scores[i][0], i))
    else:
        chosen = min(viable, key=lambda i: (-scores[i][0], i))
    _, _, R_c, t_c = scores[chosen]
    ret = np.eye(4)
    ret[:3, :3] = R_c
    ret[:3, 3] = t_c
    return ret


def construct_K(focal_x, focal_y, c_x, c_y):

    return np.array([[focal_x, 0, c_x],
                     [0, focal_y, c_y],
                     [0, 0, 1]])


def normalize_vec(v):
    return v / np.linalg.norm(v)


# The degenerate-direction guard, beside its
# machinery (the parallax gate in is_valid_triangulated_point).
# A candidate whose direction vectors are non-normalizable
# (||v|| = 0 or non-finite) has NO parallax: the guarded path
# rejects it deterministically under the parallax gate's
# attribution (the failing CHECK is the parallax check - the
# existing code table is unchanged; no new code value). The
# unguarded legacy path normalized zero/NaN vectors into NaN
# comparisons - and NaN < threshold is False, so a degenerate
# candidate PASSED every gate (a latent NaN-acceptance hazard;
# only the exactly-at-camera case was caught, by cheirality) and
# normalize_vec's 0/0 fired a RuntimeWarning on real legs.
# Normalizable candidates
# compute IDENTICALLY either way (v / ||v||, the same float
# arithmetic) - the guard is bit-transparent everywhere else, and
# the battery never reaches the degenerate path, so the
# battery is bit-identical by construction. Kill-switch
# transparency: pin 0 in memory reproduces the legacy semantics
# exactly, warning included.
TRIANGULATION_DEGENERATE_GUARD_ENABLED = 1


def is_valid_triangulated_point(point_idx, cur_frame, prev_frame, point_4d):
    """Gate every triangulation candidate against geometry and
    consistency. ACCEPTANCE is unchanged from the original: VALID iff
    every gate passes.

    All failures are reported. The original cascaded (last
    failing check won), so a behind-camera point reported as a
    reprojection failure, MIN_DEPTH_VIOLATION was unreachable in
    telemetry, and per-gate rejection attribution was impossible.
    `validation_code` is the FIRST
    failing check in the fixed priority order (cheirality, depth,
    parallax, reprojection) - the root-cause reading; `failed_codes`
    carries all of them for telemetry.

    The well-conditioned check (|point_4d[3]| < 1e-6) is
    REMOVED: dead at the only production call site - the tracker
    de-homogenizes candidates (w normalized to exactly 1.0) before
    this gate, and w == 0 candidates are dropped by the tracker's
    pre-filter (the real guard, in _triangulate). No NaN candidates
    exist on the battery (the "non-finite" rows were
    behind-camera points with inf reprojection errors).

    `reprojection_error` is the mean of the two PIXEL reprojection
    errors, consumed at creation (arithmetic unchanged).
    """
    # Constants are read through the module alias: the
    # star-imported copies defeat --override, and this gate is the
    # sweep target.
    point_3d = point_4d[:3]

    # Transform to camera coordinates for both frames
    cur_cam_coords = cur_frame.world_to_camera(point_3d)
    prev_cam_coords = prev_frame.world_to_camera(point_3d)

    cur_reprojection_error, prev_reprojection_error = get_reprojection_error(
        point_idx, cur_frame, prev_frame, point_3d)

    failed_codes = []

    # 1. Cheirality: the point must be in front of both cameras.
    if cur_cam_coords[2] <= const.MIN_DEPTH or prev_cam_coords[2] <= const.MIN_DEPTH:
        failed_codes.append(
            TRIANGULATION_VALIDATION_CODE['MIN_DEPTH_VIOLATION'])

    # 2. Depth ceiling - a latent divergence guard. 5a measured it
    #    never binds on the battery (its candidates always co-fail
    #    parallax/reprojection); it stays as cheap insurance.
    if cur_cam_coords[2] > const.MAX_DEPTH or prev_cam_coords[2] > const.MAX_DEPTH:
        failed_codes.append(
            TRIANGULATION_VALIDATION_CODE['MAX_DEPTH_VIOLATION'])

    # 3. Parallax floor (MIN_PARALLAX_DEG - the former MAX_PARALLAX
    #    was a misnomer: the check REJECTS parallax below it).
    v1 = point_3d - cur_frame.pose[:3, 3]
    v2 = point_3d - prev_frame.pose[:3, 3]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    degenerate = (TRIANGULATION_DEGENERATE_GUARD_ENABLED
                  and (n1 == 0.0 or not np.isfinite(n1)
                       or n2 == 0.0 or not np.isfinite(n2)))
    if degenerate:
        # no direction -> no parallax -> the floor cannot be
        # certified: deterministic rejection (see the flag's
        # comment above). The legacy path instead normalized into
        # NaN comparisons that PASSED this gate.
        failed_codes.append(
            TRIANGULATION_VALIDATION_CODE['MIN_PARALLAX_VIOLATION'])
    elif TRIANGULATION_DEGENERATE_GUARD_ENABLED:
        # same arithmetic as normalize_vec (v / ||v||) - identical
        # bits on every normalizable candidate
        cos_parallax = np.dot(v1 / n1, v2 / n2)
        parallax_angle = np.degrees(np.arccos(np.clip(cos_parallax,
                                                      -1.0, 1.0)))
        if parallax_angle < const.MIN_PARALLAX_DEG:
            failed_codes.append(
                TRIANGULATION_VALIDATION_CODE['MIN_PARALLAX_VIOLATION'])
    else:
        vec1 = normalize_vec(v1)
        vec2 = normalize_vec(v2)

        cos_parallax = np.dot(vec1, vec2)
        parallax_angle = np.degrees(np.arccos(np.clip(cos_parallax,
                                                      -1.0, 1.0)))

        if parallax_angle < const.MIN_PARALLAX_DEG:
            failed_codes.append(
                TRIANGULATION_VALIDATION_CODE['MIN_PARALLAX_VIOLATION'])

    # 4. Reprojection consistency, in PLAIN PIXELS in both frames (the
    #    former MAX_SQUARED_REPROJECTION_ERROR was a misnomer - the
    #    comparison was never squared; see MAX_TRI_REPROJECTION_PX).
    if cur_reprojection_error > const.MAX_TRI_REPROJECTION_PX or prev_reprojection_error > const.MAX_TRI_REPROJECTION_PX:
        failed_codes.append(
            TRIANGULATION_VALIDATION_CODE['MAX_REPROJECTION_ERROR_VIOLATION'])

    validation_code = (failed_codes[0] if failed_codes
                       else TRIANGULATION_VALIDATION_CODE['VALID'])

    result = {
        'validation_code': validation_code,
        'failed_codes': failed_codes,
        'reprojection_error': (cur_reprojection_error + prev_reprojection_error) / 2,
    }

    return result


def get_reprojection_error(point_idx, cur_frame, prev_frame, point_3d):
    match = cur_frame.matches[point_idx]

    # Pixel coordinates of the matched keypoints
    cur_2d = np.array(cur_frame.keypoints[match.queryIdx].pt)
    prev_2d = np.array(prev_frame.keypoints[match.trainIdx].pt)

    # Project the 3D point to both frames
    cur_proj = cur_frame.project_point(point_3d, normalized=False)
    prev_proj = prev_frame.project_point(point_3d, normalized=False)

    if cur_proj is None or prev_proj is None:
        return float('inf'), float('inf')

    # Compute reprojection errors in pixel image space
    cur_error = np.linalg.norm(cur_proj - cur_2d)
    prev_error = np.linalg.norm(prev_proj - prev_2d)

    return cur_error, prev_error


def triangulate(pose1, pose2, pts1, pts2):
    # Use OpenCV's triangulatePoints for vectorized computation
    points_4d = cv.triangulatePoints(pose1, pose2, pts1.T, pts2.T)
    return points_4d.T  # Transpose to match original format (N, 4)


def compute_triangulation(frame1, frame2, use_optimization=False):
    if not frame1.matches or len(frame1.matches) == 0:
        warning_log(LOG_TAG, "No matches found for triangulation")
        return np.array([])

    indices1 = [m.queryIdx for m in frame1.matches]
    indices2 = [m.trainIdx for m in frame1.matches]
    pts1 = frame1.kp_pts_norm[indices1]
    pts2 = frame2.kp_pts_norm[indices2]

    # Get projection matrices
    P1 = np.linalg.inv(frame1.pose)[:3, :]
    P2 = np.linalg.inv(frame2.pose)[:3, :]

    points_4d = triangulate(P1, P2, pts1, pts2)

    if len(points_4d) == 0:
        warning_log(LOG_TAG,
                    "No valid points found in OpenCV triangulation")

    if not use_optimization:
        return points_4d

    points_4d_optimized = []
    optimized_points_count = 0
    for i in range(len(pts1)):
        pt1 = pts1[i]
        pt2 = pts2[i]

        # Use CV result as initial guess
        initial_guess = points_4d[i][:3] / points_4d[i][3]

        # Optimize using least squares
        result = optimize_triangulation(pt1, pt2, P1, P2, initial_guess)

        if result is not None:
            # Convert back to homogeneous coordinates
            points_4d_optimized.append(np.append(result, 1.0))
            optimized_points_count += 1
        else:
            # Fall back to OpenCV result
            points_4d_optimized.append(points_4d[i])

    debug_log(
        LOG_TAG, f"Optimized {optimized_points_count} out of {len(pts1)} points")

    return np.array(points_4d_optimized)


def optimize_triangulation(pt1, pt2, P1, P2, initial_guess, max_iterations=10):
    def residual_function(point_3d):
        # Project 3D point to both cameras
        point_homo = np.append(point_3d, 1.0)

        proj1 = P1 @ point_homo
        proj2 = P2 @ point_homo

        # Normalize projections
        if proj1[2] != 0:
            proj1 = proj1[:2] / proj1[2]
        else:
            proj1 = np.array([1e6, 1e6])

        if proj2[2] != 0:
            proj2 = proj2[:2] / proj2[2]
        else:
            proj2 = np.array([1e6, 1e6])

        # Compute residuals
        residual1 = pt1 - proj1
        residual2 = pt2 - proj2

        return np.concatenate([residual1, residual2])

    try:
        result = least_squares(
            residual_function,
            initial_guess,
            max_nfev=max_iterations * 10,
            ftol=1e-4,
            xtol=1e-4,
            method='lm'
        )

        if result.success:
            return result.x
        else:
            warning_log(
                LOG_TAG, "Least squares optimization failed for triangulation")
            return None

    except Exception as e:
        warning_log(LOG_TAG, f"Least squares optimization error: {e}")
        return None


def compute_baseline_distance(frame1, frame2):
    center1 = frame1.camera_center
    center2 = frame2.camera_center

    return np.linalg.norm(center2 - center1)


def compute_new_points_ratio(cur_frame):
    """Legacy selection signal, deliberately NOT a bounded landmark ratio.

    Missing features/matches contribute no numeric ratio evidence (0).
    Keep the nonempty-data population and arithmetic unchanged here;
    replacing that signal is a separate keyframe-policy experiment.
    """
    if (cur_frame.keypoints is None or len(cur_frame.keypoints) == 0
            or cur_frame.matches is None or len(cur_frame.matches) == 0):
        return 0.0

    # ALL unmapped detections / matches with the previous FRAME. These
    # are different populations, so the retained value may exceed one.
    return cur_frame.num_tracked_features / len(cur_frame.matches)


def scaled_frame_count(count, video_fps):
    """Convert a FRAME-count constant to the frame count
    that preserves its TIME meaning at the video's fps. The constants
    were calibrated on ~30fps-equivalent content;
    at 60fps a camera moves half as far per frame, so frame-count
    thresholds must double to keep the same time/geometric window.
    30 fps is the identity anchor: scaled == count exactly (and any
    run path that never sets a video fps - the entire benchmark
    harness - is bit-identical by construction). At HIGHER fps the
    scaled value is used (ceil - 59.94 lands on the exact 2x values);
    at LOWER fps the anchor count is a FLOOR: a 2 fps stream keeps 5
    recovery frames = 2.5 s (the natural 'several consecutive checks'
    semantics) - tightening thresholds to fractions of their anchor
    (ceil alone gives 1 there) would make recovery fire on a single
    low-quality frame and let rotation insertions go frame-to-frame.
    """
    scaled = int(np.ceil(float(count) * float(video_fps) / 30.0))
    return max(int(count), scaled)


def should_insert_keyframe(cur_frame, prev_keyframe,
                           frames_since_last_keyframe=0,
                           video_fps=30.0):
    # Constants are read through the module (not the star-imported copies)
    # so benchmark threshold sweeps (--override) can tune them live.
    #
    # 1. Viewpoint change: translation baseline OR accumulated rotation
    #    since the last keyframe. Rotation-dominant motion starved the map
    #    under the translation-only criterion (forward_rotation
    #    inserted keyframes on the forced-frame backstop while its
    #    triangulation rejected ~100% of points). The rotation trigger
    #    REGULARIZES the cadence of rotating sequences (gap-capped): it
    #    fires proactively at a consistent cadence, before starvation
    #    develops, but never frame-to-frame, where degenerate pairs poison
    #    the map. Sequences with a denser baseline-driven cadence
    #    (orbit: ~2.7 frames) are naturally pre-empted.
    baseline_distance = compute_baseline_distance(cur_frame, prev_keyframe)
    R_rel = cur_frame.pose[:3, :3].T @ prev_keyframe.pose[:3, :3]
    cos_rot = (np.trace(R_rel) - 1.0) / 2.0
    rotation_deg = float(np.degrees(np.arccos(np.clip(cos_rot, -1.0, 1.0))))

    baseline_ok = const.MINIMUM_BASELINE_THRESHOLD <= baseline_distance
    # The rotation-cadence floor is a FRAME count whose
    # rationale ("never frame-to-frame, where degenerate pairs poison
    # the map") is about TIME/displacement, not frames - it scales
    # with the video fps (identity at the 30 fps calibration anchor).
    # The baseline criterion above is displacement-based and needs no
    # scaling.
    cadence_allows = frames_since_last_keyframe >= \
        scaled_frame_count(const.KEYFRAME_ROTATION_MIN_FRAME_GAP, video_fps)
    rotation_ok = (const.KEYFRAME_MINIMUM_ROTATION_DEG <= rotation_deg
                   and cadence_allows)
    viewpoint_ok = baseline_ok or rotation_ok

    # 2. Check new points ratio
    new_points_ratio = compute_new_points_ratio(cur_frame)
    new_points_ok = new_points_ratio >= const.MINIMUM_NEW_POINTS_RATIO

    # 3. Check tracking quality
    quality_ok = cur_frame.tracking_quality >= const.MINIMUM_KEYFRAME_QUALITY_THRESHOLD

    # Decision logic
    reasons = []

    if not viewpoint_ok:
        reasons.append(
            f"Baseline too small: {baseline_distance:.3f} < {const.MINIMUM_BASELINE_THRESHOLD} "
            f"or rotation trigger unmet: {rotation_deg:.2f} deg at {frames_since_last_keyframe} frames since last keyframe")
    if not new_points_ok:
        reasons.append(
            f"Not enough new points: {new_points_ratio:.2f} < {const.MINIMUM_NEW_POINTS_RATIO}")

    if not quality_ok:
        reasons.append(
            f"Tracking quality too low: {cur_frame.tracking_quality:.2f} < {const.MINIMUM_KEYFRAME_QUALITY_THRESHOLD}")

    # Keyframe insertion criteria
    critical_conditions = [viewpoint_ok]
    quality_conditions = [new_points_ok, quality_ok]

    # Insert keyframe if:
    # - All critical conditions are met AND at least one quality condition is met
    # - OR tracking quality is very low (emergency keyframe)
    should_insert = (all(critical_conditions) and any(
        quality_conditions))

    if should_insert:
        debug_log(LOG_TAG, "Keyframe statistics: "
                  f"Baseline: {baseline_distance:.3f}, "
                  f"Rotation: {rotation_deg:.2f} deg, "
                  f"New Points Ratio: {new_points_ratio:.2f}, "
                  f"Tracking Quality: {cur_frame.tracking_quality:.2f}, ")
        return True, "Quality criteria met"
    else:
        return False, "; ".join(reasons)
