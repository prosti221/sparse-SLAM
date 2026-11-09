# --- State estimator constants
MINIMUM_NUMBER_OF_INLIERS_FOR_PROJECTION_MATCHING = 10
PNP_ITERATIONS_COUNT = 1000  # Iterations for PnP RANSAC
PNP_REPROJECTION_ERROR = 8.0  # Reprojection error for PnP RANSAC
PNP_MINIMUM_INLIERS = 10  # Minimum inliers for PnP

### CAMERA INTRINSICS DIAGNOSTIC (phase 2j) ###
# Benchmark-side diagnostic ONLY - the SLAM code never reads this.
# benchmarks.harness.load_calibration multiplies fx/fy by this factor
# (principal point untouched) to measure the pipeline's sensitivity to
# wrong intrinsics and to validate the 2j calibration arc's recovery
# (perturb f by +/-20-30%, check convergence to the true value; see
# benchmarks/IMPROVEMENT.md "Phase 2j"). 1.0 is a LITERAL no-op (the
# multiply is skipped) so default runs are bit-identical. Read through
# the module alias so --override can sweep it.
FOCAL_SCALE = 1.0

### CAMERA AUTO-CALIBRATION (phase 2j) ###
# The video-feed init batch: buffer roughly this many seconds of
# frames for the GeoCalib shared-intrinsics estimate (ViPE-style),
# clamped to [MIN, MAX] frames (FPS comes from the capture; 30
# assumed when it reports none). The GeoCalib dependency itself is
# the optional 'calib' extra - see slam/calibration/geocalib_init.py.
CALIB_INIT_SECONDS = 2.0
CALIB_INIT_MIN_FRAMES = 8
CALIB_INIT_MAX_FRAMES = 32
# GeoCalib reports no k1 uncertainty; the refit treats k1's prior as
# soft with this 1-sigma (loose by design - the map must be able to
# override it once the observability gate frees).
CALIB_PRIOR_SIGMA_K1 = 0.02

# --- Intrinsics refit in the BA loop (phase 2j-4, the DroidCalib port,
# v3: BASIN SEARCH - the frozen-map least-squares refit (v1) is dead by
# six-probe measurement; see slam/calibration/refit.py's design record
# and the 2j-4 record in benchmarks/IMPROVEMENT.md). Read through the
# module alias everywhere (--override-sweepable). The ENABLED flag is
# an INT (0/1), not a bool: the --override parser is
# type(current)(value) and bool("0") is True - the trap.
CALIB_REFIT_ENABLED = 0
# Per-candidate global-BA call cap when solving into a candidate's
# basin (the crossing probe: one call lands 98% of the way - the floor
# trajectories converge in ONE call at complete-map states), and the
# relative cost-stabilization tolerance that ends a candidate's solve
# early.
CALIB_SEARCH_BA_CAP = 4
CALIB_SEARCH_J_TOL = 0.02
# The a-priori focal sigma for provided/dataset intrinsics: provenance
# only in v3 (the search compares basins by data cost, not prior
# weights; the GeoCalib uncertainty informed the INIT, which is where
# it belongs - the 2j-2 record).
CALIB_PRIOR_SIGMA_F_FRAC = 0.35
# Diagnostic (benchmark experiments only): added to the dataset k1 at
# estimate construction, so k1-recovery can be tested from perturbed
# initializations too.
CALIB_K1_INIT_OFFSET = 0.0

# --- Observability gate (phase 2j-4). DroidCalib: "the video should
# contain diverse camera motion (translation and rotations around the
# different axes) for the intrinsics to be observable". The critical
# motions (pure translation, orbital, planar rotation) are excluded by
# construction: >= 2 translation eigen-axes excludes single-axis
# translation; >= 2 accumulated-rotation axes excludes planar AND
# orbital (yaw-only) motion.
CALIB_GATE_MIN_KEYFRAMES = 12
CALIB_GATE_MIN_POINTS = 300
CALIB_GATE_MIN_TRANSL_AXES = 2
CALIB_GATE_TRANSL_AXIS_RATIO = 0.05
CALIB_GATE_MIN_ROT_AXES = 2
CALIB_GATE_MIN_ROT_DEG = 15.0

### KEYFRAME SELECTION CONSTANTS ###
MAX_NUMBER_OF_FRAMES_BETWEEN_KEYFRAMES = 10
MINIMUM_BASELINE_THRESHOLD = 2.0  # Minimum translation for new keyframe
# Minimum accumulated rotation (deg) since the last keyframe for a new
# one - rotation is viewpoint change just like translation (phase 1c:
# rotation-dominant motion starved the map under the translation-only
# criterion). Gated by a minimum inter-keyframe gap: it REGULARIZES the
# cadence of rotating sequences (forward_rotation's ragged 3.6-frame
# gaps starved its map; a consistent gap-3 cadence fixes it) but never
# fires frame-to-frame, where the resulting degenerate pairs poison the
# map (orbit with an ungated threshold: 44 -> 140 keyframes, surf
# 15.9 -> 0). Sequences whose baseline-driven cadence is already denser
# (orbit inserts every ~2.7 frames) are naturally pre-empted.
KEYFRAME_MINIMUM_ROTATION_DEG = 1.5
KEYFRAME_ROTATION_MIN_FRAME_GAP = 3
# Minimum scene coverage for new keyframe
MINIMUM_SCENE_COVERAGE_THRESHOLD = 0.30
MINIMUM_KEYFRAME_QUALITY_THRESHOLD = 0.01  # Minimum quality for new keyframe
# Minimum number of new points for keyframe selection
MINIMUM_NEW_POINTS_RATIO = 0.4

# Triangulation constants
MIN_DEPTH = 0.0  # Minimum depth for valid points
MAX_DEPTH = 1000  # 1000.0  # Maximum depth for valid points
MIN_BASELINE_RATIO = 0.04
MAX_SQUARED_REPROJECTION_ERROR = 1.0
MAX_PARALLAX = 0.65

# Window size for local bundle adjustment (number of recent keyframes
# whose poses are optimized together, with older keyframes fixed as
# boundary anchors). Phase 2b sweep (w 9/7/5, all four sequences, seed
# 42): 7 improves fr (0.0605 -> 0.0548) and orbit (1.347 -> 1.143) ATE
# while cutting tum's BA cost 35% (map_optimize 51.7 -> 33.4 ms/frame,
# fps 9.4 -> 12.5); tum ATE 0.102 -> 0.179 stays within its 0.25
# baseline (finding 14: tum single-run deltas span 0.102-0.182 across
# configs - within-baseline check, not a precise knob effect). w=5
# fails orbit/tum RPE-rot baselines (under-anchored windows let
# rotation drift).
LOCAL_MAP_WINDOW_SIZE = 7
# Interval (in keyframes) for the timer-triggered global BA. 0 disables
# the timer (phase 2a, finding 13): measured A/B showed it HURT accuracy
# on three of four sequences (translation 0.0845 -> 0.0181 ATE, orbit
# 1.72 -> 1.04, forward_rotation 0.076 -> 0.065 when removed) while
# dominating tum's BA cost (43.9s of 67s). Local BA anchors the window
# to fixed boundary keyframes, so corrections stay local and consistent
# with the map the tracker uses; the timer-triggered global re-solve
# trades well-constrained regions' accuracy to smooth error globally,
# on a cadence with no evidence of need. Loop closure (phase 3) will
# trigger global BA on loop events, where the global rearrangement is
# justified by a genuine loop constraint.
GLOBAL_BUNDLE_ADJUSTMENT_KEYFRAME_INTERVAL = 0
# Minimum number of observations a point needs for it to not be pruned
MINIMUM_OBSERVATIONS_FOR_POINT = 3
# Phase 2h step 3: redundant-keyframe culling (at most ONE older
# keyframe per insertion, after optimize/triangulate). A candidate -
# any keyframe outside the protected windows (never keyframes[0], the
# gauge anchor; never the local-BA window; never the relocation window,
# which normal relocalization matches against) - is redundant when at
# least CULL_REDUNDANCY_RATIO of its point-side map points are observed
# by at least CULL_MIN_OTHER_OBSERVERS OTHER keyframes: the information
# it carries is already carried by its neighbors (ORB-SLAM-style).
# Point-side observations only (the BA constraint registry - phase 2g)
# and keyframe observers only (tracker frames never anchor anything).
# Conservative defaults; swept via --override in the 2h A/B.
CULL_REDUNDANCY_RATIO = 0.9
CULL_MIN_OTHER_OBSERVERS = 3

# Bidirectional validation codes
TRIANGULATION_VALIDATION_CODE = {
    'VALID': 0,
    'MIN_DEPTH_VIOLATION': 1,
    'MAX_DEPTH_VIOLATION': 2,
    'MIN_BASELINE_VIOLATION': 3,
    'MAX_REPROJECTION_ERROR_VIOLATION': 4,
    'WELL_CONDITIONED_VIOLATION': 5,
    'MIN_PARALLAX_VIOLATION': 6,
}
TRIANGULATION_VALIDATION_CODE.update(
    dict([reversed(i) for i in TRIANGULATION_VALIDATION_CODE.items()]))


# --- Feature extractor constants
ORB_EXTRACTOR_NAME = 'ORB'
DNN_EXTRACTOR_NAME = 'DNN'
AKAZE_EXTRACTOR_NAME = 'AKAZE'
BINARY_DESCRIPTION_METHODS = [ORB_EXTRACTOR_NAME, AKAZE_EXTRACTOR_NAME]

ORB_NUMBER_OF_POINTS = 8000
ORB_QUALITY_LEVEL = 0.01  # Quality level for feature detection
ORB_MIN_DISTANCE = 7  # Minimum distance between features
ORB_KEYPOINT_SIZE = 20  # Size of keypoints for feature extraction


# --- Feature matching constants
KNN_K_VALUE = 2  # Number of nearest neighbors for KNN matching
LOWE_RATIO = 0.7  # Lowe's ratio test threshold for feature matching
# RANSAC threshold for essential matrix estimation
MATCHER_RANSAC_MINIMUM_INLIERS = 8  # Minimum inliers for RANSAC
MATCHER_RANSAC_THRESHOLD = 0.005
MATCHER_RANSAC_PROBABILITY = 0.999  # Probability for RANSAC
# A cheirality verdict is "decisive" when the winning branch's support
# exceeds its opposite-translation twin by this ratio. The motion prior may
# veto decisive verdicts that contradict recent motion (the backward-chain
# collapse signature); coin-flip verdicts are left alone - their errors are
# uncorrelated and partially cancel, while prior arbitration would correlate
# them into drift (measured: translation ATE 1.81 -> 3.31, orbit RPE rot
# 33 -> 43 deg when the prior arbitrates near-ties).
MATCHER_PRIOR_DECISIVE_FLIP_RATIO = 2.0

# --- Bundle adjustment constants
MINIMUM_LOCAL_KEYFRAMES = 2  # Minimum keyframes for local BA
MINIMUM_LOCAL_POINTS = 10  # Minimum points for local BA
MINIMUM_LOCAL_OBSERVATIONS_FOR_POINT = 20
# Threshold for number of ovservations needed to consider a point as outlier
LEAST_SQUARES_F_SCALE = 0.2  # Scale for least squares optimization

MINIMUM_GLOBAL_OBSERVATIONS_FOR_POINT = 50
# Early-exit threshold for BA iterations (phase 2a): the LM loop stops
# once the relative chi improvement per iteration falls below this. The
# terminate action used to be created but never registered, so every BA
# ran the full 30-iteration budget even when converged (local BA
# re-optimizes largely converged window points on every keyframe).
# 1e-4 (measured sweep, all four sequences, seed 42): stopping when
# per-iteration chi improvement becomes marginal REGULARIZES the map -
# squeezing the last 1e-4 of chi fits reprojection noise into the
# geometry. fr 0.0645 -> 0.0605, translation 0.0181 -> 0.0160, tum
# 0.160 -> 0.102, orbit surf 26.6 -> 52.0 (its ATE 1.04 -> 1.35 is the
# unclosed-loop metric; within baseline, phase 3's target). 1e-3/1e-2
# exit too early (1e-2 fails fr/orbit baselines), 1e-6 saves almost
# nothing (these problems genuinely improve for ~30 iterations).
BA_TERMINATE_GAIN_THRESHOLD = 1e-4

# --- PnP divergence guards (phase 1b, benchmarks/IMPROVEMENT.md) ---
# A PnP refinement may not: reproject worse than this (median over its own
# observations, normalized coords), move the camera farther than this
# fraction of the median observed depth, or rotate more than this many
# degrees from the seed pose. Violations are rejected and the frame keeps
# its seed pose - a diverged PnP must never poison the map (measured
# pre-1b: keyframe PnP diverging to |t| ~ 2e4, 12x beyond MAX_DEPTH).
PNP_MAX_MEDIAN_REPROJECTION_ERROR = 0.02  # ~9 px at fx=450
PNP_MAX_CENTER_STEP_RATIO = 0.5          # of median observed depth
PNP_MAX_ROTATION_STEP_DEG = 20.0

# --- Renderer constants
TRACKING_QUALITY_GRADIENT = {
    'min_hue': 0.0,    # Red (poor tracking)
    'max_hue': 0.3,    # Green (excellent tracking)
    'saturation': 0.9,
    'value': 0.9
}


# --- Map constants
TRACKING_QUALITY_WINDOW_SIZE = 5

# --- Relocalization constants
# Trigger relocalization when quality < 0.3
RELOCALIZATION_QUALITY_THRESHOLD = 0.3
RELOCALIZATION_FRAME_COUNT = 5  # After 5 consecutive low-quality frames
RELOCALIZATION_WINDOW = 10  # Search last 10 keyframes first in normal reloc
MIN_RELOCALIZATION_MATCHES = 15
MIN_RELOCALIZATION_INLIERS = 12
MIN_AGGRESSIVE_RELOCALIZATION_INLIERS = 8

# --- Reset mechanism constants
MAX_CONSECUTIVE_RELOCALIZATION_FAILURES = 2  # Reset after 3 failed attempts
# Reset after 5 successful relocs in a row
MAX_CONSECUTIVE_SUCCESSFUL_RELOCALIZATIONS = 5
