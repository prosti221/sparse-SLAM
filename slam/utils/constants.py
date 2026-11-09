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
# The adopt-margin guard: the search only adopts a candidate whose
# converged cost beats the INCUMBENT's by this relative margin. The
# gross-error signal measured 20-46% (the wrong-f incumbent's floor);
# the floor noise between plausible candidates is ~1-5% and biased -
# the tum negative-control run moved a TRUE init +3.4% before this
# guard. 10% separates the regimes.
CALIB_SEARCH_ADOPT_MARGIN = 0.10
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
# Rotation-axis criterion: per-pair DOMINANT-axis assignment (see
# gate_metrics). Two prior designs failed on real orbit and are
# recorded there: linear |rotvec| accumulation (sums pose noise
# linearly) and the eigen form (real orbit's 15 deg RPE tracking
# error is genuine rotation mass, not jitter). CALIB_GATE_MIN_ROT_DEG
# doubles as the per-axis accumulated mass and the total floor.
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

# --- Loop closure (phase 3). 3a repairs the dormant milestone-3
# machinery (the E-matrix loop edge carried UNIT-SCALE translation -
# dimensionally wrong against the metric pose graph - plus a
# camera-frame convention bug; see slam/loop_closure.py's design
# record); 3b wires the detector into the tracker. All constants read
# via the module alias everywhere (--override-sweepable; the
# star-import trap). The ENABLED flag is an INT (0/1), not a bool: the
# --override parser is type(current)(value) and bool("0") is True.
LOOP_CLOSURE_ENABLED = 1
# The CORRECTION flag (the 3c landing decision, user-selected): the
# boundary measurement showed orbit's lone end-closure is OUT of
# rigid closure's regime (every correction variant degraded its
# KF-trajectory fidelity - five variants, see the 3c record) while
# tum's dense mid-run corrections measurably help but their
# single-run margins need 3d's multi-seed chaos verdict. The
# correction is INERT by default; 3d's verdict flips this
# deliberately. Detection + verification + telemetry always run
# (RNG-transparent, zero numeric effect wherever no correction
# applies). An INT flag - the --override parser's bool("0") trap.
LOOP_CORRECTION_ENABLED = 0
# Candidate retrieval: brute-force descriptor similarity over the
# feature DB (BF cross-check; populated at add_keyframe, culled by the
# 2h machinery). The threshold was never calibrated by milestone 3 -
# 3b calibrates it on orbit (the true revisit must fire) and tum
# (must stay silent) telemetry. The exclusion window is in KEYFRAMES
# (the milestone-3 detector called the same number "frames"; resolved
# to keyframes - one query per new keyframe).
LOOP_SIMILARITY_THRESHOLD = 0.3
LOOP_EXCLUDE_RECENT_KF = 20
LOOP_MAX_CANDIDATES = 5
# Geometric verification (reloc-grade PnP against the match KF's map
# points - the metric fix; finding 12's map-anchored cure applied to
# the loop pair):
LOOP_MIN_CORRESPONDENCES = 30   # descriptor-matched map points entering PnP
LOOP_MIN_INLIERS = 15           # PnP RANSAC inliers
# Measured 2026-11-09 (3b calibration): the true orbit revisit carries
# Lowe-passed correspondences that are HALF wrong - the synthetic
# wall's repetitive texture matches similar-looking patches, and the
# PnP RANSAC separates them cleanly (42 inliers at 0.66px median
# reprojection, ratio 0.46). The 0.75 first guess (calibrated on clean
# fixtures) killed the target sequence's own closure; the degeneracy
# protection lives in the inlier count + true-pixel median + the
# rotation cross-check, not the ratio.
LOOP_MIN_INLIER_RATIO = 0.40
LOOP_MAX_MEDIAN_REPROJ_PX = 3.0  # TRUE pixels (median over inliers)
# PnP RANSAC inlier marking threshold, in pixels (converted to the
# normalized-coordinate units of the solve at the call site: the
# codebase's PnP works on undistorted normalized coords with eye(3) -
# the 2j-3 survey).
LOOP_PNP_RANSAC_THRESHOLD_PX = 5.0
# The in-module PnP RANSAC (phase 3b): cv.solvePnPRansac draws from
# OpenCV's GLOBAL RNG, and any mid-run draw would shift the stream for
# every subsequent tracker RANSAC call - breaking default-path
# bit-identity wherever the loop module merely runs (a data-dependent
# no-op is not a no-op). The deterministic replacement draws from a
# per-call numpy Generator with this seed, this many minimal-sample
# iterations (EPnP per sample, iterative refine on the winner).
LOOP_RANSAC_SEED = 1701
LOOP_RANSAC_ITERATIONS = 50
# PnP vs two-view relative-rotation agreement: the E-matrix rotation
# (independent of the map) vs the PnP rotation; catches mirrored /
# degenerate PnP solutions that reproject well.
LOOP_MAX_ROTATION_DISAGREEMENT_DEG = 20.0
# Pose-graph edge information weights. The odometry edges are the
# tautological-but-correct chain prior (computed from current poses -
# by design the loop edge is the graph's only corrective
# information). Weights: the loop edge is a FRESH, verified PnP
# constraint; the odometry chain carries accumulated error - the
# loop edge must DOMINATE for the chain to absorb the correction
# (with softer-than-odometry loop weights the loop edge would simply
# stretch and absorb its own residual, correcting nothing - the
# milestone-3 values 5/1 vs odometry 10 were backwards; 3d sweeps).
LOOP_INFO_ODOMETRY = 10.0
LOOP_INFO_ROTATION = 100.0      # loop edge rotation information
LOOP_INFO_TRANSLATION = 100.0    # loop edge translation information
# The drift-magnitude gate (phase 3c, shaped by the 3b calibration):
# a verified constraint that AGREES with the odometry chain is a
# no-op - closing it would be a harmless-but-wasteful pose graph +
# global BA. Only DISAGREEMENT with the chain justifies the global
# rearrangement (finding 13's spirit). Measured regimes: tum's
# near-revisits sit well below 0.3 (2 s of desk motion); orbit's
# true revisit drifts ~25 units over its lap. In world units.
LOOP_MIN_DRIFT_M = 0.30
# The drift CEILING (phase 3c, the orbit boundary measurement): rigid
# SE3 closure fixes ARRANGEMENT of an approximately-right map - it
# cannot recover a scale-broken, shape-wrong one. Measured on orbit's
# true revisit (rotation-true constraint, 0.95 deg vs GT): every
# correction variant DEGRADED the aligned KF-trajectory (control 1.208;
# Huber-damped graph 1.243; full-pull 1.340; +global BA 1.407/1.881) -
# the ~28-unit drift on a ~15-unit path is out of the closure regime,
# the spiral between the anchors cannot be rigidly deformed into the
# circle. tum's near-revisit corrections fire at drift 0.3-1.0 and
# measurably HELP. The ceiling keeps the correction inside the
# measured-helpful regime; out-of-regime closures skip with telemetry
# (loop_correction_out_of_regime) and the sequence stays numerically
# untouched.
LOOP_MAX_DRIFT_M = 2.0
# The correction cooldown (phase 3c, the tum runtime lesson): with
# only the drift gate, tum's near-revisits correct at every
# opportunity (163 corrections measured, ATE 0.165 -> 0.043 - genuine
# continuous re-anchoring - at 1.7 fps, a 17x runtime regression).
# After a correction, the hook skips detection entirely for this many
# keyframe insertions (the whole detector: the verification's BF
# matching is itself a per-frame cost at tum's keyframe density).
# tum: ~11 corrections instead of 163; the drift re-accumulates
# between corrections so the quality/runtime tradeoff is measured in
# the 3d A/B. orbit's single end-closure is unaffected (the counter
# starts armed).
LOOP_CORRECTION_COOLDOWN_KF = 20
