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

### FRAME INTAKE GATE (phase 8-5, the VIDEO path only) ###
# Finding 33 (kyoto confirmation): below ~1 px of matched-feature
# displacement between consecutive fed frames, the two-view geometry
# is pure noise - kyoto_original at 60 fps has a 0.0 px MEDIAN
# displacement (sub-pixel motion under ~2.3 gray-levels of appearance
# noise), the E-chain then claims 0.47 units + 0.114 deg per frame at
# ~zero true motion, and 6 map wipes happen in 30 s; the SAME
# content fed at 15 fps (2.4 px median) runs 0 resets / 21K points
# with the SAME constants. The gate (slam/core/intake.py) feeds a
# decoded frame only when its median matched-feature displacement vs
# the LAST FED frame clears this threshold. 2.0 px is the measured
# healthy edge: above it every fed pair carries real geometric
# signal; at the noise floor the skipped frames carry sub-noise
# motion and are information-free to the pose chain. The threshold
# is in pixels by design; its focal direction is benign (a LONGER
# focal magnifies displacement -> easier to clear -> feeds more, the
# safe side; a shorter one starves more, bounded by the valve below).
# Value history (all measured, user-decided): 2.0 = the phase-8-5
# adopted healthy edge (finding 33's band); 8.0 = the user's
# post-adoption cadence approximation of their 2 fps re-encode
# (stride-30 median displacement 8.6 px); 4.0 = the user's
# conclusion of the 8->2->4 sweep (2026-11-12), with finding 37's
# kyoto_2_5 pan sweep as the mid-point evidence (final |t|
# 219.0 / 185.2 / 123.3 / 101.6 map units and resets 2/3/0/0 at
# 2/4/8/16 px - 4 px composes ~16% less fake translation than the
# 2.0 edge while feeding more than the 8.0 cadence).
FRAME_INTAKE_MIN_DISPLACEMENT_PX = 4.0
# Safety valve: at a full stop displacement never clears the
# threshold, but appearance keeps drifting (lighting, dynamic
# content) - force a feed after this many seconds of container time,
# converted to DECODED frames at the capture's fps with a 3-frame
# floor so slow containers keep a regular heartbeat.
FRAME_INTAKE_MAX_GAP_S = 1.0
# A displacement MEDIAN needs a sample: fewer good matches than
# this means the gate cannot measure (scene cut, occlusion,
# featureless wall) - the frame is fed unconditionally (the
# conservative choice: the tracker must see state it may need to
# recover from, and the gate has no basis to skip).
FRAME_INTAKE_MIN_MATCHES = 10

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

# Triangulation gates. The VALUES are the measured local optimum from
# BOTH directions (2i loosening: orbit collapses; 5b tightening:
# every config degrades at least one sequence, several break
# baselines outright). Phase 5c names say what the checks are (the
# misnomers they replace are recorded in the phase 5c record).
MIN_DEPTH = 0.0  # Cheirality: minimum depth for valid points (z > 0)
MAX_DEPTH = 1000  # Latent divergence guard; never binds on the battery (5a)
# Parallax FLOOR in degrees (the former MAX_PARALLAX was a misnomer -
# the check REJECTS parallax BELOW it). Parallax controls depth
# observability, not correspondence correctness. A depth-uncertainty
# estimate additionally needs an independent image-noise model and
# assumptions about pose accuracy (phase 5d).
MIN_PARALLAX_DEG = 0.65
# Creation-gate reprojection consistency, in PLAIN PIXELS, both frames
# (the former MAX_SQUARED_REPROJECTION_ERROR was a misnomer - the
# comparison was never squared). Read through the module alias in the
# validator (the --override sweep target); the point-health path reads
# its OWN constant below (the 2i isolation property).
MAX_TRI_REPROJECTION_PX = 1.0
# Point-health threshold for the pruning path (Point.is_good and the
# frame tracking-quality counts). Deliberately a SEPARATE name from
# the creation gate's constant (phase 5c): the paths compare
# different things (the creation gate: plain pixels of two-view
# consistency; this path: a running average over a pixel/normalized
# unit mixture - documented in the 2i Q2 record, re-designed by 5e).
# Same value 1.0 as before the split - behavior-preserving. The
# name-bound imports in point.py/frame.py isolate this path from
# --override sweeps of the creation gate (and vice versa).
MAX_POINT_AVG_REPROJECTION_ERROR = 1.0

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

# Validation codes (phase 5c: per-gate reporting - the cascade is
# gone; validation_code is the FIRST failing check, failed_codes
# carries all). The historical numbers are kept for cross-referencing
# old reports: 3 (MIN_BASELINE) and 5 (WELL_CONDITIONED) are retired
# with their dead checks - neither was ever emitted (MIN_BASELINE was
# never implemented; WELL_CONDITIONED was unreachable: the tracker
# de-homogenizes candidates before the gate).
TRIANGULATION_VALIDATION_CODE = {
    'VALID': 0,
    'MIN_DEPTH_VIOLATION': 1,
    'MAX_DEPTH_VIOLATION': 2,
    'MAX_REPROJECTION_ERROR_VIOLATION': 4,
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
# Phase 8-2: the projection-matching search radius, in ANGULAR terms.
# The pixel radius is resolved per frame as
#   max(PROJECTION_MATCH_MIN_RADIUS_PX,
#       round(fx * tan(PROJECTION_MATCH_RADIUS_DEG)))
# from the LIVE K (so a 2j-4 refit's focal propagates here too).
# Measurement provenance (finding 32, walking_salem diagnosis): the
# hard-coded 5 px was implicitly tuned to the benchmark cameras - at
# tum_xyz's fx=518 it is 0.553 deg of angular tolerance, which is the
# regime where the pipeline was measured healthy (per-frame E-chain
# rotation noise median 0.522 deg -> 4.7 px displacement; observations
# on 80% of frames). The SAME noise (0.557 deg median) on the
# walking_salem config video (fx=1449.6) is a 14.1 px displacement -
# 3x outside the fixed radius (0.198 deg tolerance) - collapsing
# map-observation coverage to 19% of frames and killing per-frame PnP
# anchoring. Normalizing to the measured-healthy angular tolerance
# gives that video ~14 px. The FLOOR keeps every benchmark focal
# (fr/orbit/translation fx=450 -> 4.32 px; tum_xyz fx=517.3 -> 4.97 px;
# both round below 5) at exactly the historical 5 px, so the frozen
# battery is bit-identical BY CONSTRUCTION. The descriptor gate
# (Hamming < 20 at the call site; random-pair distance ~128) remains
# the correspondence validator - the radius only sets how far the
# search may look, never what is accepted.
PROJECTION_MATCH_RADIUS_DEG = 0.55
PROJECTION_MATCH_MIN_RADIUS_PX = 5
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
# Phase 8-3: with the hygiene clear in _handle_relocalization_check
# (genuinely healthy tracking clears both streak counters), these caps
# count CONSECUTIVE recovery cycles with no healthy frame between them.
# The success cap is the ineffective-recovery reset: it fires only when
# a whole streak of recoveries was accepted yet quality never once
# recovered - at least cap x RELOCALIZATION_FRAME_COUNT consecutive
# low-quality frames at the needs_recovery cadence - never on successes
# spread across healthy tracking (finding 30's lifetime bug; finding
# 32's walking_salem cascade: the cumulative counter wiped the map 7x
# in 600 frames when better anchoring raised recovery frequency).
MAX_CONSECUTIVE_RELOCALIZATION_FAILURES = 2  # consecutive failed attempts -> reset
# consecutive quality-failing recoveries -> reset
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
# The CORRECTION flag (3d's verdict, 2026-11-09): the chaos-protocol
# matrix (finding 14's tool) found the zero-regression configuration -
# correction ON with the ceiling at the measured regime boundary (see
# LOOP_MAX_DRIFT_M): tum improves materially, orbit / fr / translation
# stay bit-identical (blocked or silent), and the whole matrix is
# SEED-FLAT (all four seeds bit-identical per config - the phase-2
# determinism work plus robust RANSAC make the pipeline
# seed-insensitive; the finding-14 chaos lives on the CONFIG axis).
# The flip was the landing decision the 3c boundary deferred to this
# verdict. An INT flag - the --override parser's bool("0") trap.
LOOP_CORRECTION_ENABLED = 1
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
# The drift CEILING (3d's verdict: the measured regime boundary). The
# 3c story was "drift magnitude is not the separator" (tum's
# beneficial corrections reach 10-40 units, orbit's harmful closure is
# ~28 - the distributions overlap); the chaos matrix then measured
# the CEILING SWEEP on tum and found the boundary anyway: corrections
# above ~22 are harmful even on TUM (ceiling 25: ate 0.175 vs the
# plateau's 0.1335), the [18, 22] band is a stable plateau (identical
# numbers - the same 12 corrections fire, the same 8 block), and
# orbit's lone ~28-unit closure sits on the blocked side (bit-identical
# to control at ceiling 20). tum's landed readout: ate 0.165 -> 0.1335,
# ate_kf 0.166 -> 0.041. The regime separator exists - it just was not
# where the first measurement looked (tum's FULL distribution overlaps
# orbit's closure; tum's BENEFICIAL band does not).
LOOP_MAX_DRIFT_M = 20.0
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
