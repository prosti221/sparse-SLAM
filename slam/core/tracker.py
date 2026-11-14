import numpy as np
import cv2 as cv
from scipy.spatial import cKDTree
from collections import defaultdict
from typing import List, Optional, Tuple

from slam.core.point import Point
from slam.core.frame import Frame
from slam.core.map import Map
from slam.core.observation import Observation
from slam.utils.utils import *
from slam.utils.constants import *
from slam.utils import constants as const
from slam.utils.logger import *
from slam.utils import telemetry
from slam_native import MatchInputs, match_projected
from slam.features.matcher import *
from slam.features.feature_extractor import FeatureExtractor
# Loop-closure and rotation-model flags live beside their machinery
# (slam/loop_closure, slam/features/matcher); read at CALL time via
# module aliases so probes can pin them in memory for A/B.
import slam.loop_closure as loop_closure_mod
import slam.features.matcher as matcher_mod

# The reset TELEPORTS the live trajectory to the world origin
# (cur_frame.pose = identity) — everything after a reset is then
# incomparable to anything before it. Flag 1 = re-gauge continuation:
# keep the pre-reset pose as the new map's anchor, so the live
# trajectory stays in ONE world frame and the recovery archive
# (whose poses share that frame) re-anchors as a no-op instead of a
# second teleport. INT flag (the --override bool("0") trap); read
# at CALL time so probes can pin it in memory for A/B.
RESET_GAUGE_CONTINUITY_ENABLED = 1

# Per-correction attribution instrumentation: flag 1 records the
# correction's measured magnitudes (drift, per-stage max pose move,
# BA outcome, map size) into the telemetry VALUE store — observation
# only, no behavior change at either value. Flag 0 (the shipped
# default) writes nothing anywhere, so the battery is byte-identical
# by construction. INT flag (the --override bool("0") trap); read at
# CALL time so probes can pin it in memory for A/B.
LOOP_CORRECTION_ATTRIB_ENABLED = 0

LOG_TAG = 'Tracker'


class Tracker:
    def __init__(self, map: Map, feature_extraction_method: str = ORB_EXTRACTOR_NAME, enable_multiscale=False, enable_ba=True):
        self.feature_extraction_method = feature_extraction_method
        self.step = 0
        self.cur_frame: Frame = None
        self.prev_frame: Frame = None
        self.map: Map = map

        self.velocity = np.eye(4)
        self.iterations_since_last_keyframe = 0
        self.consecutive_successful_relocalizations = 0
        self.consecutive_relocalization_failures = 0

        self.enable_ba = enable_ba

        self.feature_extractor = FeatureExtractor(feature_extraction_method)
        self.feature_extractor.set_multiscale_enabled(enable_multiscale)
        # The loop-closure machinery's per-kind norm + similarity scale
        # resolve from the Map's extractor_name at call time.
        self.map.extractor_name = feature_extraction_method

        # The intrinsics refiter (the DroidCalib port). Created ONLY
        # when the flag is set AND the run path wired the live
        # estimate onto the map - the module is not even imported
        # otherwise. INT flag because the benchmark override parser
        # is type(current)(value) and bool("0") would parse True.
        self.calib_refiner = None
        if const.CALIB_REFIT_ENABLED and map.intrinsics is not None:
            from slam.calibration.refit import IntrinsicsRefiner
            self.calib_refiner = IntrinsicsRefiner(map, map.intrinsics)

        # Set by the loop-closure hook when a correction moved
        # keyframe poses mid-update; consumed AFTER the constant-
        # velocity update in update() (which would otherwise bake the
        # teleport into the next prediction - the reloc precedent,
        # relocated because the hook runs at insertion time, before
        # the velocity line).
        self._loop_correction_applied = False
        # Insertions since the last correction (the cooldown starts
        # ARMED so a mid-run closure is never blocked; a correction
        # resets it to 0).
        self._loop_kfs_since_correction = const.LOOP_CORRECTION_COOLDOWN_KF

        # The rotation-dominant pair-pose gate: VIDEO-PATH ONLY -
        # slam.py enables it where it wires the intake gate (the
        # coherence signal exists only at the fed cadence, where
        # pairs carry >= 8 px of displacement above the ORB
        # quantization noise; stride-1 harness feeds keep the default
        # OFF and are bit-identical by construction).
        self.rotation_model_enabled = False
        # The t-coherence EMA state (see
        # _update_rotation_model_gate): EMA of |cos| between
        # consecutive frame-chain pair E-t directions, the previous
        # pair's unit t, and the sticky engagement flag.
        self._rot_model_ema = None
        self._rot_model_prev_t = None
        self._rot_model_engaged = False

    def update(self, new_frame: Frame) -> bool:
        # Extract features from the new frame — unless the caller
        # already attached them (the intake gate carries every fed
        # frame's extraction; the [26a] probe measured the double
        # extraction at ~23% of the live-path wall on ALIKED). The
        # harness path (bare frames) extracts exactly as before.
        if new_frame.keypoints is None:
            with telemetry.stage("extract"):
                new_kps, new_descs = self.feature_extractor.extract(
                    new_frame.gray_image)
            new_frame.set_features(new_kps, new_descs)

        # If this is the first frame, initialize the current frame
        if not self.cur_frame:
            self.cur_frame = new_frame
            self.map.add_keyframe(self.cur_frame)
            return True

        self.prev_frame = self.cur_frame
        self.cur_frame = new_frame

        # Predict the initial pose estimate using constant velocity model (more stable)
        self.cur_frame.pose = self._get_next_pose_estimate()

        # Try to refine with feature matching if possible. The motion prior
        # for the essential-matrix disambiguation is the previous
        # frame-to-frame relative pose; the identity carries no
        # information (run start / post-reloc reset), so branch selection
        # is then purely geometric.
        motion_prior = None if np.array_equal(
            self.velocity, np.eye(4)) else self.velocity
        with telemetry.stage("match"):
            Rt = match_features_between_frames(
                self.cur_frame, self.prev_frame, self.feature_extraction_method,
                motion_prior=motion_prior)
        # The t-coherence gate: the legacy E path ALWAYS runs first,
        # identically (same cv-RNG consumption, same match data, same
        # events) - the EMA is fed by its translation directions, and
        # only the COMMITTED pose step swaps when the gate is engaged.
        Rt = self._update_rotation_model_gate(Rt)
        if Rt is not None:
            # Refine the kinematic estimate with the feature-based relative
            # pose: Twc_cur = Twc_prev @ T_{prev<-cur}
            refined_pose = self.prev_frame.pose @ Rt
            self.cur_frame.pose = refined_pose
            debug_log(LOG_TAG, "Pose estimate refined with feature matching")
        else:
            telemetry.event("match_fallback")
            debug_log(
                LOG_TAG, "Using kinematic pose estimate (feature matching failed)")

        # Project the visible map points onto current frame for matching by projection
        with telemetry.stage("project"):
            projected_points = self._project_visible_map_points()

        # Match the projected points with the current frame's keypoints
        with telemetry.stage("proj_match"):
            observations = self._match_projected_points(projected_points)

        # Register the projection matches (map-point side AND frame side)
        # before pose refinement: refine_pose_pnp reads the frame's
        # observations. This registration used to happen inside
        # _handle_keyframe_insertion - i.e. AFTER the per-frame PnP - which
        # is why that PnP read the previous keyframe's stale observations
        # and refined the wrong frame entirely.
        for obs in observations:
            obs.point.add_observation(obs)

        # Refine the current frame's pose against the map. The E-chained
        # estimate is only the SEED; PnP over the frame's map-point
        # observations is the map-anchoring step - the long-baseline
        # constraint that per-frame essential matrices cannot provide.
        # Divergent solutions are guarded and rejected.
        if len(observations) >= 10:
            with telemetry.stage("pnp"):
                pose_refined, input_count = self.map.g2o_optimizer.refine_pose_pnp(
                    self.cur_frame)
            telemetry.event("pnp_frame_ok" if pose_refined else "pnp_frame_fail")
            if pose_refined:
                debug_log(
                    LOG_TAG, f"Pose refined with PnP using {input_count} input correspondences")
            else:
                debug_log(LOG_TAG, "Pose refinement failed")

        # Check & handle keyframe insertion criteria
        self._handle_keyframe_insertion()

        self.step += 1
        self.iterations_since_last_keyframe += 1
        self.map.update_tracking_quality()

        # Update the velocity based on the current and previous frame poses
        self.velocity = np.linalg.inv(
            self.prev_frame.pose) @ self.cur_frame.pose

        # A loop correction moved keyframe poses mid-update (including
        # this frame's, if it was just inserted and queried as the
        # loop keyframe) - the constant-velocity line above has the
        # teleport baked in. Reset it (the reloc precedent): the next
        # prediction is the corrected pose itself, and matching
        # re-establishes motion from there.
        if self._loop_correction_applied:
            self.velocity = np.eye(4)
            self._loop_correction_applied = False

        # Check for relocalization if tracking quality is consistently low
        with telemetry.stage("relocate"):
            self._handle_relocalization_check()

        return self.cur_frame.is_keyframe

    def _update_rotation_model_gate(self, Rt):
        """The t-coherence rotation-dominant gate.

        On rotation-only motion the E-chain pair pose is degenerate -
        with true t~0 the epipolar constraint holds for ANY
        translation, so every pair commits unit-normalized |t|=1.0
        in noise-driven directions (on a pan in place: accumulated
        fake translation of hundreds of map units). What separates
        rotation-only motion from real translation is TEMPORAL -
        real translation keeps the committed t direction coherent
        pair after pair, rotation-only commits noise. This gate
        tracks exactly that: an EMA of |cos(t_k, t_{k-1})| over
        consecutive frame-chain pair E-t directions.

        Semantics (sticky both ways): ENGAGE the rotation model when
        the EMA drops below ROTATION_MODEL_T_COHERENCE_ENGAGE; re-arm
        legacy when it recovers above
        ROTATION_MODEL_T_COHERENCE_REARM. No basis (run start, a
        failed pair) never updates the EMA and never engages - the
        legacy estimate always stands when the gate cannot
        arbitrate. The legacy E estimate ALWAYS runs first and feeds
        the EMA even while engaged (so real translation re-arms);
        the engaged commit swaps only the pose step for
        rotation_model_pose_between_frames' [R | 0] and its consensus
        matches.

        VIDEO-PATH ONLY (self.rotation_model_enabled): the coherence
        signal exists only at the fed cadence; harness-path feeds
        keep the legacy path bit-identical by construction.

        Returns the pair pose to COMMIT (the legacy Rt, or the
        rotation-model override).
        """
        if (not self.rotation_model_enabled
                or not matcher_mod.ROTATION_MODEL_ENABLED):
            return Rt

        # --- feed the EMA from the legacy E translation directions ---
        if Rt is not None:
            t = Rt[:3, 3]
            norm = float(np.linalg.norm(t))
            if norm > 1e-9:
                if self._rot_model_prev_t is not None:
                    c = abs(float(np.dot(t, self._rot_model_prev_t)) / norm)
                    a = matcher_mod.ROTATION_MODEL_EMA_ALPHA
                    self._rot_model_ema = (
                        c if self._rot_model_ema is None
                        else a * self._rot_model_ema + (1.0 - a) * c)
                self._rot_model_prev_t = t.copy()

        if self._rot_model_ema is None:
            # No basis yet: the run start / every pair so far failed.
            return Rt

        # --- sticky engage / re-arm ---
        if not self._rot_model_engaged:
            if self._rot_model_ema < matcher_mod.ROTATION_MODEL_T_COHERENCE_ENGAGE:
                self._rot_model_engaged = True
                telemetry.event("pair_rotation_model_engage")
                debug_log(LOG_TAG, f"Rotation model ENGAGED (t-coherence "
                          f"EMA {self._rot_model_ema:.3f})")
        elif self._rot_model_ema > matcher_mod.ROTATION_MODEL_T_COHERENCE_REARM:
            self._rot_model_engaged = False
            telemetry.event("pair_rotation_model_rearm")
            debug_log(LOG_TAG, f"Rotation model re-armed legacy (t-coherence "
                      f"EMA {self._rot_model_ema:.3f})")

        if not self._rot_model_engaged or Rt is None:
            return Rt

        Rt_rot = rotation_model_pose_between_frames(
            self.cur_frame, self.prev_frame,
            self.feature_extraction_method)
        if Rt_rot is None:
            # No rotation consensus for this pair: the legacy
            # estimate stands (the no-basis principle).
            return Rt
        return Rt_rot

    def _on_keyframe_inserted(self):
        with telemetry.stage("map_optimize"):
            self.map.optimize(self.enable_ba)

        with telemetry.stage("triangulate"):
            triangulated_points = self._triangulate()
        self.map.add_points(triangulated_points)

        # Bound map growth - at most one redundant older keyframe per
        # insertion (never the gauge anchor or the BA/relocation
        # windows). Runs after triangulation so the candidate scan
        # sees the full fresh map.
        with telemetry.stage("cull"):
            self.map.cull_redundant_keyframe()

        # The intrinsics refit hook. Runs AFTER the local BA of this
        # insertion (the refit alternates with it), inside its own
        # profile stage. A LOCKED gate or a frozen estimate is a
        # no-op by construction.
        if self.calib_refiner is not None:
            with telemetry.stage("calib"):
                self.calib_refiner.on_keyframe_inserted()

        # The loop-closure detection hook (telemetry-only; the
        # correction path is flag-gated below). Keyframe cadence -
        # one query per insertion, on the freshly inserted keyframe,
        # after the map state is final for this insertion. The module
        # is RNG-transparent by construction, so this stage is
        # numerically inert wherever no loop verifies.
        with telemetry.stage("loop"):
            self._handle_loop_closure_detection()

    def _filter_duplicate_keypoints(self):
        # Latent empty-path bug (exposed by the 2j-4 volume-scene
        # probe): a keyframe-insertion frame can carry no matches at
        # all (sparse scenes; matches is None until set_match_data).
        # Wall-textured benchmark sequences never hit this.
        if self.cur_frame.matches is None:
            return
        for match in self.cur_frame.matches:
            cur_keyframe_kp = self.map.cur_keyframe.keypoints[match.trainIdx]
            if cur_keyframe_kp not in self.map.cur_keyframe.keypoint_to_point_map:
                continue

            point = self.map.cur_keyframe.get_point_by_keypoint(
                cur_keyframe_kp)

            if point is None:
                continue

            self.cur_frame.kp_unique_mask[match.queryIdx] = False

            # CONSISTENCY CORE only: the old code unconditionally
            # wrote the frame-side registry, which OVERWROTE the
            # existing (projection-matched) observation
            # for the same (point, frame) with a different measurement:
            # the two registries then disagreed, and insertion-time PnP
            # consumed a different pixel than BA's constraint for the
            # same point. Now: if projection matching already
            # registered this pair, keep its observation and only mask
            # the keypoint.
            if point.is_observed_by(self.cur_frame.frame_id):
                telemetry.event("dedup_skip_existing")
                continue

            # Dedup-only matches stay FRAME-SIDE ONLY, exactly as
            # before: they are load-bearing for insertion-time PnP and
            # tracking-quality counts, but deliberately NOT promoted to
            # full (point-side) observations. Promotion was measured
            # and REJECTED twice, across six gate configurations: the
            # inter-frame chains that help tum poison orbit's map, the
            # genuinely-consistent fr constraints destabilize
            # insertion/pruning dynamics everywhere, and the promoted
            # observations x culling interaction destroys tum's win.
            # Full record in benchmarks/IMPROVEMENT.md.
            self.cur_frame.add_point_observation(
                Observation(self.cur_frame, point, match.queryIdx))

        # Filter out matches that are already triangulated in the global map.
        pre_filter_length = len(self.cur_frame.matches)
        self.cur_frame.matches = [
            m for m in self.cur_frame.matches if self.cur_frame.kp_unique_mask[m.queryIdx]]

        debug_log(
            LOG_TAG, f"Removing duplicate points before triangulation. From {pre_filter_length} to {len(self.cur_frame.matches)}")

    def _handle_keyframe_insertion(self):
        # Observations were already registered in update() before the
        # per-frame PnP.
        should_insert, result = should_insert_keyframe(
            self.cur_frame, self.map.cur_keyframe,
            frames_since_last_keyframe=self.iterations_since_last_keyframe,
            video_fps=self.map.video_fps)

        # The forced-insert backstop is a FRAME count with a TIME
        # meaning (~333 ms at the 30 fps calibration anchor); it
        # scales with the video fps (identity at 30 - the harness
        # never sets one). Measured on walking_salem: this path never
        # fires (the baseline criterion pre-empts at ~3 frames), but
        # the normalization keeps its time meaning at any fps.
        max_frames_between_kfs = scaled_frame_count(
            const.MAX_NUMBER_OF_FRAMES_BETWEEN_KEYFRAMES, self.map.video_fps)
        if not should_insert and self.iterations_since_last_keyframe < max_frames_between_kfs:
            debug_log(
                LOG_TAG, f"Skipping keyframe insertion criteria not met. {result}")
            return
        debug_log(
            LOG_TAG, f"Inserting keyframe after {self.iterations_since_last_keyframe} iterations.")

        # Match features between current frame and previous keyframe
        with telemetry.stage("kf_match"):
            match_features_between_frames(
                self.cur_frame, self.map.cur_keyframe, self.feature_extraction_method)

        # We filter out matches that have already been triangulated.
        self._filter_duplicate_keypoints()

        # Promote current frame to keyframe
        self.map.add_keyframe(self.cur_frame)

        self.iterations_since_last_keyframe = 0
        self._on_keyframe_inserted()

    def _project_visible_map_points(self) -> List[Tuple[Point, np.ndarray]]:
        # Batch-projected: project_points does the per-point math as
        # one GEMM. In-front filtering matches the per-point
        # semantics exactly (behind-camera -> skipped).
        points = self.map.points
        if not points:
            return []

        pts_3d = np.array([mp.pt_3d for mp in points], dtype=np.float64)
        uv = self.cur_frame.project_points(pts_3d, normalized=False)

        # Behind-camera rows are NaN; keep the rest as (point, pixel-row)
        # pairs. The uv rows are read-only for consumers.
        finite_idx = np.flatnonzero(np.isfinite(uv).all(axis=1))
        return [(points[i], uv[i]) for i in finite_idx]

    def _match_projected_points(
        self, projected_points: List[Tuple[Point, np.ndarray]],
        dist_thresh: int = None
    ) -> List[Observation]:
        # The search radius is ANGULAR, resolved per call from the
        # live K (a refit's focal propagates here). The historical
        # hard-coded 5 px is the floor: every benchmark focal
        # (450-518) resolves BELOW it and stays at exactly 5 px
        # (bit-identical battery), while long-focal videos get the
        # tolerance the pipeline was measured healthy at (0.55 deg -
        # see PROJECTION_MATCH_RADIUS_DEG's provenance). The descriptor
        # gate below (the per-kind reloc threshold) remains the
        # correspondence validator; the radius only widens the search.
        if dist_thresh is None:
            fx = float(self.cur_frame.K[0, 0])
            dist_thresh = max(
                const.PROJECTION_MATCH_MIN_RADIUS_PX,
                int(round(fx * np.tan(np.radians(
                    const.PROJECTION_MATCH_RADIUS_DEG)))))
        _, descriptors = self.cur_frame.get_keypoints_descriptors()
        kp_coords = self.cur_frame.kp_pts
        if kp_coords is None or len(kp_coords) == 0:
            warning_log(
                LOG_TAG, "No keypoints available for projection matching")
            # Latent bug (exposed by the 2j-4 volume-scene probe, whose
            # sparse frames can have no keypoints): this path returned
            # a TUPLE of arrays while the normal path returns a LIST of
            # Observations - the caller then iterated ndarrays and
            # crashed on obs.point. Wall-textured benchmark sequences
            # never hit this (every frame sees billboards), so the fix
            # is behavior-neutral for the suite.
            return []

        points = [mp for mp, _uv in projected_points]
        # The C3a native batch (route B): the candidate pre-filter, the
        # neighborhood search, and the best-descriptor accept run in
        # ONE native call over flat SOA; the object-graph side (the
        # Observation objects, the running averages, the unique mask)
        # stays Python below. The distance arithmetic is the
        # pre-registered route-B formulation (HAMMING popcount exact;
        # L2 in f64 - near-threshold pairs may flip vs cv.norm's float
        # SIMD, the accepted tolerance).
        desc_width = 0
        for p in points:
            if p.descriptor is not None:
                desc_width = int(np.ascontiguousarray(p.descriptor).nbytes)
                break
        if desc_width == 0:
            # No candidate point carries a descriptor: nothing can
            # match (the Python path's d1-is-None skip, wholesale).
            return []
        M = len(points)
        point_desc = np.zeros((M, desc_width), dtype=np.uint8)
        has_desc = np.zeros(M, dtype=np.uint8)
        for i, p in enumerate(points):
            if p.descriptor is not None:
                point_desc[i] = np.ascontiguousarray(
                    p.descriptor).view(np.uint8).reshape(-1)
                has_desc[i] = 1

        uv = np.empty((M, 2), dtype=np.float64)
        for i, (_p, uv_proj) in enumerate(projected_points):
            uv[i] = uv_proj

        already = np.fromiter(
            (1 if self.cur_frame.frame_id in p.observations else 0
             for p in points), dtype=np.uint8, count=M)

        kp_desc = np.ascontiguousarray(descriptors).view(np.uint8).reshape(
            len(descriptors), -1)
        desc_kind = 1 if EXTRACTOR_DESCRIPTOR_KINDS[
            self.feature_extraction_method] == "binary" else 2
        reloc_thresh = const.EXTRACTOR_RELOC_DESC_THRESHOLDS[
            EXTRACTOR_DESCRIPTOR_KINDS[self.feature_extraction_method]]
        H, W = self.cur_frame.image.shape[:2]

        point_rows, kp_indices, repro_errors, n_matches = match_projected(
            MatchInputs(
                point_uv=uv, point_desc=point_desc,
                point_has_desc=has_desc,
                kp_coords=np.ascontiguousarray(kp_coords, dtype=np.float64),
                kp_desc=kp_desc, already_observed=already,
                desc_kind=desc_kind,
                search_radius=float(dist_thresh), reloc_thresh=reloc_thresh,
                margin=10.0, width=float(W), height=float(H)))

        observations = []
        for r in range(n_matches):
            mp = points[int(point_rows[r])]
            best_idx = int(kp_indices[r])
            repro_error = float(repro_errors[r])
            observation = Observation(self.cur_frame, mp, best_idx)
            observations.append(observation)
            mp.update_reprojection_error(repro_error)
            self.cur_frame.kp_unique_mask[best_idx] = False
        return observations

    def _triangulate(self) -> List[Point]:
        if not (self.map.cur_keyframe and self.map.prev_keyframe):
            warning_log(LOG_TAG, "Not enough keyframes to triangulate points.")
            return []

        # Triangulate points
        points_4d = compute_triangulation(
            self.map.cur_keyframe, self.map.prev_keyframe, use_optimization=False)

        # Latent empty-path bug (exposed by the 2j-4 volume-scene
        # probe): compute_triangulation returns a 1-D empty array on
        # its no-matches warning path, and the [:, 3] indexing below
        # then crashes. Wall-textured sequences never hit this.
        if points_4d.size == 0:
            return []

        valid_points = points_4d[:, 3] != 0
        points_4d = points_4d[valid_points]
        points_4d = points_4d / points_4d[:, 3:]

        points = []
        valid_indices = np.where(valid_points)[0]

        rejected_points = 0
        validation_results = defaultdict(lambda: 0)

        for i, idx in enumerate(valid_indices):
            if idx >= len(self.map.cur_keyframe.matches):
                error_log(LOG_TAG, f"Index {idx} out of bounds for matches.")
                break

            point_4d = points_4d[i]

            point_validation_result = is_valid_triangulated_point(
                idx, self.map.cur_keyframe, self.map.prev_keyframe, point_4d)

            failed_codes = point_validation_result['failed_codes']
            if failed_codes:
                rejected_points += 1
                # PER-GATE counts (multi-failures count in every
                # bucket - a cascade would hide all but the last
                # failure and make cheirality unreachable in
                # telemetry).
                for c in failed_codes:
                    validation_results[TRIANGULATION_VALIDATION_CODE[c]] += 1
                continue

            point = self._create_point_and_register_observations(
                point_4d, idx)
            point.update_reprojection_error(
                point_validation_result['reprojection_error'])

            points.append(point)

        validation_str = "".join(
            [f'{k}: {v}, ' for k, v in validation_results.items()])
        debug_log(
            LOG_TAG,
            f"Triangulated {len(points)} points, rejected {rejected_points} points. "
            f"Per-gate failure counts (multi-failures count in every bucket): {validation_str}"
        )
        telemetry.event("tri_accepted", len(points))
        telemetry.event("tri_rejected", rejected_points)
        for code_name, count in validation_results.items():
            telemetry.event(f"tri_reject_{code_name}", count)
        return points

    def _create_point_and_register_observations(self, point_4d: np.ndarray, point_idx: int) -> Point:
        match = self.map.cur_keyframe.matches[point_idx]
        point = Point(
            np.array(point_4d)[:3],
            self.cur_frame.get_color_value_for_keypoint(match.queryIdx),
            self.map.cur_keyframe.get_keypoints_descriptors()[
                1][match.queryIdx],
        )

        # Current keyframe observation
        kp_idx_cur = match.queryIdx
        cur_frame_obs = Observation(self.map.cur_keyframe, point, kp_idx_cur)
        point.add_observation(cur_frame_obs)

        # Previous keyframe observation
        kp_idx_prev = match.trainIdx
        prev_frame_obs = Observation(
            self.map.prev_keyframe, point, kp_idx_prev)
        point.add_observation(prev_frame_obs)

        return point

    def _get_next_pose_estimate(self):
        # Constant-velocity prediction: Twc_prev @ (T_prev<-cur_prev).
        # Kept as its own method - the recovery audit wraps it by name
        # to record the predicted pose.
        return self.prev_frame.pose @ self.velocity

    def _handle_loop_closure_detection(self):
        """The loop-closure hook at KEYFRAME cadence - one query per
        insertion on the freshly inserted keyframe (a step-based
        cadence re-queried the same cur_keyframe repeatedly).
        Detection -> reloc-grade PnP verification -> the
        drift-magnitude gate -> the correction: pose graph (gauge at
        KF0) -> global BA (the justified trigger) -> the motion-model
        reset.

        The module is cv-RNG-transparent by construction (no cv
        RANSAC anywhere in the path), so the default pipeline's
        numerics are untouched wherever no loop verifies AND
        corrects. LOOP_CLOSURE_ENABLED is an INT flag (the
        --override parser's bool("0") trap)."""
        if not const.LOOP_CLOSURE_ENABLED:
            return

        # The correction cooldown (the dense-keyframe runtime lesson):
        # skip the whole detector - not just the correction - while
        # armed, because the verification's BF matching is itself a
        # per-frame cost at dense keyframe rates. The cooldown gates
        # only when the correction path is ARMED: with the correction
        # inert (LOOP_CORRECTION_ENABLED=0) detection runs
        # unthrottled.
        if const.LOOP_CORRECTION_ENABLED and (
                self._loop_kfs_since_correction
                < const.LOOP_CORRECTION_COOLDOWN_KF):
            self._loop_kfs_since_correction += 1
            telemetry.event("loop_hook_cooldown")
            return
        self._loop_kfs_since_correction += 1

        current_keyframe = self.map.cur_keyframe
        if current_keyframe is None:
            return

        candidates = self.map.detect_loop_closures(current_keyframe)
        if not candidates:
            debug_log(LOG_TAG, "No loop closure candidates detected")
            return

        verified_loops = self.map.loop_detector.verify_loop_candidates(
            candidates)
        if not verified_loops:
            debug_log(LOG_TAG, "No loop closures verified")
            return
        info_log(LOG_TAG, f"Verified {len(verified_loops)} loop "
                 f"closure(s)")

        # The correction flag: everything above is live telemetry
        # (RNG-transparent, zero numeric effect); everything below is
        # the correction path, inert by default.
        if not const.LOOP_CORRECTION_ENABLED:
            debug_log(LOG_TAG, "Correction inert "
                      "(LOOP_CORRECTION_ENABLED=0)")
            return

        # --- the drift-magnitude gate then the correction ---
        # A verified constraint that AGREES with the odometry chain is
        # a no-op - closing it would be a harmless-but-wasteful pose
        # graph + global BA. Only DISAGREEMENT with the chain
        # justifies the global rearrangement.
        constraints = []
        attrib = LOOP_CORRECTION_ATTRIB_ENABLED
        for loop in verified_loops:
            match_kf = self.map.get_keyframe_by_id(loop.match_keyframe_id)
            if match_kf is None:
                continue
            # The constraint's implied pose of the query: M(q,m) @
            # pose_m = pose_q_pnp (the fresh PnP estimate the
            # measurement was built from).
            implied = loop.relative_pose @ match_kf.pose
            drift = float(np.linalg.norm(
                implied[:3, 3] - current_keyframe.pose[:3, 3]))
            if attrib:
                telemetry.value("loop_attr_drift", drift)
            if drift < const.LOOP_MIN_DRIFT_M:
                telemetry.event("loop_correction_skipped")
                debug_log(LOG_TAG, f"Verified loop agrees with the "
                          f"chain (drift {drift:.3f} < "
                          f"{const.LOOP_MIN_DRIFT_M}) - no correction")
                continue
            if drift > const.LOOP_MAX_DRIFT_M:
                telemetry.event("loop_correction_out_of_regime")
                debug_log(LOG_TAG, f"Verified loop drift {drift:.3f} "
                          f"exceeds the closure regime "
                          f"({const.LOOP_MAX_DRIFT_M}) - rigid SE3 "
                          f"closure cannot help a map this far gone "
                          f"(measured on orbit); skipping")
                continue
            constraints.append(
                self.map.pose_graph_optimizer
                .create_constraint_from_verified_loop(loop))

        if not constraints:
            return

        # The pose graph distributes the loop error across the chain
        # (odometry edges as the tautological prior, gauge at KF0).
        graph_pre = ([kf.pose.copy() for kf in self.map.keyframes]
                     if attrib else None)
        if not self.map.pose_graph_optimizer.optimize_with_loop_constraints(
                constraints):
            telemetry.event("loop_correction_failed")
            warning_log(LOG_TAG, "Pose graph optimization failed")
            return
        telemetry.event("loop_corrected")
        info_log(LOG_TAG, f"Loop correction applied "
                 f"({len(constraints)} constraint(s))")
        if attrib:
            telemetry.value("loop_attr_constraints", len(constraints))
            graph_move = max(
                (float(np.linalg.norm(
                    (kf.pose[:3, 3] - pre[:3, 3])))
                 for kf, pre in zip(self.map.keyframes, graph_pre)),
                default=0.0)
            telemetry.value("loop_attr_graph_maxmove", graph_move)
            telemetry.value("loop_attr_map_points", len(self.map.points))

        # Re-align the map points: the pose graph moved keyframe
        # poses, so the existing triangulations are stale - the global
        # BA re-solves poses+points together ([25b]/[25c]: the BA is
        # the measured per-event destroyer yet load-bearing for the
        # keyframe trajectory - skipping it collapsed KF-ATE up to
        # 9x on the A/B). gated=True opts THIS call site into the
        # acceptance gate — the solve writes back only if it improved
        # the objective it minimizes; a diverged rearrangement
        # restores the re-anchored pre-solve state instead of
        # teleporting the map. The Map's PERIODIC global BA stays
        # ungated (bit-transparent legacy path).
        ba_pre = ([kf.pose.copy() for kf in self.map.keyframes]
                  if attrib else None)
        ba_ok = self.map.g2o_optimizer.global_bundle_adjustment(gated=True)
        if ba_ok:
            telemetry.event("loop_global_ba")
        else:
            warning_log(LOG_TAG, "Global BA after loop correction "
                        "failed - map may be inconsistent")
        if attrib:
            telemetry.value("loop_attr_ba_ok", 1.0 if ba_ok else 0.0)
            ba_move = max(
                (float(np.linalg.norm(
                    (kf.pose[:3, 3] - pre[:3, 3])))
                 for kf, pre in zip(self.map.keyframes, ba_pre)),
                default=0.0)
            telemetry.value("loop_attr_ba_maxmove", ba_move)

        # Motion-model consistency (the reloc precedent): this
        # frame's pose just jumped. update()'s constant-velocity line
        # runs AFTER this hook and would bake the teleport into the
        # next prediction - the flag resets it there.
        self._loop_correction_applied = True
        # The cooldown restarts: no further corrections for
        # LOOP_CORRECTION_COOLDOWN_KF insertions.
        self._loop_kfs_since_correction = 0

    def _handle_relocalization_check(self):
        # Recovery-streak hygiene: 'consecutive' means consecutive.
        # Genuinely healthy tracking (map quality at or above the
        # recovery threshold) clears BOTH streak counters, so the
        # success counter reaches its cap only across recovery cycles
        # with NO healthy frame anywhere between them - a
        # demonstrated-ineffective recovery streak, never successes
        # spread across healthy gaps. The clear is deliberately gated
        # on QUALITY, not on needs_recovery():
        # on_relocalization_successful resets
        # consecutive_low_quality_frames, so the frames after every
        # successful recovery are always 'no recovery needed' - gating
        # the clear there would make the cap unreachable dead code.
        # The failure counter gets the same lifetime: a healthy frame
        # ends a failure streak too. No telemetry event is added so
        # frozen-control event comparisons stay exact wherever this
        # changes nothing observable.
        if self.map.avg_tracking_quality >= const.RELOCALIZATION_QUALITY_THRESHOLD:
            self.consecutive_successful_relocalizations = 0
            self.consecutive_relocalization_failures = 0
            return
        if not self.map.needs_recovery():
            return

        # Reaching the cap now means a full streak of recoveries was
        # accepted yet quality never once recovered between them (the
        # hygiene clear above) - each recovery demonstrably failed to
        # restore tracking. The reset is the last resort for a map
        # that cannot support tracking, not a penalty for successful
        # recoveries.
        debug_log(
            LOG_TAG, f"Checking consecutive successful relocs: {self.consecutive_successful_relocalizations}/{MAX_CONSECUTIVE_SUCCESSFUL_RELOCALIZATIONS}")
        if self.consecutive_successful_relocalizations >= MAX_CONSECUTIVE_SUCCESSFUL_RELOCALIZATIONS:
            warning_log(
                LOG_TAG, f"{self.consecutive_successful_relocalizations} consecutive relocalizations without quality recovery - map cannot support tracking, resetting")
            self._reset_system()
            return

        debug_log(LOG_TAG, f"Map needs recovery - attempting relocalization")
        if self._attempt_relocalization():
            debug_log(LOG_TAG, "Relocalization successful!")
            return

        debug_log(LOG_TAG, "Relocalization failed, trying aggressive approach")
        if self._attempt_aggressive_relocalization():
            debug_log(LOG_TAG, "Aggressive relocalization successful!")
            return

        # The frozen pre-reset maps are the LAST recovery resort
        # before the failure path. A verified cross-map match
        # (the loop-grade PnP stack against an archived keyframe's
        # map points) re-anchors the ENTIRE live map into the
        # archive's world frame: the emitted trajectory continues
        # there instead of re-originating at identity (the measured
        # amnesia alternative: total map loss plus an immediate
        # post-reset re-collapse). No archives exist on
        # any run that never reset (the whole curated battery), so
        # this returns before any event or work there.
        if self._attempt_archive_relocalization():
            debug_log(LOG_TAG, "Archive relocalization successful - "
                      "map re-anchored into the pre-reset world frame")
            return

        # Relocalization failed completely. NOTE: latent bug fixed in
        # 2j-4 (exposed by the sparse volume-scene probes): this
        # counter was incremented through self.map - the attribute
        # lives on the TRACKER (initialized next to the success
        # counter). No benchmark sequence ever reached this path (it
        # crashed on entry), so the fix is behavior-neutral for the
        # suite; the reloc_failed telemetry event becomes observable
        # for the first time.
        self.consecutive_relocalization_failures += 1
        telemetry.event("reloc_failed")
        warning_log(
            LOG_TAG, f"Relocalization failed - tracking lost ({self.consecutive_relocalization_failures}/{MAX_CONSECUTIVE_RELOCALIZATION_FAILURES})")

        # Check if we should reset the system due to consecutive failures
        if self.consecutive_relocalization_failures >= MAX_CONSECUTIVE_RELOCALIZATION_FAILURES:
            self._reset_system()
            return

        warning_log(
            LOG_TAG, "Continuing with degraded tracking - try returning to previously mapped areas")

    def _attempt_relocalization(self) -> bool:
        telemetry.event("reloc_attempt")
        # Search through recent keyframes (last N)
        recent_keyframes = self.map.keyframes[-RELOCALIZATION_WINDOW:] if len(
            self.map.keyframes) > RELOCALIZATION_WINDOW else self.map.keyframes

        for kf in reversed(recent_keyframes):
            # Recovery runs AFTER insertion, so the newest keyframe
            # can BE this frame. Matching a frame against itself has
            # zero baseline - the two-view geometry is undefined - yet
            # identical images still fit an essential matrix with
            # hundreds of epipolar inliers and can propose a
            # catastrophic ~180deg rotation. A frame is never its own
            # relocalization reference; the next-newest keyframe is
            # tried exactly as before. No telemetry event is added so
            # frozen control comparisons stay exact wherever no
            # self-pair was ever accepted.
            if kf is self.cur_frame:
                continue
            pose, inliers = self._match_frame_to_keyframe(self.cur_frame, kf)
            if inliers >= MIN_RELOCALIZATION_INLIERS:
                telemetry.event("reloc_success")
                self.cur_frame.pose = pose
                # The pose just jumped to the matched keyframe's frame.
                # Feeding that jump into the constant-velocity model makes
                # the next prediction repeat the teleport (and explode);
                # reset it and let feature matching re-establish motion.
                self.velocity = np.eye(4)
                self.consecutive_successful_relocalizations += 1
                self.map.on_relocalization_successful(kf.frame_id)
                return True
        return False

    def _attempt_aggressive_relocalization(self) -> bool:
        telemetry.event("reloc_aggressive_attempt")
        for kf in reversed(self.map.keyframes):
            # Same exclusion as _attempt_relocalization: the
            # permissive matcher's relaxed gates admit zero-baseline
            # self-pairs even more readily than the normal path.
            if kf is self.cur_frame:
                continue
            pose, inliers = self._match_frame_to_keyframe_permissive(
                self.cur_frame, kf)
            if inliers >= MIN_AGGRESSIVE_RELOCALIZATION_INLIERS:
                telemetry.event("reloc_aggressive_success")
                # Reset pose and potentially clear recent bad keyframes
                self.map.remove_keyframes_after(kf.frame_id)
                self.cur_frame.pose = pose
                # See _attempt_relocalization: never feed the pose jump
                # into the constant-velocity model.
                self.velocity = np.eye(4)
                self.consecutive_successful_relocalizations += 1
                self.map.on_relocalization_successful(kf.frame_id)
                return True
        return False

    def _attempt_archive_relocalization(self) -> bool:
        """The frozen pre-reset maps as the last recovery resort
        (archive-and-re-anchor instead of amnesia).

        Candidates are the archived keyframes, newest archive first,
        newest keyframe first within each (the reset just happened -
        the current view is nearest the pre-reset map's END). Each
        candidate runs the loop-grade verification stack
        (`verify_archive_recovery`: descriptor match against the
        candidate's archived map points -> deterministic PnP RANSAC ->
        the inlier / ratio / median-TRUE-px gates -> the E-vs-PnP
        rotation cross-check). There is no E-chain proposal here by
        design: the PnP pose IS the estimate, because it is METRIC in
        the archive's world frame (PnP resolves the monocular scale
        against 3D points; an E-chain relative pose composed onto the
        archived keyframe would carry unit-scale translation).

        A verified pose re-anchors the ENTIRE live map into the
        archive's world frame (see _reanchor_map_to), then commits
        exactly like the other recovery paths: velocity reset, the
        success counter, the quality-streak clear + local BA via
        on_relocalization_successful. The scan bound caps the
        candidates examined per cycle (early-out on the first
        verified match). The attempt event fires only when archives
        EXIST, so recovery on a run that never reset returns before
        any event or work.
        """
        if not loop_closure_mod.RECOVERY_ARCHIVE_ENABLED:
            return False
        if not self.map.archives:
            return False
        telemetry.event("reloc_archive_attempt")
        scanned = 0
        for archive in reversed(self.map.archives):
            for kf in reversed(archive.keyframes):
                if scanned >= loop_closure_mod.RECOVERY_ARCHIVE_SCAN_LIMIT:
                    debug_log(LOG_TAG, "Archive scan bound reached "
                              "without a verified match")
                    return False
                scanned += 1
                pose_pnp = self.map.loop_detector.verify_archive_recovery(
                    self.cur_frame, kf, archive.points)
                if pose_pnp is None:
                    continue
                telemetry.event("reloc_archive_success")
                info_log(LOG_TAG, f"Archive relocalization: verified "
                         f"against archived keyframe {kf.frame_id} "
                         f"({scanned} candidates scanned) - re-anchoring "
                         f"the live map into the pre-reset world frame")
                self._reanchor_map_to(pose_pnp)
                self.cur_frame.pose = pose_pnp
                # The reloc precedent: never feed the pose jump into
                # the constant-velocity model.
                self.velocity = np.eye(4)
                self.consecutive_successful_relocalizations += 1
                self.map.on_relocalization_successful(kf.frame_id)
                return True
        return False

    def _reanchor_map_to(self, pose_pnp: np.ndarray):
        """Rigidly move the ENTIRE live map into a recovered
        world frame (the archive's). The verified PnP pose is the
        current frame's pose in the ARCHIVE frame; the live map's
        poses are in its own (identity-origin) frame; G transforms
        every live keyframe pose and point position so the map and
        the recovered pose live in ONE frame (a mixed-frame map
        would corrupt every downstream projection and BA solve).

        RIGID only - the live map's internal scale is preserved; if
        the live and archive scales disagree, the re-anchored live
        geometry sits in the archive frame at its own scale (a
        recorded-open Sim3 question). The persistent local-BA graph
        is invalidated BEFORE the write (external pose/point
        mutations must never mix with the persistent graph's
        retained estimates), and the add_points dedup set is rebuilt
        from the moved geometry.
        """
        # Invalidation BEFORE any external write.
        self.map.g2o_optimizer.reset_persistent_graph()
        G = pose_pnp @ np.linalg.inv(self.cur_frame.pose)
        for kf in self.map.keyframes:
            kf.pose = G @ kf.pose
        R, t = G[:3, :3], G[:3, 3]
        for pt in self.map.points:
            pt.pt_3d = R @ pt.pt_3d + t
        # Observations read Point.pt_3d through a live property, so
        # they follow automatically; only the dedup set needs the
        # rebuild.
        self.map.rebuild_point_coords()

    def _match_frame_to_keyframe(self, frame: Frame, keyframe: Frame) -> Tuple[Optional[np.ndarray], int]:
        matches, Rt = match_features(
            frame, keyframe, self.feature_extraction_method)

        if len(matches) < MIN_RELOCALIZATION_MATCHES:
            return None, 0

        # Extract matched points
        indices_f1 = [m.queryIdx for m in matches]
        indices_f2 = [m.trainIdx for m in matches]

        pts_f1_norm = frame.kp_pts_norm[indices_f1]
        pts_f2_norm = keyframe.kp_pts_norm[indices_f2]

        # Estimate pose using RANSAC
        E, mask, _ = estimate_essential_matrix(pts_f1_norm, pts_f2_norm)
        if E is None:
            return None, 0

        inliers = np.sum(mask) if mask is not None else 0

        if inliers >= MIN_RELOCALIZATION_INLIERS:
            if Rt is None:
                return None, 0
            # Rt is the pose of this frame expressed in the keyframe's frame
            # (T_{kf<-frame}), so: Twc_frame = Twc_keyframe @ Rt
            pose = keyframe.pose @ Rt
            # The proposal is checked against the map BEFORE any
            # state is committed (recovery once accepted high-inlier
            # E-proposals - including ~180-deg half-turns - with no
            # geometric check and no rollback; the per-frame PnP
            # guard rejected the pose only AFTER commit). A rejected
            # candidate returns exactly like a failed match - the
            # caller's loop falls through to the next candidate, the
            # pose was never written, the streaks were never cleared,
            # and the aggressive path's remove_keyframes_after never
            # ran (rollback is factored out). The gate rejects only
            # on positive contradiction and accepts wherever it
            # cannot validate (see validate_map_anchored_pose); the
            # telemetry event fires only on rejection so
            # frozen-control comparisons stay exact wherever the
            # gate is silent.
            if not self.map.loop_detector.validate_map_anchored_pose(
                    frame, keyframe, pose):
                telemetry.event("reloc_rejected_validation")
                debug_log(LOG_TAG, "Recovery proposal rejected by "
                          "map-anchored validation - trying next candidate")
                return None, 0
            return pose, inliers

        return None, 0

    def _match_frame_to_keyframe_permissive(self, frame: Frame, keyframe: Frame) -> Tuple[Optional[np.ndarray], int]:
        # Temporarily adjust matcher constants for more permissive matching
        original_threshold = MATCHER_RANSAC_THRESHOLD
        original_min_inliers = MATCHER_RANSAC_MINIMUM_INLIERS

        # Make matching more permissive
        const.MATCHER_RANSAC_THRESHOLD = 0.01
        const.MATCHER_RANSAC_MINIMUM_INLIERS = 5

        try:
            result = self._match_frame_to_keyframe(frame, keyframe)
        finally:
            # Restore original values
            const.MATCHER_RANSAC_THRESHOLD = original_threshold
            const.MATCHER_RANSAC_MINIMUM_INLIERS = original_min_inliers

        return result

    def _reset_system(self):
        telemetry.event("system_reset")
        info_log(LOG_TAG, "System reset triggered - clearing map and restarting")

        # Archive-and-re-anchor replaces amnesia: the pre-reset map
        # is FROZEN as the recovery machinery's last-resort
        # relocalization target BEFORE the clear - the objects move
        # into the archive intact (poses, points, descriptors,
        # observations). The current frame is excluded: it becomes
        # the new origin immediately after, and the archive must not
        # alias an object the live pipeline keeps mutating.
        self.map.archive_current_state(exclude_frame=self.cur_frame)

        # Reset the map completely
        self.map.reset()

        # Reset tracker state
        self.step = 0
        self.velocity = np.eye(4)
        self.iterations_since_last_keyframe = 0
        self.consecutive_successful_relocalizations = 0
        self.consecutive_relocalization_failures = 0

        # Set current frame as new origin. Flag 0 = the teleport: the
        # live trajectory re-originates at identity. Flag 1 keeps the
        # pre-reset pose instead: the new map's gauge CONTINUES the
        # old world frame (the pose-graph gauge anchors at KF0's
        # CURRENT pose wherever it is; the E-chain/PnP compose
        # relative motions; the harness's Umeyama is global — all
        # gauge-agnostic). The recovery archive lives in that same
        # frame, so the archive-reloc re-anchor degenerates to a
        # no-op instead of a second teleport.
        if not RESET_GAUGE_CONTINUITY_ENABLED:
            self.cur_frame.pose = np.eye(4)
        # (flag 1: keep self.cur_frame.pose — the pre-reset estimate)
        self.prev_frame = None

        # Add current frame as first keyframe of new map
        self.map.add_keyframe(self.cur_frame)

        info_log(LOG_TAG, "System reset complete - starting new map")
