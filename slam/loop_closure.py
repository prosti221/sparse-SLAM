"""
Loop Closure Detection and Optimization for SLAM.

The loop-closure verification is reloc-grade map-anchored PnP: an
earlier two-view-essential-matrix verification was doubly wrong - the
E-translation is UNIT-SCALE (monocular scale) while the graph's
odometry edges are metric world-frame transforms (a ~3.5-unit-apart
revisit pair would be dragged to ~1 unit apart - map shrinkage, not
closure), and the camera-frame Rt does not match the graph's
measurement form M(a,b) = pose_a @ inv(pose_b). Its verification
gates compounded it: the reprojection check compared NORMALIZED
coordinates against a "2.0 pixels" constant (~1000 px - it filtered
nothing) and cheirality used np.all (a single outlier killed a
genuine loop).

The landed verification: unseeded PnP against the match KF's map
points, gated on inliers / ratio / median TRUE-pixel reprojection,
with the PnP rotation cross-checked against the independent two-view
E rotation (catches mirrored / degenerate PnP solutions that reproject
well). See LoopClosureDetector._estimate_loop_constraint.

RANSAC determinism / RNG transparency: the module NEVER
draws from OpenCV's global RNG - solvePnPRansac / findEssentialMat
would advance the stream for every subsequent tracker RANSAC call,
breaking default-path bit-identity wherever the module merely runs
(a data-dependent no-op is not a no-op). The PnP RANSAC here draws
from a per-call numpy Generator seeded from a constant, and the
rotation cross-check's essential matrix is a normalized 8-point over
the (already clean) PnP inlier set. The module is deterministic AND
cv-RNG-transparent by construction.
"""
from typing import List, Tuple, Dict, Optional, NamedTuple, TYPE_CHECKING
from uuid import UUID
import numpy as np
import cv2 as cv
from slam.core.frame import Frame
if TYPE_CHECKING:
    from slam.core.map import Map
from slam.utils import constants as const
from slam.utils import telemetry
from slam.utils.logger import debug_log, warning_log
from slam.utils.utils import extractRtFromE

LOG_TAG = 'LoopClosure'

# Recovery-validation kill-switch. MODULE-level for now (the
# constants.py migration awaits the user's file stabilizing). An INT
# flag, not a bool - the --override parser's type(current)(value)
# trap (bool("0") is True). Read via this module's namespace at CALL
# time so probes can patch it in memory for A/B.
RECOVERY_VALIDATION_ENABLED = 1

# Archive kill-switch + archive-query scan bound (same module-level
# placement / INT-flag convention / call-time reads). OFF reproduces
# legacy amnesia exactly: no archive is created at reset, and the
# archive recovery returns before any work. The scan bound caps the
# archived-keyframe candidates examined per recovery cycle (newest-
# first, early-out on the first verified match); resets are rare so
# archives are few, but a salem-scale pre-reset map holds 150+
# keyframes and a recovery spiral must not pay for all of them every
# cycle.
RECOVERY_ARCHIVE_ENABLED = 1
RECOVERY_ARCHIVE_SCAN_LIMIT = 40

# Point re-anchor kill-switch (same module-level placement / INT-flag
# / call-time-read convention). The mechanism: the pose graph moves
# keyframe poses while the map points stay triangulated under the
# OLD poses, so the global BA that follows starts from an
# inconsistent state (moved cameras, stale structure) and lands
# wherever LM takes it - measured as single-keyframe teleports of
# 21-160 map units per solve and 2-3x KF-ATE explosions (the same
# stage REPAIRS dramatically when it happens to land in the right
# basin). The re-anchor moves each point with its earliest observing
# keyframe's rigid correction BEFORE the BA, making the consistent
# basin the starting point.
#
# MEASURED: REJECTED - standalone (failed the adoption contract on
# all 8 sequence x host legs: the stale points were LOAD-BEARING,
# anchoring the per-frame stream in the pre-correction gauge, and
# propagating corrections consistently amplified harmful ones), and
# again in combination with the graph correction + BA acceptance
# gate (the re-anchor is the destabilizer: reset storms on two
# sequences in BOTH normalization regimes). Stays OFF - the
# IMPROVEMENT.md records hold the full evidence.
LOOP_POINT_REANCHOR_ENABLED = 0


class LoopClosureCandidate(NamedTuple):
    """Represents a potential loop closure between two keyframes"""
    query_keyframe_id: UUID
    match_keyframe_id: UUID
    similarity_score: float
    geometric_verified: bool = False
    relative_pose: Optional[np.ndarray] = None


class KeyframeFeatureDatabase:
    """
    Stores and manages features for all keyframes for efficient similarity search.

    Initially uses simple storage - can be extended with vocabulary trees,
    compression, or other optimizations.
    """

    def __init__(self):
        # keyframe_id -> feature descriptors
        self.feature_storage: Dict[UUID, np.ndarray] = {}
        # Maintains temporal order for exclusion
        self.keyframe_order: List[UUID] = []

    def add_keyframe_features(self, keyframe_id: UUID, descriptors: np.ndarray):
        """Store features for a keyframe"""
        self.feature_storage[keyframe_id] = descriptors.copy()
        self.keyframe_order.append(keyframe_id)
        debug_log(
            LOG_TAG, f"Stored features for keyframe {keyframe_id} ({len(descriptors)} features)")

    def remove_keyframe_features(self, keyframe_id: UUID):
        """Remove a keyframe's stored features: a culled keyframe must
        stop being a loop-closure
        candidate, or find_similar_keyframes would keep matching future
        frames against a dead keyframe."""
        if keyframe_id in self.feature_storage:
            del self.feature_storage[keyframe_id]
        if keyframe_id in self.keyframe_order:
            self.keyframe_order.remove(keyframe_id)

    def get_keyframe_features(self, keyframe_id: UUID) -> Optional[np.ndarray]:
        """Retrieve features for a keyframe"""
        return self.feature_storage.get(keyframe_id)

    def find_similar_keyframes(self, query_descriptors: np.ndarray,
                               exclude_recent: int = 20,
                               similarity_threshold: float = 0.1,
                               extractor_name: str = const.ORB_EXTRACTOR_NAME) -> List[LoopClosureCandidate]:
        """
        Find keyframes similar to query using direct feature matching.

        Args:
            query_descriptors: Feature descriptors from current frame
            exclude_recent: Number of most recent keyframes to exclude
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of potential loop closure candidates
        """
        if len(self.keyframe_order) <= exclude_recent:
            return []

        # Exclude recent keyframes. NOTE: exclude_recent=0
        # must mean "no exclusion", but a naive slice
        # [:-exclude_recent] degenerates to [:-0] = the EMPTY list
        # (a Python gotcha the exclusion pin caught).
        if exclude_recent <= 0:
            candidate_keyframes = self.keyframe_order
        else:
            candidate_keyframes = self.keyframe_order[:-exclude_recent]

        candidates = []
        for kf_id in candidate_keyframes:
            kf_descriptors = self.feature_storage.get(kf_id)
            if kf_descriptors is None:
                continue

            # Compute similarity score (simple ratio of matched features)
            similarity = self._compute_similarity_score(
                query_descriptors, kf_descriptors, extractor_name)

            if similarity >= similarity_threshold:
                candidates.append(LoopClosureCandidate(
                    query_keyframe_id=None,  # Will be set by caller
                    match_keyframe_id=kf_id,
                    similarity_score=similarity
                ))

        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)

        debug_log(
            LOG_TAG, f"Found {len(candidates)} potential loop candidates")
        return candidates

    def _compute_similarity_score(self, desc1: np.ndarray, desc2: np.ndarray,
                                  extractor_name: str = const.ORB_EXTRACTOR_NAME) -> float:
        """
        Compute similarity between two feature sets using multiple metrics.
        Returns a score between 0-1 where higher values indicate more similar scenes.
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return 0.0

        try:
            # The norm is per descriptor KIND (the ORB-era
            # NORM_HAMMING is an ORB-Hamming pin: float descriptors
            # crash batchDistance on it), and the distance scale
            # likewise - both read at call time via the module alias
            # (probe-pinnable).
            kind = const.EXTRACTOR_DESCRIPTOR_KINDS[extractor_name]
            norm = cv.NORM_HAMMING if kind == 'binary' else cv.NORM_L2
            scale = const.LOOP_SIMILARITY_DISTANCE_SCALE[kind]
            bf = cv.BFMatcher(norm, crossCheck=True)
            matches = bf.match(desc1, desc2)

            if len(matches) == 0:
                return 0.0

            # Extract match distances for quality assessment
            distances = [m.distance for m in matches]
            avg_distance = np.mean(distances)
            min_distance = np.min(distances)

            # Quality metrics
            match_ratio = len(matches) / min(len(desc1), len(desc2))

            # Distance-based quality (lower distance = better match)
            # Normalize by typical good match threshold (30 for ORB)
            distance_quality = max(0, 1 - (avg_distance / scale))

            # Combine metrics: prioritize match ratio but weight by quality
            # High match ratio with poor distances = coincidental matches
            similarity = match_ratio * (0.7 + 0.3 * distance_quality)

            return min(similarity, 1.0)

        except Exception as e:
            warning_log(LOG_TAG, f"Error computing similarity: {e}")
            return 0.0


class LoopClosureDetector:
    """
    Main orchestrator for loop closure detection and verification.
    Thresholds live in the constants module (--override-sweepable;
    read via the module alias at CALL time).
    """

    def __init__(self, map: "Map", feature_db: KeyframeFeatureDatabase):
        self.map = map
        self.feature_db = feature_db

    def detect_potential_loops(self, current_frame: Frame) -> List[LoopClosureCandidate]:
        """Query the feature DB for keyframes visually similar to the
        given (just-inserted) keyframe. One query per new keyframe
        (an earlier step-based cadence re-queried the same
        cur_keyframe repeatedly); the exclusion
        window is in KEYFRAMES (an earlier version called the
        same number "frames" - resolved to keyframes, sweepable). The
        query keyframe itself is never a candidate (the pipeline adds
        it to the DB at insertion, before this runs).
        """
        _, descriptors = current_frame.get_keypoints_descriptors()
        if descriptors is None or len(descriptors) == 0:
            return []

        candidates = self.feature_db.find_similar_keyframes(
            descriptors,
            exclude_recent=const.LOOP_EXCLUDE_RECENT_KF,
            similarity_threshold=const.LOOP_SIMILARITY_THRESHOLD,
            extractor_name=self.map.extractor_name)
        candidates = candidates[:const.LOOP_MAX_CANDIDATES]

        out = []
        for cand in candidates:
            if cand.match_keyframe_id == current_frame.frame_id:
                continue
            out.append(LoopClosureCandidate(
                query_keyframe_id=current_frame.frame_id,
                match_keyframe_id=cand.match_keyframe_id,
                similarity_score=cand.similarity_score))

        if out:
            telemetry.event("loop_detected")
        debug_log(
            LOG_TAG, f"Detected {len(out)} potential loop closures")
        return out

    def verify_loop_candidates(self, candidates: List[LoopClosureCandidate]) -> List[LoopClosureCandidate]:
        """Geometric verification via the reloc-grade PnP constraint
        (see _estimate_loop_constraint). A verified candidate carries
        the pose-graph edge measurement M(query, match) =
        pose_q_pnp @ inv(pose_match) in relative_pose."""
        verified_loops = []

        for candidate in candidates:
            # Get the keyframes
            query_kf = self.map.get_keyframe_by_id(candidate.query_keyframe_id)
            match_kf = self.map.get_keyframe_by_id(candidate.match_keyframe_id)

            if query_kf is None or match_kf is None:
                continue

            relative_pose = self._estimate_loop_constraint(query_kf,
                                                           match_kf)
            if relative_pose is not None:
                verified_loops.append(LoopClosureCandidate(
                    query_keyframe_id=candidate.query_keyframe_id,
                    match_keyframe_id=candidate.match_keyframe_id,
                    similarity_score=candidate.similarity_score,
                    geometric_verified=True,
                    relative_pose=relative_pose
                ))

        if verified_loops:
            telemetry.event("loop_verified")
        debug_log(
            LOG_TAG, f"Verified {len(verified_loops)} loop closures")
        return verified_loops

    def _estimate_loop_constraint(self, query_kf: Frame,
                                  match_kf: Frame) -> Optional[np.ndarray]:
        """The metric loop constraint between two keyframes.

        Descriptor-match the query KF's keypoints against the MATCH
        KF'S MAP POINTS (the point-side registry - the 2f/2g lesson),
        solve unseeded PnP (RANSAC; the estimate must be independent of
        the drifted chain) in the codebase's undistorted normalized
        coordinates, gate on inliers / ratio / median TRUE-pixel
        reprojection, then cross-check the PNP ROTATION against the
        independent two-view E rotation over the same correspondences
        (the match-side observations give the second view without a
        second descriptor match; catches mirrored / degenerate PnP
        solutions that reproject well).

        Returns the pose-graph edge measurement M(query, match) =
        pose_q_pnp @ inv(pose_match) - the drift-free relative pose as
        seen by the match KF's neighborhood geometry (the loop edge is
        the graph's only corrective information; the odometry edges
        are the tautological-by-design chain prior). None on any gate
        failure (telemetry carries the reason).
        """
        # --- the match KF's map points (point-side registry) ---
        corr = []  # (pt_3d, descriptor, Point) rows
        for pt in self.map.points:
            if pt.is_observed_by(match_kf.frame_id) and pt.descriptor is not None:
                corr.append((pt.pt_3d, pt.descriptor, pt))
        if len(corr) < const.LOOP_MIN_CORRESPONDENCES:
            telemetry.event("loop_rejected_correspondences")
            debug_log(LOG_TAG, f"Match keyframe carries only "
                      f"{len(corr)} described map points")
            return None

        # --- query-keypoint -> map-point correspondences ---
        _, query_desc = query_kf.get_keypoints_descriptors()
        if query_desc is None or len(query_desc) == 0:
            telemetry.event("loop_rejected_correspondences")
            return None
        point_descs = np.asarray([c[1] for c in corr])
        # The norm is per descriptor KIND (the ORB-era NORM_HAMMING is
        # an ORB-Hamming pin: float descriptors crash batchDistance on
        # it). Read at call time via the module alias.
        kind = const.EXTRACTOR_DESCRIPTOR_KINDS[self.map.extractor_name]
        norm = cv.NORM_HAMMING if kind == 'binary' else cv.NORM_L2
        bf = cv.BFMatcher(norm)
        knn = bf.knnMatch(query_desc, point_descs, k=2)
        q_idx, c_idx = [], []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < const.LOWE_RATIO * n.distance:
                q_idx.append(m.queryIdx)
                c_idx.append(m.trainIdx)
        if len(q_idx) < const.LOOP_MIN_CORRESPONDENCES:
            telemetry.event("loop_rejected_correspondences")
            debug_log(LOG_TAG, f"Only {len(q_idx)} map-point "
                      f"correspondences (need "
                      f"{const.LOOP_MIN_CORRESPONDENCES})")
            return None

        # --- deterministic PnP RANSAC over the correspondences (see
        #     the module docstring: cv.solvePnPRansac would advance
        #     OpenCV's global RNG stream; this one draws from a
        #     per-call numpy Generator) ---
        obj = np.asarray([corr[c][0] for c in c_idx], dtype=np.float32)
        img = query_kf.kp_pts_norm[q_idx].astype(np.float32)
        rvec, tvec, inliers = self._pnp_ransac(obj, img,
                                               query_kf.K[0, 0])
        if rvec is None or len(inliers) < const.LOOP_MIN_INLIERS:
            telemetry.event("loop_rejected_pnp")
            return None
        if len(inliers) / len(q_idx) < const.LOOP_MIN_INLIER_RATIO:
            telemetry.event("loop_rejected_gates")
            debug_log(LOG_TAG, f"PnP inlier ratio "
                      f"{len(inliers) / len(q_idx):.2f} below "
                      f"{const.LOOP_MIN_INLIER_RATIO}")
            return None

        # --- median inlier reprojection in TRUE pixels ---
        R_pnp, _ = cv.Rodrigues(rvec)
        obj_in = obj[inliers]
        Xc = (R_pnp @ obj_in.T).T + tvec.ravel()
        proj = Xc[:, :2] / Xc[:, 2:3]
        err_px = (np.linalg.norm(proj - img[inliers], axis=1)
                  * query_kf.K[0, 0])
        if np.median(err_px) > const.LOOP_MAX_MEDIAN_REPROJ_PX:
            telemetry.event("loop_rejected_gates")
            debug_log(LOG_TAG, f"PnP median reprojection "
                      f"{np.median(err_px):.2f}px above "
                      f"{const.LOOP_MAX_MEDIAN_REPROJ_PX}")
            return None

        pose_q_pnp = self._twc_from_pnp(rvec, tvec)
        R_pnp_rel = (np.linalg.inv(match_kf.pose) @ pose_q_pnp)[:3, :3]

        # --- rotation cross-check (the degeneracy guard) ---
        if not self._rotation_agrees(match_kf, img, inliers,
                                     [corr[c] for c in
                                      np.asarray(c_idx)[inliers]],
                                     R_pnp_rel):
            telemetry.event("loop_rejected_rotation")
            return None

        measurement = pose_q_pnp @ np.linalg.inv(match_kf.pose)
        debug_log(LOG_TAG,
                  f"Loop constraint {query_kf.frame_id} -> "
                  f"{match_kf.frame_id}: {len(inliers)}/{len(q_idx)} "
                  f"inliers, median reproj {np.median(err_px):.2f}px")
        return measurement

    def _twc_from_pnp(self, rvec, tvec) -> np.ndarray:
        """solvePnP returns the world-to-camera transform; the Frame
        convention is camera-to-world (pose)."""
        R, _ = cv.Rodrigues(rvec)
        T_cw = np.eye(4)
        T_cw[:3, :3] = R
        T_cw[:3, 3] = tvec.ravel()
        return np.linalg.inv(T_cw)

    def _pnp_ransac(self, obj: np.ndarray, img: np.ndarray,
                    fx: float):
        """Deterministic PnP RANSAC. cv.solvePnPRansac
        draws from OpenCV's GLOBAL RNG, and any mid-run draw would
        shift the stream for every subsequent tracker RANSAC call -
        breaking default-path bit-identity wherever the module merely
        runs (a data-dependent no-op is not a no-op). This RANSAC
        draws from a per-call numpy Generator seeded from a constant:
        the module is cv-RNG-transparent by construction and its own
        behavior is deterministic across runs. Minimal samples solve
        EPNP (deterministic, closed-form); the winner refines
        iteratively over its inliers. Returns (rvec, tvec, inlier
        indices); (None, None, None) when no sample yields a pose."""
        rng = np.random.default_rng(const.LOOP_RANSAC_SEED)
        n = len(obj)
        if n < 4:
            return None, None, None
        thresh = const.LOOP_PNP_RANSAC_THRESHOLD_PX / fx

        def _mark(rvec, tvec):
            R, _ = cv.Rodrigues(rvec)
            Xc = (R @ obj.T).T + tvec.ravel()
            z = Xc[:, 2]
            with np.errstate(invalid="ignore", divide="ignore"):
                proj = np.where((z > 1e-9)[:, None],
                                Xc[:, :2] / np.where(
                                    z > 1e-9, z, 1.0)[:, None], 1e9)
            err = np.linalg.norm(proj - img, axis=1)
            return np.where((z > 1e-9) & (err < thresh))[0]

        best_rvec = best_tvec = None
        best_inliers = None
        best_count = 0
        for _ in range(const.LOOP_RANSAC_ITERATIONS):
            idx = rng.choice(n, size=4, replace=False)
            try:
                ok, rvec, tvec = cv.solvePnP(
                    obj[idx], img[idx], np.eye(3), np.zeros(5),
                    flags=cv.SOLVEPNP_EPNP)
            except cv.error:
                continue
            if not ok:
                continue
            inliers = _mark(rvec, tvec)
            if len(inliers) > best_count:
                best_count = len(inliers)
                best_rvec, best_tvec, best_inliers = rvec, tvec, inliers

        if best_inliers is None or len(best_inliers) < 4:
            return None, None, None

        # Refine on the winner's inliers (iterative LM from the
        # sample solution - deterministic), then re-mark so the
        # returned inlier set matches the refined pose.
        try:
            ok, rvec, tvec = cv.solvePnP(
                obj[best_inliers], img[best_inliers], np.eye(3),
                np.zeros(5), rvec=best_rvec, tvec=best_tvec,
                useExtrinsicGuess=True, flags=cv.SOLVEPNP_ITERATIVE)
        except cv.error:
            ok = False
        if ok:
            inliers = _mark(rvec, tvec)
            if len(inliers) >= len(best_inliers):
                return rvec, tvec, inliers
        return best_rvec, best_tvec, best_inliers

    def _essential_8point(self, pts_q: np.ndarray,
                          pts_m: np.ndarray) -> Optional[np.ndarray]:
        """Normalized 8-point essential matrix over an already-clean
        inlier set (Hartley normalization, SVD, essential-matrix
        projection). No RANSAC - the PnP inlier set IS the filter
        (the RNG-transparency constraint). Both inputs are
        undistorted normalized coordinates, so the result composes
        directly with extractRtFromE(K = eye(3))."""
        pts_q = np.asarray(pts_q, dtype=np.float64)
        pts_m = np.asarray(pts_m, dtype=np.float64)
        if len(pts_q) < 8:
            return None

        def _hartley(pts):
            c = pts.mean(axis=0)
            mean_d = np.linalg.norm(pts - c, axis=1).mean()
            if mean_d < 1e-12:
                return np.eye(3), pts
            s = np.sqrt(2.0) / mean_d
            T = np.array([[s, 0.0, -s * c[0]],
                          [0.0, s, -s * c[1]],
                          [0.0, 0.0, 1.0]])
            homo = T @ np.column_stack([pts, np.ones(len(pts))]).T
            return T, homo.T[:, :2]

        _, q = _hartley(pts_q)
        _, m = _hartley(pts_m)
        # x_match^T E x_query = 0 (the f2<-f1 convention of
        # extractRtFromE).
        A = np.column_stack([
            m[:, 0] * q[:, 0], m[:, 0] * q[:, 1], m[:, 0],
            m[:, 1] * q[:, 0], m[:, 1] * q[:, 1], m[:, 1],
            q[:, 0], q[:, 1], np.ones(len(q))])
        try:
            _, _, Vt = np.linalg.svd(A)
            F = Vt[-1].reshape(3, 3)
            U, _, Vt2 = np.linalg.svd(F)
            return U @ np.diag([1.0, 1.0, 0.0]) @ Vt2
        except np.linalg.LinAlgError:
            return None

    def _rotation_agrees(self, match_kf: Frame, img: np.ndarray,
                         inliers: np.ndarray, corr_inlier_rows,
                         R_pnp_rel: np.ndarray) -> bool:
        """Two-view rotation agreement over the PnP inlier set: the
        match-side OBSERVATIONS of the same map points give the
        second view (no second descriptor match needed; measured
        keypoint positions, not a projection that would bake in the
        match KF's pose), the E rotation is estimated independently of
        the map, and a mirrored / degenerate PnP shows up as a
        rotation disagreement. The threshold is generous - this is a
        degeneracy guard, not a precision gate.

        The essential matrix is the normalized 8-POINT over the
        already-clean PnP inlier set (findEssentialMat's
        RANSAC draws from OpenCV's global RNG - see the module
        docstring's RNG-transparency note)."""
        pts_q = img[inliers]
        pts_m = []
        for _, _, pt in corr_inlier_rows:
            obs = pt.observations.get(match_kf.frame_id)
            if obs is None:
                return False
            pts_m.append(obs.pt_2d_norm)
        pts_m = np.asarray(pts_m, dtype=np.float64)
        if len(pts_q) < 8:
            return False

        E = self._essential_8point(pts_q, pts_m)
        if E is None:
            return False
        Rt2 = extractRtFromE(pts_q, pts_m, E, np.eye(3))
        if Rt2 is None:
            return False
        # Rt2 is T_{match<-query} (the f2<-f1 convention of
        # extractRtFromE); R_pnp_rel is the same relative rotation from
        # the PnP / map side.
        dR = Rt2[:3, :3].T @ R_pnp_rel
        ang = np.degrees(np.arccos(
            np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
        if ang > const.LOOP_MAX_ROTATION_DISAGREEMENT_DEG:
            debug_log(LOG_TAG, f"Rotation disagreement {ang:.1f} deg "
                      f"above {const.LOOP_MAX_ROTATION_DISAGREEMENT_DEG}")
            return False
        return True

    def map_anchored_pnp_pose(self, frame: Frame, match_kf: Frame,
                              points) -> Tuple[Optional[np.ndarray],
                                               Optional[dict]]:
        """Shared front: the reloc-grade map-anchored PnP over
        a candidate keyframe's described map points, parameterized by
        the POINT SOURCE (the live map for recovery validation, an
        ArchivedMap's points for archive recovery).

        Descriptor-match the frame against the candidate's point-side
        described map points (BF knn + LOWE_RATIO, no cv-RNG draws),
        deterministic PnP RANSAC (`_pnp_ransac` - seeded per call),
        gate on the loop-measured quality stack (inliers / ratio /
        median TRUE-pixel reprojection).

        Returns (pose_pnp, diagnostics) on success - the pose is
        METRIC in the point set's world frame (PnP resolves the
        monocular scale against the 3D points; the E-chain cannot -
        why the archive recovery commits the PnP pose, not an E-chain
        proposal). Returns (None, {'reason': ...}) on every gate
        failure or inability: a low-quality PnP cannot arbitrate
        anything (the no-basis principle).
        """
        corr = []  # (pt_3d, descriptor, Point) rows
        for pt in points:
            if pt.is_observed_by(match_kf.frame_id) and pt.descriptor is not None:
                corr.append((pt.pt_3d, pt.descriptor, pt))
        if len(corr) < const.LOOP_MIN_CORRESPONDENCES:
            return None, {'reason': 'sparse_candidate',
                          'described_points': len(corr)}

        _, frame_desc = frame.get_keypoints_descriptors()
        if frame_desc is None or len(frame_desc) == 0:
            return None, {'reason': 'featureless_frame'}
        point_descs = np.asarray([c[1] for c in corr])
        # Same per-kind norm seam as the loop constraint (above).
        kind = const.EXTRACTOR_DESCRIPTOR_KINDS[self.map.extractor_name]
        norm = cv.NORM_HAMMING if kind == 'binary' else cv.NORM_L2
        bf = cv.BFMatcher(norm)
        knn = bf.knnMatch(frame_desc, point_descs, k=2)
        q_idx, c_idx = [], []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < const.LOWE_RATIO * n.distance:
                q_idx.append(m.queryIdx)
                c_idx.append(m.trainIdx)
        if len(q_idx) < const.LOOP_MIN_CORRESPONDENCES:
            return None, {'reason': 'few_correspondences',
                          'correspondences': len(q_idx)}

        obj = np.asarray([corr[c][0] for c in c_idx], dtype=np.float32)
        img = frame.kp_pts_norm[q_idx].astype(np.float32)
        rvec, tvec, inliers = self._pnp_ransac(obj, img, frame.K[0, 0])
        if rvec is None or len(inliers) < const.LOOP_MIN_INLIERS:
            return None, {'reason': 'pnp_failed',
                          'inliers': 0 if rvec is None else len(inliers)}
        if len(inliers) / len(q_idx) < const.LOOP_MIN_INLIER_RATIO:
            return None, {'reason': 'low_inlier_ratio',
                          'ratio': len(inliers) / len(q_idx)}

        # median inlier reprojection in TRUE pixels
        R_pnp, _ = cv.Rodrigues(rvec)
        obj_in = obj[inliers]
        Xc = (R_pnp @ obj_in.T).T + tvec.ravel()
        proj = Xc[:, :2] / Xc[:, 2:3]
        err_px = (np.linalg.norm(proj - img[inliers], axis=1)
                  * frame.K[0, 0])
        if np.median(err_px) > const.LOOP_MAX_MEDIAN_REPROJ_PX:
            return None, {'reason': 'median_reprojection',
                          'median_err_px': float(np.median(err_px))}

        pose_pnp = self._twc_from_pnp(rvec, tvec)
        diag = {'inliers': inliers, 'q_idx': q_idx, 'c_idx': c_idx,
                'img': img, 'corr': corr, 'obj': obj,
                'n_correspondences': len(q_idx),
                'median_err_px': float(np.median(err_px))}
        return pose_pnp, diag

    def verify_archive_recovery(self, frame: Frame, match_kf: Frame,
                                points) -> Optional[np.ndarray]:
        """The loop-grade verification stack for an ARCHIVE
        recovery candidate - map_anchored_pnp_pose plus the E-vs-PnP
        ROTATION CROSS- CHECK (the archive path has no E-chain proposal
        for the disagreement check; the PnP IS the estimator,
        so it gets the loop detector's degeneracy guard: the archived
        keyframe's own observations of the inlier points give the
        match-side view, and a mirrored / degenerate PnP shows up as a
        rotation disagreement).

        Returns the metric map-anchored pose (in the ARCHIVE's world
        frame), or None on any gate failure.
        """
        pose_pnp, diag = self.map_anchored_pnp_pose(frame, match_kf, points)
        if pose_pnp is None:
            debug_log(LOG_TAG, f"Archive candidate rejected: "
                      f"{diag.get('reason')}")
            return None
        R_pnp_rel = (np.linalg.inv(match_kf.pose) @ pose_pnp)[:3, :3]
        corr_rows = [diag['corr'][c] for c in
                     np.asarray(diag['c_idx'])[diag['inliers']]]
        if not self._rotation_agrees(match_kf, diag['img'],
                                     diag['inliers'], corr_rows,
                                     R_pnp_rel):
            debug_log(LOG_TAG, "Archive candidate rejected: PnP "
                      "rotation cross-check failed")
            return None
        debug_log(LOG_TAG, f"Archive candidate verified: "
                  f"{len(diag['inliers'])}/{diag['n_correspondences']} "
                  f"inliers, median reproj "
                  f"{diag['median_err_px']:.2f}px")
        return pose_pnp

    def validate_map_anchored_pose(self, frame: Frame, match_kf: Frame,
                                   proposed_pose: np.ndarray) -> bool:
        """Recovery validation: check a recovery proposal
        against the map BEFORE the caller commits anything.

        The defect class: both recovery paths accepted
        candidates on epipolar inlier support alone and committed
        immediately - RPY frames 434/553 accepted 686/565-inlier
        half-turns that survived as 175.8/179.4-deg stream errors
        DESPITE the per-frame PnP guard rejecting them afterwards;
        recovery had no rollback (local BA unwound 579/681, the rest
        persisted). An earlier fix removed the zero-baseline self-pair
        admission
        path; this closes the remaining no-validation architecture.

        REJECT only on POSITIVE contradiction (the
        correspondence/PnP/quality work lives in
        map_anchored_pnp_pose; this wrapper keeps the original
        contract bit-for-bit): a HIGH-QUALITY PnP whose rotation
        disagrees with the proposal by more than
        LOOP_MAX_ROTATION_DISAGREEMENT_DEG. Every inability to
        validate - sparse candidate, too few Lowe matches, PnP
        failure, or a low-quality PnP that cannot arbitrate - ACCEPTS
        exactly as the legacy path (the intake gate's no-basis
        principle: young / post-reset sparse maps keep their
        recovery).

        Returns True (accept) / False (reject). Fires no telemetry;
        the tracker hook owns the rejection event so it fires only on
        actual rejections (frozen-control
        comparisons stay exact wherever the gate is silent).
        """
        if not RECOVERY_VALIDATION_ENABLED:
            # Flag OFF: bit-transparent - return before constructing
            # a matcher or touching anything.
            return True

        pose_pnp, diag = self.map_anchored_pnp_pose(
            frame, match_kf, self.map.points)
        if pose_pnp is None:
            debug_log(LOG_TAG, f"Validation: cannot arbitrate "
                      f"({diag.get('reason')}) - no basis to validate, "
                      "accepting")
            return True

        # --- the positive contradiction: a high-quality map-anchored
        #     pose disagreeing with the proposal. The angle is between
        #     ABSOLUTE rotations (both poses are world-frame); a
        #     half-turn proposal shows up as ~180 deg against the
        #     map-anchored truth.
        dR = proposed_pose[:3, :3].T @ pose_pnp[:3, :3]
        ang = np.degrees(np.arccos(
            np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
        if ang > const.LOOP_MAX_ROTATION_DISAGREEMENT_DEG:
            debug_log(LOG_TAG, f"Validation: REJECTING recovery "
                      f"proposal - map-anchored PnP disagrees by "
                      f"{ang:.1f} deg (bound "
                      f"{const.LOOP_MAX_ROTATION_DISAGREEMENT_DEG}), "
                      f"{len(diag['inliers'])}/{diag['n_correspondences']} "
                      f"inliers, median reproj "
                      f"{diag['median_err_px']:.2f}px")
            return False
        debug_log(LOG_TAG, f"Validation: proposal agrees with the map "
                  f"({ang:.1f} deg, {len(diag['inliers'])}/"
                  f"{diag['n_correspondences']} inliers) - accepting")
        return True

class PoseGraphConstraint:
    """A pose-graph LOOP edge (the odometry/chain edges are built
    inside the optimizer from consecutive keyframes). The edge
    information weights are applied by the optimizer from the
    LOOP_INFO_* constants - an earlier per-constraint information
    matrix was never consumed (a dead field; removed)."""

    def __init__(self, from_keyframe_id: UUID, to_keyframe_id: UUID,
                 relative_pose: np.ndarray):
        self.from_keyframe_id = from_keyframe_id
        self.to_keyframe_id = to_keyframe_id
        self.relative_pose = relative_pose


class PoseGraphOptimizer:
    """
    Handles global pose graph optimization with loop closure constraints.
    Extends the existing BA framework.
    """

    def __init__(self, map: "Map"):
        self.map = map

    def optimize_with_loop_constraints(self, loop_constraints: List[PoseGraphConstraint]) -> bool:
        """Run the SE3 pose graph over the loop constraints (odometry
        edges built internally from consecutive keyframes; the loop
        edges carry the verified metric constraints).

        On success, the write-back RE-ANCHORS the map
        points with the correction (see _reanchor_points_with_
        correction) so the global BA that follows starts from a
        consistent state instead of the moved-cameras/stale-structure
        initialization measured to produce 21-160-unit keyframe
        teleports. Flag OFF = the pre-17 passthrough, exactly."""
        debug_log(
            LOG_TAG, f"Starting pose graph optimization with {len(loop_constraints)} loop constraints")

        # Convert constraints to the (from, to, measurement) form
        # expected by the optimizer's pose graph
        pose_graph_constraints = [
            (constraint.from_keyframe_id,
             constraint.to_keyframe_id,
             constraint.relative_pose)
            for constraint in loop_constraints
        ]

        if not LOOP_POINT_REANCHOR_ENABLED:
            return self.map.g2o_optimizer.pose_graph_optimization(
                pose_graph_constraints)

        # The pre-correction poses (the correction IS the delta;
        # captured BEFORE the g2o write-back moves anything).
        pre_poses = {kf.frame_id: kf.pose.copy()
                     for kf in self.map.keyframes}
        ok = self.map.g2o_optimizer.pose_graph_optimization(
            pose_graph_constraints)
        if ok:
            self._reanchor_points_with_correction(pre_poses)
        return ok

    def _reanchor_points_with_correction(self,
                                          pre_poses: dict) -> None:
        """Move every map point with the rigid
        correction of its EARLIEST observing keyframe, after the pose
        graph moved the keyframes and before the global BA runs.

        A point triangulated under the old poses must follow the
        keyframe that anchors its observations; the earliest observer
        approximates the triangulating reference (the ORB-SLAM2
        reference-KF practice). Multi-keyframe points get the single
        earliest delta — an approximation the BA refines from a
        CONSISTENT basin instead of teleporting keyframes to explain
        stale structure (measured, both hosts: 21-160-unit per-solve
        keyframe moves from the stale initialization).

        No RNG draws; pure linear algebra on existing state (the
        module's cv-RNG-transparency contract holds).
        """
        # insertion order -> the earliest-observer tiebreak
        kf_order = {kf.frame_id: i
                    for i, kf in enumerate(self.map.keyframes)}
        reanchored = 0
        for pt in self.map.points:
            best_order, best_id = None, None
            for fid in pt.observations.keys():
                order = kf_order.get(fid)
                if order is not None and (best_order is None
                                          or order < best_order):
                    best_order, best_id = order, fid
            if best_id is None:
                continue        # no live keyframe observes it
            pre = pre_poses.get(best_id)
            kf = self.map.get_keyframe_by_id(best_id)
            if pre is None or kf is None:
                continue
            delta = kf.pose @ np.linalg.inv(pre)
            p = np.asarray(pt.pt_3d, dtype=np.float64).reshape(3)
            pt.pt_3d = delta[:3, :3] @ p + delta[:3, 3]
            reanchored += 1
        debug_log(LOG_TAG, f"Re-anchored {reanchored} map points "
                  f"with the loop correction")

    def create_constraint_from_verified_loop(self, verified_loop: LoopClosureCandidate) -> PoseGraphConstraint:
        """Convert a verified loop closure to a pose graph constraint.
        The verified candidate's relative_pose IS the edge measurement
        M(query, match) = pose_q_pnp @ inv(pose_match) (see
        _estimate_loop_constraint)."""
        return PoseGraphConstraint(
            from_keyframe_id=verified_loop.query_keyframe_id,
            to_keyframe_id=verified_loop.match_keyframe_id,
            relative_pose=verified_loop.relative_pose
        )
