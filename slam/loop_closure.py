"""
Loop Closure Detection and Optimization for SLAM
"""
from typing import List, Tuple, Dict, Optional, NamedTuple, TYPE_CHECKING
from uuid import UUID
import numpy as np
import cv2 as cv
from slam.core.frame import Frame
if TYPE_CHECKING:
    from slam.core.map import Map
from slam.utils.logger import debug_log, warning_log, error_log

LOG_TAG = 'LoopClosure'


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

    def get_keyframe_features(self, keyframe_id: UUID) -> Optional[np.ndarray]:
        """Retrieve features for a keyframe"""
        return self.feature_storage.get(keyframe_id)

    def find_similar_keyframes(self, query_descriptors: np.ndarray,
                               exclude_recent: int = 20,
                               similarity_threshold: float = 0.1) -> List[LoopClosureCandidate]:
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

        # Exclude recent keyframes
        candidate_keyframes = self.keyframe_order[:-exclude_recent]

        candidates = []
        for kf_id in candidate_keyframes:
            kf_descriptors = self.feature_storage.get(kf_id)
            if kf_descriptors is None:
                continue

            # Compute similarity score (simple ratio of matched features)
            similarity = self._compute_similarity_score(
                query_descriptors, kf_descriptors)

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

    def _compute_similarity_score(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """
        Compute similarity between two feature sets using multiple metrics.
        Returns a score between 0-1 where higher values indicate more similar scenes.
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return 0.0

        try:
            # Use BFMatcher with cross-check for bidirectional consistency
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
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
            distance_quality = max(0, 1 - (avg_distance / 60.0))

            # Combine metrics: prioritize match ratio but weight by quality
            # High match ratio with poor distances = coincidental matches
            similarity = match_ratio * (0.7 + 0.3 * distance_quality)

            return min(similarity, 1.0)

        except Exception as e:
            warning_log(LOG_TAG, f"Error computing similarity: {e}")
            return 0.0

    def cleanup_old_features(self, keep_recent: int = 100):
        """Remove features from old keyframes to save memory"""
        if len(self.keyframe_order) <= keep_recent:
            return

        # Remove oldest keyframes beyond the keep limit
        to_remove = self.keyframe_order[:-keep_recent]
        for kf_id in to_remove:
            if kf_id in self.feature_storage:
                del self.feature_storage[kf_id]

        self.keyframe_order = self.keyframe_order[-keep_recent:]
        debug_log(
            LOG_TAG, f"Cleaned up features, keeping {len(self.keyframe_order)} recent keyframes")


class LoopClosureDetector:
    """
    Main orchestrator for loop closure detection and verification.
    """

    def __init__(self, map: "Map", feature_db: KeyframeFeatureDatabase):
        self.map = map
        self.feature_db = feature_db

        # Configuration parameters
        self.similarity_threshold = 0.3  # Minimum similarity for candidate
        self.min_temporal_separation = 20  # Minimum frames between loop keyframes
        self.max_candidates = 5  # Maximum candidates to verify
        # Minimum matches for geometric verification
        self.min_matches_for_verification = 15

    def detect_potential_loops(self, current_frame: Frame) -> List[LoopClosureCandidate]:
        """
        Main entry point for loop detection.
        Called periodically during tracking.
        """
        # Get features from current frame (not yet a keyframe)
        _, descriptors = current_frame.get_keypoints_descriptors()
        if descriptors is None or len(descriptors) == 0:
            return []

        # Find similar keyframes
        candidates = self.feature_db.find_similar_keyframes(
            descriptors,
            exclude_recent=self.min_temporal_separation,
            similarity_threshold=self.similarity_threshold
        )

        # Limit number of candidates
        candidates = candidates[:self.max_candidates]

        # Set query keyframe (will be set when frame becomes keyframe)
        for cand in candidates:
            # Create new candidate with current frame as query
            # Note: current_frame.frame_id will be set when it becomes a keyframe
            pass

        debug_log(
            LOG_TAG, f"Detected {len(candidates)} potential loop closures")
        return candidates

    def verify_loop_candidates(self, candidates: List[LoopClosureCandidate]) -> List[LoopClosureCandidate]:
        """
        Verify candidates using geometric consistency checks.
        """
        verified_loops = []

        for candidate in candidates:
            # Get the keyframes
            query_kf = self.map.get_keyframe_by_id(candidate.query_keyframe_id)
            match_kf = self.map.get_keyframe_by_id(candidate.match_keyframe_id)

            if query_kf is None or match_kf is None:
                continue

            # Perform geometric verification
            if self._verify_geometric_consistency(query_kf, match_kf):
                # Estimate relative pose
                relative_pose = self._estimate_relative_pose(
                    query_kf, match_kf)
                if relative_pose is not None:
                    verified_candidate = LoopClosureCandidate(
                        query_keyframe_id=candidate.query_keyframe_id,
                        match_keyframe_id=candidate.match_keyframe_id,
                        similarity_score=candidate.similarity_score,
                        geometric_verified=True,
                        relative_pose=relative_pose
                    )
                    verified_loops.append(verified_candidate)

        debug_log(LOG_TAG, f"Verified {len(verified_loops)} loop closures")
        return verified_loops

    def _verify_geometric_consistency(self, kf1: Frame, kf2: Frame) -> bool:
        """
        Check if two keyframes are geometrically consistent using multi-view geometry.
        Performs comprehensive verification including:
        - Feature matching with RANSAC outlier rejection
        - Essential matrix estimation and validation
        - Pose consistency checking
        - 3D point triangulation validation
        """
        try:
            # Step 1: Feature matching with geometric model selection
            from slam.features.matcher import match_features
            matches, Rt = match_features(kf1, kf2, "ORB")

            if len(matches) < self.min_matches_for_verification:
                debug_log(
                    LOG_TAG, f"Insufficient matches for verification: {len(matches)}")
                return False

            # Step 2: Verify pose estimation was successful
            if Rt is None:
                debug_log(LOG_TAG, "Pose estimation failed during verification")
                return False

            # Step 3: Validate the relative pose makes geometric sense
            if not self._validate_relative_pose(Rt):
                debug_log(LOG_TAG, "Relative pose validation failed")
                return False

            # Step 4: Check triangulation consistency
            if not self._validate_triangulation_consistency(kf1, kf2, matches, Rt):
                debug_log(LOG_TAG, "Triangulation consistency check failed")
                return False

            debug_log(
                LOG_TAG, f"Geometric verification passed for keyframes {kf1.frame_id} and {kf2.frame_id}")
            return True

        except Exception as e:
            warning_log(LOG_TAG, f"Geometric verification failed: {e}")
            return False

    def _estimate_relative_pose(self, kf1: Frame, kf2: Frame) -> Optional[np.ndarray]:
        """
        Estimate relative pose between two keyframes.
        """
        try:
            from slam.features.matcher import match_features
            matches, Rt = match_features(kf1, kf2, "ORB")

            if Rt is not None:
                return Rt
            else:
                # Fallback pose estimation
                return self._fallback_pose_estimation(kf1, kf2)

        except Exception as e:
            warning_log(LOG_TAG, f"Pose estimation failed: {e}")
            return None

    def _validate_relative_pose(self, relative_pose: np.ndarray) -> bool:
        """
        Validate that the relative pose makes geometric sense.
        Checks for reasonable translation and rotation magnitudes.
        """
        if relative_pose is None or relative_pose.shape != (4, 4):
            return False

        # Extract rotation and translation
        R = relative_pose[:3, :3]
        t = relative_pose[:3, 3]

        # Check rotation matrix validity (should be orthogonal)
        R_check = np.dot(R, R.T)
        rotation_valid = np.allclose(R_check, np.eye(3), atol=1e-6)

        if not rotation_valid:
            debug_log(LOG_TAG, "Invalid rotation matrix in relative pose")
            return False

        # Check translation magnitude (should be reasonable, not too large or zero)
        translation_magnitude = np.linalg.norm(t)
        translation_valid = 0.01 < translation_magnitude < 50.0  # 1cm to 50m range

        if not translation_valid:
            debug_log(
                LOG_TAG, f"Invalid translation magnitude: {translation_magnitude}")
            return False

        return True

    def _validate_triangulation_consistency(self, kf1: Frame, kf2: Frame,
                                            matches: List, relative_pose: np.ndarray) -> bool:
        """
        Validate triangulation consistency by checking reprojection errors
        and 3D point quality for a subset of matched features.
        """
        try:
            # Sample a subset of matches for validation (don't check all to save time)
            sample_size = min(20, len(matches))
            sample_matches = matches[:sample_size]

            # Get corresponding points
            pts1_indices = [m.queryIdx for m in sample_matches]
            pts2_indices = [m.trainIdx for m in sample_matches]

            pts1_norm = kf1.kp_pts_norm[pts1_indices]
            pts2_norm = kf2.kp_pts_norm[pts2_indices]

            # Triangulate sample points
            points_4d = self._triangulate_points(
                pts1_norm, pts2_norm, relative_pose)
            if points_4d is None:
                return False

            # Convert to 3D and check validity
            points_3d = points_4d[:, :3] / points_4d[:, 3:]

            # Check that points are in front of both cameras
            points_in_front = self._check_points_in_front_of_cameras(
                points_3d, relative_pose)

            # Check reprojection errors
            reprojection_valid = self._check_reprojection_errors(
                kf1, kf2, points_3d, pts1_norm, pts2_norm)

            return points_in_front and reprojection_valid

        except Exception as e:
            warning_log(
                LOG_TAG, f"Triangulation consistency check failed: {e}")
            return False

    def _triangulate_points(self, pts1_norm: np.ndarray, pts2_norm: np.ndarray,
                            relative_pose: np.ndarray) -> Optional[np.ndarray]:
        """
        Triangulate 3D points from normalized correspondences and relative pose.
        """
        try:
            # Set up projection matrices
            P1 = np.eye(3, 4)  # First camera at origin
            P2 = relative_pose[:3, :]  # Second camera relative pose

            # Triangulate using OpenCV
            points_4d = cv.triangulatePoints(P1, P2, pts1_norm.T, pts2_norm.T)
            return points_4d.T  # Transpose to get points as rows

        except Exception as e:
            warning_log(LOG_TAG, f"Point triangulation failed: {e}")
            return None

    def _check_points_in_front_of_cameras(self, points_3d: np.ndarray,
                                          relative_pose: np.ndarray) -> bool:
        """
        Check that triangulated points are in front of both cameras.
        """
        try:
            # Camera 1 (origin): check Z > 0
            z1_positive = np.all(points_3d[:, 2] > 0)

            # Camera 2: transform points and check Z > 0
            R = relative_pose[:3, :3]
            t = relative_pose[:3, 3]
            points_cam2 = (R @ points_3d.T).T + t
            z2_positive = np.all(points_cam2[:, 2] > 0)

            return z1_positive and z2_positive

        except Exception as e:
            warning_log(LOG_TAG, f"Camera position check failed: {e}")
            return False

    def _check_reprojection_errors(self, kf1: Frame, kf2: Frame, points_3d: np.ndarray,
                                   pts1_norm: np.ndarray, pts2_norm: np.ndarray) -> bool:
        """
        Check reprojection errors for triangulated points.
        """
        try:
            # Reproject to both cameras
            proj1 = kf1.project_points(points_3d, normalized=True)
            proj2 = kf2.project_points(points_3d, normalized=True)

            # Check if any projections failed (NaN values from points behind camera)
            if np.any(np.isnan(proj1)) or np.any(np.isnan(proj2)):
                error_log(
                    LOG_TAG, f"Some points could not be projected (behind camera)")
                return False

            # Compute reprojection errors
            errors1 = np.linalg.norm(proj1 - pts1_norm, axis=1)
            errors2 = np.linalg.norm(proj2 - pts2_norm, axis=1)

            # Check that most points have reasonable reprojection error
            max_error = 2.0  # pixels
            good_points1 = np.sum(errors1 < max_error) / len(errors1) > 0.8
            good_points2 = np.sum(errors2 < max_error) / len(errors2) > 0.8

            return good_points1 and good_points2

        except Exception as e:
            warning_log(LOG_TAG, f"Reprojection error check failed: {e}")
            return False

    def _fallback_pose_estimation(self, kf1: Frame, kf2: Frame) -> Optional[np.ndarray]:
        """
        Fallback pose estimation using simpler methods.
        """
        # Placeholder - could use PNP or other methods
        return None


class PoseGraphConstraint:
    """Represents a constraint in the pose graph (loop closure or odometry)"""

    def __init__(self, from_keyframe_id: UUID, to_keyframe_id: UUID,
                 relative_pose: np.ndarray, information_matrix: np.ndarray):
        self.from_keyframe_id = from_keyframe_id
        self.to_keyframe_id = to_keyframe_id
        self.relative_pose = relative_pose
        self.information_matrix = information_matrix


class PoseGraphOptimizer:
    """
    Handles global pose graph optimization with loop closure constraints.
    Extends the existing BA framework.
    """

    def __init__(self, map: "Map"):
        self.map = map

    def optimize_with_loop_constraints(self, loop_constraints: List[PoseGraphConstraint]) -> bool:
        """
        Perform pose graph optimization with loop closure constraints.

        This extends the existing global BA to include loop edges.
        """
        debug_log(
            LOG_TAG, f"Starting pose graph optimization with {len(loop_constraints)} loop constraints")

        # Convert constraints to the format expected by pose graph optimization
        pose_graph_constraints = []
        for constraint in loop_constraints:
            pose_graph_constraints.append((
                constraint.from_keyframe_id,
                constraint.to_keyframe_id,
                constraint.relative_pose
            ))

        # Use the new pose graph optimization method
        return self.map.g2o_optimizer.pose_graph_optimization(pose_graph_constraints)

    def create_constraint_from_verified_loop(self, verified_loop: LoopClosureCandidate) -> PoseGraphConstraint:
        """
        Convert a verified loop closure to a pose graph constraint.
        """
        # Create information matrix (uncertainty estimate)
        # For now, use identity with some uncertainty
        information = np.eye(6) * 100.0  # High confidence

        return PoseGraphConstraint(
            from_keyframe_id=verified_loop.query_keyframe_id,
            to_keyframe_id=verified_loop.match_keyframe_id,
            relative_pose=verified_loop.relative_pose,
            information_matrix=information
        )
