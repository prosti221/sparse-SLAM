from collections import defaultdict
from itertools import dropwhile, islice
from typing import List, Tuple, Set
from uuid import UUID
import numpy as np
from slam.utils.logger import info_log, debug_log, error_log, warning_log
from slam.utils.constants import *
from slam.utils import constants as const
from slam.utils import telemetry
from slam.core.point import Point
from slam.core.frame import Frame
from slam.core.observation import Observation
from slam.ba.bundle_adjustment_g2o import G2OBundleAdjustment
from slam.loop_closure import LoopClosureCandidate, KeyframeFeatureDatabase, LoopClosureDetector, PoseGraphOptimizer

LOG_TAG = 'Map'


class Map:
    def __init__(self):
        self.points: List[Point] = []
        self.keyframes: List[Frame] = []
        self.point_coords: Set[Tuple[float, float, float]] = set()

        self.points_by_id: dict[UUID, Point] = {}
        self.keyframes_by_id: dict[UUID, Frame] = {}

        # Covisibility graph: {keyframe_id: {keyframe_id: shared_observations_count}}
        self.covisibility_graph: dict[UUID, dict[UUID, int]] = defaultdict(
            lambda: defaultdict(int))

        # Cache for covisibility scores to avoid recomputation
        self._covisibility_score_cache: dict[Tuple[UUID, UUID], int] = {}

        # Tracking quality history
        self.tracking_quality_history: dict[UUID, float] = {}

        # Recovery state tracking
        self.consecutive_low_quality_frames = 0

        # Bundle adjustment optimizer
        self.g2o_optimizer = G2OBundleAdjustment(self)

        # Loop closure components
        self.feature_db = KeyframeFeatureDatabase()
        self.loop_detector = LoopClosureDetector(self, self.feature_db)
        self.pose_graph_optimizer = PoseGraphOptimizer(self)

    def add_points(self, points: List[Point]):
        new_points = []

        for pt in points:
            coord_key = tuple(np.round(pt.pt_3d, 3))
            if coord_key not in self.point_coords:
                self.point_coords.add(coord_key)
                new_points.append(pt)
                self.points_by_id[pt.point_id] = pt

                # Update covisibility graph for this point
                self._add_point_to_covisibility_graph(pt)
            else:
                warning_log(
                    LOG_TAG, f"Point with coordinates {coord_key} already exists. Skipping.")

        self.points.extend(new_points)

    def add_keyframe(self, kf: Frame):
        if kf.frame_id in self.keyframes_by_id:
            warning_log(
                LOG_TAG, f"Keyframe {kf.frame_id} already exists. Skipping.")
            return

        kf.is_keyframe = True
        self.keyframes.append(kf)
        self.keyframes_by_id[kf.frame_id] = kf

        # Store features for loop closure detection
        _, descriptors = kf.get_keypoints_descriptors()
        if descriptors is not None:
            self.feature_db.add_keyframe_features(kf.frame_id, descriptors)

        # Initialize covisibility connections for new keyframe
        self.covisibility_graph[kf.frame_id] = defaultdict(int)

        # Update covisibility graph based on existing points observed by this keyframe
        for point_id in kf.observed_points_list:
            if point_id in self.points_by_id:
                point = self.points_by_id[point_id]
                self._add_keyframe_to_point_covisibility(kf.frame_id, point)

    def optimize(self, enable_ba=True):
        # First refine the current keyframe pose with PnP
        self.g2o_optimizer.refine_pose_pnp()

        if enable_ba:
            success = False
            perform_global_ba = self.should_perform_global_bundle_adjustment(
                const.GLOBAL_BUNDLE_ADJUSTMENT_KEYFRAME_INTERVAL)
            if perform_global_ba:
                telemetry.event("ba_global_call")
                success = self.g2o_optimizer.global_bundle_adjustment()
            else:
                telemetry.event("ba_local_call")
                success = self.g2o_optimizer.local_bundle_adjustment(
                    self.keyframes[-1].frame_id,
                    window_size=const.LOCAL_MAP_WINDOW_SIZE,
                    fix_points=False
                )

            telemetry.event("ba_success" if success else "ba_failure")
            if success:
                debug_log(
                    LOG_TAG, "BA was successful, pruning outlier points...")
            else:
                warning_log(LOG_TAG, "BA failed")

        self.prune_outlier_points()

    def _add_point_to_covisibility_graph(self, point: Point):
        observing_kfs = point.observing_keyframes

        # Connect all pairs of keyframes that observe this point
        for i, kf1_id in enumerate(observing_kfs):
            for kf2_id in observing_kfs[i+1:]:
                # Increment connection strength
                self.covisibility_graph[kf1_id][kf2_id] += 1
                self.covisibility_graph[kf2_id][kf1_id] += 1

                # Invalidate cached scores for these keyframes
                self._invalidate_covisibility_cache(kf1_id, kf2_id)

    def _add_keyframe_to_point_covisibility(self, new_kf_id: UUID, point: Point):
        observing_kfs = point.observing_keyframes

        for other_kf_id in observing_kfs:
            if other_kf_id != new_kf_id:
                self.covisibility_graph[new_kf_id][other_kf_id] += 1
                self.covisibility_graph[other_kf_id][new_kf_id] += 1

                # Invalidate cached scores
                self._invalidate_covisibility_cache(new_kf_id, other_kf_id)

    def _remove_point_from_covisibility_graph(self, point: Point):
        observing_kfs = point.observing_keyframes

        # Remove connections between all pairs of keyframes that observe this point
        for i, kf1_id in enumerate(observing_kfs):
            # Remove point from kf observation (only for existing
            # keyframes). NOTE (phase 2g): a direct del leaves STALE
            # keypoint_to_point_map entries that _filter_duplicate_
            # keypoints can later match against, resurrecting pruned
            # points as phantom observations. Cleaning them (via
            # remove_point_observation) was MEASURED AND REJECTED: the
            # stale entries act as a de facto "don't re-triangulate
            # where a point already died" guard, and clearing them floods
            # fast-rotation sequences with junk re-triangulations
            # (orbit surf 50.8 -> 6.1, a baseline failure). The
            # phantom-observation behavior stays until the map-management
            # item replaces it with an explicit redetection policy.
            if kf1_id in self.keyframes_by_id:
                del self.keyframes_by_id[kf1_id].observed_points[point.point_id]

            for kf2_id in observing_kfs[i+1:]:
                # Only update covisibility for keyframes that exist
                if kf1_id not in self.keyframes_by_id or kf2_id not in self.keyframes_by_id:
                    continue

                # Decrement connection strength
                self.covisibility_graph[kf1_id][kf2_id] -= 1
                self.covisibility_graph[kf2_id][kf1_id] -= 1

                # Remove connection if it reaches zero
                if self.covisibility_graph[kf1_id][kf2_id] <= 0:
                    del self.covisibility_graph[kf1_id][kf2_id]
                if self.covisibility_graph[kf2_id][kf1_id] <= 0:
                    del self.covisibility_graph[kf2_id][kf1_id]

                # Invalidate cached scores
                self._invalidate_covisibility_cache(kf1_id, kf2_id)

    def _invalidate_covisibility_cache(self, kf1_id: UUID, kf2_id: UUID):
        cache_key1 = (kf1_id, kf2_id)
        cache_key2 = (kf2_id, kf1_id)

        if cache_key1 in self._covisibility_score_cache:
            del self._covisibility_score_cache[cache_key1]
        if cache_key2 in self._covisibility_score_cache:
            del self._covisibility_score_cache[cache_key2]

    def get_local_keyframes(self, reference_frame_id: UUID, window_size: int = 5) -> List[Frame]:
        if reference_frame_id not in self.keyframes_by_id:
            error_log(
                LOG_TAG, f"Reference keyframe {reference_frame_id} not found")
            return []

        return self.keyframes[-window_size:]

    def get_local_points(self, local_keyframes: List[Frame]) -> List[Point]:
        local_points = set()
        for kf in local_keyframes:
            for obs in kf.observed_points.values():
                local_points.add(obs.point)
        return local_points

    def get_observations(self, local_keyframes: List[Frame]) -> List[Observation]:
        return [obs for kf in local_keyframes for obs in kf.observed_points.values()]

    def prune_outlier_points(self):
        redundant_points = bad_points = 0
        for idx in reversed(range(len(self.points))):
            point = self.points[idx]

            # Rethink this later...
            """
            #observing_kfs = set(point.observing_keyframes)
            #common_kfs = observing_kfs.intersection(local_kf_ids)

            # Look for outdated points that have lost relevance
            if len(common_kfs) == 0 and point.num_observations < MINIMUM_OBSERVATIONS_FOR_POINT and False:
                if self.remove_point_by_index(idx):
                    redundant_points += 1
                    continue
            """

            # Look for bad points that have a low avg reprojection error
            if not point.is_good:
                if self.remove_point_by_index(idx):
                    bad_points += 1

        telemetry.event("points_pruned", bad_points)
        debug_log(
            LOG_TAG, f"Removed {bad_points} bad points, and {redundant_points} redundant points")

    def remove_point_by_index(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self.points):
            error_log(LOG_TAG, f"Index {idx} out of range for points list")
            return False

        point = self.points[idx]
        point_id = point.point_id

        # Remove from covisibility graph
        self._remove_point_from_covisibility_graph(point)

        # Remove from points list
        del self.points[idx]

        # Remove from points_by_id dictionary
        if point_id in self.points_by_id:
            del self.points_by_id[point_id]

        # Remove coordinate from point_coords set
        coord_key = tuple(np.round(point.pt_3d, 3))
        if coord_key in self.point_coords:
            self.point_coords.remove(coord_key)

        return True

    def _compute_covisibility_score(self, kf1: Frame, kf2: Frame) -> int:
        cache_key = (kf1.frame_id, kf2.frame_id)

        if cache_key in self._covisibility_score_cache:
            debug_log(LOG_TAG, f"Cache hit for covisibility score!")
            return self._covisibility_score_cache[cache_key]

        # Use the precomputed count from the covisibility graph
        shared_count = self.covisibility_graph[kf1.frame_id].get(
            kf2.frame_id, 0)

        # Cache the result
        self._covisibility_score_cache[cache_key] = shared_count
        self._covisibility_score_cache[(
            kf2.frame_id, kf1.frame_id)] = shared_count

        return shared_count

    def should_perform_global_bundle_adjustment(self, interval: int) -> bool:
        # Phase 2a (finding 13): interval-triggered global BA is DISABLED
        # (interval <= 0). Measured A/B across all four sequences (seed 42,
        # full record in benchmarks/IMPROVEMENT.md): turning it off IMPROVED
        # three of four (translation ATE 0.0845 -> 0.0181, orbit 1.72 ->
        # 1.04, forward_rotation 0.076 -> 0.065) at a large speed win, and
        # cost only +0.005 ATE on tum_xyz (0.177 -> 0.182, within
        # baseline). The local BA anchors window poses to FIXED boundary
        # keyframes, keeping corrections local; a global re-solve trades
        # accuracy in well-constrained regions to smooth systematic error
        # across the whole map, on a fixed cadence with no evidence of
        # need. Global BA remains available (and justified) on loop
        # closure - Phase 3 triggers it on loop events, not on a timer.
        return interval > 0 and len(self.keyframes) % interval == 0

    def get_keyframe_by_id(self, frame_id: UUID) -> Frame:
        return self.keyframes_by_id.get(frame_id, None)

    def get_point_by_id(self, point_id: UUID) -> Point:
        return self.points_by_id.get(point_id, None)

    def update_tracking_quality(self):
        self.tracking_quality_history = {
            kf_id: kf.tracking_quality for kf_id, kf in self.keyframes_by_id.items()}

        # Update consecutive low quality frame counter for recovery detection
        if self.avg_tracking_quality < const.RELOCALIZATION_QUALITY_THRESHOLD:
            self.consecutive_low_quality_frames += 1
        else:
            self.consecutive_low_quality_frames = 0

    def needs_recovery(self) -> bool:
        return (self.avg_tracking_quality < const.RELOCALIZATION_QUALITY_THRESHOLD and
                self.consecutive_low_quality_frames >= const.RELOCALIZATION_FRAME_COUNT)

    def on_relocalization_successful(self, reference_keyframe_id: UUID):
        self.consecutive_low_quality_frames = 0
        # Trigger local BA to refine the area
        self.optimize(enable_ba=True)
        info_log(
            LOG_TAG, f"Successfully relocalized to keyframe {reference_keyframe_id}")

    def reset(self):
        info_log(LOG_TAG, "Resetting map - clearing all data")

        # Clear all collections
        self.points.clear()
        self.keyframes.clear()
        self.point_coords.clear()
        self.points_by_id.clear()
        self.keyframes_by_id.clear()
        self.covisibility_graph.clear()
        self._covisibility_score_cache.clear()
        self.tracking_quality_history.clear()

        # Reset state counters
        self.consecutive_low_quality_frames = 0

        # Reinitialize components that might need it
        self.feature_db = KeyframeFeatureDatabase()
        self.loop_detector = type(self.loop_detector)(self, self.feature_db)
        self.pose_graph_optimizer = type(self.pose_graph_optimizer)(self)

        debug_log(LOG_TAG, "Map reset complete")

    def remove_keyframes_after(self, reference_keyframe_id: UUID):
        """Remove every keyframe inserted AFTER the reference keyframe
        (aggressive relocalization drops the diverged branch it just
        recovered from). Phase 2h repair: this used to compare frame_id
        UUIDs with <= - frame ids are random uuid4 values, so their
        ordering is meaningless and the backwards walk removed a RANDOM
        suffix of the keyframe list. The keyframes list is insertion
        ordered, so position IS the temporal order."""
        try:
            ref_idx = next(i for i, kf in enumerate(self.keyframes)
                           if kf.frame_id == reference_keyframe_id)
        except StopIteration:
            return

        for kf in list(self.keyframes[ref_idx + 1:]):
            debug_log(
                LOG_TAG, f"Removing bad keyframe {kf.frame_id} after relocalization")
            self.remove_keyframe_by_id(kf.frame_id)

    def detect_loop_closures(self, current_frame: Frame) -> List['LoopClosureCandidate']:
        return self.loop_detector.detect_potential_loops(current_frame)

    def get_covisibility_keyframes(self, keyframe_id: UUID, min_shared_points: int = 15) -> List[Frame]:
        if keyframe_id not in self.covisibility_graph:
            return []

        covisible_kfs = []
        for kf_id, shared_count in self.covisibility_graph[keyframe_id].items():
            if shared_count >= min_shared_points and kf_id in self.keyframes_by_id:
                covisible_kfs.append(
                    (self.keyframes_by_id[kf_id], shared_count))

        # Sort by shared points count (descending)
        covisible_kfs.sort(key=lambda x: x[1], reverse=True)
        return [kf for kf, _ in covisible_kfs]

    def cleanup_covisibility_cache(self):
        self._covisibility_score_cache.clear()
        debug_log(LOG_TAG, "Cleaned up covisibility score cache")

    def remove_keyframe_by_index(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self.keyframes):
            error_log(LOG_TAG, f"Index {idx} out of range for keyframes list")
            return False

        keyframe = self.keyframes[idx]
        return self._remove_keyframe(keyframe, idx)

    def remove_keyframe_by_id(self, keyframe_id: UUID) -> bool:
        if keyframe_id not in self.keyframes_by_id:
            error_log(LOG_TAG, f"Keyframe {keyframe_id} not found")
            return False

        keyframe = self.keyframes_by_id[keyframe_id]

        # Find the index of this keyframe
        try:
            idx = self.keyframes.index(keyframe)
        except ValueError:
            error_log(
                LOG_TAG, f"Keyframe {keyframe_id} not found in keyframes list")
            return False

        return self._remove_keyframe(keyframe, idx)

    def _remove_keyframe(self, keyframe: Frame, idx: int) -> bool:
        """Phase 2h machinery repair. Not invoked by the default
        pipeline today (the redundancy-culling hook lands with step 3);
        the live caller is the aggressive-relocalization reset path via
        remove_keyframes_after. Repairs vs the draft:

        - The orphan threshold is the MINIMUM_OBSERVATIONS_FOR_POINT
          constant (was the literal 2 beside an "or whatever minimum
          threshold you use" placeholder), counted over KEYFRAME
          observations only: point.observations also accumulates
          never-promoted tracker frames (their poses are never
          optimized, and they are never removed), and those must not
          mask a point that has lost its keyframe anchors - keyframes
          are the only source of BA edges and covisibility.
        - The KeyframeFeatureDatabase entry is removed (a culled
          keyframe must stop being a loop-closure candidate).
        - Orphan removal is a single reversed pass (was list.index per
          orphan - O(n^2) on a growing map).
        - The 2f persistent local-BA graph needs NO special handling:
          its per-call reconcile rebuilds target membership from the
          current map state, so this keyframe's vertex (and the
          orphaned points' vertices) cascade-drop on the next local BA.
        """
        keyframe_id = keyframe.frame_id

        # Remove observations of this keyframe from all points
        # (point-side is the authoritative registry - phase 2g; the
        # keyframe's own frame-side entries, including dedup-only
        # matches, die with it). Collect points that fall below the
        # keyframe-observation minimum.
        orphan_ids = set()
        for point in self.points:
            if keyframe_id in point.observations:
                point.remove_observation(keyframe_id)
                kf_observers = sum(
                    1 for fid in point.observations
                    if fid in self.keyframes_by_id)
                if kf_observers < const.MINIMUM_OBSERVATIONS_FOR_POINT:
                    orphan_ids.add(point.point_id)

        # Remove points that no longer have enough keyframe
        # observations (single reversed pass: del by index only shifts
        # the tail). This also drops them from the covisibility graph,
        # points_by_id and point_coords via remove_point_by_index.
        removed_points_count = 0
        for point_idx in reversed(range(len(self.points))):
            if self.points[point_idx].point_id in orphan_ids:
                if self.remove_point_by_index(point_idx):
                    removed_points_count += 1

        # Remove from covisibility graph (the keyframe's own edges go
        # wholesale; shared counts among survivors were already
        # decremented for the removed points above)
        self._remove_keyframe_from_covisibility_graph(keyframe_id)

        # Remove from keyframes list
        del self.keyframes[idx]

        # Remove from keyframes_by_id dictionary
        if keyframe_id in self.keyframes_by_id:
            del self.keyframes_by_id[keyframe_id]

        # Remove from tracking quality history
        if keyframe_id in self.tracking_quality_history:
            del self.tracking_quality_history[keyframe_id]

        # Stop being a loop-closure candidate
        self.feature_db.remove_keyframe_features(keyframe_id)

        telemetry.event("kf_culled")
        telemetry.event("points_culled", removed_points_count)
        debug_log(
            LOG_TAG, f"Removed keyframe {keyframe_id} and {removed_points_count} associated points")
        return True

    def cull_redundant_keyframe(self) -> bool:
        """Phase 2h step 3: bound map growth by removing redundant
        keyframes (ORB-SLAM-style). Called once per keyframe insertion,
        AFTER optimize/triangulate, so the candidate scan sees the full
        fresh map. A candidate is redundant when at least
        CULL_REDUNDANCY_RATIO of its point-side map points are observed
        by at least CULL_MIN_OTHER_OBSERVERS OTHER keyframes - the
        information it carries is already carried by its neighbors. At
        most ONE keyframe (the oldest redundant candidate) is culled
        per insertion. Protected, never culled:

        - keyframes[0] (the gauge anchor),
        - the local-BA window (LOCAL_MAP_WINDOW_SIZE most recent),
        - the relocation window (RELOCALIZATION_WINDOW most recent -
          normal relocalization matches features against these;
          shrinking that pool mid-run would blind the recovery path).

        Registry semantics (phases 2g/2h lessons): point-side
        observations only (the frame-side registry holds dedup-only
        matches and phantom pruned points), and KEYFRAME observers only
        (never-promoted tracker frames are never optimized and never
        removed - they must not count as anchors).
        """
        protected_tail = max(const.LOCAL_MAP_WINDOW_SIZE,
                             const.RELOCALIZATION_WINDOW)
        n = len(self.keyframes)
        # Candidates: keyframes[1:n-protected_tail] (oldest first).
        end = n - protected_tail
        if end <= 1:
            return False
        candidates = {kf.frame_id for kf in self.keyframes[1:end]}

        # One pass over live points: bucket each point by which
        # candidate keyframes observe it (point-side registry), and
        # count the point's OTHER keyframe observers once (a point
        # needs >= CULL_MIN_OTHER_OBSERVERS+1 keyframe observers in
        # total to be redundant evidence for anyone).
        candidate_points: dict[UUID, list] = defaultdict(list)
        for point in self.points:
            kf_observers = [fid for fid in point.observations
                            if fid in self.keyframes_by_id]
            n_others = len(kf_observers) - 1
            for fid in kf_observers:
                if fid in candidates:
                    candidate_points[fid].append((point, n_others))

        for kf in self.keyframes[1:end]:
            bucket = candidate_points.get(kf.frame_id)
            if not bucket:
                # No point-side observations (all its points pruned):
                # ratio undefined. Conservative - skip (it may still
                # serve relocalization matching).
                continue
            redundant = sum(1 for _, n_others in bucket
                             if n_others >= const.CULL_MIN_OTHER_OBSERVERS)
            if redundant >= const.CULL_REDUNDANCY_RATIO * len(bucket):
                debug_log(
                    LOG_TAG, f"Culling redundant keyframe {kf.frame_id} "
                    f"({redundant}/{len(bucket)} points redundantly observed)")
                return self.remove_keyframe_by_id(kf.frame_id)
        return False

    def _remove_keyframe_from_covisibility_graph(self, keyframe_id: UUID):
        if keyframe_id not in self.covisibility_graph:
            return

        # Get all keyframes connected to this one
        connected_keyframes = list(self.covisibility_graph[keyframe_id].keys())

        # Remove this keyframe from all other keyframes' connections
        for other_kf_id in connected_keyframes:
            if other_kf_id in self.covisibility_graph:
                if keyframe_id in self.covisibility_graph[other_kf_id]:
                    del self.covisibility_graph[other_kf_id][keyframe_id]

            # Invalidate cache entries involving this keyframe
            self._invalidate_covisibility_cache(keyframe_id, other_kf_id)

        # Remove the keyframe's entry from the covisibility graph
        del self.covisibility_graph[keyframe_id]

    # Properties
    @property
    def cur_keyframe(self):
        if len(self.keyframes) > 0:
            return self.keyframes[-1]
        return None

    @property
    def prev_keyframe(self):
        if len(self.keyframes) > 1:
            return self.keyframes[-2]
        return None

    @property
    def avg_tracking_quality(self) -> float:
        if not self.tracking_quality_history:
            return 0.0

        local_window_values = list(
            self.tracking_quality_history.values())[-const.LOCAL_MAP_WINDOW_SIZE:]

        total_quality = sum(local_window_values) / len(local_window_values)
        return total_quality
