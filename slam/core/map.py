from collections import defaultdict
import numpy as np
from slam.utils.logger import debug_log, error_log, warning_log
from slam.utils.constants import *
from slam.core.point import Point
from slam.core.frame import Frame
from slam.ba.bundle_adjustment_g2o import G2OBundleAdjustment
from typing import List, Tuple, Set
from uuid import UUID

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
        self.g2o_optimizer = G2OBundleAdjustment(self)

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
            error_log(
                LOG_TAG, f"Keyframe {kf.frame_id} already exists. Skipping.")
            return

        kf.is_keyframe = True
        self.keyframes.append(kf)
        self.keyframes_by_id[kf.frame_id] = kf

        # Initialize covisibility connections for new keyframe
        self.covisibility_graph[kf.frame_id] = defaultdict(int)

        # Update covisibility graph based on existing points observed by this keyframe
        for point_id in kf.observed_points_list:
            if point_id in self.points_by_id:
                point = self.points_by_id[point_id]
                self._add_keyframe_to_point_covisibility(kf.frame_id, point)

        self.update_tracking_quality(kf.frame_id, kf.tracking_quality)

    def optimize(self):
        # Decide if we need to optimize globally
        # TODO: Figure out better criterias for deciding this, maybe based on the current tracking quality?
        success = False
        perform_global_ba = self.should_perform_global_bundle_adjustment(
            GLOBAL_BUNDLE_ADJUSTMENT_KEYFRAME_INTERVAL)
        if (perform_global_ba):
            success = self.g2o_optimizer.global_bundle_adjustment()
        else:
            success = self.g2o_optimizer.local_bundle_adjustment(
                self.keyframes[-1].frame_id,
                window_size=len(self.keyframes),
                fix_points=False
            )
        if success:
            debug_log(LOG_TAG, "BA was successful, pruning outlier points...")
            self.prune_outlier_points()
        else:
            warning_log(LOG_TAG, "BA failed")

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
            for kf2_id in observing_kfs[i+1:]:
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

        # Get covisible keyframes directly from the graph
        covisible_kfs = self.covisibility_graph[reference_frame_id]

        # Sort by covisibility strength (number of shared observations)
        keyframe_scores = []
        ref_kf = self.keyframes_by_id[reference_frame_id]

        for kf_id, shared_count in covisible_kfs.items():
            if kf_id in self.keyframes_by_id:
                kf = self.keyframes_by_id[kf_id]
                # Use the shared count directly instead of recomputing
                keyframe_scores.append((shared_count, kf))

        # Sort by score (descending) and take top window_size
        keyframe_scores.sort(key=lambda x: x[0], reverse=True)
        local_keyframes = [kf for _, kf in keyframe_scores[:window_size]]

        # Always include reference keyframe
        if ref_kf not in local_keyframes:
            local_keyframes.append(ref_kf)

        return local_keyframes

    def get_local_points(self, local_keyframes: List[Frame], min_observations: int = 2) -> List[Point]:
        local_kf_ids = {kf.frame_id for kf in local_keyframes}
        local_points = []

        for point in self.points:
            # Check if point is observed by any local keyframe
            observing_kfs = set(point.observing_keyframes)
            common_kfs = observing_kfs.intersection(local_kf_ids)

            if len(common_kfs) >= min_observations:
                local_points.append(point)

        return local_points

    def get_observations(
        self,
        local_keyframes: List[Frame],
        local_points: List[Point],
        normalize_points=False
    ) -> List[Tuple[int, int, np.ndarray]]:
        observations = []

        # Create index mappings
        kf_id_to_idx = {kf.frame_id: i for i, kf in enumerate(local_keyframes)}
        point_id_to_idx = {pt.point_id: i for i, pt in enumerate(local_points)}

        for point in local_points:
            point_idx = point_id_to_idx[point.point_id]

            for kf_id, (keypoint_idx, pt_2d) in point.observations.items():
                if kf_id in kf_id_to_idx:
                    kf_idx = kf_id_to_idx[kf_id]
                    if normalize_points:
                        pt_2d = self.keyframes[kf_idx].normalize_keypoint(
                            pt_2d)
                    observations.append((point_idx, kf_idx, pt_2d))

        return observations

    def prune_outlier_points(self):
        local_keyframes = self.get_local_keyframes(
            self.cur_keyframe.frame_id, LOCAL_MAP_WINDOW_SIZE)
        local_kf_ids = {kf.frame_id for kf in local_keyframes}

        redundant_points = bad_points = 0
        for idx in reversed(range(len(self.points))):
            point = self.points[idx]

            observing_kfs = set(point.observing_keyframes)
            common_kfs = observing_kfs.intersection(local_kf_ids)

            # Look for outdated points that have lost relevance
            if len(common_kfs) == 0 and point.num_observations < MINIMUM_OBSERVATIONS_FOR_POINT:
                if self.remove_point_by_index(idx):
                    redundant_points += 1
                    continue

            # Look for bad points that have a low avg reprojection error
            if not point.is_good:
                if self.remove_point_by_index(idx):
                    bad_points += 1

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
        return len(self.keyframes) % interval == 0

    def get_keyframe_by_id(self, frame_id: UUID) -> Frame:
        return self.keyframes_by_id.get(frame_id, None)

    def get_point_by_id(self, point_id: UUID) -> Point:
        return self.points_by_id.get(point_id, None)

    def update_tracking_quality(self, keyframe_id: UUID, quality: float):
        self.tracking_quality_history[keyframe_id] = quality

    def needs_recovery(self) -> bool:
        return self.avg_tracking_quality < TRACKING_QUALITY_THRESHOLD

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

    def get_statistics(self) -> dict:
        stats = {
            'total_keyframes': len(self.keyframes),
            'total_points': len(self.points),
            'avg_observations_per_point': np.mean([p.num_observations for p in self.points]) if self.points else 0,
            'avg_covisibility_connections': np.mean([len(connections) for connections in self.covisibility_graph.values()]) if self.covisibility_graph else 0,
            'avg_tracking_quality': self.avg_tracking_quality,
            'needs_recovery': self.needs_recovery(),
        }
        return stats

    def remove_keyframe_by_index(self, idx: int) -> bool:
        """Remove a keyframe by its index in the keyframes list."""
        if idx < 0 or idx >= len(self.keyframes):
            error_log(LOG_TAG, f"Index {idx} out of range for keyframes list")
            return False

        keyframe = self.keyframes[idx]
        return self._remove_keyframe(keyframe, idx)

    def remove_keyframe_by_id(self, keyframe_id: UUID) -> bool:
        """Remove a keyframe by its UUID."""
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
        """Internal method to remove a keyframe and clean up all references."""
        keyframe_id = keyframe.frame_id

        # Remove observations of this keyframe from all points
        points_to_remove = []
        for point in self.points:
            if keyframe_id in point.observations:
                point.remove_observation(keyframe_id)
                # If point has too few observations after removal, mark for deletion
                if point.num_observations < 2:  # or whatever minimum threshold you use
                    points_to_remove.append(point)

        # Remove points that no longer have enough observations
        removed_points_count = 0
        for point in points_to_remove:
            try:
                point_idx = self.points.index(point)
                if self.remove_point_by_index(point_idx):
                    removed_points_count += 1
            except ValueError:
                # Point was already removed
                continue

        # Remove from covisibility graph
        self._remove_keyframe_from_covisibility_graph(keyframe_id)

        # Remove from keyframes list
        del self.keyframes[idx]

        # Remove from keyframes_by_id dictionary
        if keyframe_id in self.keyframes_by_id:
            del self.keyframes_by_id[keyframe_id]

        # Remove from tracking quality history
        if keyframe_id in self.tracking_quality_history:
            del self.tracking_quality_history[keyframe_id]

        debug_log(
            LOG_TAG, f"Removed keyframe {keyframe_id} and {removed_points_count} associated points")
        return True

    def _remove_keyframe_from_covisibility_graph(self, keyframe_id: UUID):
        """Remove a keyframe from the covisibility graph and clean up connections."""
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
            self.tracking_quality_history.values())[-LOCAL_MAP_WINDOW_SIZE:]

        total_quality = sum(local_window_values) / len(local_window_values)
        return total_quality
