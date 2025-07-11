from collections import defaultdict
import numpy as np
from logger import debug_log, error_log

LOG_TAG = 'Map'


class Map:
    def __init__(self):
        self.points = []       # All known 3D points
        self.point_coords = set()  # Track coordinates to avoid duplicates
        self.keyframes = []    # All added keyframes

        # Enhanced tracking for bundle adjustment
        self.points_by_id = {}  # {point_id: Point}
        self.keyframes_by_id = {}  # {keyframe_id: Frame}

        # Covisibility graph for efficient local BA
        self.covisibility_graph = defaultdict(
            set)  # {kf_id: set of connected kf_ids}

    def add_points(self, points):
        new_points = []

        for pt in points:
            # Create a hashable key from coordinates
            coord_key = tuple(np.round(pt.pt_3d, 3))
            if coord_key not in self.point_coords:
                self.point_coords.add(coord_key)
                new_points.append(pt)

                # Add to tracking dictionaries
                self.points_by_id[pt.point_id] = pt

        self.points.extend(new_points)

        # Update covisibility graph
        self._update_covisibility_graph()

        return new_points

    def add_keyframe(self, kf):
        self.keyframes.append(kf)
        self.keyframes_by_id[kf.frame_id] = kf

        # Update covisibility graph
        self._update_covisibility_graph()

    def get_local_keyframes(self, reference_frame_id, window_size=5):
        if reference_frame_id not in self.keyframes_by_id:
            error_log(
                LOG_TAG, f"Reference keyframe {reference_frame_id} not found")
            return []

        # Get covisible keyframes
        covisible_kfs = self.covisibility_graph[reference_frame_id]

        # Sort by covisibility strength (number of shared observations)
        keyframe_scores = []
        ref_kf = self.keyframes_by_id[reference_frame_id]

        for kf_id in covisible_kfs:
            if kf_id in self.keyframes_by_id:
                kf = self.keyframes_by_id[kf_id]
                score = self._compute_covisibility_score(ref_kf, kf)
                keyframe_scores.append((score, kf))

        # Sort by score (descending) and take top window_size
        keyframe_scores.sort(key=lambda x: x[0], reverse=True)
        local_keyframes = [kf for _, kf in keyframe_scores[:window_size]]

        # Always include reference keyframe
        if ref_kf not in local_keyframes:
            local_keyframes.append(ref_kf)

        return local_keyframes

    def get_local_points(self, local_keyframes, min_observations=2):
        local_kf_ids = {kf.frame_id for kf in local_keyframes}
        local_points = []

        for point in self.points:
            # Check if point is observed by any local keyframe
            observing_kfs = set(point.get_observing_keyframes())
            common_kfs = observing_kfs.intersection(local_kf_ids)

            if len(common_kfs) >= min_observations:
                local_points.append(point)

        return local_points

    def get_observations(self, local_keyframes, local_points):
        observations = []

        # Create index mappings
        kf_id_to_idx = {kf.frame_id: i for i,
                        kf in enumerate(local_keyframes)}
        point_id_to_idx = {pt.point_id: i for i, pt in enumerate(local_points)}

        for point in local_points:
            point_idx = point_id_to_idx[point.point_id]

            for kf_id, (keypoint_idx, pt_2d) in point.observations.items():
                if kf_id in kf_id_to_idx:
                    kf_idx = kf_id_to_idx[kf_id]
                    observations.append((point_idx, kf_idx, pt_2d))

        return observations

    def remove_outlier_points(self, outlier_threshold=3.0, min_observations=3):
        good_points = []
        removed_count = 0

        for point in self.points:
            if point.is_good_point(min_observations, outlier_threshold):
                good_points.append(point)
            else:
                removed_count += 1
                # Remove from tracking
                if point.point_id in self.points_by_id:
                    del self.points_by_id[point.point_id]

        self.points = good_points

        # Update coordinate set
        self.point_coords = set()
        for pt in self.points:
            coord_key = tuple(np.round(pt.pt_3d, 3))
            self.point_coords.add(coord_key)

        debug_log(LOG_TAG, f"Removed {removed_count} outlier points")

    def _update_covisibility_graph(self):
        self.covisibility_graph.clear()

        # For each point, connect keyframes that observe it
        for point in self.points:
            observing_kfs = point.get_observing_keyframes()

            # Connect all pairs of keyframes that observe this point
            for i, kf1_id in enumerate(observing_kfs):
                for kf2_id in observing_kfs[i+1:]:
                    self.covisibility_graph[kf1_id].add(kf2_id)
                    self.covisibility_graph[kf2_id].add(kf1_id)

    def _compute_covisibility_score(self, kf1, kf2):
        shared_points = 0

        for point in self.points:
            if (point.is_observed_by(kf1.frame_id) and
                    point.is_observed_by(kf2.frame_id)):
                shared_points += 1

        return shared_points

    def should_perform_global_bundle_adjustment(self, interval):
        return len(self.keyframes) % interval == 0

    def get_keyframe_by_id(self, frame_id):
        return self.keyframes_by_id.get(frame_id, None)

    def get_point_by_id(self, point_id):
        return self.points_by_id.get(point_id, None)

    def get_statistics(self):
        stats = {
            'total_keyframes': len(self.keyframes),
            'total_points': len(self.points),
            'avg_observations_per_point': np.mean([p.num_observations for p in self.points]) if self.points else 0,
            'avg_covisibility_connections': np.mean([len(connections) for connections in self.covisibility_graph.values()]) if self.covisibility_graph else 0
        }
        return stats
