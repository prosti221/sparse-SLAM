import uuid
import numpy as np


class Point:
    def __init__(self, pt_3d, color=None, descriptor=None):
        self.point_id = uuid.uuid4()
        self.pt_3d = pt_3d
        self.color = color
        self.descriptor = descriptor

        # Track which keyframes observe this point
        self.observations = {}  # {keyframe_id: (keypoint_idx, 2d_point)}

        # Track quality metrics
        self.num_observations = 0
        self.average_reprojection_error = 0.0
        self.is_outlier = False

    def add_observation(self, keyframe_id, keypoint_idx, pt_2d):
        self.observations[keyframe_id] = (keypoint_idx, pt_2d)
        self.num_observations = len(self.observations)

    def remove_observation(self, keyframe_id):
        if keyframe_id in self.observations:
            del self.observations[keyframe_id]
            self.num_observations = len(self.observations)

    def get_observing_keyframes(self):
        return list(self.observations.keys())

    def get_3d_position(self):
        return self.pt_3d.copy()

    def is_observed_by(self, keyframe_id):
        return keyframe_id in self.observations

    def get_observation_in_keyframe(self, keyframe_id):
        return self.observations.get(keyframe_id, None)

    def update_reprojection_error(self, error):
        if self.num_observations > 0:
            self.average_reprojection_error = (
                self.average_reprojection_error *
                (self.num_observations - 1) + error
            ) / self.num_observations

    def is_good_point(self, min_observations=3, max_reproj_error=2.0):
        return (self.num_observations >= min_observations and
                self.average_reprojection_error < max_reproj_error and
                not self.is_outlier)
