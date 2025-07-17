import uuid
import numpy as np
from typing import List, Tuple, Mapping
from uuid import UUID


class Point:
    def __init__(self, pt_3d: np.ndarray, color: np.ndarray = None, descriptor: np.ndarray = None):
        self.point_id: UUID = uuid.uuid4()
        self.pt_3d: np.ndarray = pt_3d
        self.color: np.ndarray = color
        self.descriptor: np.ndarray = descriptor

        # Track which keyframes observe this point
        self.observations = {}  # {frame_id: (keypoint_idx, 2d_point)}

        # Track quality metrics
        self.num_observations = 0
        self.average_reprojection_error = 0.0
        self.is_outlier = False

    def add_observation(self, frame_id: UUID, keypoint_idx: int, pt_2d: np.ndarray) -> None:
        self.observations[frame_id] = (keypoint_idx, pt_2d)
        self.num_observations = len(self.observations)

    def remove_observation(self, frame_id: UUID):
        if frame_id in self.observations:
            del self.observations[frame_id]
            self.num_observations = len(self.observations)

    def get_observing_keyframes(self) -> List[UUID]:
        return list(self.observations.keys())

    def is_observed_by(self, frame_id: UUID) -> bool:
        return frame_id in self.observations

    def get_observation_in_keyframe(self, frame_id: UUID) -> Mapping[UUID, Tuple[int, np.ndarray]]:
        return self.observations.get(frame_id, None)

    def update_reprojection_error(self, error: float) -> None:
        if self.num_observations > 0:
            self.average_reprojection_error = (
                self.average_reprojection_error *
                (self.num_observations - 1) + error
            ) / self.num_observations

    def is_good_point(self, min_observations: int = 2, max_reproj_error: float = 2.0) -> bool:
        return (self.num_observations >= min_observations and
                self.average_reprojection_error < max_reproj_error and
                not self.is_outlier)
