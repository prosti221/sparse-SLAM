import uuid
import numpy as np
from typing import List, Tuple, Dict
from uuid import UUID
from slam.utils.constants import MAX_SQUARED_REPROJECTION_ERROR
from slam.core.observation import Observation


class Point:
    def __init__(self, pt_3d: np.ndarray, color: np.ndarray = None, descriptor: np.ndarray = None):
        self.point_id: UUID = uuid.uuid4()
        self.pt_3d: np.ndarray = pt_3d
        self.color: np.ndarray = color
        self.descriptor: np.ndarray = descriptor

        # Track which keyframes observe this point
        self.observations: Dict[UUID, Observation] = {}

        # Track quality metrics
        self.average_reprojection_error = 0.0

    def add_observation(self, observation: Observation) -> None:
        self.observations[observation.frame_id] = observation

    def remove_observation(self, frame_id: UUID):
        if frame_id in self.observations:
            del self.observations[frame_id]
            self.num_observations = len(self.observations)

    def is_observed_by(self, frame_id: UUID) -> bool:
        return frame_id in self.observations

    def get_observation_in_keyframe(self, frame_id: UUID) -> Observation:
        return self.observations.get(frame_id, None)

    def update_reprojection_error(self, error: float) -> None:
        if self.num_observations > 0:
            self.average_reprojection_error = (
                self.average_reprojection_error *
                (self.num_observations - 1) + error
            ) / self.num_observations

    @property
    def observing_keyframes(self) -> List[UUID]:
        return list(self.observations.keys())

    @property
    def is_good(self) -> bool:
        return self.average_reprojection_error < MAX_SQUARED_REPROJECTION_ERROR

    @property
    def num_observations(self) -> int:
        return len(self.observations)
