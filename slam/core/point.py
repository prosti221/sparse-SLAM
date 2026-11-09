import uuid
import numpy as np
from typing import List, Tuple, Dict
from uuid import UUID
from slam.utils.constants import MAX_POINT_AVG_REPROJECTION_ERROR
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
        observation.frame.add_point_observation(observation)
        self._update_observing_keyframe_quality()

    def remove_observation(self, frame_id: UUID):
        if frame_id in self.observations:
            del self.observations[frame_id]
            # NOTE (finding 16): this used to re-assign
            # self.num_observations here - an AttributeError, since
            # num_observations is a setter-less property that computes
            # len(self.observations) live. The assignment is simply
            # gone; the property reads through.
        self._update_observing_keyframe_quality()

    def is_observed_by(self, frame_id: UUID) -> bool:
        return frame_id in self.observations

    def get_observation_in_keyframe(self, frame_id: UUID) -> Observation:
        return self.observations.get(frame_id, None)

    def update_reprojection_error(self, error: float,
                                  update_quality: bool = True) -> None:
        if self.num_observations > 0:
            self.average_reprojection_error = (
                self.average_reprojection_error *
                (self.num_observations - 1) + error
            ) / self.num_observations
        # Phase 2i (P1): the BA error-fold recomputes every point's
        # errors back-to-back, and flagging ALL observers on EVERY fold
        # step was quadratic in a point's observation count (~780k
        # idempotent attribute-sets per BA call on translation, whose
        # long-lived points carry ~21 observations). The flag is
        # idempotent and lazily consumed (Frame.tracking_quality
        # recomputes on next read), so batch folds pass update_quality=
        # False and flag each point's observers once, afterwards
        # (preserving the exact set: only points whose fold actually
        # updated get flagged - see _recompute_point_errors).
        if update_quality:
            self._update_observing_keyframe_quality()

    def _update_observing_keyframe_quality(self):
        for obs in self.observations.values():
            obs.frame.tracking_quality_needs_update = True

    @property
    def observing_keyframes(self) -> List[UUID]:
        return list(self.observations.keys())

    @property
    def is_good(self) -> bool:
        # The point-health path's OWN threshold (phase 5c split; the
        # unit-mixture caveat is recorded in the 2i Q2 record until 5e
        # re-designs this path).
        return self.average_reprojection_error < MAX_POINT_AVG_REPROJECTION_ERROR

    @property
    def num_observations(self) -> int:
        return len(self.observations)
