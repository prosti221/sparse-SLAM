import itertools

import numpy as np
from typing import List, Dict
from slam.utils.constants import MAX_POINT_AVG_REPROJECTION_ERROR
from slam.core.observation import Observation

# u64 identity (migration locked decision 3): native handles ARE
# these ids; creation is main-thread-only and count() is atomic
# under the GIL.
_next_point_id = itertools.count(1)


class Point:
    def __init__(self, pt_3d: np.ndarray, color: np.ndarray = None, descriptor: np.ndarray = None):
        self.point_id: int = next(_next_point_id)
        self.pt_3d: np.ndarray = pt_3d
        self.color: np.ndarray = color
        self.descriptor: np.ndarray = descriptor

        # Track which keyframes observe this point
        self.observations: Dict[int, Observation] = {}

        # Track quality metrics
        self.average_reprojection_error = 0.0

    def add_observation(self, observation: Observation) -> None:
        self.observations[observation.frame_id] = observation
        observation.frame.add_point_observation(observation)
        self._update_observing_keyframe_quality()

    def remove_observation(self, frame_id: int):
        if frame_id in self.observations:
            del self.observations[frame_id]
            # num_observations is a setter-less property that computes
            # len(self.observations) live - nothing to re-assign here;
            # the property reads through.
        self._update_observing_keyframe_quality()

    def is_observed_by(self, frame_id: int) -> bool:
        return frame_id in self.observations

    def update_reprojection_error(self, error: float,
                                  update_quality: bool = True) -> None:
        if self.num_observations > 0:
            self.average_reprojection_error = (
                self.average_reprojection_error *
                (self.num_observations - 1) + error
            ) / self.num_observations
        # The BA error-fold recomputes every point's
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
    def observing_keyframes(self) -> List[int]:
        return list(self.observations.keys())

    @property
    def is_good(self) -> bool:
        # The point-health path's OWN threshold (deliberately
        # separate from the creation gate's; see the constants note).
        return self.average_reprojection_error < MAX_POINT_AVG_REPROJECTION_ERROR

    @property
    def num_observations(self) -> int:
        return len(self.observations)
