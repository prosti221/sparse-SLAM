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
        # Staging geometry: the geometry home is the map's native point
        # registry (read through the pt_3d property below once the
        # point is registered by Map.add_points, which sets _map).
        # Unregistered points (pre-registration, removed, or plain
        # test-constructed) read and write this slot.
        self._pt_3d = pt_3d
        self._map = None
        self.color: np.ndarray = color
        self.descriptor: np.ndarray = descriptor

        # Track which keyframes observe this point
        self.observations: Dict[int, Observation] = {}

        # Track quality metrics
        self.average_reprojection_error = 0.0

    @property
    def pt_3d(self):
        m = self._map
        if m is not None and m._registry_rows is not None:
            row = m._registry_rows.get(self.point_id)
            if row is not None:
                # A live view row of the registry buffer - the
                # geometry home. Production code rebinds pt_3d, never
                # mutates the returned array in place.
                return m._positions_view[row]
        return self._pt_3d

    @pt_3d.setter
    def pt_3d(self, value):
        m = self._map
        if m is not None and m._registry_rows is not None \
                and self.point_id in m._registry_rows:
            m._point_registry.update_positions(
                [self.point_id],
                np.asarray(value, dtype=np.float64).reshape(1, 3))
        else:
            self._pt_3d = value

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
