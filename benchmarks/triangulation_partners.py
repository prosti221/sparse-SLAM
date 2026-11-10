"""Phase 6d: bounded partner experiments, NOT production defaults.

Keep adjacent insertion matching/dedup/PnP/BA intact. At triangulation
time evaluate at most six older live KFs: four recent, plus two with the
most authoritative shared 3px-consistent map measurements (>=10).
New-point matches use deterministic BF/Lowe/one-to-one matching, followed
by the UNCHANGED production DLT and creation gates, not extra RANSAC.
This is an explicit new-pair matching policy, not just a projection swap.
"""
from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import copy
from dataclasses import dataclass
from unittest.mock import patch

import cv2 as cv
import numpy as np

from benchmarks.keyframe_audit import _authoritative_observations, distribution
from slam.core.observation import Observation
from slam.core.point import Point
from slam.core.tracker import Tracker
from slam.utils import constants as const, telemetry
from slam.utils.utils import compute_triangulation, is_valid_triangulated_point


RECENT_OLDER = 4
COVISIBLE_OLDER = 2
MIN_SHARED = 10
MIN_NEW_POINTS = 10
MODES = ("shadow", "best_pair", "adjacent_plus_one")
_active = False


def shared_counts(current, world):
    """Live, two-sided, collision-free observations; no lazy quality reads.

    This is measured map consistency, not a calibrated inlier probability.
    Reference-side consistency is also checked before counting an anchor.
    """
    observations, _ = _authoritative_observations(current, world)
    if not observations:
        return {}
    predicted = current.project_points(np.array([o.point.pt_3d for o in observations]), False)
    measured = np.array([current.keypoints[o.kp_idx].pt for o in observations])
    consistent = [o for o, error in zip(observations, np.linalg.norm(predicted - measured, axis=1))
                  if np.isfinite(error) and error <= 3.0]
    candidates = defaultdict(list)
    for obs in consistent:
        for fid, other in obs.point.observations.items():
            reference = world.keyframes_by_id.get(fid)
            if reference is not None and reference is not current and other.frame is reference:
                candidates[fid].append(other)
    counts = {}
    for fid, possible in candidates.items():
        reference = world.keyframes_by_id[fid]
        authoritative, _ = _authoritative_observations(reference, world)
        identities = {id(o) for o in authoritative}
        possible = [o for o in possible if id(o) in identities]
        if not possible:
            continue
        uv = reference.project_points(np.array([o.point.pt_3d for o in possible]), False)
        measured = np.array([reference.keypoints[o.kp_idx].pt for o in possible])
        error = np.linalg.norm(uv - measured, axis=1)
        counts[fid] = int(np.sum(np.isfinite(error) & (error <= 3.0)))
    return counts


def shortlist(current, adjacent, world):
    """Latest wins ties; UUIDs and stale covisibility caches never rank KFs."""
    older = [kf for kf in world.keyframes if kf is not current and kf is not adjacent
             and world.keyframes_by_id.get(kf.frame_id) is kf]
    counts = shared_counts(current, world)
    recent = list(reversed(older[-RECENT_OLDER:]))
    rank = {kf.frame_id: i for i, kf in enumerate(world.keyframes)}
    covisible = sorted((kf for kf in older if counts.get(kf.frame_id, 0) >= MIN_SHARED),
                       key=lambda kf: (counts[kf.frame_id], rank[kf.frame_id]), reverse=True)
    # Two ADDITIONAL covisible views; the recent pool is always available
    # even when tracking supplied no authoritative mapped measurements.
    extra = [kf for kf in covisible if kf not in recent][:COVISIBLE_OLDER]
    return recent + extra, counts


def occupied_indices(frame):
    """Conservative endpoint reservation, including stale/frame-only entries.

    Do not resurrect pruned points, repair registries or promote observations.
    Checking both registries avoids overwriting even inconsistent histories.
    """
    keypoints = frame.keypoints if frame.keypoints is not None else []
    occupied = {i for i, kp in enumerate(keypoints) if kp in frame.keypoint_to_point_map}
    occupied.update(o.kp_idx for o in frame.observed_points.values())
    if frame.kp_unique_mask is not None:
        occupied.update(i for i, unique in enumerate(frame.kp_unique_mask) if not unique)
    return occupied


def match_new_pair(current, reference, extractor):
    """query=current, train=reference, original descriptor order preserved.

    Ratio-test on ALL descriptors, THEN reserve mapped endpoints; removing
    competitors before kNN would change Lowe's population. BF only: even
    float descriptors must not invoke randomized FLANN/index construction.
    """
    a, b = current.descriptors, reference.descriptors
    if (a is None or b is None or len(a) == 0 or len(b) < 2
            or current.keypoints is None or reference.keypoints is None
            or len(current.keypoints) != len(a) or len(reference.keypoints) != len(b)):
        return []
    norm = cv.NORM_HAMMING if extractor in const.BINARY_DESCRIPTION_METHODS else cv.NORM_L2
    matcher = cv.BFMatcher(norm, crossCheck=False)
    reserved_query, reserved_train = occupied_indices(current), occupied_indices(reference)
    query_seen, train_seen = set(), set()
    result = []
    for neighbors in matcher.knnMatch(a, b, k=2):
        if len(neighbors) != 2:
            continue
        m, n = neighbors
        # Same order and greedy uniqueness as the production ratio stage.
        if (m.distance < const.LOWE_RATIO * n.distance
                and m.queryIdx not in query_seen and m.trainIdx not in train_seen):
            query_seen.add(m.queryIdx)
            train_seen.add(m.trainIdx)
            if m.queryIdx not in reserved_query and m.trainIdx not in reserved_train:
                result.append(m)
    return result


@dataclass
class PairProposal:
    reference: object
    matches: list
    points: dict
    errors: dict
    failures: dict
    zero_w: int

    def available(self, blocked=()):
        blocked = set(blocked)
        return [i for i in self.points if self.matches[i].queryIdx not in blocked]

    def summary(self, current, blocked=()):
        available = self.available(blocked)
        rays = []
        for frame, indices in ((current, [self.matches[i].queryIdx for i in available]),
                               (self.reference, [self.matches[i].trainIdx for i in available])):
            xy = frame.kp_pts_norm[indices] if indices else np.empty((0, 2))
            ray = np.column_stack([xy, np.ones(len(xy))]) @ frame.pose[:3, :3].T
            ray /= np.linalg.norm(ray, axis=1, keepdims=True)
            rays.append(ray)
        angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(*rays), axis=1),
                                      np.sum(rays[0] * rays[1], axis=1)))
        return {"matches": len(self.matches), "accepted": len(self.points),
                "available_accepted": len(available), "zero_w": self.zero_w,
                "failed_gates": dict(Counter(const.TRIANGULATION_VALIDATION_CODE[code]
                                              for codes in self.failures.values() for code in codes)),
                "accepted_bearing_parallax_deg": distribution(angles)}


def evaluate_pair(current, reference, matches):
    """Read-only shallow view binds the correct pair's matches for helpers.

    No actual Frame.matches/map ordering/registries/cache/quality mutation.
    Arithmetic and finite-w handling mirror Tracker._triangulate exactly.
    """
    view = copy(current)
    view.matches = matches
    homogeneous = compute_triangulation(view, reference, use_optimization=False)
    points, errors, failures = {}, {}, {}
    zero_w = 0
    if homogeneous.size:
        valid = homogeneous[:, 3] != 0
        zero_w = int(sum(~valid))
        normalized = homogeneous[valid] / homogeneous[valid, 3:]
        for index, point in zip(np.flatnonzero(valid), normalized):
            index = int(index)
            validation = is_valid_triangulated_point(index, view, reference, point)
            if validation["failed_codes"]:
                failures[index] = validation["failed_codes"]
            else:
                points[index] = point
                errors[index] = validation["reprojection_error"]
    return PairProposal(reference, matches, points, errors, failures, zero_w)


def choose_pair(adjacent, older, blocked=(), supplement=False):
    """Single interpretable score: gate-passing NEW matches, latest tie wins.

    >=10 distinct accepted matches supplies minimal new-pair overlap AND
    useful point geometry; mapped support is not a universal requirement.
    Adjacent wins ties in replacement; no fewer points than its proposal.
    """
    eligible = [p for p in older if len(p.available(blocked)) >= MIN_NEW_POINTS]
    if not eligible:
        return None if supplement else adjacent
    best = max(eligible, key=lambda p: len(p.available(blocked)))
    if supplement:
        return best
    return best if len(best.points) > len(adjacent.points) else adjacent


def materialize(current, proposal, blocked=()):
    """Register ONLY new points on the true query/train frames, both sides.

    Same descriptor/color/error fold as production. No dedup promotion,
    mask-history rewrite or coordinate-dedup cleanup (Map.add_points owns
    that unchanged behavior). Blocked current keypoints cannot be reused.
    """
    blocked = set(blocked)
    result = []
    reserved_query, reserved_train = occupied_indices(current), occupied_indices(proposal.reference)
    for i in proposal.available(blocked):
        m = proposal.matches[i]
        if m.queryIdx in reserved_query or m.trainIdx in reserved_train:
            raise RuntimeError("Triangulation proposal would overwrite an occupied endpoint")
        point = Point(proposal.points[i][:3], current.get_color_value_for_keypoint(m.queryIdx),
                      current.descriptors[m.queryIdx])
        point.add_observation(Observation(current, point, m.queryIdx))
        point.add_observation(Observation(proposal.reference, point, m.trainIdx))
        point.update_reprojection_error(proposal.errors[i])
        result.append(point)
        reserved_query.add(m.queryIdx)
        reserved_train.add(m.trainIdx)
    failures = {i: codes for i, codes in proposal.failures.items()
                if proposal.matches[i].queryIdx not in blocked}
    telemetry.event("tri_accepted", len(result))
    telemetry.event("tri_rejected", len(failures))
    for code, count in Counter(code for codes in failures.values() for code in codes).items():
        telemetry.event(f"tri_reject_{const.TRIANGULATION_VALIDATION_CODE[code]}", count)
    return result


class TriangulationPartnerExperiment:
    """Nest OUTSIDE KeyframeAudit so its cohort hook sees final output.

    Shadow evaluates proposals before the original creation exactly once.
    No map/pose/RNG writes from diagnostics. Interventions may naturally
    change future poses, retention/culling/loop history; constants stay fixed.
    """

    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError(f"Unknown partner mode: {mode}")
        self.mode = mode
        self.rows = []
        self._frame_index = -1

    def __enter__(self):
        global _active
        if _active:
            raise RuntimeError("TriangulationPartnerExperiment contexts cannot overlap")
        _active = True
        self._stack = ExitStack()
        original = Tracker._triangulate
        original_update = Tracker.update

        def update(tracker, frame):
            self._frame_index += 1
            return original_update(tracker, frame)

        def triangulate(tracker):
            current, adjacent = tracker.map.cur_keyframe, tracker.map.prev_keyframe
            if current is None or adjacent is None:
                return original(tracker)
            references, counts = shortlist(current, adjacent, tracker.map)
            base = evaluate_pair(current, adjacent, current.matches or [])
            older = [evaluate_pair(current, reference, match_new_pair(
                current, reference, tracker.feature_extraction_method)) for reference in references]
            base_queries = {base.matches[i].queryIdx for i in base.points}
            # Temporal tie order even for extra covisible shortlist members.
            rank = {kf.frame_id: i for i, kf in enumerate(tracker.map.keyframes)}
            older.sort(key=lambda p: rank[p.reference.frame_id], reverse=True)
            best = choose_pair(base, older)
            extra = choose_pair(base, older, base_queries, supplement=True)
            row = {"frame_index": self._frame_index,
                   "reference_frame_indices": [getattr(kf, "_bench_frame_index", None)
                                               for kf in references],
                   "adjacent_frame_index": getattr(adjacent, "_bench_frame_index", None),
                   "adjacent": base.summary(current),
                   "older": [{"frame_index": getattr(p.reference, "_bench_frame_index", None),
                              "shared_consistent": counts.get(p.reference.frame_id, 0),
                              **p.summary(current),
                              "after_adjacent_accepted": len(p.available(base_queries))} for p in older],
                   "best_pair_frame_index": getattr(best.reference, "_bench_frame_index", None),
                   "best_pair_is_older": best is not base,
                   "supplement_frame_index": (getattr(extra.reference, "_bench_frame_index", None)
                                              if extra is not None else None),
                   "supplement_available": len(extra.available(base_queries)) if extra is not None else 0,
                   "mode": self.mode}
            if self.mode == "best_pair" and best is not base:
                result = materialize(current, best)
            else:
                result = original(tracker)
                if len(result) != len(base.points):
                    raise RuntimeError("Adjacent proposal no longer mirrors production creation")
                if self.mode == "adjacent_plus_one" and extra is not None:
                    result += materialize(current, extra, base_queries)
            row["created"] = len(result)
            self.rows.append(row)
            return result

        self._stack.enter_context(patch.object(Tracker, "_triangulate", triangulate))
        self._stack.enter_context(patch.object(Tracker, "update", update))
        return self

    def __exit__(self, *exc):
        global _active
        try:
            return self._stack.__exit__(*exc)
        finally:
            _active = False
