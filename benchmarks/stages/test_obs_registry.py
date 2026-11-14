"""The C4 observation registry: drift pins against the Python
point->observations dicts after every event class, and the cull scan's
equivalence to the Python reference scan."""
import numpy as np
import pytest

from benchmarks.stages.helpers import make_K, make_frame, set_keypoints, yaw_pose
from slam.core.map import Map
from slam.core.observation import Observation
from slam.core.point import Point


def _desc(fill):
    return np.full(32, fill, dtype=np.uint8)


def _attach(frame, point, kp_idx):
    obs = Observation(frame, point, kp_idx)
    point.add_observation(obs)
    return obs


def _observers(reg, pid):
    """The point's live (frame_id, kp_idx) pairs as a sorted set (the
    grouped native order is unspecified - comparisons are set-wise)."""
    fids, kis, starts, counts = reg.observers_for_points([pid])
    s = int(starts[0])
    return sorted(zip(fids[s:s + counts[0]].tolist(),
                      kis[s:s + counts[0]].tolist()))


def _dict_pairs(point):
    return sorted((fid, obs.kp_idx) for fid, obs in point.observations.items())


def _mixed_world():
    """Two keyframes + one tracker frame + three points observed in
    varying combinations, registered through the real funnels."""
    world = Map()
    kfs = []
    for i, yaw in enumerate((0.1, 0.3)):
        kf = set_keypoints(make_frame(yaw_pose(yaw, (0.2, 0.0, 0.0)),
                                      K=make_K()), [(30, 30), (60, 30)])
        world.add_keyframe(kf)
        kfs.append(kf)
    tracker_frame = set_keypoints(make_frame(K=make_K()), [(30, 30)])

    points = [
        Point(np.array([0.10, 0.0, 5.0]), None, _desc(0)),
        Point(np.array([0.20, 0.0, 5.0]), None, _desc(1)),
        Point(np.array([0.30, 0.0, 5.0]), None, _desc(2)),
    ]
    _attach(kfs[0], points[0], 0)
    _attach(kfs[1], points[0], 0)
    _attach(tracker_frame, points[0], 0)   # tracker frames register too
    _attach(kfs[0], points[1], 1)
    _attach(kfs[1], points[2], 0)
    world.add_points(points)
    return world, kfs, tracker_frame, points


def test_add_events_mirror_the_python_dicts():
    world, _kfs, _tf, points = _mixed_world()
    reg = world._obs_registry
    assert reg.size() == sum(len(p.observations) for p in points) == 5
    for p in points:
        assert _observers(reg, p.point_id) == _dict_pairs(p)


def test_duplicate_pair_adds_overwrite_in_place():
    """One observation per (point, frame) pair, and a re-add OVERWRITES
    the pair's row (the dict semantics: kp_idx/coords update, size and
    row order unchanged)."""
    world, kfs, _tf, points = _mixed_world()
    reg = world._obs_registry
    n0 = reg.size()
    dup = Observation(kfs[0], points[0], 1)  # same pair, different kp_idx
    points[0].add_observation(dup)
    world._add_observation_events([dup])
    assert reg.size() == n0
    assert reg.has_pair(points[0].point_id, kfs[0].frame_id)
    assert _dict_pairs(points[0]) == _observers(reg, points[0].point_id)
    assert _observers(reg, points[0].point_id)[0][1] == 1


def test_point_removal_drops_only_that_point():
    world, _kfs, _tf, points = _mixed_world()
    reg = world._obs_registry
    world.remove_point_by_index(1)
    assert reg.size() == 4
    assert _observers(reg, points[1].point_id) == []
    for p in (points[0], points[2]):
        assert _observers(reg, p.point_id) == _dict_pairs(p)


def test_keyframe_removal_drops_the_frame_from_every_point():
    world, kfs, _tf, points = _mixed_world()
    reg = world._obs_registry
    live_ids = {p.point_id for p in world.points}
    assert world.remove_keyframe_by_id(kfs[1].frame_id)
    # The orphan cascade removes points that observed the removed
    # keyframe and fell below MINIMUM_OBSERVATIONS_FOR_POINT; points
    # that never observed it survive regardless of their observer
    # count (the measured dict semantics). The registry mirrors both:
    for p in points:
        if p.point_id in live_ids and p in world.points:
            assert _observers(reg, p.point_id) == _dict_pairs(p)
        else:
            assert _observers(reg, p.point_id) == []
        assert all(fid != kfs[1].frame_id
                   for fid, _ki in _observers(reg, p.point_id))
    # the pair map is erased with the rows: a re-add of the same pair
    # on a surviving point lands as a fresh row (the dict allows the
    # same cycle)
    survivor = next(p for p in world.points)
    _attach(kfs[0], survivor, 0)
    world._add_observation_events([survivor.observations[kfs[0].frame_id]])
    assert _observers(reg, survivor.point_id) == _dict_pairs(survivor)


def test_reset_clears():
    world, _kfs, _tf, points = _mixed_world()
    world.reset()
    assert world._obs_registry.size() == 0
    assert not world._obs_registry.has_pair(points[0].point_id, 0)


def _py_cull_scan(world, candidate_ids, min_other):
    """The pre-C4 Python scan, verbatim, as the oracle."""
    candidates = set(candidate_ids)
    buckets = {fid: [] for fid in candidate_ids}
    for point in world.points:
        kf_observers = [fid for fid in point.observations
                        if fid in world.keyframes_by_id]
        n_others = len(kf_observers) - 1
        for fid in kf_observers:
            if fid in candidates:
                buckets[fid].append(n_others)
    return ([len(b) for b in buckets.values()],
            [sum(1 for n in b if n >= min_other) for b in buckets.values()])


def test_cull_scan_matches_the_python_reference():
    """The native aggregate equals the Python scan exactly - including
    the tracker-frame exclusion (never-promoted frames are never
    anchors), the min_other boundary, and empty buckets."""
    world, kfs, _tf, points = _mixed_world()
    # a fourth point observed ONLY by tracker frames + one keyframe:
    # in the candidate's bucket, but its n_others counts keyframes only
    p4 = Point(np.array([0.40, 0.0, 5.0]), None, _desc(3))
    _attach(kfs[1], p4, 0)
    world.add_points([p4])

    candidates = [kf.frame_id for kf in world.keyframes]
    kf_ids = list(world.keyframes_by_id)
    min_other = 2  # the boundary: a 3-observer point is redundant,
                   # a 2-observer point is not
    got = world._obs_registry.cull_scan(candidates, kf_ids, min_other)
    want = _py_cull_scan(world, candidates, min_other)
    assert got[0].tolist() == want[0]
    assert got[1].tolist() == want[1]
    # the boundary itself: min_other-1 never crosses
    got_b = world._obs_registry.cull_scan(candidates, kf_ids, min_other - 1)
    want_b = _py_cull_scan(world, candidates, min_other - 1)
    assert got_b[1].tolist() == want_b[1]
    assert sum(got_b[1]) > sum(got[1])


def test_cull_scan_query_order_is_the_return_order():
    """The counts are parallel to the QUERY order (the oldest-first
    selection reads slot i)."""
    world, kfs, _tf, _points = _mixed_world()
    ids = [kf.frame_id for kf in world.keyframes]
    bs, _rc = world._obs_registry.cull_scan(ids, ids, 2)
    # kf0's bucket rows: p0,p1; kf1's: p0,p2. The tracker frame's row
    # is live in the registry but never a bucket row (candidates are
    # keyframes only).
    assert bs.tolist() == [2, 2]
