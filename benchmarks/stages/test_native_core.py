"""[C3b] the point registry drift pins.

The C1 MapCore window (and its eight pins) retired with migration C5a:
the push model was write-only in production - the C2a fold and the C3
correspondence take Python-assembled arrays, and the C3b/C4 registries
superseded the window as the native mirror. The record lives in
benchmarks/IMPROVEMENT.md; these pins keep the SURVIVING machinery
(the point registry) guarded.
"""
import numpy as np
import pytest

slam_native = pytest.importorskip("slam_native")

from benchmarks.stages.helpers import (  # noqa: E402
    make_K, make_world_points)
from slam.core.map import Map  # noqa: E402
from slam.core.point import Point  # noqa: E402


def test_point_registry_matches_map_after_events():
    """[C3b] the drift guard: the native point registry must equal the
    Python graph's geometry + descriptors after every event class -
    add, cull, the BA write-back, and rebuild_point_coords."""
    K = make_K()
    m = Map()
    reg_probe = m._point_registry
    assert reg_probe is None  # lazily created on the first add

    pts3d = make_world_points(6, z_range=(7.0, 9.0), rng_seed=13)
    points = [Point(x.copy(), None, np.full(32, i, dtype=np.uint8))
              for i, x in enumerate(pts3d)]
    m.add_points(points)
    reg = m._point_registry
    assert reg is not None and reg.size() == 6
    pv = np.asarray(reg.positions_view())
    for i, pt in enumerate(m.points):
        np.testing.assert_array_equal(pv[i], pt.pt_3d)
        assert reg.has(pt.point_id)

    # cull (the remove funnel): every removed id leaves the registry.
    # NOTE the registry's rows are in ITS internal order (swap-erase
    # compaction reorders); the drift comparison goes BY ID (the
    # consumers map ids to rows via row_of).
    m.remove_point_by_index(2)
    assert reg.size() == 5
    assert not reg.has(points[2].point_id)
    assert reg.has(points[5].point_id)
    pv = np.asarray(reg.positions_view())
    for pt in m.points:
        np.testing.assert_array_equal(pv[reg.row_of(pt.point_id)], pt.pt_3d)

    # the BA write-back class: bulk position update
    moved = np.array([[99.0, 98.0, 97.0]] * len(m.points))
    for i, pt in enumerate(m.points):
        pt.pt_3d = moved[i].copy()
    reg.update_positions([pt.point_id for pt in m.points], moved)
    pv = np.asarray(reg.positions_view())
    for pt in m.points:
        np.testing.assert_array_equal(pv[reg.row_of(pt.point_id)], pt.pt_3d)

    # rebuild_point_coords class: the re-anchoring moves every point
    new_pts = [pt.pt_3d + 1.0 for pt in m.points]
    for pt, n in zip(m.points, new_pts):
        pt.pt_3d = n
    m.rebuild_point_coords()
    pv = np.asarray(reg.positions_view())
    for pt in m.points:
        np.testing.assert_array_equal(pv[reg.row_of(pt.point_id)], pt.pt_3d)

    # reset clears
    m.reset()
    assert reg.size() == 0


def test_point_registry_views_are_live():
    """[C3b] the registry's views read the LIVE storage: an update
    through the registry's own blocks is visible in the Python-held
    view object (the zero-copy contract the consumers rely on)."""
    from slam_native import PointRegistry
    reg = PointRegistry(32)
    ids = [5]
    pos = np.array([[1.0, 2.0, 3.0]])
    desc = np.zeros((1, 32), dtype=np.uint8)
    reg.register_points(ids, pos, desc, np.array([1], dtype=np.uint8))
    view = reg.positions_view()
    np.testing.assert_array_equal(np.asarray(view)[0], [1.0, 2.0, 3.0])
    reg.update_positions(ids, np.array([[4.0, 5.0, 6.0]]))
    np.testing.assert_array_equal(np.asarray(view)[0], [4.0, 5.0, 6.0])
