"""The point registry's contract with the Python Point objects.

Migration C6 inverted the C3b model: the registry IS the geometry
home and Point.pt_3d reads it (the per-point Python twin and its
write-back sync loops are gone - the drift-guard pins that compared
the two retired with that record). These pins guard the surviving
contract: the property's registered/staging resolution, the setter's
routing, the map's bulk funnels, and the post-reset archive snapshot.
"""
import numpy as np
import pytest

slam_native = pytest.importorskip("slam_native")

from benchmarks.stages.helpers import (  # noqa: E402
    make_world_points)
from slam.core.map import Map  # noqa: E402
from slam.core.point import Point  # noqa: E402


def test_property_resolves_by_registration():
    """A registered point reads the registry row (shared memory with
    the registry buffer); an unregistered point reads its staging
    slot; registration switches the source."""
    m = Map()
    pt = Point(np.array([1.0, 2.0, 3.0]))
    assert pt.pt_3d is pt._pt_3d  # unregistered: the staging object

    m.add_points([pt])
    row = m._registry_rows[pt.point_id]
    val = pt.pt_3d
    assert np.shares_memory(val, m._positions_view)
    np.testing.assert_array_equal(val, [1.0, 2.0, 3.0])
    assert row == m._point_registry.row_of(pt.point_id)


def test_setter_routes_by_registration():
    """The registered setter funnels through update_positions (the
    staging slot untouched); the unregistered setter writes staging."""
    m = Map()
    unreg = Point(np.array([0.0, 0.0, 0.0]))
    unreg.pt_3d = np.array([9.0, 9.0, 9.0])
    np.testing.assert_array_equal(unreg._pt_3d, [9.0, 9.0, 9.0])

    pts3d = make_world_points(2, z_range=(7.0, 9.0), rng_seed=5)
    reg_pt = Point(pts3d[0].copy())
    m.add_points([reg_pt])
    staging_before = reg_pt._pt_3d

    reg_pt.pt_3d = np.array([42.0, 41.0, 40.0])
    np.testing.assert_array_equal(
        m._positions_view[m._registry_rows[reg_pt.point_id]],
        [42.0, 41.0, 40.0])
    np.testing.assert_array_equal(staging_before, pts3d[0])

    # and the unregistered point's write never touched the registry
    assert m._point_registry.size() == 1


def test_map_bulk_funnels():
    """positions_of returns row copies in the given order; a
    set_positions write is visible to every property read."""
    m = Map()
    pts3d = make_world_points(4, z_range=(7.0, 9.0), rng_seed=7)
    pts = [Point(x.copy()) for x in pts3d]
    m.add_points(pts)

    got = m.positions_of([pts[2], pts[0], pts[2]])
    np.testing.assert_array_equal(got[0], pts3d[2])
    np.testing.assert_array_equal(got[1], pts3d[0])
    np.testing.assert_array_equal(got[2], pts3d[2])
    assert not np.shares_memory(got, m._positions_view)  # a copy

    moved = np.arange(12, dtype=np.float64).reshape(4, 3) + 100.0
    m.set_positions(pts, moved)
    for i, pt in enumerate(pts):
        np.testing.assert_array_equal(pt.pt_3d, moved[i])


def test_view_survives_registry_growth():
    """Capacity growth reallocates the buffer; the map's rows+view
    pair refreshes with it, so property reads stay correct."""
    m = Map()
    n = 200  # the registry's initial capacity is 64
    pts = [Point(np.array([float(i), 0.0, 1.0])) for i in range(n)]
    m.add_points(pts)
    for i in (0, 63, 64, 199):
        np.testing.assert_array_equal(pts[i].pt_3d, [float(i), 0.0, 1.0])

    m.set_positions(pts[100:], np.full((n - 100, 3), 7.0))
    np.testing.assert_array_equal(pts[100].pt_3d, [7.0, 7.0, 7.0])
    np.testing.assert_array_equal(pts[199].pt_3d, [7.0, 7.0, 7.0])
    np.testing.assert_array_equal(pts[0].pt_3d, [0.0, 0.0, 1.0])


def test_removed_point_reads_last_known_geometry():
    """The removal snapshots the LIVE row into the staging slot before
    the row is dropped: an unregistered read returns the point's
    LAST-KNOWN geometry (what the removed twin's attr held), not
    registration-time geometry."""
    m = Map()
    pt = Point(np.array([1.0, 1.0, 1.0]))
    m.add_points([pt])
    m.set_positions([pt], np.array([[5.0, 5.0, 5.0]]))
    np.testing.assert_array_equal(pt.pt_3d, [5.0, 5.0, 5.0])

    m.remove_point_by_index(0)
    np.testing.assert_array_equal(pt.pt_3d, [5.0, 5.0, 5.0])
    assert pt.point_id not in m._registry_rows


def test_positions_of_mixed_population():
    """Frame-side observed_points can hold unregistered points
    (coordinate-duplicate rejections, removed points). positions_of
    resolves the mixed population: rows for registered, staging
    snapshots for the rest - the values the old attr reads gave."""
    m = Map()
    reg_pt = Point(np.array([1.0, 2.0, 3.0]))
    # Same ROUNDED coord key as reg_pt (the dedup gate), different
    # value so the staging fallback is distinguishable.
    dup = Point(np.array([1.0, 2.0, 3.0004]))
    m.add_points([reg_pt, dup])
    assert dup.point_id not in m._registry_rows  # never registered
    assert reg_pt.point_id in m._registry_rows

    got = m.positions_of([dup, reg_pt, dup])
    np.testing.assert_array_equal(got[0], [1.0, 2.0, 3.0004])
    np.testing.assert_array_equal(got[1], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(got[2], [1.0, 2.0, 3.0004])


def test_archive_snapshot_survives_reset():
    """archive_current_state snapshots the LIVE rows into the staging
    slots BEFORE the reset clears the registry: the archive's SAME
    Point objects keep serving the recovery relocalization reads
    afterwards (ids are never reused, so they can never false-hit a
    new registration)."""
    from benchmarks.stages.helpers import make_frame

    m = Map()
    m.add_keyframe(make_frame())
    pts3d = make_world_points(3, z_range=(7.0, 9.0), rng_seed=11)
    pts = [Point(x.copy()) for x in pts3d]
    m.add_points(pts)
    m.set_positions(pts, pts3d + 10.0)

    m.archive_current_state()
    m.reset()

    assert len(m.archives) == 1
    for i, pt in enumerate(m.archives[0].points):
        np.testing.assert_array_equal(pt.pt_3d, pts3d[i] + 10.0)


def test_registry_views_are_live():
    """The registry's views read the LIVE storage: an update through
    the registry's own blocks is visible in the Python-held view
    object (the zero-copy contract the consumers rely on)."""
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
