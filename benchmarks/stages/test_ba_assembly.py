"""The C5a BA-path assembly pins: the native assembly must derive the
fold + BA inputs from the obs registry's rows in the EXACT Python
orders, and the fold fed through it must be bit-exact with the
per-object Python extraction."""
import numpy as np
import pytest

slam_native = pytest.importorskip("slam_native")

from benchmarks.stages.helpers import (  # noqa: E402
    make_K, make_frame, make_world_points, project_gt, set_keypoints,
    yaw_pose)
from slam.core.map import Map  # noqa: E402
from slam.core.observation import Observation  # noqa: E402
from slam.core.point import Point  # noqa: E402


def _assembly_fixture():
    """Three keyframes + one tracker frame; four points with varying
    observation patterns (a point observed by the tracker frame too,
    a gap pattern), registered through the real funnels."""
    K = make_K()
    m = Map()
    kfs = []
    world = make_world_points(4, z_range=(7.0, 9.0), rng_seed=5)
    points = [Point(X.copy(), None, np.zeros(32, dtype=np.uint8))
              for X in world]
    for i in range(3):
        T = yaw_pose(0.0, (0.0, 0.0, 0.5 * i))
        kf = make_frame(T, K)
        uvs = [project_gt(T, X, K)[0] for X in world]
        set_keypoints(kf, uvs)
        m.add_keyframe(kf)
        kfs.append(kf)
    tracker = make_frame(K)  # never promoted
    set_keypoints(tracker, [(10.0, 10.0)])

    for ki, kf in enumerate(kfs):
        for pi, pt in enumerate(points):
            if (pi + ki) % 3 == 0:
                continue  # observation gaps
            pt.add_observation(Observation(kf, pt, pi))
    m.add_points(points)
    # a post-registration observation (the proj-match site's shape)
    points[1].add_observation(Observation(tracker, points[1], 0))
    m._add_observation_events([points[1].observations[tracker.frame_id]])
    return m, kfs, tracker, points


def _py_derivation(m, local_points, local_keyframes):
    """The per-object Python derivations the assembly replaced, as the
    oracle (the pre-C5a code, verbatim in meaning)."""
    observations = [obs for p in local_points
                    for obs in p.observations.values()]
    target_kf_ids = {obs.frame.frame_id for obs in observations}
    frames = [kf for kf in m.keyframes if kf.frame_id in target_kf_ids]
    frame_row = {kf.frame_id: i for i, kf in enumerate(frames)}
    point_row = {p.point_id: i for i, p in enumerate(local_points)}
    edge_obs = [obs for obs in observations
                if obs.frame.frame_id in frame_row]
    frames_fold = list(dict.fromkeys(ob.frame for ob in observations))
    points_fold = list(dict.fromkeys(ob.point for ob in observations))
    anchor = next(i for i, kf in enumerate(m.keyframes)
                  if kf.frame_id in target_kf_ids)
    return dict(
        observations=observations,
        frames=frames, frame_row=frame_row, point_row=point_row,
        edge_obs=edge_obs, frames_fold=frames_fold,
        points_fold=points_fold,
        anchor_slot=frame_row[m.keyframes[anchor].frame_id],
        uv=np.array([ob.pt_2d_norm for ob in observations],
                    dtype=np.float64),
        edge_uv=np.array([ob.pt_2d_norm for ob in edge_obs],
                         dtype=np.float64))


def _run_assembly(m, local_points, local_keyframes):
    kf_ids = [kf.frame_id for kf in m.keyframes]
    local_ids = [p.point_id for p in local_points]
    rows_map = m._registry_rows
    local_rows = [rows_map[pid] for pid in local_ids]
    local_avg = [p.average_reprojection_error for p in local_points]
    return m._obs_registry.assemble_local_ba(
        kf_ids, [kf.frame_id for kf in local_keyframes],
        local_ids, local_rows,
        m._point_registry.positions_view(),
        np.array(local_avg, dtype=np.float64))


def test_assembly_sequence_and_orders_match_the_python():
    m, kfs, _tracker, _points = _assembly_fixture()
    local_keyframes = kfs
    local_point_set = m.get_local_points(local_keyframes)
    local_points = [p for p in m.points if p in local_point_set]
    want = _py_derivation(m, local_points, local_keyframes)
    (f_of, f_op, f_norm, f_rows, f_avg, f_nobs, f_slot, fold_frame_ids,
     ba_frame_ids, anchor_a, in_window, e_fr, e_pr, e_uv,
     b_pos) = _run_assembly(m, local_points, local_keyframes)

    # the observation sequence (uv order == the Python list order)
    np.testing.assert_array_equal(np.asarray(f_norm), want["uv"])
    # the first-appearance orders == dict.fromkeys
    assert list(fold_frame_ids) == [f.frame_id for f in want["frames_fold"]]
    fold_points = [local_points[i].point_id for i in f_slot]
    assert fold_points == [p.point_id for p in want["points_fold"]]
    assert list(f_nobs) == [len(p.observations) for p in want["points_fold"]]
    # the fold point arrays reorder the inputs by the subset slots
    np.testing.assert_array_equal(
        np.asarray(f_avg),
        np.array([[p.average_reprojection_error for p in local_points][i]
                  for i in f_slot]))
    # the fold rows index the registry block: re-gathering at fold time
    # reproduces the fold points' object positions (the fold-time
    # gather contract - the solve moves the geometry after assembly)
    pv = np.asarray(m._point_registry.positions_view())
    np.testing.assert_array_equal(
        pv[np.asarray(f_rows)],
        np.array([p.pt_3d for p in want["points_fold"]], dtype=np.float64))
    # the BA derivations
    assert list(ba_frame_ids) == [kf.frame_id for kf in want["frames"]]
    assert int(anchor_a[0]) == want["anchor_slot"]
    window_ids = {kf.frame_id for kf in local_keyframes}
    assert list(in_window) == [1 if fid in window_ids else 0
                               for fid in ba_frame_ids]
    assert list(e_fr) == [want["frame_row"][ob.frame.frame_id]
                          for ob in want["edge_obs"]]
    assert list(e_pr) == [want["point_row"][ob.point.point_id]
                          for ob in want["edge_obs"]]
    np.testing.assert_array_equal(np.asarray(e_uv), want["edge_uv"])
    # the BA positions gather == the per-point reads (a bit copy)
    np.testing.assert_array_equal(
        np.asarray(b_pos),
        np.array([p.pt_3d for p in local_points], dtype=np.float64))


def test_assembly_excludes_tracker_frames_from_edges_but_keeps_the_fold():
    """The edge filter (keyframe frames only) and the fold's frames
    (tracker frames included, first-appearance) are DIFFERENT views of
    the same sequence."""
    m, kfs, tracker, points = _assembly_fixture()
    local_keyframes = kfs
    local_point_set = m.get_local_points(local_keyframes)
    local_points = [p for p in m.points if p in local_point_set]
    (f_of, f_op, f_norm, f_rows, f_avg, f_nobs, f_slot, fold_frame_ids,
     ba_frame_ids, _anchor_a, _in_window, e_fr, _e_pr, _e_uv,
     _b_pos) = _run_assembly(m, local_points, local_keyframes)
    assert tracker.frame_id in list(fold_frame_ids)   # the fold sees it
    assert tracker.frame_id not in list(ba_frame_ids)  # no edge, no vertex
    n_edges = len(e_fr)
    n_edges_python = sum(
        1 for ob in ([ob for p in local_points
                      for ob in p.observations.values()])
        if ob.frame.frame_id in {kf.frame_id for kf in m.keyframes})
    assert n_edges == n_edges_python


def test_registry_uv_equals_the_dicts_pt2d_norm():
    """[C5a] the NEW pin class: the registry's stored (xn, yn) must
    equal the Observation objects' pt_2d_norm - the fold and the BA
    consume the registry's copies now (the C4 pins compared only
    (fid, kp_idx))."""
    m, kfs, tracker, points = _assembly_fixture()
    # the sequence uv == the dicts' pt_2d_norm, in order
    local_point_set = m.get_local_points(kfs)
    local_points = [p for p in m.points if p in local_point_set]
    (f_of, _f_op, f_norm, *_rest) = _run_assembly(m, local_points, kfs)
    observations = [obs for p in local_points
                    for obs in p.observations.values()]
    np.testing.assert_array_equal(
        np.asarray(f_norm),
        np.array([ob.pt_2d_norm for ob in observations], dtype=np.float64))
    # and the event-class coverage: a duplicate overwrite updates the uv
    dup = Observation(kfs[0], points[0], 1)
    points[0].add_observation(dup)
    m._add_observation_events([dup])
    (_f_of2, _f_op2, f_norm2, *_r2) = _run_assembly(m, local_points, kfs)
    observations2 = [obs for p in local_points
                     for obs in p.observations.values()]
    np.testing.assert_array_equal(
        np.asarray(f_norm2),
        np.array([ob.pt_2d_norm for ob in observations2], dtype=np.float64))


def test_assembly_zero_row_local_point_is_defensive():
    """A local point with a registry row but ZERO live observation rows
    contributes a vertex but no fold slot (unreachable through the
    funnels - every live point keeps a keyframe row - but the branch
    must behave)."""
    from slam_native import ObsRegistry
    reg = ObsRegistry()  # no rows at all
    (f_of, _f_op, _f_norm, f_rows, f_avg, f_nobs, f_slot, _ffids,
     ba_frame_ids, anchor_a, _win, e_fr, _e_pr, _e_uv,
     b_pos) = reg.assemble_local_ba(
        [100], [100], [7], [0], np.array([[1.0, 2.0, 3.0]]),
        np.array([0.5]))
    assert len(f_of) == 0            # no sequence rows
    assert list(f_slot) == []        # excluded from the fold subset
    np.testing.assert_array_equal(np.asarray(b_pos)[0], [1.0, 2.0, 3.0])
    assert list(ba_frame_ids) == []  # no frame appears in the sequence
    assert int(anchor_a[0]) == -1    # no anchor
    assert len(e_fr) == 0


def test_fold_via_assembly_is_bit_exact():
    """The fold through the C5a assembly must produce EXACTLY the
    per-object path's averages and flag set (same kernel, bit-equal
    inputs)."""
    world_points = make_world_points(12, z_range=(7.0, 10.0))
    m_ref = _fold_map(6, world_points)
    m_new = _fold_map(6, world_points)
    kfs_ref = m_ref[1]
    pts_ref = m_ref[2]
    m_ref, kfs_ref, pts_ref = m_ref[0], kfs_ref, pts_ref
    m_new, kfs_new, pts_new = m_new

    obs_ref = [obs for p in pts_ref for obs in p.observations.values()]
    m_ref.g2o_optimizer._recompute_point_errors(obs_ref)

    local_points = [p for p in m_new.points
                    if p in m_new.get_local_points(kfs_new)]
    kf_ids = [kf.frame_id for kf in m_new.keyframes]
    local_ids = [p.point_id for p in local_points]
    rows_map = m_new._registry_rows
    asm = m_new._obs_registry.assemble_local_ba(
        kf_ids, [kf.frame_id for kf in kfs_new], local_ids,
        [rows_map[pid] for pid in local_ids],
        m_new._point_registry.positions_view(),
        np.array([p.average_reprojection_error for p in local_points],
                 dtype=np.float64))
    m_new.g2o_optimizer._recompute_point_errors(local_points, assembly=asm)

    for a, b in zip(pts_ref, pts_new):
        assert a.average_reprojection_error == b.average_reprojection_error
    flagged_ref = [kf.tracking_quality_needs_update for kf in kfs_ref]
    flagged_new = [kf.tracking_quality_needs_update for kf in kfs_new]
    assert flagged_ref == flagged_new


def _fold_map(n_kfs, world_points):
    """The test_pnp_ba fold fixture's shape: one point BEHIND every
    camera (its fold never updates; its observers must never flag)."""
    K = make_K()
    m = Map()
    kfs = []
    points = [Point(X.copy(), None, np.zeros(32, dtype=np.uint8))
              for X in world_points]
    points.append(Point(np.array([0.0, 0.0, -1.0]), None,
                        np.zeros(32, dtype=np.uint8)))
    for i in range(n_kfs):
        T = yaw_pose(0.0, (0.0, 0.0, 0.5 * i))
        kf = make_frame(T, K)
        uvs = [project_gt(T, X, K)[0] for X in world_points]
        uvs.append((320.0, 240.0))
        set_keypoints(kf, uvs)
        for j, pt in enumerate(points):
            pt.add_observation(Observation(kf, pt, j))
        m.add_keyframe(kf)
        kfs.append(kf)
    m.add_points(points)
    return m, kfs, points
