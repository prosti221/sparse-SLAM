"""[C2b1] the vendored-g2o drift measurement.

A fixed synthetic SE3 BA graph (deterministic fixture) solved twice
from IDENTICAL inputs: once via g2o-python (the production reference,
a fresh-build graph mirroring `_optimize_with_g2o`'s config exactly),
once via the native `solve_se3_ba` (the vendored g2o 1.1.0 - the same
source g2o-python builds from). The measured estimate drift is
RECORDED here (pre-registration says C2b1 records, C2b2's plan sets
the tolerance band from this measurement); the pin holds the recorded
bound so it cannot silently grow.
"""
import numpy as np
import pytest

slam_native = pytest.importorskip("slam_native")
from slam_native import BASolveInputs, solve_se3_ba  # noqa: E402

g2o = pytest.importorskip("g2o")


def _scene(n_frames=6, n_points=20):
    """Keyframes on a circle looking at a point cloud ahead; ground
    truth by the same normalized model the solver's CameraParameters
    (focal 1.0, principal (0,0)) applies."""
    rng = np.random.default_rng(42)
    gt_T_cw = np.zeros((n_frames, 4, 4))
    for i in range(n_frames):
        theta = 2.0 * np.pi * i / n_frames
        c = np.array([4.0 * np.cos(theta), 4.0 * np.sin(theta), 0.0])
        # look at the origin: z-axis points from c to origin
        z = -c / np.linalg.norm(c)
        x = np.cross(np.array([0.0, 0.0, 1.0]), z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=1)  # camera-to-world columns
        gt_T_cw[i, :3, :3] = R
        gt_T_cw[i, :3, 3] = c
        gt_T_cw[i, 3, 3] = 1.0
    gt_pts = np.column_stack([
        rng.uniform(-2, 2, n_points),
        rng.uniform(-2, 2, n_points),
        rng.uniform(4, 8, n_points),
    ])

    obs_frame, obs_point, obs_uv = [], [], []
    for i in range(n_frames):
        T_wc = np.linalg.inv(gt_T_cw[i])
        for j in range(n_points):
            cam = (T_wc @ np.append(gt_pts[j], 1.0))[:3]
            if cam[2] <= 0.1:
                continue
            obs_frame.append(i)
            obs_point.append(j)
            obs_uv.append(cam[:2] / cam[2])
    return (gt_T_cw, gt_pts, np.array(obs_frame, dtype=np.int64),
            np.array(obs_point, dtype=np.int64),
            np.array(obs_uv, dtype=np.float64))


def _perturb(gt_T_cw, gt_pts, seed=7):
    rng = np.random.default_rng(seed)
    T0 = gt_T_cw.copy()
    pts0 = gt_pts.copy()
    for i in range(len(T0)):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = 0.02
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        dR = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
        R_new = dR @ T0[i, :3, :3]
        T0[i, :3, :3] = R_new
        T0[i, :3, 3] += rng.uniform(-0.05, 0.05, 3)
    pts0 += rng.normal(0, 0.05, pts0.shape)
    return T0, pts0


def _solve_g2o_python(T0, pts0, obs_frame, obs_point, obs_uv, fixed_rows):
    """The production reference: a fresh-build graph with EXACTLY the
    `_optimize_with_g2o` configuration. T0 rows are WORLD-TO-CAMERA
    (the vertex estimate convention; the production build feeds
    SE3Quat from inv(kf.pose))."""
    import slam.utils.constants as const
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    terminate = g2o.SparseOptimizerTerminateAction()
    terminate.set_gain_threshold(const.BA_TERMINATE_GAIN_THRESHOLD)
    opt.set_algorithm(solver)
    opt.add_post_iteration_action(terminate)
    cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)
    cam.set_id(0)
    opt.add_parameter(cam)

    F, M = len(T0), len(pts0)
    for i in range(F):
        v = g2o.VertexSE3Expmap()
        v.set_id(i)
        v.set_estimate(g2o.SE3Quat(T0[i, :3, :3], T0[i, :3, 3]))
        v.set_fixed(i in fixed_rows)
        opt.add_vertex(v)
    for j in range(M):
        v = g2o.VertexPointXYZ()
        v.set_id(F + j)
        v.set_estimate(pts0[j])
        v.set_fixed(False)
        v.set_marginalized(True)
        opt.add_vertex(v)
    info = np.eye(2)
    kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
    for k in range(len(obs_frame)):
        e = g2o.EdgeProjectXYZ2UV()
        e.set_parameter_id(0, 0)
        e.set_vertex(0, opt.vertex(F + obs_point[k]))
        e.set_vertex(1, opt.vertex(obs_frame[k]))
        e.set_measurement(obs_uv[k])
        e.set_information(info)
        e.set_robust_kernel(kernel)
        assert opt.add_edge(e)
    opt.initialize_optimization()
    iters = opt.optimize(30)
    assert iters > 0
    T_out = np.zeros((F, 4, 4))
    for i in range(F):
        est = opt.vertex(i).estimate()
        T_out[i, :3, :3] = est.rotation().matrix()
        T_out[i, :3, 3] = est.translation()
        T_out[i, 3, 3] = 1.0
    pts_out = np.array([opt.vertex(F + j).estimate() for j in range(M)])
    return T_out, pts_out


def _solve_native(T0, pts0, obs_frame, obs_point, obs_uv, fixed_rows):
    F, M = len(T0), len(pts0)
    fixed = np.zeros(F, dtype=np.uint8)
    for i in fixed_rows:
        fixed[i] = 1
    inp = BASolveInputs(
        T_wc=np.ascontiguousarray(T0), fixed=fixed,
        positions=np.ascontiguousarray(pts0), points_fixed=0,
        obs_frame=obs_frame, obs_point=obs_point,
        obs_uv=np.ascontiguousarray(obs_uv), max_iterations=30)
    T_out, pts_out, iters, chi2 = solve_se3_ba(inp)
    assert iters > 0
    return np.asarray(T_out), np.asarray(pts_out), iters


def test_native_vs_g2o_python_drift_recorded():
    gt_T_cw, gt_pts, obs_frame, obs_point, obs_uv = _scene()
    T0_cw, pts0 = _perturb(gt_T_cw, gt_pts)
    # The solver convention is world-to-camera (inv of the c2w pose).
    T0 = np.stack([np.linalg.inv(T0_cw[i]) for i in range(len(T0_cw))])
    gt_T = np.stack([np.linalg.inv(gt_T_cw[i]) for i in range(len(gt_T_cw))])
    fixed_rows = {0}  # the gauge anchor: the first frame

    ref_T, ref_pts = _solve_g2o_python(T0, pts0, obs_frame, obs_point,
                                       obs_uv, fixed_rows)
    nat_T, nat_pts, _ = _solve_native(T0, pts0, obs_frame, obs_point,
                                      obs_uv, fixed_rows)

    dT = np.abs(ref_T - nat_T).max()
    dpts_rel = (np.abs(ref_pts - nat_pts).max() /
                np.abs(ref_pts).max())
    # The recorded bound (C2b1): the two solves agree to <1e-6 - the
    # same solver source and config, so the solutions coincide. The
    # ML estimate is NOT guaranteed closer to the ground truth than
    # the perturbed init (that assertion was wrong and is removed);
    # the convergence invariant is the objective itself:
    # chi2(post) <= chi2(pre), which Levenberg guarantees.
    assert dpts_rel < 1e-6, f"point estimate drift {dpts_rel:.3e}"
    assert dT < 1e-6, f"pose estimate drift {dT:.3e}"

    homo = np.column_stack(
        [pts0[obs_point], np.ones(len(obs_point))])  # (O, 4)
    cam = np.einsum('nij,nj->ni', T0[obs_frame], homo)[:, :3]
    uv0 = cam[:, :2] / cam[:, 2:3]
    r0 = uv0 - obs_uv
    chi2_pre = float((r0 * r0).sum())
    homo1 = np.column_stack(
        [nat_pts[obs_point], np.ones(len(obs_point))])
    inp_cam = np.einsum('nij,nj->ni', nat_T[obs_frame], homo1)[:, :3]
    uv1 = inp_cam[:, :2] / inp_cam[:, 2:3]
    r1 = uv1 - obs_uv
    chi2_post = float((r1 * r1).sum())
    assert chi2_post <= chi2_pre, (chi2_pre, chi2_post)


def test_native_solve_is_deterministic():
    gt_T_cw, gt_pts, obs_frame, obs_point, obs_uv = _scene()
    T0_cw, pts0 = _perturb(gt_T_cw, gt_pts)
    T0 = np.stack([np.linalg.inv(T0_cw[i]) for i in range(len(T0_cw))])
    a = _solve_native(T0, pts0, obs_frame, obs_point, obs_uv, {0})
    b = _solve_native(T0, pts0, obs_frame, obs_point, obs_uv, {0})
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]  # iterations
