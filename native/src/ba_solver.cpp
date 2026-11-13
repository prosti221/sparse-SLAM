#include "ba_solver.h"

#include <cmath>
#include <memory>
#include <stdexcept>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/parameter_cameraparameters.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>

namespace slarnative {

namespace {
constexpr double kHuberDelta = 2.4497425365372686;  // sqrt(5.991)
constexpr double kGainThreshold = 1e-4;             // BA_TERMINATE_GAIN_THRESHOLD
}  // namespace

py::object solve_se3_ba(const BASolveInputs& in) {
  const auto T_in = in.T_wc.unchecked<3>();
  const auto fx_in = in.fixed.unchecked<1>();
  const auto pos_in = in.positions.unchecked<2>();
  const auto ofr = in.obs_frame.unchecked<1>();
  const auto opt_in = in.obs_point.unchecked<1>();
  const auto uv_in = in.obs_uv.unchecked<2>();

  const long F = T_in.shape(0);
  const long M = pos_in.shape(0);
  const long O = ofr.shape(0);

  if (in.fixed.shape(0) != F || in.T_wc.shape(1) != 4 ||
      in.T_wc.shape(2) != 4 || pos_in.shape(1) != 3 ||
      opt_in.shape(0) != O || in.obs_uv.shape(0) != O ||
      in.obs_uv.shape(1) != 2) {
    throw std::invalid_argument(
        "solve_se3_ba: shapes must be T_wc (F,4,4), fixed (F,), "
        "positions (M,3), obs_* (O,), obs_uv (O,2)");
  }
  if (in.max_iterations <= 0) {
    throw std::invalid_argument("solve_se3_ba: max_iterations must be > 0");
  }
  for (long i = 0; i < O; ++i) {
    if (ofr(i) < 0 || ofr(i) >= F || opt_in(i) < 0 || opt_in(i) >= M) {
      throw std::invalid_argument(
          "solve_se3_ba: obs rows out of range");
    }
  }

  // Outputs first (allocation needs the GIL), filled inside the
  // released section.
  py::array_t<double> T_out({F, 4L, 4L});
  py::array_t<double> pos_out({M, 3L});
  auto T_buf = T_out.mutable_unchecked<3>();
  auto pos_buf = pos_out.mutable_unchecked<2>();

  int iterations = -1;
  double chi2 = 0.0;

  {
    py::gil_scoped_release release;

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    using LSE = g2o::LinearSolverEigen<
        g2o::BlockSolver_6_3::PoseMatrixType>;  // LinearSolverEigenSE3
    auto linear = std::make_unique<LSE>();
    auto block = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear));
    auto levenberg = std::make_unique<g2o::OptimizationAlgorithmLevenberg>(
        std::move(block));
    std::shared_ptr<g2o::OptimizationAlgorithm> algo(std::move(levenberg));
    optimizer.setAlgorithm(algo);

    auto terminate = std::make_shared<g2o::SparseOptimizerTerminateAction>();
    terminate->setGainThreshold(kGainThreshold);
    optimizer.addPostIterationAction(terminate);

    auto cam = std::make_shared<g2o::CameraParameters>(
        1.0, g2o::Vector2(0.0, 0.0), 0.0);
    cam->setId(0);
    optimizer.addParameter(cam);

    // Frame vertices: ids 0..F-1, estimate = WORLD-TO-CAMERA (the
    // production build's convention).
    for (long i = 0; i < F; ++i) {
      Eigen::Matrix3d R;
      Eigen::Vector3d t;
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) R(r, c) = T_in(i, r, c);
        t(r) = T_in(i, r, 3);
      }
      auto v = std::make_shared<g2o::VertexSE3Expmap>();
      v->setId(static_cast<int>(i));
      v->setEstimate(g2o::SE3Quat(R, t));
      v->setFixed(fx_in(i) != 0);
      optimizer.addVertex(v);
    }

    // Point vertices: ids F..F+M-1, marginalized (the Python path's
    // VertexPointXYZ.set_marginalized(True)).
    for (long j = 0; j < M; ++j) {
      auto v = std::make_shared<g2o::VertexPointXYZ>();
      v->setId(static_cast<int>(F + j));
      Eigen::Vector3d p(pos_in(j, 0), pos_in(j, 1), pos_in(j, 2));
      v->setEstimate(p);
      v->setFixed(in.points_fixed != 0);
      v->setMarginalized(true);
      optimizer.addVertex(v);
    }

    // Edges in the observation order (the Python build's order):
    // vertex(0) = point, vertex(1) = frame, measurement = the
    // normalized 2d coords, information eye(2), a fresh Huber kernel.
    Eigen::Matrix2d info = Eigen::Matrix2d::Identity();
    for (long i = 0; i < O; ++i) {
      auto e = std::make_shared<g2o::EdgeProjectXYZ2UV>();
      e->setParameterId(0, 0);
      e->setVertex(0, optimizer.vertex(static_cast<int>(F + opt_in(i))));
      e->setVertex(1, optimizer.vertex(static_cast<int>(ofr(i))));
      e->setMeasurement(
          g2o::Vector2(uv_in(i, 0), uv_in(i, 1)));
      e->setInformation(info);
      auto huber = std::make_shared<g2o::RobustKernelHuber>();
      huber->setDelta(kHuberDelta);
      e->setRobustKernel(huber);
      optimizer.addEdge(e);
    }

    optimizer.initializeOptimization();
    iterations = optimizer.optimize(in.max_iterations);
    if (iterations < 0) {
      throw std::runtime_error("solve_se3_ba: optimizer failed");
    }
    chi2 = optimizer.chi2();

    for (long i = 0; i < F; ++i) {
      const auto v = std::static_pointer_cast<const g2o::VertexSE3Expmap>(
          optimizer.vertex(static_cast<int>(i)));
      const g2o::SE3Quat& est = v->estimate();
      const Eigen::Matrix3d R = est.rotation().toRotationMatrix();
      const Eigen::Vector3d& t = est.translation();
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T_buf(i, r, c) = R(r, c);
        T_buf(i, r, 3) = t(r);
      }
      for (int c = 0; c < 4; ++c) T_buf(i, 3, c) = (c == 3) ? 1.0 : 0.0;
    }
    // NOTE: T_out rows are the WORLD-TO-CAMERA estimates - the same
    // convention as the input; the Python write-back maps them to
    // camera-to-world with inv() exactly as the production code does.
    for (long j = 0; j < M; ++j) {
      const auto v = std::static_pointer_cast<const g2o::VertexPointXYZ>(
          optimizer.vertex(static_cast<int>(F + j)));
      const Eigen::Vector3d& p = v->estimate();
      pos_buf(j, 0) = p(0);
      pos_buf(j, 1) = p(1);
      pos_buf(j, 2) = p(2);
    }
  }

  py::object it_obj = py::int_(iterations);
  py::object chi_obj = py::float_(chi2);
  return py::make_tuple(T_out, pos_out, it_obj, chi_obj);
}

}  // namespace slarnative
