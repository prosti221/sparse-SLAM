#include "persistent_ba.h"
#include "ba_constants.h"

#include <cmath>
#include <set>
#include <stdexcept>

namespace slarnative {

PersistentLocalBA::PersistentLocalBA(bool verbose) : verbose_(verbose) {
  ensure_optimizer();
}

void PersistentLocalBA::ensure_optimizer() {
  optimizer_ = std::make_shared<g2o::SparseOptimizer>();
  optimizer_->setVerbose(verbose_);

  using LSE = g2o::LinearSolverEigen<
      g2o::BlockSolver_6_3::PoseMatrixType>;  // LinearSolverEigenSE3
  auto linear = std::make_unique<LSE>();
  auto block = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear));
  std::shared_ptr<g2o::OptimizationAlgorithm> algo =
      std::make_shared<g2o::OptimizationAlgorithmLevenberg>(std::move(block));
  optimizer_->setAlgorithm(algo);

  auto terminate = std::make_shared<g2o::SparseOptimizerTerminateAction>();
  terminate->setGainThreshold(kGainThreshold);
  optimizer_->addPostIterationAction(terminate);

  auto cam = std::make_shared<g2o::CameraParameters>(
      1.0, g2o::Vector2(0.0, 0.0), 0.0);
  cam->setId(0);
  optimizer_->addParameter(cam);

  next_vertex_id_ = 0;
}

void PersistentLocalBA::reset() {
  // The wholesale invalidation: a fresh optimizer and empty caches -
  // exactly what the Python path's reset_persistent_graph produced.
  kf_vertices_.clear();
  pt_vertices_.clear();
  edges_by_pt_.clear();
  edges_by_kf_.clear();
  ensure_optimizer();
}

void PersistentLocalBA::drop_stale(
    const std::vector<uint64_t>& frame_ids,
    const std::vector<uint64_t>& point_ids) {
  std::set<uint64_t> target_kf(frame_ids.begin(), frame_ids.end());
  std::set<uint64_t> target_pt(point_ids.begin(), point_ids.end());

  for (auto it = kf_vertices_.begin(); it != kf_vertices_.end();) {
    if (target_kf.count(it->first)) {
      ++it;
      continue;
    }
    const uint64_t fid = it->first;
    optimizer_->removeVertex(it->second, false);  // cascades the edges
    it = kf_vertices_.erase(it);
    auto registry = edges_by_kf_.find(fid);
    if (registry != edges_by_kf_.end()) {
      for (uint64_t pid : registry->second) {
        auto pt_pairs = edges_by_pt_.find(pid);
        if (pt_pairs != edges_by_pt_.end()) {
          pt_pairs->second.erase(fid);
          if (pt_pairs->second.empty()) edges_by_pt_.erase(pt_pairs);
        }
      }
      edges_by_kf_.erase(registry);
    }
  }

  for (auto it = pt_vertices_.begin(); it != pt_vertices_.end();) {
    if (target_pt.count(it->first)) {
      ++it;
      continue;
    }
    const uint64_t pid = it->first;
    optimizer_->removeVertex(it->second, false);
    it = pt_vertices_.erase(it);
    auto registry = edges_by_pt_.find(pid);
    if (registry != edges_by_pt_.end()) {
      for (uint64_t fid : registry->second) {
        auto kf_pairs = edges_by_kf_.find(fid);
        if (kf_pairs != edges_by_kf_.end()) {
          kf_pairs->second.erase(pid);
          if (kf_pairs->second.empty()) edges_by_kf_.erase(kf_pairs);
        }
      }
      edges_by_pt_.erase(registry);
    }
  }
}

py::object PersistentLocalBA::solve_local(
    const std::vector<uint64_t>& frame_ids,
    const std::vector<uint64_t>& point_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> T_wc,
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> in_window,
    int anchor_row,
    uint8_t points_fixed,
    const std::vector<uint64_t>& obs_frame_row,
    const std::vector<uint64_t>& obs_point_row,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_uv,
    int max_iterations) {
  const long F = static_cast<long>(frame_ids.size());
  const long M = static_cast<long>(point_ids.size());
  const long O = static_cast<long>(obs_frame_row.size());

  if (T_wc.ndim() != 3 || T_wc.shape(0) != F || T_wc.shape(1) != 4 ||
      T_wc.shape(2) != 4) {
    throw std::invalid_argument("solve_local: T_wc must be (F, 4, 4)");
  }
  if (positions.ndim() != 2 || positions.shape(0) != M ||
      positions.shape(1) != 3) {
    throw std::invalid_argument("solve_local: positions must be (M, 3)");
  }
  if (in_window.ndim() != 1 || in_window.shape(0) != F) {
    throw std::invalid_argument("solve_local: in_window must be (F,)");
  }
  if (obs_point_row.size() != obs_frame_row.size() || obs_uv.ndim() != 2 ||
      obs_uv.shape(0) != O || obs_uv.shape(1) != 2) {
    throw std::invalid_argument(
        "solve_local: obs arrays must be (O,) / (O, 2) of equal length");
  }
  if (F > 0 && (anchor_row < 0 || anchor_row >= F)) {
    throw std::invalid_argument("solve_local: anchor_row out of range");
  }
  if (max_iterations <= 0) {
    throw std::invalid_argument("solve_local: max_iterations must be > 0");
  }
  {
    std::set<uint64_t> seen;
    for (uint64_t id : frame_ids) {
      if (!seen.insert(id).second)
        throw std::invalid_argument("solve_local: duplicate frame id");
    }
    seen.clear();
    for (uint64_t id : point_ids) {
      if (!seen.insert(id).second)
        throw std::invalid_argument("solve_local: duplicate point id");
    }
  }

  // Outputs first (allocation needs the GIL), filled after the solve.
  py::array_t<double> T_out({F, 4L, 4L});
  py::array_t<double> pos_out({M, 3L});

  int iterations = -1;
  size_t actual_edges = 0;
  {
    py::gil_scoped_release release;

    const auto T_in = T_wc.unchecked<3>();
    const auto pos_in = positions.unchecked<2>();
    const auto win_in = in_window.unchecked<1>();
    const auto uv_in = obs_uv.unchecked<2>();

    drop_stale(frame_ids, point_ids);

    // New point vertices in map order (marginalized; fixed per
    // fix_points) - the Python reconcile's creation order.
    for (long j = 0; j < M; ++j) {
      const uint64_t pid = point_ids[j];
      if (pt_vertices_.count(pid)) continue;
      auto v = std::make_shared<g2o::VertexPointXYZ>();
      v->setId(next_vertex_id_++);
      v->setEstimate(Eigen::Vector3d(pos_in(j, 0), pos_in(j, 1), pos_in(j, 2)));
      v->setFixed(points_fixed != 0);
      v->setMarginalized(true);
      optimizer_->addVertex(v);
      pt_vertices_.emplace(pid, std::move(v));
    }

    // New keyframe vertices in map order.
    for (long i = 0; i < F; ++i) {
      const uint64_t fid = frame_ids[i];
      if (kf_vertices_.count(fid)) continue;
      Eigen::Matrix3d R;
      Eigen::Vector3d t;
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) R(r, c) = T_in(i, r, c);
        t(r) = T_in(i, r, 3);
      }
      auto v = std::make_shared<g2o::VertexSE3Expmap>();
      v->setId(next_vertex_id_++);
      v->setEstimate(g2o::SE3Quat(R, t));
      optimizer_->addVertex(v);
      kf_vertices_.emplace(fid, std::move(v));
    }

    // The delta edge pass in the observations list order.
    Eigen::Matrix2d info = Eigen::Matrix2d::Identity();
    for (long i = 0; i < O; ++i) {
      const uint64_t pid = point_ids[obs_point_row[i]];
      const uint64_t fid = frame_ids[obs_frame_row[i]];
      auto pt_pairs = edges_by_pt_.find(pid);
      if (pt_pairs != edges_by_pt_.end() && pt_pairs->second.count(fid)) {
        continue;  // the pair is already in the graph
      }
      auto e = std::make_shared<g2o::EdgeProjectXYZ2UV>();
      e->setParameterId(0, 0);
      e->setVertex(0, pt_vertices_.at(pid));
      e->setVertex(1, kf_vertices_.at(fid));
      e->setMeasurement(g2o::Vector2(uv_in(i, 0), uv_in(i, 1)));
      e->setInformation(info);
      auto huber = std::make_shared<g2o::RobustKernelHuber>();
      huber->setDelta(kHuberDelta);
      e->setRobustKernel(huber);
      if (optimizer_->addEdge(e)) {
        // Register only if the edge entered the graph: a failed edge
        // stays "missing" so the next reconcile re-creates it (the
        // Python path's exact rule).
        edges_by_pt_[pid].insert(fid);
        edges_by_kf_[fid].insert(pid);
      }
    }

    // The fixed-flag refresh (the anchor slides with the window).
    for (long i = 0; i < F; ++i) {
      kf_vertices_.at(frame_ids[i])->setFixed(
          i == anchor_row || win_in(i) == 0);
    }

    optimizer_->initializeOptimization();
    iterations = optimizer_->optimize(max_iterations);
    actual_edges = optimizer_->edges().size();

    auto T_buf = T_out.mutable_unchecked<3>();
    for (long i = 0; i < F; ++i) {
      const g2o::SE3Quat& est = kf_vertices_.at(frame_ids[i])->estimate();
      const Eigen::Matrix3d R = est.rotation().toRotationMatrix();
      const Eigen::Vector3d& t = est.translation();
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T_buf(i, r, c) = R(r, c);
        T_buf(i, r, 3) = t(r);
      }
      for (int c = 0; c < 4; ++c) T_buf(i, 3, c) = (c == 3) ? 1.0 : 0.0;
    }
    auto pos_buf = pos_out.mutable_unchecked<2>();
    for (long j = 0; j < M; ++j) {
      const Eigen::Vector3d& p = pt_vertices_.at(point_ids[j])->estimate();
      pos_buf(j, 0) = p(0);
      pos_buf(j, 1) = p(1);
      pos_buf(j, 2) = p(2);
    }
  }

  return py::make_tuple(T_out, pos_out, py::int_(iterations),
                        py::int_(static_cast<long>(actual_edges)));
}

size_t PersistentLocalBA::num_edges() const {
  return optimizer_->edges().size();
}

size_t PersistentLocalBA::fixed_pose_count() const {
  size_t n = 0;
  for (const auto& [id, v] : kf_vertices_) {
    if (v->fixed()) ++n;
  }
  return n;
}

std::vector<std::pair<uint64_t, uint64_t>>
PersistentLocalBA::graph_edge_pairs() const {
  // Resolved from the ACTUAL edges via the vertex maps (the g2o id
  // spaces differ from the object ids; the shared pointers are the
  // join key).
  std::unordered_map<const g2o::HyperGraph::Vertex*, uint64_t> pt_by_ptr;
  std::unordered_map<const g2o::HyperGraph::Vertex*, uint64_t> kf_by_ptr;
  pt_by_ptr.reserve(pt_vertices_.size());
  kf_by_ptr.reserve(kf_vertices_.size());
  for (const auto& [pid, v] : pt_vertices_) pt_by_ptr.emplace(v.get(), pid);
  for (const auto& [fid, v] : kf_vertices_) kf_by_ptr.emplace(v.get(), fid);

  std::vector<std::pair<uint64_t, uint64_t>> out;
  out.reserve(optimizer_->edges().size());
  for (const auto& e : optimizer_->edges()) {
    const auto& vs = e->vertices();
    if (vs.size() < 2) continue;
    auto pt_it = pt_by_ptr.find(vs[0].get());
    auto kf_it = kf_by_ptr.find(vs[1].get());
    if (pt_it == pt_by_ptr.end() || kf_it == kf_by_ptr.end()) continue;
    out.emplace_back(pt_it->second, kf_it->second);
  }
  return out;
}

std::vector<std::pair<uint64_t, uint64_t>>
PersistentLocalBA::bookkeeping_pairs() const {
  std::vector<std::pair<uint64_t, uint64_t>> out;
  for (const auto& [pid, fids] : edges_by_pt_) {
    for (uint64_t fid : fids) out.emplace_back(pid, fid);
  }
  return out;
}

std::vector<double> PersistentLocalBA::frame_estimate_T_wc(uint64_t id) const {
  auto it = kf_vertices_.find(id);
  if (it == kf_vertices_.end()) throw std::out_of_range("unknown frame id");
  const g2o::SE3Quat& est = it->second->estimate();
  const Eigen::Matrix3d R = est.rotation().toRotationMatrix();
  const Eigen::Vector3d& t = est.translation();
  std::vector<double> out(16, 0.0);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) out[r * 4 + c] = R(r, c);
    out[r * 4 + 3] = t(r);
  }
  out[15] = 1.0;
  return out;
}

std::vector<double> PersistentLocalBA::point_estimate(uint64_t id) const {
  auto it = pt_vertices_.find(id);
  if (it == pt_vertices_.end()) throw std::out_of_range("unknown point id");
  const Eigen::Vector3d& p = it->second->estimate();
  return {p(0), p(1), p(2)};
}

}  // namespace slarnative
