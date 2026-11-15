#pragma once

#include "ba_constants.h"  // kHuberDelta
#include "error_fold.h"  // c_arr

#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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

// The persistent local-BA graph (migration C2b2b): the g2o optimizer,
// the vertex caches, and the edge-pair bookkeeping live HERE across
// calls. One `solve_local` transaction replaces the Python reconcile
// with the EXACT op order of the g2o-python path it ports:
//   1. the stale-vertex removal cascade (drops the removed vertices'
//      edges and their pair-registry entries),
//   2. new point vertices in map order (marginalized),
//   3. new keyframe vertices in map order,
//   4. the delta edge pass in the observations list order (a pair
//      already in the registry is skipped; a FAILED add_edge stays
//      "missing" so the next reconcile re-creates it),
//   5. the fixed-flag refresh (the gauge anchor + window boundary),
//   6. initializeOptimization + optimize,
//   7. extraction of ALL current vertex estimates.
// With identical creation order and the same monotonic vertex-id
// counter semantics, the solve sequence is identical to g2o-python's
// (the C2b1 zero-drift precedent) - bit-identity per solve is the
// acceptance bar. The Python side keeps membership/anchor
// computation, the write-back, the fold, and the telemetry.
class PersistentLocalBA {
 public:
  explicit PersistentLocalBA(bool verbose);

  void reset();

  // T_wc / positions are the CURRENT map geometry, consumed ONLY for
  // vertices not yet in the graph (existing vertices keep their
  // in-graph estimates - the persistent semantics).
  py::object solve_local(
      const std::vector<uint64_t>& frame_ids,  // target KF membership, map order
      const std::vector<uint64_t>& point_ids,  // target point membership, map order
      py::array_t<double, py::array::c_style | py::array::forcecast> T_wc,
      py::array_t<double, py::array::c_style | py::array::forcecast> positions,
      py::array_t<uint8_t, py::array::c_style | py::array::forcecast> in_window,
      int anchor_row,
      uint8_t points_fixed,
      const std::vector<uint64_t>& obs_frame_row,  // rows into frame_ids
      const std::vector<uint64_t>& obs_point_row,  // rows into point_ids
      py::array_t<double, py::array::c_style | py::array::forcecast> obs_uv,
      int max_iterations);

  size_t num_frames() const { return kf_vertices_.size(); }
  size_t num_points() const { return pt_vertices_.size(); }
  size_t num_edges() const;
  size_t fixed_pose_count() const;
  bool has_frame(uint64_t id) const { return kf_vertices_.count(id) != 0; }
  bool has_point(uint64_t id) const { return pt_vertices_.count(id) != 0; }

  // The robust kernel scale (D3): the delta for NEWLY-created edges,
  // and an update pass over the existing edges' kernels (the graph
  // persists across solves - a mid-run delta change must reach the
  // in-graph edges). The default is kHuberDelta (behaviorally
  // identical until the Python side passes a scaled delta).
  void set_huber_delta(double delta);
  double huber_delta() const { return huber_delta_; }

  // Both views of the (point, keyframe) edge pairs, for the stage
  // pin that asserts the registry mirrors the graph.
  std::vector<std::pair<uint64_t, uint64_t>> graph_edge_pairs() const;
  std::vector<std::pair<uint64_t, uint64_t>> bookkeeping_pairs() const;

  // The graph's estimates (world-to-camera 4x4 for frames; 3-vectors
  // for points) - the write-back invariant's verification surface.
  std::vector<double> frame_estimate_T_wc(uint64_t id) const;
  std::vector<double> point_estimate(uint64_t id) const;

 private:
  void ensure_optimizer();
  void drop_stale(const std::vector<uint64_t>& frame_ids,
                  const std::vector<uint64_t>& point_ids);

  std::shared_ptr<g2o::SparseOptimizer> optimizer_;
  bool verbose_;

  std::unordered_map<uint64_t, std::shared_ptr<g2o::VertexSE3Expmap>> kf_vertices_;
  std::unordered_map<uint64_t, std::shared_ptr<g2o::VertexPointXYZ>> pt_vertices_;
  // The pair registry (the Python path's _graph_edges_by_pt/_kf).
  // The optimizer owns the edges STRONGLY (HyperGraph::edges_ is an
  // EdgeSet of shared_ptr); no separate owner registry is needed.
  std::unordered_map<uint64_t, std::unordered_set<uint64_t>> edges_by_pt_;
  std::unordered_map<uint64_t, std::unordered_set<uint64_t>> edges_by_kf_;
  double huber_delta_ = kHuberDelta;
  int next_vertex_id_ = 0;
};

}  // namespace slarnative
