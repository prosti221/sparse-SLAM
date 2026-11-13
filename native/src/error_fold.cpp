#include "error_fold.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace slarnative {

// The error-fold kernel (migration C2a): a pure function over the
// BA-input SOA. Mirrors the Python op order exactly:
//   pass 1 - group observations by frame in FIRST-APPEARANCE order,
//            project as a per-row 4-term index-order dot
//            (pose_inv @ [X, Y, Z, 1]), skip behind-camera points,
//            norm = sqrt(dx*dx + dy*dy) elementwise;
//   pass 2 - apply the errors in the ORIGINAL observation order, each
//            point's running average with n = its GLOBAL observation
//            count (Point.update_reprojection_error semantics with
//            update_quality=False).
// The ONLY order-sensitive arithmetic is the 4-term dot - numpy/BLAS
// may contract or reorder it relative to the scalar index-order loop,
// so the port's drift is measured against the Python oracle on the
// stage fixtures and the pin tolerance is set from that measurement.

FoldOutputs error_fold(const FoldInputs& in) {
  const auto pose = in.poses_inv.unchecked<3>();
  const auto ofr = in.obs_frame.unchecked<1>();
  const auto ppos = in.point_pos.unchecked<2>();
  const auto onrm = in.obs_norm.unchecked<2>();
  const auto opt = in.obs_point.unchecked<1>();
  const auto pavg = in.point_avg.unchecked<1>();
  const auto pnobs = in.point_nobs.unchecked<1>();

  const long F = pose.shape(0);
  const long O = ofr.shape(0);
  const long M = pavg.shape(0);

  if (in.point_pos.shape(0) != M || in.point_pos.shape(1) != 3 ||
      in.obs_norm.shape(0) != O || in.obs_norm.shape(1) != 2 ||
      opt.shape(0) != O || pnobs.shape(0) != M) {
    throw std::invalid_argument(
        "error_fold: arrays must be point_pos (M,3), obs_norm (O,2), "
        "obs_frame/obs_point (O,), M-sized point state");
  }

  FoldOutputs out;
  out.err = py::array_t<double>(O);
  out.new_avg = py::array_t<double>(M);
  out.updated = py::array_t<uint8_t>(M);
  auto err = out.err.mutable_unchecked<1>();
  auto navg = out.new_avg.mutable_unchecked<1>();
  auto upd = out.updated.mutable_unchecked<1>();

  for (long i = 0; i < M; ++i) {
    navg(i) = pavg(i);
    upd(i) = 0;
  }
  for (long i = 0; i < O; ++i) err(i) = NAN;

  // The GIL is released for the whole fold: the inputs stay alive via
  // the call arguments and nothing Python is touched below.
  py::gil_scoped_release release;

  // Pass 1: first-appearance frame grouping. The frames are few (the
  // local window), so a linear scan per distinct frame is fine and
  // preserves the Python dict's insertion order exactly.
  std::vector<int64_t> frame_order;
  std::vector<long> counts;
  std::vector<std::vector<long>> groups;
  for (long i = 0; i < O; ++i) {
    const int64_t f = ofr(i);
    long slot = -1;
    for (size_t g = 0; g < frame_order.size(); ++g) {
      if (frame_order[g] == f) {
        slot = static_cast<long>(g);
        break;
      }
    }
    if (slot < 0) {
      frame_order.push_back(f);
      groups.emplace_back();
      slot = static_cast<long>(groups.size()) - 1;
    }
    groups[slot].push_back(i);
  }

  for (const auto& group : groups) {
    const long fi = ofr(group[0]);
    for (long i : group) {
      const long pi = opt(i);
      const double X = ppos(pi, 0), Y = ppos(pi, 1), Z = ppos(pi, 2);
      // Row-wise 4-term index-order dot: pose_inv row @ [X, Y, Z, 1].
      double cam[4];
      for (int r = 0; r < 4; ++r) {
        cam[r] = pose(fi, r, 0) * X + pose(fi, r, 1) * Y +
                 pose(fi, r, 2) * Z + pose(fi, r, 3);
      }
      if (!(cam[2] > 0.0)) {
        // Behind camera: the Python path leaves NaN there too
        // (project_points pre-fills NaN and skips invalid rows).
        continue;
      }
      const double xn = cam[0] / cam[2];
      const double yn = cam[1] / cam[2];
      if (!(std::isfinite(xn) && std::isfinite(yn))) {
        // project_points' isfinite mask on the projected coords.
        continue;
      }
      const double dx = xn - onrm(i, 0);
      const double dy = yn - onrm(i, 1);
      err(i) = std::sqrt(dx * dx + dy * dy);
    }
  }

  // Pass 2: apply in the original observation order.
  for (long i = 0; i < O; ++i) {
    if (std::isnan(err(i))) continue;
    const long pi = opt(i);
    const int64_t n = pnobs(pi);
    if (n > 0) {
      navg(pi) = (navg(pi) * static_cast<double>(n - 1) + err(i)) /
                 static_cast<double>(n);
      upd(pi) = 1;
    }
  }

  return out;
}

}  // namespace slarnative
