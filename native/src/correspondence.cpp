#include "correspondence.h"

#include <cmath>
#include <stdexcept>

namespace slarnative {

namespace {

inline double hamming_dist(const uint8_t* a, const uint8_t* b, int w) {
  int d = 0;
  for (int i = 0; i < w; ++i) {
    const uint8_t x = static_cast<uint8_t>(a[i] ^ b[i]);
    d += __builtin_popcount(x);
  }
  return static_cast<double>(d);
}

inline double l2_dist(const uint8_t* a, const uint8_t* b, int w) {
  // f32 descriptors widened to f64, accumulated in f64 (the
  // pre-registered route-B formulation).
  double s = 0.0;
  const float* fa = reinterpret_cast<const float*>(a);
  const float* fb = reinterpret_cast<const float*>(b);
  for (int i = 0; i < w / 4; ++i) {
    const double d = static_cast<double>(fa[i]) - static_cast<double>(fb[i]);
    s += d * d;
  }
  return std::sqrt(s);
}

}  // namespace

py::object match_projected(const MatchInputs& in) {
  const auto uv_in = in.point_uv.unchecked<2>();
  const auto pd_in = in.point_desc.unchecked<2>();
  const auto phas_in = in.point_has_desc.unchecked<1>();
  const auto kc_in = in.kp_coords.unchecked<2>();
  const auto kd_in = in.kp_desc.unchecked<2>();
  const auto obs_in = in.already_observed.unchecked<1>();

  const long M = uv_in.shape(0);
  const long K = kc_in.shape(0);
  const long W = pd_in.shape(1);
  if (kd_in.shape(1) != W) {
    throw std::invalid_argument(
        "match_projected: point and keypoint descriptor widths must match");
  }
  if (in.point_has_desc.shape(0) != M || obs_in.shape(0) != M) {
    throw std::invalid_argument("match_projected: (M,) arrays must be M-sized");
  }
  if (in.kp_desc.shape(0) != K) {
    throw std::invalid_argument("match_projected: kp_desc must be (K, W)");
  }
  if (in.desc_kind != 1 && in.desc_kind != 2) {
    throw std::invalid_argument(
        "match_projected: desc_kind must be 1 (HAMMING) or 2 (L2)");
  }
  if (in.desc_kind == 2 && W % 4 != 0) {
    throw std::invalid_argument(
        "match_projected: L2 descriptor width must be a multiple of 4");
  }

  // The outputs: accepted pairs in the candidate order. The count is
  // bounded by M; allocate the max and slice in Python.
  py::array_t<int64_t> point_rows(M);
  py::array_t<int64_t> kp_indices(M);
  py::array_t<double> repro_errors(M);
  auto pr_buf = point_rows.mutable_unchecked<1>();
  auto ki_buf = kp_indices.mutable_unchecked<1>();
  auto re_buf = repro_errors.mutable_unchecked<1>();

  long n_matches = 0;
  {
    py::gil_scoped_release release;

    const int w = static_cast<int>(W);
    for (long m = 0; m < M; ++m) {
      const double u_proj = uv_in(m, 0);
      const double v_proj = uv_in(m, 1);
      if (!std::isfinite(u_proj) || !std::isfinite(v_proj)) {
        continue;  // behind camera
      }
      if (!(in.margin <= u_proj && u_proj < in.width - in.margin &&
            in.margin <= v_proj && v_proj < in.height - in.margin)) {
        continue;  // out of bounds
      }
      if (obs_in(m) != 0) {
        continue;  // the frame already observes this point
      }
      if (phas_in(m) == 0) {
        continue;  // no descriptor to match with
      }

      long best_idx = -1;
      double best_dist = 0.0;
      for (long k = 0; k < K; ++k) {
        const double du = kc_in(k, 0) - u_proj;
        const double dv = kc_in(k, 1) - v_proj;
        if (du * du + dv * dv > in.search_radius * in.search_radius) {
          // The neighborhood bound: the keypoint-to-projection PIXEL
          // distance (the spatial ball, r = dist_thresh) - a separate
          // quantity from the descriptor accept threshold.
          continue;
        }
        const double d = (in.desc_kind == 1)
                             ? hamming_dist(&pd_in(m, 0), &kd_in(k, 0), w)
                             : l2_dist(&pd_in(m, 0), &kd_in(k, 0), w);
        if (best_idx == -1 || d < best_dist) {
          best_dist = d;
          best_idx = k;
        }
      }

      if (best_idx != -1 && best_dist < in.reloc_thresh) {
        // The reprojection error: the pixel distance between the
        // accepted keypoint and the projection, in the EXACT
        // formulation the Python path used (np.linalg.norm on a
        // 2-vector = sqrt(fma(dy, dy, dx*dx)) - the FMA form matched
        // numpy on 20000/20000 inputs; the elementwise form differs on
        // 8.8% and its ULP shifts cascade through the point-quality
        // averages into culling).
        const double du = kc_in(best_idx, 0) - u_proj;
        const double dv = kc_in(best_idx, 1) - v_proj;
        pr_buf(n_matches) = m;
        ki_buf(n_matches) = best_idx;
        re_buf(n_matches) = std::sqrt(std::fma(dv, dv, du * du));
        ++n_matches;
      }
    }
  }

  return py::make_tuple(point_rows, kp_indices, repro_errors,
                        py::int_(n_matches));
}

}  // namespace slarnative
