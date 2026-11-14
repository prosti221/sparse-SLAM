#pragma once

#include "error_fold.h"  // c_arr

#include <cstdint>
#include <vector>

namespace slarnative {

struct MatchInputs {
  c_arr<double> point_uv;      // (M, 2) projected pixel coords; NaN rows
                               // = behind camera (skipped)
  c_arr<int64_t> point_rows;   // (M,) the candidates' rows into the
                               // descriptor/has-desc blocks
  c_arr<uint8_t> point_desc;   // (B, W) the descriptor BLOCK (the
                               // registry's live rows; read via point_rows)
  c_arr<uint8_t> point_has_desc;  // (B,) 0/1
  c_arr<double> kp_coords;     // (K, 2)
  c_arr<uint8_t> kp_desc;      // (K, W)
  c_arr<uint8_t> already_observed;  // (M,) 0/1 - the current frame already
                                    // observes this point
  int desc_kind;               // 1 = uint8 HAMMING (popcount), 2 = f32 L2
  double search_radius;        // the SPATIAL ball radius in pixels
                               // (the angular-resolved dist_thresh)
  double reloc_thresh;         // the DESCRIPTOR accept threshold (per kind)
  double margin;
  double width;   // the frame's W (the bounds check is margin <= u < W-margin)
  double height;  // the frame's H
};

// The C3a projection-match kernel (route B): the candidate pre-filter,
// the neighborhood search (brute force over the frame's keypoints),
// the per-candidate best-descriptor search, and the threshold accept.
// Returns (point_rows, kp_indices, repro_errors) for the ACCEPTED
// pairs, in the candidate order (the Python side creates the
// Observation objects and updates the object graph).
//
// Distance arithmetic (pre-registered, route B): HAMMING is exact
// popcount; L2 is f32-widened-to-f64 accumulation - strictly more
// accurate than cv.norm's runtime-dispatched float SIMD (which matched
// neither an f32 nor an f64 numpy formulation). Near-threshold pairs
// may flip; that is the accepted tolerance.
py::object match_projected(const MatchInputs& in);

}  // namespace slarnative
