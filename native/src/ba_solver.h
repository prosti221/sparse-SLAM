#pragma once

#include "error_fold.h"  // brings c_arr + the py aliases

namespace slarnative {

struct BASolveInputs {
  c_arr<double> T_wc;        // (F, 4, 4) initial WORLD-TO-CAMERA - the
                             // g2o vertex estimate convention (the
                             // production build feeds SE3Quat from
                             // inv(kf.pose); kf.pose is camera-to-world)
  c_arr<uint8_t> fixed;      // (F,) per-frame fixed flags (gauge anchor)
  c_arr<double> positions;   // (M, 3) initial point positions
  uint8_t points_fixed;      // fix_points for ALL point vertices
  c_arr<int64_t> obs_frame;  // (O,) frame rows
  c_arr<int64_t> obs_point;  // (O,) point rows
  c_arr<double> obs_uv;      // (O, 2) normalized measurements
  int max_iterations;
};

// The C2b1 smoke solve: a FRESH-BUILD SE3 BA graph solved with the
// vendored g2o (library version 1.1.0 - the same source g2o-python
// 0.0.12 builds from), configured EXACTLY like the Python path:
// LinearSolverEigen<6x6> + BlockSolver_6_3 + Levenberg, Huber
// (sqrt(5.991)) per edge, information eye(2), CameraParameters
// (focal 1.0, principal (0,0)), the terminate action at the gain
// threshold. Returns (T_wc (F,4,4), positions (M,3), iterations,
// chi2) - the estimates in the SAME world-to-camera convention as the
// input; the Python write-back maps them back with inv().
py::object solve_se3_ba(const BASolveInputs& in);

}  // namespace slarnative
