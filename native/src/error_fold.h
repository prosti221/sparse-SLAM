#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace slarnative {

// Input arrays are C-contiguous with forced cast (the Python call site
// hands native NumPy arrays; the kernel must never copy on hot paths).
template <typename T>
using c_arr = py::array_t<T, py::array::c_style | py::array::forcecast>;

struct FoldInputs {
  c_arr<double> poses_inv;   // (F, 4, 4) world-to-camera, per frame row
  c_arr<int64_t> obs_frame;  // (O,) row index into poses_inv
  c_arr<double> point_pos;   // (M, 3) world positions, per point row
  c_arr<double> obs_norm;    // (O, 2) target normalized coords
  c_arr<int64_t> obs_point;  // (O,) row index into the point arrays
  c_arr<double> point_avg;   // (M,) running-average input state
  c_arr<int64_t> point_nobs; // (M,) GLOBAL observation count per point
};

struct FoldOutputs {
  py::array_t<double> err;
  py::array_t<double> new_avg;
  py::array_t<uint8_t> updated;
};

// The error-fold kernel (migration C2a). See error_fold.cpp for the
// op-order contract.
FoldOutputs error_fold(const FoldInputs& in);

}  // namespace slarnative
