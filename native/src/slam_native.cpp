#include <cmath>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "map_core.h"
#include "error_fold.h"

namespace py = pybind11;

// C0 smoke kernel: the [26d] vectorized row norm, computed per row in
// strict scalar order. With -ffp-contract=off the multiply-add cannot
// fuse into an FMA, so the result is bit-identical to numpy's
// elementwise sqrt(x*x + y*y) on every machine - this is the FP
// contract made executable (the bridge pin asserts it).
static py::array_t<double> row_norms(
    py::array_t<double, py::array::c_style | py::array::forcecast> d) {
  if (d.ndim() != 2 || d.shape(1) != 2) {
    throw std::invalid_argument("row_norms expects an (N, 2) float64 array");
  }
  const auto in = d.unchecked<>();
  py::array_t<double> out(d.shape(0));
  auto obuf = out.mutable_unchecked<>();
  for (py::ssize_t i = 0; i < in.shape(0); ++i) {
    const double x = in(i, 0);
    const double y = in(i, 1);
    obuf(i) = std::sqrt(x * x + y * y);
  }
  return out;
}

PYBIND11_MODULE(slam_native, m) {
  m.doc() = "sparse-SLAM native core (C0: build infra + FP contract)";
  m.def("row_norms", &row_norms, py::arg("d"),
        "Elementwise sqrt(x*x + y*y) per row, strict scalar order.");
  m.attr("__version__") = "0.1.0";

  py::class_<slarnative::MapCore>(m, "MapCore")
      .def(py::init<>())
      .def("push_window", &slarnative::MapCore::push_window,
           py::arg("frame_ids"), py::arg("poses"), py::arg("Ks"),
           py::arg("k1s"), py::arg("kp_offsets"), py::arg("kp_pts"),
           py::arg("kp_pts_norm"), py::arg("frame_desc_bytes"),
           py::arg("frame_desc_dtype"), py::arg("point_ids"),
           py::arg("positions"), py::arg("point_desc_bytes"),
           py::arg("point_desc_dtype"), py::arg("point_num_obs"),
           py::arg("obs_frame_ids"), py::arg("obs_point_ids"),
           py::arg("obs_kp_idx"), py::arg("obs_pt2d_norm"))
      .def("clear", &slarnative::MapCore::clear)
      .def("num_frames", &slarnative::MapCore::num_frames)
      .def("num_points", &slarnative::MapCore::num_points)
      .def("num_obs", &slarnative::MapCore::num_obs)
      .def("has_frame", &slarnative::MapCore::has_frame)
      .def("has_point", &slarnative::MapCore::has_point)
      .def("frame_pose", &slarnative::MapCore::frame_pose)
      .def("frame_K", &slarnative::MapCore::frame_K)
      .def("frame_k1", &slarnative::MapCore::frame_k1)
      .def("frame_descriptors", &slarnative::MapCore::frame_descriptors)
      .def("point_descriptor", &slarnative::MapCore::point_descriptor)
      .def("frame_desc_meta", &slarnative::MapCore::frame_desc_meta)
      .def("point_desc_meta", &slarnative::MapCore::point_desc_meta)
      .def("point_position", &slarnative::MapCore::point_position)
      .def("point_avg_err", &slarnative::MapCore::point_avg_err)
      .def("window_frame_ids", &slarnative::MapCore::window_frame_ids)
      .def("window_point_ids", &slarnative::MapCore::window_point_ids)
      .def("window_kp_idx", &slarnative::MapCore::window_kp_idx)
      .def("window_pt2d_norm", &slarnative::MapCore::window_pt2d_norm)
      .def("update_point_errors", &slarnative::MapCore::update_point_errors);

  // Unpacks the C++ FoldOutputs struct into a Python tuple at the
  // boundary: (per-obs errors, per-point new averages, updated flags).
  m.def("error_fold",
        [](const slarnative::FoldInputs& in) {
          slarnative::FoldOutputs out = slarnative::error_fold(in);
          return py::make_tuple(out.err, out.new_avg, out.updated);
        },
        py::arg("in"),
        "The error-fold kernel: per-obs errors, per-point new averages, "
        "updated flags.");

  py::class_<slarnative::FoldInputs>(m, "FoldInputs")
      .def(py::init([](slarnative::c_arr<double> poses_inv,
                       slarnative::c_arr<int64_t> obs_frame,
                       slarnative::c_arr<double> obs_pt3d,
                       slarnative::c_arr<double> obs_norm,
                       slarnative::c_arr<int64_t> obs_point,
                       slarnative::c_arr<double> point_avg,
                       slarnative::c_arr<int64_t> point_nobs) {
             return slarnative::FoldInputs{
                 std::move(poses_inv), std::move(obs_frame), std::move(obs_pt3d),
                 std::move(obs_norm),  std::move(obs_point), std::move(point_avg),
                 std::move(point_nobs)};
           }),
           py::arg("poses_inv"), py::arg("obs_frame"), py::arg("point_pos"),
           py::arg("obs_norm"), py::arg("obs_point"), py::arg("point_avg"),
           py::arg("point_nobs"));
}
