#include <cmath>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "map_core.h"
#include "error_fold.h"
#include "ba_solver.h"
#include "persistent_ba.h"
#include "correspondence.h"
#include "point_registry.h"

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

  py::class_<slarnative::BASolveInputs>(m, "BASolveInputs")
      .def(py::init([](slarnative::c_arr<double> T_wc,
                       slarnative::c_arr<uint8_t> fixed,
                       slarnative::c_arr<double> positions,
                       uint8_t points_fixed,
                       slarnative::c_arr<int64_t> obs_frame,
                       slarnative::c_arr<int64_t> obs_point,
                       slarnative::c_arr<double> obs_uv,
                       int max_iterations) {
             return slarnative::BASolveInputs{
                 std::move(T_wc),      std::move(fixed),
                 std::move(positions), points_fixed,
                 std::move(obs_frame), std::move(obs_point),
                 std::move(obs_uv),    max_iterations};
           }),
           py::arg("T_wc"), py::arg("fixed"), py::arg("positions"),
           py::arg("points_fixed"), py::arg("obs_frame"),
           py::arg("obs_point"), py::arg("obs_uv"),
           py::arg("max_iterations"));

  m.def("solve_se3_ba", &slarnative::solve_se3_ba, py::arg("in"),
        "The C2b1 smoke solve: a fresh-build SE3 BA graph with the "
        "vendored g2o, configured like the Python path. Returns "
        "(T_cw, positions, iterations, chi2).");

  py::class_<slarnative::PersistentLocalBA>(m, "PersistentLocalBA")
      .def(py::init<bool>(), py::arg("verbose"))
      .def("reset", &slarnative::PersistentLocalBA::reset)
      .def(
          "solve_local",
          &slarnative::PersistentLocalBA::solve_local,
          py::arg("frame_ids"), py::arg("point_ids"), py::arg("T_wc"),
          py::arg("positions"), py::arg("in_window"), py::arg("anchor_row"),
          py::arg("points_fixed"), py::arg("obs_frame_row"),
          py::arg("obs_point_row"), py::arg("obs_uv"),
          py::arg("max_iterations"))
      .def("num_frames", &slarnative::PersistentLocalBA::num_frames)
      .def("num_points", &slarnative::PersistentLocalBA::num_points)
      .def("num_edges", &slarnative::PersistentLocalBA::num_edges)
      .def("fixed_pose_count", &slarnative::PersistentLocalBA::fixed_pose_count)
      .def("has_frame", &slarnative::PersistentLocalBA::has_frame)
      .def("has_point", &slarnative::PersistentLocalBA::has_point)
      .def("graph_edge_pairs", &slarnative::PersistentLocalBA::graph_edge_pairs)
      .def("bookkeeping_pairs", &slarnative::PersistentLocalBA::bookkeeping_pairs)
      .def("frame_estimate_T_wc", &slarnative::PersistentLocalBA::frame_estimate_T_wc)
      .def("point_estimate", &slarnative::PersistentLocalBA::point_estimate);

  py::class_<slarnative::MatchInputs>(m, "MatchInputs")
      .def(py::init([](slarnative::c_arr<double> point_uv,
                       slarnative::c_arr<uint8_t> point_desc,
                       slarnative::c_arr<uint8_t> point_has_desc,
                       slarnative::c_arr<double> kp_coords,
                       slarnative::c_arr<uint8_t> kp_desc,
                       slarnative::c_arr<uint8_t> already_observed,
                       int desc_kind, double search_radius,
                       double reloc_thresh, double margin, double width,
                       double height) {
             return slarnative::MatchInputs{
                 std::move(point_uv),         std::move(point_desc),
                 std::move(point_has_desc),   std::move(kp_coords),
                 std::move(kp_desc),          std::move(already_observed),
                 desc_kind,                   search_radius,
                 reloc_thresh,                margin,
                 width,                       height};
           }),
           py::arg("point_uv"), py::arg("point_desc"), py::arg("point_has_desc"),
           py::arg("kp_coords"), py::arg("kp_desc"), py::arg("already_observed"),
           py::arg("desc_kind"), py::arg("search_radius"), py::arg("reloc_thresh"),
           py::arg("margin"), py::arg("width"), py::arg("height"));

  m.def("match_projected", &slarnative::match_projected, py::arg("in"),
        "The C3a projection-match kernel (route B). Returns "
        "(point_rows, kp_indices, repro_errors, n_matches).");

  py::class_<slarnative::PointRegistry>(m, "PointRegistry")
      .def(py::init<int>(), py::arg("desc_width"))
      .def("register_points", &slarnative::PointRegistry::register_points,
           py::arg("ids"), py::arg("positions"), py::arg("desc_bytes"),
           py::arg("has_desc"))
      .def("update_positions", &slarnative::PointRegistry::update_positions,
           py::arg("ids"), py::arg("positions"))
      .def("remove_points", &slarnative::PointRegistry::remove_points,
           py::arg("ids"))
      .def("clear", &slarnative::PointRegistry::clear)
      .def("size", &slarnative::PointRegistry::size)
      .def("has", &slarnative::PointRegistry::has)
      .def("desc_width", &slarnative::PointRegistry::desc_width)
      .def("positions_view", &slarnative::PointRegistry::positions_view)
      .def("descriptors_view", &slarnative::PointRegistry::descriptors_view)
      .def("has_desc_view", &slarnative::PointRegistry::has_desc_view)
      .def("row_of", &slarnative::PointRegistry::row_of)
      .def("ids", &slarnative::PointRegistry::ids);
}
