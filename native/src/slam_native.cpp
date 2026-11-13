#include <cmath>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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
}
