#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace slarnative {

// A window record: one frame's projection metadata. kp_pts /
// kp_pts_norm / desc_bytes are per-frame copies sliced out of the
// caller's concatenated arrays (window-sized, replaced on every push).
struct FrameRec {
  uint64_t id;
  double pose[16];  // row-major 4x4
  double K[9];      // row-major 3x3
  double k1;
  std::vector<double> kp_pts;
  std::vector<double> kp_pts_norm;
  std::vector<uint8_t> desc_bytes;
  uint64_t desc_byte_width = 0;
  int desc_dtype = 0;  // 0 = none, 1 = uint8, 2 = float32
};

struct PointRec {
  uint64_t id;
  double position[3];
  double avg_err = 0.0;
  uint64_t num_observations = 0;
  std::vector<uint8_t> desc_bytes;
  uint64_t desc_byte_width = 0;
  int desc_dtype = 0;  // 0 = none, 1 = uint8, 2 = float32
};

// One observation in EXACTLY the order the producer pushed it - the
// error-fold's pass-2 walks the observation list in insertion order.
struct ObsRec {
  uint64_t frame_id;
  uint64_t point_id;
  uint64_t kp_idx;
  double xn, yn;
};

// The native-side map window (migration C1). REPLACE-semantics bulk
// transactions only: the Python object graph remains the working state
// and pushes the BA-input window at `Map.get_observations`; no
// granular accessors exist, so there is no chatty boundary and no
// staleness (every push fully rewrites). Handles ARE the Python u64
// ids - no mapping layer. The C2 fold and the C3 correspondence read
// this storage directly.
class MapCore {
 public:
  // Replace the whole window. Validates every shape and id uniqueness
  // before mutating (a rejected push leaves the previous window
  // intact).
  void push_window(
      const std::vector<uint64_t>& frame_ids,
      py::array_t<double, py::array::c_style | py::array::forcecast> poses,
      py::array_t<double, py::array::c_style | py::array::forcecast> Ks,
      py::array_t<double, py::array::c_style | py::array::forcecast> k1s,
      py::array_t<int64_t, py::array::c_style | py::array::forcecast>
          kp_offsets,
      py::array_t<double, py::array::c_style | py::array::forcecast> kp_pts,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          kp_pts_norm,
      py::array_t<uint8_t, py::array::c_style | py::array::forcecast>
          frame_desc_bytes,
      int frame_desc_dtype,
      const std::vector<uint64_t>& point_ids,
      py::array_t<double, py::array::c_style | py::array::forcecast> positions,
      py::array_t<uint8_t, py::array::c_style | py::array::forcecast>
          point_desc_bytes,
      int point_desc_dtype,
      py::array_t<int64_t, py::array::c_style | py::array::forcecast>
          point_num_obs,
      const std::vector<uint64_t>& obs_frame_ids,
      const std::vector<uint64_t>& obs_point_ids,
      py::array_t<int64_t, py::array::c_style | py::array::forcecast> obs_kp_idx,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          obs_pt2d_norm);

  void clear();

  size_t num_frames() const { return frames_.size(); }
  size_t num_points() const { return points_.size(); }
  size_t num_obs() const { return obs_.size(); }

  bool has_frame(uint64_t id) const;
  bool has_point(uint64_t id) const;

  std::vector<double> frame_pose(uint64_t id) const;      // 16 doubles
  std::vector<double> frame_K(uint64_t id) const;         // 9 doubles
  double frame_k1(uint64_t id) const;
  // Raw descriptor bytes + the dtype code (0 none, 1 uint8, 2 float32).
  std::vector<uint8_t> frame_descriptors(uint64_t id) const;  // n * width
  std::vector<uint8_t> point_descriptor(uint64_t id) const;   // width
  std::pair<int, uint64_t> frame_desc_meta(uint64_t id) const;
  std::pair<int, uint64_t> point_desc_meta(uint64_t id) const;
  std::vector<double> point_position(uint64_t id) const;  // 3 doubles
  double point_avg_err(uint64_t id) const;

  // The stored observation order, as pushed.
  std::vector<uint64_t> window_frame_ids() const;
  std::vector<uint64_t> window_point_ids() const;
  std::vector<int64_t> window_kp_idx() const;
  py::array_t<double> window_pt2d_norm() const;

  // The error-fold's write-back: per-observation errors in push order,
  // applied to each point's running average with EXACTLY the Python
  // recurrence (Point.update_reprojection_error): n = the point's
  // GLOBAL observation count (pushed with the window), and
  // avg = (avg*(n-1) + err) / n when n > 0.
  void update_point_errors(const std::vector<uint64_t>& point_ids,
                           const std::vector<double>& errors);

 private:
  std::vector<FrameRec> frames_;
  std::vector<PointRec> points_;
  std::vector<ObsRec> obs_;
  std::unordered_map<uint64_t, size_t> frame_idx_;
  std::unordered_map<uint64_t, size_t> point_idx_;
};

}  // namespace slarnative
