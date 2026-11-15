#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace slarnative {

// The live map-point registry (migration C3b): the FULL point set's
// geometry and descriptors held NATIVELY in capacity-managed numpy
// storage, updated incrementally at the map's event points (add,
// remove, the BA write-back, rebuild) so the per-frame consumers
// (projection matching, the error fold, the BA window push) read
// zero-copy VIEWS instead of re-assembling Python-object arrays on
// every call. The registry is the geometry home: the Point objects'
// pt_3d resolves here through the map's rows index (the per-point
// Python twin and its write-back sync are gone).
class PointRegistry {
 public:
  explicit PointRegistry(int desc_width);

  // Upsert: new ids are appended (map insertion order preserved);
  // known ids get their geometry refreshed. desc_bytes is (N, W) raw
  // descriptor bytes (zero rows for no-descriptor points).
  void register_points(const std::vector<uint64_t>& ids,
                       py::array_t<double, py::array::c_style | py::array::forcecast>
                           positions,
                       py::array_t<uint8_t, py::array::c_style | py::array::forcecast>
                           desc_bytes,
                       py::array_t<uint8_t, py::array::c_style | py::array::forcecast>
                           has_desc);

  // Bulk position update by id (the BA write-back, rebuild_point_coords).
  void update_positions(const std::vector<uint64_t>& ids,
                        py::array_t<double, py::array::c_style | py::array::forcecast>
                            positions);

  // Remove by id (culls, orphaned points): swap-erase compaction with
  // row-map maintenance.
  void remove_points(const std::vector<uint64_t>& ids);

  void clear();

  size_t size() const { return ids_.size(); }
  bool has(uint64_t id) const { return row_of_.count(id) != 0; }
  int desc_width() const { return desc_width_; }

  // Zero-copy views over the live storage (the row count is the
  // live size; the caller must not hold them across mutations that
  // compact).
  py::array positions_view() const;
  py::array descriptors_view() const;
  py::array has_desc_view() const;

  // Row index for an id (the consumers' obs->row mappings), or -1.
  long row_of(uint64_t id) const;

  std::vector<uint64_t> ids() const;

 private:
  void ensure_capacity(size_t need_rows);

  int desc_width_;
  size_t used_ = 0;
  std::vector<uint64_t> ids_;                       // row -> id
  std::unordered_map<uint64_t, size_t> row_of_;     // id -> row
  // Capacity-managed numpy storage (stable memory for the views).
  py::array_t<double> positions_;
  py::array_t<uint8_t> descriptors_;
  py::array_t<uint8_t> has_desc_;
};

}  // namespace slarnative
