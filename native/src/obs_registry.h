#pragma once

#include "error_fold.h"  // c_arr

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace slarnative {

// The observation registry (migration C4): the point-side observation
// graph (the authoritative registry per the measured 2f semantics)
// held NATIVELY, append-only (tombstone removals - the row order is
// the insertion order, stable forever: the error fold's pass-2 order
// contract). The events are bulk and transactional: accepted matches
// add observations, point removals tombstone the point's rows,
// keyframe removals tombstone the frame's observations from every
// point, reset clears. The Python point->observations dicts stay the
// working state; the drift pins compare the two after every event
// class.
class ObsRegistry {
 public:
  // Bulk add: (point_id, frame_id, kp_idx, xn, yn) rows. Duplicate
  // (point, frame) pairs are ignored (the dicts' semantics: one obs
  // per pair).
  void add_observations(const std::vector<uint64_t>& point_ids,
                        const std::vector<uint64_t>& frame_ids,
                        py::array_t<int64_t, py::array::c_style | py::array::forcecast>
                            kp_idx,
                        py::array_t<double, py::array::c_style | py::array::forcecast>
                            pt2d_norm);

  // Tombstone every observation of the given POINTS (the remove
  // funnel) or of the given FRAMES (the keyframe-removal orphan
  // pass). Tombstoned rows are skipped by the queries; the storage
  // is never compacted (the order contract).
  void remove_by_points(const std::vector<uint64_t>& point_ids);
  void remove_by_frames(const std::vector<uint64_t>& frame_ids);

  void clear();

  // The checkpoint boundary (M7): the EXACT row stream -
  // (point_id, frame_id, kp_idx, xn, yn, live) per row IN STORAGE
  // ORDER (tombstones included: the row positions are the pass-2
  // contract; the pair map rebuilds). Add-order re-simulation cannot
  // reconstruct the positions of re-observed pairs (the native keeps
  // the FIRST-add position; the dicts hold the LAST values).
  void restore(const std::vector<uint64_t>& point_ids,
               const std::vector<uint64_t>& frame_ids,
               py::array_t<int64_t, py::array::c_style | py::array::forcecast>
                   kp_idx,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                   pt2d_norm,
               const std::vector<uint8_t>& live);
  py::object serialize();

  size_t size() const { return live_; }  // the live rows
  bool has_pair(uint64_t point_id, uint64_t frame_id) const;

  // The cull scan's input (the first consumer): per point id, its
  // LIVE keyframe observers. Returned as parallel arrays
  // (point_row_index_into_the_query, frame_id) grouped by point in
  // the query's order.
  py::object observers_for_points(
      const std::vector<uint64_t>& point_ids) const;

  // The cull scan aggregate (the first consumer, bulk): per candidate
  // keyframe IN QUERY ORDER, (bucket_size, redundant_count) where
  // bucket_size = the # of points the candidate keyframe-observes and
  // redundant_count = the # of those whose OTHER keyframe observers
  // >= min_other. Observer counts are KEYFRAME observers only
  // (tracker frames never count as anchors - the measured 2f
  // semantics).
  py::object cull_scan(const std::vector<uint64_t>& candidate_ids,
                       const std::vector<uint64_t>& keyframe_ids,
                       int64_t min_other) const;

  // The C5 BA-path assembly (bulk): the local-BA + error-fold inputs
  // derived from the registry's rows in the EXACT Python orders.
  // Inputs: kf ids in map-keyframe order, the window kf subset, the
  // local point ids in map order (the frame-side get_local_points set,
  // Python-authored - its dedup-only entries have no rows), their
  // registry rows, the registry's positions block, the running avgs.
  // Returns (fold arrays..., ba arrays...) - see the binding.
  py::object assemble_local_ba(
      const std::vector<uint64_t>& kf_ids,
      const std::vector<uint64_t>& window_kf_ids,
      const std::vector<uint64_t>& local_point_ids,
      const std::vector<int64_t>& local_point_rows,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          positions,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          point_avg) const;

 private:
  struct ObsRow {
    uint64_t point_id;
    uint64_t frame_id;
    int64_t kp_idx;
    double xn, yn;
    bool live;
  };
  std::vector<ObsRow> rows_;
  // The (point, frame) pair -> row index (the duplicate-pair check).
  std::unordered_map<uint64_t, std::unordered_map<uint64_t, size_t>> pair_row_;
  size_t live_ = 0;
};

}  // namespace slarnative
