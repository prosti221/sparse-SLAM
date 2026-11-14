#include "obs_registry.h"

#include <stdexcept>

namespace slarnative {

void ObsRegistry::add_observations(
    const std::vector<uint64_t>& point_ids,
    const std::vector<uint64_t>& frame_ids,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> kp_idx,
    py::array_t<double, py::array::c_style | py::array::forcecast> pt2d_norm) {
  const long N = static_cast<long>(point_ids.size());
  if (frame_ids.size() != point_ids.size() || kp_idx.shape(0) != N ||
      pt2d_norm.shape(0) != N || pt2d_norm.shape(1) != 2) {
    throw std::invalid_argument(
        "add_observations: parallel arrays must be (N,) / (N, 2)");
  }
  const auto ki_in = kp_idx.unchecked<1>();
  const auto un_in = pt2d_norm.unchecked<2>();
  {
    py::gil_scoped_release release;
    for (long i = 0; i < N; ++i) {
      const uint64_t pid = point_ids[i];
      const uint64_t fid = frame_ids[i];
      auto& per_point = pair_row_[pid];
      auto existing = per_point.find(fid);
      if (existing != per_point.end()) {
        // One observation per (point, frame) pair - and a re-add
        // OVERWRITES in place (the dict semantics: the kp_idx and the
        // normalized coords update, the row keeps its position).
        ObsRow& row = rows_[existing->second];
        row.kp_idx = ki_in(i);
        row.xn = un_in(i, 0);
        row.yn = un_in(i, 1);
        continue;
      }
      per_point.emplace(fid, rows_.size());
      rows_.push_back(ObsRow{pid, fid, ki_in(i), un_in(i, 0), un_in(i, 1),
                             true});
      ++live_;
    }
  }
}

void ObsRegistry::remove_by_points(const std::vector<uint64_t>& point_ids) {
  {
    py::gil_scoped_release release;
    for (uint64_t pid : point_ids) {
      auto it = pair_row_.find(pid);
      if (it == pair_row_.end()) continue;
      for (const auto& [fid, row] : it->second) {
        if (rows_[row].live) {
          rows_[row].live = false;
          --live_;
        }
      }
      pair_row_.erase(it);
    }
  }
}

void ObsRegistry::remove_by_frames(const std::vector<uint64_t>& frame_ids) {
  std::unordered_set<uint64_t> doomed(frame_ids.begin(), frame_ids.end());
  {
    py::gil_scoped_release release;
    for (auto& row : rows_) {
      if (row.live && doomed.count(row.frame_id)) {
        row.live = false;
        --live_;
        auto it = pair_row_.find(row.point_id);
        if (it != pair_row_.end()) it->second.erase(row.frame_id);
      }
    }
  }
}

void ObsRegistry::clear() {
  rows_.clear();
  pair_row_.clear();
  live_ = 0;
}

bool ObsRegistry::has_pair(uint64_t point_id, uint64_t frame_id) const {
  auto it = pair_row_.find(point_id);
  return it != pair_row_.end() && it->second.count(frame_id) != 0;
}

py::object ObsRegistry::observers_for_points(
    const std::vector<uint64_t>& point_ids) const {
  // Parallel arrays grouped by point in the query's order: for query
  // slot i, the rows [starts[i], starts[i] + counts[i]) hold
  // (frame_id, kp_idx) of the point's LIVE observations.
  const size_t Q = point_ids.size();
  std::vector<int64_t> frame_ids;
  std::vector<int64_t> kp_indices;
  std::vector<int64_t> starts(Q, 0);
  std::vector<int64_t> counts(Q, 0);
  frame_ids.reserve(live_);
  kp_indices.reserve(live_);

  for (size_t q = 0; q < Q; ++q) {
    starts[q] = static_cast<int64_t>(frame_ids.size());
    auto it = pair_row_.find(point_ids[q]);
    if (it != pair_row_.end()) {
      for (const auto& [fid, row] : it->second) {
        if (!rows_[row].live) continue;
        frame_ids.push_back(static_cast<int64_t>(rows_[row].frame_id));
        kp_indices.push_back(rows_[row].kp_idx);
      }
    }
    counts[q] = static_cast<int64_t>(frame_ids.size()) - starts[q];
  }

  py::array_t<int64_t> fids(static_cast<long>(frame_ids.size()));
  py::array_t<int64_t> kis(static_cast<long>(kp_indices.size()));
  py::array_t<int64_t> st(static_cast<long>(Q));
  py::array_t<int64_t> ct(static_cast<long>(Q));
  {
    auto f = fids.mutable_unchecked<1>();
    for (size_t i = 0; i < frame_ids.size(); ++i) f(i) = frame_ids[i];
    auto k = kis.mutable_unchecked<1>();
    for (size_t i = 0; i < kp_indices.size(); ++i) k(i) = kp_indices[i];
    auto s = st.mutable_unchecked<1>();
    for (size_t i = 0; i < Q; ++i) s(i) = starts[i];
    auto c = ct.mutable_unchecked<1>();
    for (size_t i = 0; i < Q; ++i) c(i) = counts[i];
  }
  return py::make_tuple(fids, kis, st, ct);
}

py::object ObsRegistry::cull_scan(
    const std::vector<uint64_t>& candidate_ids,
    const std::vector<uint64_t>& keyframe_ids,
    int64_t min_other) const {
  // Pass 1: per point, its LIVE KEYFRAME-observer count (tracker
  // frames never count). Pass 2: bucket the rows whose frame is a
  // candidate; a point is redundant evidence for that candidate when
  // its other keyframe observers >= min_other. Integer/set logic -
  // exactly replicable of the Python scan.
  std::unordered_set<uint64_t> kf_set(keyframe_ids.begin(),
                                      keyframe_ids.end());
  std::unordered_map<uint64_t, size_t> cand_slot;
  for (size_t i = 0; i < candidate_ids.size(); ++i) {
    cand_slot.emplace(candidate_ids[i], i);
  }
  std::unordered_map<uint64_t, int64_t> kf_obs_count;
  {
    py::gil_scoped_release release;
    for (const auto& row : rows_) {
      if (row.live && kf_set.count(row.frame_id)) {
        ++kf_obs_count[row.point_id];
      }
    }
  }
  std::vector<int64_t> bucket(candidate_ids.size(), 0);
  std::vector<int64_t> redundant(candidate_ids.size(), 0);
  for (const auto& row : rows_) {
    if (!row.live) continue;
    auto ci = cand_slot.find(row.frame_id);
    if (ci == cand_slot.end()) continue;
    auto oc = kf_obs_count.find(row.point_id);
    if (oc == kf_obs_count.end()) continue;
    ++bucket[ci->second];
    if (oc->second - 1 >= min_other) ++redundant[ci->second];
  }
  py::array_t<int64_t> bs(static_cast<long>(bucket.size()));
  py::array_t<int64_t> rc(static_cast<long>(redundant.size()));
  {
    auto b = bs.mutable_unchecked<1>();
    for (size_t i = 0; i < bucket.size(); ++i) b(i) = bucket[i];
    auto r = rc.mutable_unchecked<1>();
    for (size_t i = 0; i < redundant.size(); ++i) r(i) = redundant[i];
  }
  return py::make_tuple(bs, rc);
}

}  // namespace slarnative
