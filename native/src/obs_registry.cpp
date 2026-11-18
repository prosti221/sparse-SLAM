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

void ObsRegistry::restore(
    const std::vector<uint64_t>& point_ids,
    const std::vector<uint64_t>& frame_ids,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> kp_idx,
    py::array_t<double, py::array::c_style | py::array::forcecast> pt2d_norm,
    const std::vector<uint8_t>& live) {
  const long N = static_cast<long>(point_ids.size());
  if (frame_ids.size() != point_ids.size() || kp_idx.shape(0) != N ||
      pt2d_norm.shape(0) != N || pt2d_norm.shape(1) != 2 ||
      live.size() != point_ids.size()) {
    throw std::invalid_argument(
        "restore: parallel arrays must be (N,) / (N, 2)");
  }
  clear();
  const auto ki_in = kp_idx.unchecked<1>();
  const auto un_in = pt2d_norm.unchecked<2>();
  {
    py::gil_scoped_release release;
    rows_.reserve(N);
    for (long i = 0; i < N; ++i) {
      const bool is_live = live[i] != 0;
      rows_.push_back(ObsRow{point_ids[i], frame_ids[i], ki_in(i),
                             un_in(i, 0), un_in(i, 1), is_live});
      if (is_live) {
        pair_row_[point_ids[i]].emplace(frame_ids[i], rows_.size() - 1);
        ++live_;
      }
    }
  }
}

py::object ObsRegistry::serialize() {
  const long NN = static_cast<long>(rows_.size());
  auto pids = py::array_t<uint64_t>(NN);
  auto fids = py::array_t<uint64_t>(NN);
  auto kps = py::array_t<int64_t>(NN);
  auto norms = py::array_t<double>({NN, 2L});
  auto lives = py::array_t<uint8_t>(NN);
  const auto N = NN;
  {
    py::gil_scoped_release release;
    auto pi = pids.mutable_unchecked<1>();
    auto fi = fids.mutable_unchecked<1>();
    auto ki = kps.mutable_unchecked<1>();
    auto ni = norms.mutable_unchecked<2>();
    auto li = lives.mutable_unchecked<1>();
    for (size_t i = 0; i < N; ++i) {
      pi(i) = rows_[i].point_id;
      fi(i) = rows_[i].frame_id;
      ki(i) = rows_[i].kp_idx;
      ni(i, 0) = rows_[i].xn;
      ni(i, 1) = rows_[i].yn;
      li(i) = rows_[i].live ? 1 : 0;
    }
  }
  return py::make_tuple(pids, fids, kps, norms, lives);
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

py::object ObsRegistry::assemble_local_ba(
    const std::vector<uint64_t>& kf_ids,
    const std::vector<uint64_t>& window_kf_ids,
    const std::vector<uint64_t>& local_point_ids,
    const std::vector<int64_t>& local_point_rows,
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<double, py::array::c_style | py::array::forcecast> point_avg)
    const {
  const size_t P = local_point_ids.size();
  if (local_point_rows.size() != P ||
      static_cast<size_t>(point_avg.shape(0)) != P) {
    throw std::invalid_argument(
        "assemble_local_ba: the local point arrays must be parallel");
  }
  const size_t B = static_cast<size_t>(positions.shape(0));
  if (positions.shape(1) != 3) {
    throw std::invalid_argument("assemble_local_ba: positions must be (B, 3)");
  }
  const auto pos_in = positions.unchecked<2>();
  for (size_t i = 0; i < P; ++i) {
    if (local_point_rows[i] < 0 ||
        static_cast<size_t>(local_point_rows[i]) >= B) {
      throw std::invalid_argument(
          "assemble_local_ba: local point row out of the positions block");
    }
  }

  std::unordered_set<uint64_t> local_set(local_point_ids.begin(),
                                         local_point_ids.end());
  std::unordered_map<uint64_t, size_t> local_index;
  for (size_t i = 0; i < P; ++i) {
    local_index.emplace(local_point_ids[i], i);
  }
  std::unordered_set<uint64_t> kf_set(kf_ids.begin(), kf_ids.end());
  std::unordered_set<uint64_t> window_set(window_kf_ids.begin(),
                                          window_kf_ids.end());

  // The fold's point subset: local points with >= 1 live row, in the
  // local (map) order - the first-appearance order restricted to
  // appearing points (the sequence is point-major).
  std::unordered_map<uint64_t, int64_t> live_count;
  for (const auto& row : rows_) {
    if (row.live && local_set.count(row.point_id)) {
      ++live_count[row.point_id];
    }
  }
  std::vector<size_t> fold_slots;
  std::unordered_map<uint64_t, size_t> slot_of;
  for (size_t i = 0; i < P; ++i) {
    if (live_count.count(local_point_ids[i])) {
      slot_of.emplace(local_point_ids[i], fold_slots.size());
      fold_slots.push_back(i);
    }
  }
  const size_t PF = fold_slots.size();

  // The point-major observation sequence: local points in map order,
  // each point's live rows in global row-index order (= the dicts'
  // insertion order; rows overwrite in place, dicts overwrite in
  // place).
  std::vector<int64_t> start(PF + 1, 0);
  for (size_t j = 0; j < PF; ++j) {
    start[j + 1] = start[j] + live_count[local_point_ids[fold_slots[j]]];
  }
  const long O = static_cast<long>(start[PF]);
  std::vector<int64_t> cursor(start.begin(), start.end() - 1);
  std::vector<uint64_t> seq_point(static_cast<size_t>(O));
  std::vector<uint64_t> seq_frame(static_cast<size_t>(O));
  std::vector<double> seq_uv(static_cast<size_t>(O) * 2);
  for (const auto& row : rows_) {
    if (!row.live) continue;
    auto so = slot_of.find(row.point_id);
    if (so == slot_of.end()) continue;
    const int64_t at = cursor[so->second]++;
    seq_point[static_cast<size_t>(at)] = row.point_id;
    seq_frame[static_cast<size_t>(at)] = row.frame_id;
    seq_uv[static_cast<size_t>(at) * 2] = row.xn;
    seq_uv[static_cast<size_t>(at) * 2 + 1] = row.yn;
  }

  // The fold's first-appearance frame/point indices over the sequence
  // (dict.fromkeys equivalence).
  std::unordered_map<uint64_t, int64_t> first_frame, first_point;
  std::vector<uint64_t> fold_frames;
  std::vector<int64_t> f_obs_frame(static_cast<size_t>(O));
  std::vector<int64_t> f_obs_point(static_cast<size_t>(O));
  for (long i = 0; i < O; ++i) {
    const size_t u = static_cast<size_t>(i);
    auto fit = first_frame.find(seq_frame[u]);
    if (fit == first_frame.end()) {
      fit = first_frame.emplace(seq_frame[u],
                                static_cast<int64_t>(fold_frames.size()))
                .first;
      fold_frames.push_back(seq_frame[u]);
    }
    f_obs_frame[u] = fit->second;
    auto pit = first_point.find(seq_point[u]);
    if (pit == first_point.end()) {
      pit = first_point.emplace(seq_point[u],
                                static_cast<int64_t>(first_point.size()))
                .first;
    }
    f_obs_point[u] = pit->second;
  }

  // The fold's point arrays in the subset order (avgs reordered; nobs =
  // the live-row counts = the dict sizes; fold_rows = the subset's
  // registry rows - the fold re-gathers positions from the LIVE block
  // at fold time, because the solve moves the geometry after this
  // assembly runs).
  py::array_t<double> f_avg(static_cast<long>(PF));
  py::array_t<int64_t> f_nobs(static_cast<long>(PF));
  py::array_t<int64_t> f_slot(static_cast<long>(PF));
  py::array_t<int64_t> f_rows(static_cast<long>(PF));
  {
    auto fa = f_avg.mutable_unchecked<1>();
    auto fn = f_nobs.mutable_unchecked<1>();
    auto fs = f_slot.mutable_unchecked<1>();
    auto fr = f_rows.mutable_unchecked<1>();
    for (size_t j = 0; j < PF; ++j) {
      const size_t i = fold_slots[j];
      fa(static_cast<long>(j)) = point_avg.unchecked<1>()(
          static_cast<long>(i));
      fn(static_cast<long>(j)) = live_count[local_point_ids[i]];
      fs(static_cast<long>(j)) = static_cast<int64_t>(i);
      fr(static_cast<long>(j)) = local_point_rows[i];
    }
  }

  // The BA edges: the sequence filtered to keyframe frames, list order
  // preserved (the delta-edge contract). Frame rows are the BA frames'
  // slots (map-kf order n target); point rows are the local slots.
  std::unordered_map<uint64_t, int64_t> ba_frame_row;
  std::vector<uint64_t> ba_frames;
  std::vector<uint8_t> ba_in_window;
  for (uint64_t fid : kf_ids) {
    if (first_frame.count(fid)) {
      ba_frame_row.emplace(fid, static_cast<int64_t>(ba_frames.size()));
      ba_frames.push_back(fid);
      ba_in_window.push_back(window_set.count(fid) ? 1 : 0);
    }
  }
  // The anchor: the FIRST keyframe in map order among the target
  // frames (the involved_keyframes[0] gauge semantics).
  int64_t anchor = -1;
  for (uint64_t fid : kf_ids) {
    if (first_frame.count(fid)) {
      anchor = ba_frame_row[fid];
      break;
    }
  }
  std::vector<int64_t> e_frame;
  std::vector<int64_t> e_point;
  std::vector<double> e_uv;
  e_uv.reserve(static_cast<size_t>(O) * 2);
  for (long i = 0; i < O; ++i) {
    const size_t u = static_cast<size_t>(i);
    auto fit = ba_frame_row.find(seq_frame[u]);
    if (fit == ba_frame_row.end()) continue;  // a tracker frame
    e_frame.push_back(fit->second);
    e_point.push_back(
        static_cast<int64_t>(local_index[seq_point[u]]));
    e_uv.push_back(seq_uv[u * 2]);
    e_uv.push_back(seq_uv[u * 2 + 1]);
  }

  auto list_from_u64 = [](const std::vector<uint64_t>& v) {
    py::list l;
    for (uint64_t x : v) l.append(py::int_(static_cast<long long>(x)));
    return l;
  };
  auto list_from_i64 = [](const std::vector<int64_t>& v) {
    py::list l;
    for (int64_t x : v) l.append(py::int_(static_cast<long long>(x)));
    return l;
  };

  py::array_t<int64_t> f_of(static_cast<long>(O));
  py::array_t<int64_t> f_op(static_cast<long>(O));
  py::array_t<double> f_norm({static_cast<long>(O), 2L});
  {
    auto of = f_of.mutable_unchecked<1>();
    auto op = f_op.mutable_unchecked<1>();
    auto on = f_norm.mutable_unchecked<2>();
    for (long i = 0; i < O; ++i) {
      const size_t u = static_cast<size_t>(i);
      of(i) = f_obs_frame[u];
      op(i) = f_obs_point[u];
      on(i, 0) = seq_uv[u * 2];
      on(i, 1) = seq_uv[u * 2 + 1];
    }
  }
  // The BA's positions over ALL local points (map order) - the vertex
  // set includes zero-row points (defensive; none in practice).
  py::array_t<double> b_pos({static_cast<long>(P), 3L});
  {
    auto bp = b_pos.mutable_unchecked<2>();
    for (size_t i = 0; i < P; ++i) {
      const long row = local_point_rows[i];
      for (int c = 0; c < 3; ++c) {
        bp(static_cast<long>(i), c) = pos_in(row, c);
      }
    }
  }

  py::array_t<double> e_uvs({static_cast<long>(e_frame.size()), 2L});
  {
    auto eu = e_uvs.mutable_unchecked<2>();
    for (size_t i = 0; i < e_frame.size(); ++i) {
      eu(static_cast<long>(i), 0) = e_uv[i * 2];
      eu(static_cast<long>(i), 1) = e_uv[i * 2 + 1];
    }
  }
  py::array_t<int64_t> anchor_a(1);
  anchor_a.mutable_unchecked<1>()(0) = anchor;
  py::array_t<uint8_t> win_a(static_cast<long>(ba_in_window.size()));
  {
    auto w = win_a.mutable_unchecked<1>();
    for (size_t i = 0; i < ba_in_window.size(); ++i) {
      w(static_cast<long>(i)) = ba_in_window[i];
    }
  }
  return py::make_tuple(
      f_of, f_op, f_norm, f_rows, f_avg, f_nobs, f_slot,
      list_from_u64(fold_frames),
      list_from_u64(ba_frames), anchor_a, win_a,
      list_from_i64(e_frame), list_from_i64(e_point), e_uvs, b_pos);
}

}  // namespace slarnative
