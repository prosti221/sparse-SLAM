#include "map_core.h"

#include <stdexcept>

namespace slarnative {

namespace {

void require(bool ok, const char* msg) {
  if (!ok) throw std::invalid_argument(msg);
}

}  // namespace

void MapCore::push_window(
    const std::vector<uint64_t>& frame_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> poses,
    py::array_t<double, py::array::c_style | py::array::forcecast> Ks,
    py::array_t<double, py::array::c_style | py::array::forcecast> k1s,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> kp_offsets,
    py::array_t<double, py::array::c_style | py::array::forcecast> kp_pts,
    py::array_t<double, py::array::c_style | py::array::forcecast> kp_pts_norm,
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
        obs_pt2d_norm) {
  const size_t F = frame_ids.size();
  const size_t Nkp = static_cast<size_t>(kp_pts.shape(0));
  const size_t M = point_ids.size();
  const size_t O = obs_frame_ids.size();

  require(poses.ndim() == 3 && poses.shape(0) == static_cast<long>(F) &&
              poses.shape(1) == 4 && poses.shape(2) == 4,
          "poses must be (F, 4, 4)");
  require(Ks.ndim() == 3 && Ks.shape(0) == static_cast<long>(F) &&
              Ks.shape(1) == 3 && Ks.shape(2) == 3,
          "Ks must be (F, 3, 3)");
  require(k1s.ndim() == 1 && k1s.shape(0) == static_cast<long>(F),
          "k1s must be (F,)");
  require(kp_offsets.ndim() == 1 &&
              kp_offsets.shape(0) == static_cast<long>(F) + 1,
          "kp_offsets must be (F+1,)");
  require(kp_pts.ndim() == 2 && kp_pts.shape(1) == 2 &&
              kp_pts_norm.shape(0) == kp_pts.shape(0) &&
              kp_pts_norm.shape(1) == 2,
          "kp_pts / kp_pts_norm must be (N, 2) with matching N");
  const auto kp_off = kp_offsets.unchecked<1>();
  require(kp_off(F) == static_cast<int64_t>(Nkp),
          "kp_offsets[-1] must equal the concatenated keypoint count");
  for (size_t i = 0; i < F; ++i) {
    require(kp_off(i) <= kp_off(i + 1), "kp_offsets must be nondecreasing");
  }
  require(frame_desc_dtype >= 0 && frame_desc_dtype <= 2,
          "frame_desc_dtype must be 0 (none), 1 (uint8) or 2 (float32)");
  require(point_desc_dtype >= 0 && point_desc_dtype <= 2,
          "point_desc_dtype must be 0 (none), 1 (uint8) or 2 (float32)");
  require(positions.ndim() == 2 && positions.shape(0) == static_cast<long>(M) &&
              positions.shape(1) == 3,
          "positions must be (M, 3)");
  require(point_num_obs.ndim() == 1 &&
              point_num_obs.shape(0) == static_cast<long>(M),
          "point_num_obs must be (M,)");
  require(obs_point_ids.size() == O,
          "obs_frame_ids and obs_point_ids must have equal length");
  require(obs_kp_idx.ndim() == 1 && obs_kp_idx.shape(0) == static_cast<long>(O),
          "obs_kp_idx must be (O,)");
  require(obs_pt2d_norm.ndim() == 2 &&
              obs_pt2d_norm.shape(0) == static_cast<long>(O) &&
              obs_pt2d_norm.shape(1) == 2,
          "obs_pt2d_norm must be (O, 2)");
  require(frame_desc_bytes.ndim() == 2,
          "frame_desc_bytes must be (N, W) - pass a (0, 0) matrix when none");
  require(point_desc_bytes.ndim() == 2,
          "point_desc_bytes must be (M, W) - pass a (0, 0) matrix when none");

  {
    std::unordered_map<uint64_t, bool> seen_f, seen_p;
    seen_f.reserve(F);
    seen_p.reserve(M);
    for (uint64_t id : frame_ids) {
      require(id != 0, "frame id 0 is reserved");
      require(seen_f.emplace(id, true).second, "duplicate frame id in window");
    }
    for (uint64_t id : point_ids) {
      require(id != 0, "point id 0 is reserved");
      require(seen_p.emplace(id, false).second, "duplicate point id in window");
    }
  }

  // Validated - mutate. Replace semantics: the previous window is fully
  // discarded.
  clear();

  const auto pose_in = poses.unchecked<3>();
  const auto K_in = Ks.unchecked<3>();
  const auto k1_in = k1s.unchecked<1>();
  const auto kp_in = kp_pts.unchecked<2>();
  const auto kpn_in = kp_pts_norm.unchecked<2>();
  const auto pos_in = positions.unchecked<2>();
  const auto nobs_in = point_num_obs.unchecked<1>();
  const auto okp_in = obs_kp_idx.unchecked<1>();
  const auto onorm_in = obs_pt2d_norm.unchecked<2>();
  const auto fdesc_in = frame_desc_bytes.unchecked<2>();
  const uint64_t frame_desc_w = static_cast<uint64_t>(frame_desc_bytes.shape(1));
  const auto pdesc_in = point_desc_bytes.unchecked<2>();
  const uint64_t point_desc_w = static_cast<uint64_t>(point_desc_bytes.shape(1));

  // The GIL is released for the bulk copies: the inputs stay alive via
  // the call arguments and nothing Python is touched below.
  py::gil_scoped_release release;

  frames_.resize(F);
  for (size_t i = 0; i < F; ++i) {
    FrameRec& f = frames_[i];
    f.id = frame_ids[i];
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        f.pose[r * 4 + c] = pose_in(i, r, c);
      }
    }
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        f.K[r * 3 + c] = K_in(i, r, c);
      }
    }
    f.k1 = k1_in(i);
    const size_t b = static_cast<size_t>(kp_off(i));
    const size_t e = static_cast<size_t>(kp_off(i + 1));
    f.kp_pts.resize((e - b) * 2);
    f.kp_pts_norm.resize((e - b) * 2);
    for (size_t k = b; k < e; ++k) {
      f.kp_pts[(k - b) * 2] = kp_in(k, 0);
      f.kp_pts[(k - b) * 2 + 1] = kp_in(k, 1);
      f.kp_pts_norm[(k - b) * 2] = kpn_in(k, 0);
      f.kp_pts_norm[(k - b) * 2 + 1] = kpn_in(k, 1);
    }
    f.desc_dtype = frame_desc_dtype;
    f.desc_byte_width = frame_desc_w;
    if (frame_desc_w) {
      const size_t n = e - b;
      f.desc_bytes.resize(n * frame_desc_w);
      for (size_t k = 0; k < n; ++k) {
        for (uint64_t w = 0; w < frame_desc_w; ++w) {
          f.desc_bytes[k * frame_desc_w + w] = fdesc_in(b + k, w);
        }
      }
    }
    frame_idx_.emplace(f.id, i);
  }

  points_.resize(M);
  for (size_t i = 0; i < M; ++i) {
    PointRec& p = points_[i];
    p.id = point_ids[i];
    p.position[0] = pos_in(i, 0);
    p.position[1] = pos_in(i, 1);
    p.position[2] = pos_in(i, 2);
    p.num_observations = static_cast<uint64_t>(nobs_in(i));
    p.desc_dtype = point_desc_dtype;
    p.desc_byte_width = point_desc_w;
    if (point_desc_w) {
      p.desc_bytes.resize(point_desc_w);
      for (uint64_t w = 0; w < point_desc_w; ++w) {
        p.desc_bytes[w] = pdesc_in(i, w);
      }
    }
    point_idx_.emplace(p.id, i);
  }

  obs_.resize(O);
  for (size_t i = 0; i < O; ++i) {
    ObsRec& o = obs_[i];
    o.frame_id = obs_frame_ids[i];
    o.point_id = obs_point_ids[i];
    o.kp_idx = static_cast<uint64_t>(okp_in(i));
    o.xn = onorm_in(i, 0);
    o.yn = onorm_in(i, 1);
  }
}

void MapCore::clear() {
  frames_.clear();
  points_.clear();
  obs_.clear();
  frame_idx_.clear();
  point_idx_.clear();
}

bool MapCore::has_frame(uint64_t id) const { return frame_idx_.count(id); }

bool MapCore::has_point(uint64_t id) const { return point_idx_.count(id); }

std::vector<double> MapCore::frame_pose(uint64_t id) const {
  auto it = frame_idx_.find(id);
  if (it == frame_idx_.end()) throw std::out_of_range("unknown frame id");
  return std::vector<double>(frames_[it->second].pose,
                             frames_[it->second].pose + 16);
}

std::vector<double> MapCore::frame_K(uint64_t id) const {
  auto it = frame_idx_.find(id);
  if (it == frame_idx_.end()) throw std::out_of_range("unknown frame id");
  return std::vector<double>(frames_[it->second].K, frames_[it->second].K + 9);
}

double MapCore::frame_k1(uint64_t id) const {
  auto it = frame_idx_.find(id);
  if (it == frame_idx_.end()) throw std::out_of_range("unknown frame id");
  return frames_[it->second].k1;
}

std::vector<uint8_t> MapCore::frame_descriptors(uint64_t id) const {
  auto it = frame_idx_.find(id);
  if (it == frame_idx_.end()) throw std::out_of_range("unknown frame id");
  return frames_[it->second].desc_bytes;
}

std::vector<uint8_t> MapCore::point_descriptor(uint64_t id) const {
  auto it = point_idx_.find(id);
  if (it == point_idx_.end()) throw std::out_of_range("unknown point id");
  return points_[it->second].desc_bytes;
}

std::pair<int, uint64_t> MapCore::frame_desc_meta(uint64_t id) const {
  auto it = frame_idx_.find(id);
  if (it == frame_idx_.end()) throw std::out_of_range("unknown frame id");
  return {frames_[it->second].desc_dtype, frames_[it->second].desc_byte_width};
}

std::pair<int, uint64_t> MapCore::point_desc_meta(uint64_t id) const {
  auto it = point_idx_.find(id);
  if (it == point_idx_.end()) throw std::out_of_range("unknown point id");
  return {points_[it->second].desc_dtype, points_[it->second].desc_byte_width};
}

std::vector<double> MapCore::point_position(uint64_t id) const {
  auto it = point_idx_.find(id);
  if (it == point_idx_.end()) throw std::out_of_range("unknown point id");
  return std::vector<double>(points_[it->second].position,
                             points_[it->second].position + 3);
}

double MapCore::point_avg_err(uint64_t id) const {
  auto it = point_idx_.find(id);
  if (it == point_idx_.end()) throw std::out_of_range("unknown point id");
  return points_[it->second].avg_err;
}

std::vector<uint64_t> MapCore::window_frame_ids() const {
  std::vector<uint64_t> out;
  out.reserve(obs_.size());
  for (const ObsRec& o : obs_) out.push_back(o.frame_id);
  return out;
}

std::vector<uint64_t> MapCore::window_point_ids() const {
  std::vector<uint64_t> out;
  out.reserve(obs_.size());
  for (const ObsRec& o : obs_) out.push_back(o.point_id);
  return out;
}

std::vector<int64_t> MapCore::window_kp_idx() const {
  std::vector<int64_t> out;
  out.reserve(obs_.size());
  for (const ObsRec& o : obs_) out.push_back(static_cast<int64_t>(o.kp_idx));
  return out;
}

py::array_t<double> MapCore::window_pt2d_norm() const {
  py::array_t<double> out({static_cast<long>(obs_.size()), 2L});
  auto buf = out.mutable_unchecked<2>();
  for (size_t i = 0; i < obs_.size(); ++i) {
    buf(i, 0) = obs_[i].xn;
    buf(i, 1) = obs_[i].yn;
  }
  return out;
}

void MapCore::update_point_errors(const std::vector<uint64_t>& point_ids,
                                  const std::vector<double>& errors) {
  require(point_ids.size() == errors.size(),
          "update_point_errors: ids and errors must have equal length");
  py::gil_scoped_release release;
  for (size_t i = 0; i < point_ids.size(); ++i) {
    auto it = point_idx_.find(point_ids[i]);
    if (it == point_idx_.end()) continue;
    PointRec& p = points_[it->second];
    const uint64_t n = p.num_observations;
    if (n > 0) {
      p.avg_err = (p.avg_err * static_cast<double>(n - 1) + errors[i]) /
                  static_cast<double>(n);
    }
  }
}

}  // namespace slarnative
