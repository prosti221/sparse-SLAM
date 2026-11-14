#include "point_registry.h"

#include <stdexcept>

namespace slarnative {

PointRegistry::PointRegistry(int desc_width) : desc_width_(desc_width) {
  if (desc_width_ <= 0) {
    throw std::invalid_argument(
        "PointRegistry: descriptor width must be positive");
  }
  // The initial capacity; ensure_capacity grows geometrically.
  positions_ = py::array_t<double>({64L, 3L});
  descriptors_ = py::array_t<uint8_t>({64L, static_cast<long>(desc_width_)});
  has_desc_ = py::array_t<uint8_t>(64L);
}

void PointRegistry::ensure_capacity(size_t need_rows) {
  const size_t cap = positions_.shape(0);
  if (need_rows <= cap) return;
  size_t next = cap;
  while (next < need_rows) next *= 2;
  py::array_t<double> np({static_cast<long>(next), 3L});
  py::array_t<uint8_t> nd({static_cast<long>(next), static_cast<long>(desc_width_)});
  py::array_t<uint8_t> nh(static_cast<long>(next));
  if (used_) {
    auto src = positions_.unchecked<2>();
    auto dst = np.mutable_unchecked<2>();
    for (size_t r = 0; r < used_; ++r) {
      for (int c = 0; c < 3; ++c) dst(r, c) = src(r, c);
    }
    auto sd = descriptors_.unchecked<2>();
    auto dd = nd.mutable_unchecked<2>();
    for (size_t r = 0; r < used_; ++r) {
      for (int c = 0; c < desc_width_; ++c) dd(r, c) = sd(r, c);
    }
    auto sh = has_desc_.unchecked<1>();
    auto dh = nh.mutable_unchecked<1>();
    for (size_t r = 0; r < used_; ++r) dh(r) = sh(r);
  }
  positions_ = std::move(np);
  descriptors_ = std::move(nd);
  has_desc_ = std::move(nh);
}

void PointRegistry::register_points(
    const std::vector<uint64_t>& ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> desc_bytes,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> has_desc) {
  const long N = static_cast<long>(ids.size());
  if (positions.ndim() != 2 || positions.shape(0) != N ||
      positions.shape(1) != 3) {
    throw std::invalid_argument("register_points: positions must be (N, 3)");
  }
  if (desc_bytes.ndim() != 2 || desc_bytes.shape(0) != N ||
      desc_bytes.shape(1) != desc_width_) {
    throw std::invalid_argument(
        "register_points: desc_bytes must be (N, W) with W = "
        "the registry's descriptor width");
  }
  if (has_desc.ndim() != 1 || has_desc.shape(0) != N) {
    throw std::invalid_argument("register_points: has_desc must be (N,)");
  }

  size_t n_new = 0;
  for (uint64_t id : ids) {
    if (!row_of_.count(id)) ++n_new;
  }
  ensure_capacity(used_ + n_new);

  const auto pos_in = positions.unchecked<2>();
  const auto db_in = desc_bytes.unchecked<2>();
  const auto hd_in = has_desc.unchecked<1>();
  {
    py::gil_scoped_release release;
    auto pos = positions_.mutable_unchecked<2>();
    auto desc = descriptors_.mutable_unchecked<2>();
    auto hd = has_desc_.mutable_unchecked<1>();
    for (long i = 0; i < N; ++i) {
      const uint64_t id = ids[i];
      auto it = row_of_.find(id);
      size_t row;
      if (it == row_of_.end()) {
        row = used_++;
        ids_.push_back(id);
        row_of_.emplace(id, row);
      } else {
        row = it->second;
      }
      for (int c = 0; c < 3; ++c) pos(row, c) = pos_in(i, c);
      for (int c = 0; c < desc_width_; ++c) desc(row, c) = db_in(i, c);
      hd(row) = hd_in(i);
    }
  }
}

void PointRegistry::update_positions(
    const std::vector<uint64_t>& ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> positions) {
  const long N = static_cast<long>(ids.size());
  if (positions.ndim() != 2 || positions.shape(0) != N ||
      positions.shape(1) != 3) {
    throw std::invalid_argument("update_positions: positions must be (N, 3)");
  }
  const auto pos_in = positions.unchecked<2>();
  {
    py::gil_scoped_release release;
    auto pos = positions_.mutable_unchecked<2>();
    for (long i = 0; i < N; ++i) {
      auto it = row_of_.find(ids[i]);
      if (it == row_of_.end()) {
        throw std::out_of_range("update_positions: unknown point id");
      }
      for (int c = 0; c < 3; ++c) pos(it->second, c) = pos_in(i, c);
    }
  }
}

void PointRegistry::remove_points(const std::vector<uint64_t>& ids) {
  {
    py::gil_scoped_release release;
    auto pos = positions_.mutable_unchecked<2>();
    auto desc = descriptors_.mutable_unchecked<2>();
    auto hd = has_desc_.mutable_unchecked<1>();
    for (uint64_t id : ids) {
      auto it = row_of_.find(id);
      if (it == row_of_.end()) continue;
      const size_t row = it->second;
      const size_t last = used_ - 1;
      if (row != last) {
        // swap-erase the last row into the hole, fix the moved id's row
        const uint64_t moved_id = ids_[last];
        for (int c = 0; c < 3; ++c) pos(row, c) = pos(last, c);
        for (int c = 0; c < desc_width_; ++c) desc(row, c) = desc(last, c);
        hd(row) = hd(last);
        ids_[row] = moved_id;
        row_of_[moved_id] = row;
      }
      ids_.pop_back();
      row_of_.erase(it);
      --used_;
    }
  }
}

void PointRegistry::clear() {
  ids_.clear();
  row_of_.clear();
  used_ = 0;
}

py::array PointRegistry::positions_view() const {
  // A zero-copy view over the live rows (keeps the registry's storage
  // alive as the base).
  return py::array_t<double>({static_cast<long>(used_), 3L},
                             positions_.data(), positions_);
}

py::array PointRegistry::descriptors_view() const {
  return py::array_t<uint8_t>(
      {static_cast<long>(used_), static_cast<long>(desc_width_)},
      descriptors_.data(), descriptors_);
}

py::array PointRegistry::has_desc_view() const {
  return py::array_t<uint8_t>({static_cast<long>(used_)},
                              has_desc_.data(), has_desc_);
}

long PointRegistry::row_of(uint64_t id) const {
  auto it = row_of_.find(id);
  return it == row_of_.end() ? -1 : static_cast<long>(it->second);
}

std::vector<uint64_t> PointRegistry::ids() const { return ids_; }

}  // namespace slarnative
