#include "packed.h"
#include <cassert>
#include <spdlog/spdlog.h>

// #define DBG_PACKED

void PackedA::update() {

  // Update small-machine columns.
  if (_inst.num_small_machines != 0) {
#ifdef DBG_PACKED
    spdlog::info("[PackedA]: There are small machines.");
#endif

    // Regenerate configs only when the largest small-machine makespan grows.
    if (_inst.t[_inst.num_small_machines - 1] > _last_C) {
#ifdef DBG_PACKED
      spdlog::info("[PackedA]: I will regenerate.");
#endif
      std::vector<int> tmp_conf;
      std::vector<int> tmp_cost;

      // Enumerate small-machine configs for the current caps.
      auto emit = [&](VectorXi &conf, int cost) {
        // tmp_conf.emplace_back(conf);
        // std::cout << "Found config " << conf.transpose() << " with cost " <<
        // cost << std::endl;
        tmp_conf.insert(tmp_conf.end(), conf.data(), conf.data() + conf.size());
        tmp_cost.push_back(cost);
      };
      enumerate_small::optimal(_inst, emit);

      const auto num_configs = tmp_conf.size();
      if (num_configs == 0) [[unlikely]] {
        // found no possible configs for small machines
        // TODO: Can this edge case legitimately happen? Should be a violation
        spdlog::error("[PackedA::update] found no configs for small machines");
        std::abort();
        _hatS_cost.resize(0);
        _hatS.resize(0, 0);
        _num_small_cols = 0;
      } else {
        // finalize buffer, store in memory
        spdlog::info("[PackedA::update] found {} configs for small machines",
                     num_configs);
        _last_C = _inst.t[_inst.num_small_machines - 1];
        finalize_small(tmp_conf, tmp_cost);
      }
    }

    // update offset using cost
    _small_offset.resize(_inst.num_small_machines + 1);
    _small_offset[0] = 0;
    auto it = _hatS_cost.begin();
    for (int k = 0; k < _inst.num_small_machines; ++k) {
      // determine small block width with sliding window binary search
      it = std::upper_bound(
          it, _hatS_cost.end(),
          _inst.t[k]); // = first element violating load constraint
      auto block_size = std::distance(_hatS_cost.begin(), it);
      block_size = block_size < 0 ? 0 : block_size;
      _small_offset[k + 1] = _small_offset[k] + block_size;

#ifdef DBG_PACKED
      spdlog::info("[PackedA::update] machine k={} has width={}", k,
                   block_size);
#endif
    }
    _num_small_cols = _small_offset.back();
  } else {
    // no small machines present
#ifdef DBG_PACKED
    spdlog::info("[PackedA]: There are no small machines.");
#endif
    _hatS_cost.resize(0);
    _hatS.resize(0, 0);
    _num_small_cols = 0;

    _small_offset.resize(1);
    _small_offset[0] = 0;
  }

  const bool big_present = _inst.num_small_machines != _inst.m.size();
  int big_max_d = 0;
  // lazy big machine build, only once then const
  if (big_present) {
    if (!_big_built) [[unlikely]] {
#ifdef DBG_PACKED
      spdlog::info("building big machines bc _inst.num_small_machines={} != "
                   "{}=_inst.m.size()",
                   _inst.num_small_machines, _inst.m.size());
#endif
      size_t col = 0;

      // enumerate all big machine configs given pivot element a and job counts
      // as caps
      enumerate_big::optimal(_inst, [&](VectorXi &x, int cost) {
        _hatB.col(col) = x;
        _hatB_col_cost_mod_a[col] = cost % _inst.a;
        ++col;
      });

      _hatB_slack_Delta = std::max(_hatB.maxCoeff(), _inst.a);

      _big_built = true;
    }

    Eigen::ArrayXi d(_hatB_col_cost_mod_a.size());
    Eigen::Map<const Eigen::ArrayXi> big_residues(
        _inst.big_residue.touched.data(), _inst.big_residue.touched.size());
    for (int k = 0; k < _inst.big_residue.size(); ++k) {
      d = big_residues(k) - _hatB_col_cost_mod_a.array();

      d += (d < 0).cast<int>() * _inst.a;
      int local_max = d.maxCoeff();
      big_max_d = std::max(local_max, big_max_d);
    }
  }

  _num_big_cols = _hatB.cols() * _inst.big_residue.size();

  _big_end = _num_small_cols + _num_big_cols;
  _slack_end = _big_end + (_num_big_cols != 0 ? (_hatB.cols() + 2) : 0);

  _num_blocks = _inst.num_small_machines +
                (_inst.big_residue.size() > 0
                     ? _inst.big_residue.size() + 1
                     : (small_id_small_omit_slack ? 0 : 1));
#ifdef DBG_PACKED
  spdlog::info("[PackedA] _inst.big_residue.size()={} so _num_blocks={}",
               _inst.big_residue.size(), _num_blocks);

  spdlog::info(
      "[PackedA] _num_small_cols={} and _num_big_cols={} so _big_end={}",
      _num_small_cols, _num_big_cols, _big_end);
#endif

  if (_num_small_cols > 0) {
    _Delta = std::max((big_present)*_hatB_slack_Delta,
                      std::max(big_max_d, _hatS.maxCoeff()));
  } else {
    _Delta = std::max((big_present)*_hatB_slack_Delta, big_max_d);
  }
  for (size_t k = 0; k < _num_blocks; ++k) {
    const auto off = get_block_offset(k);
    assert(off <= _slack_end);
  }

#ifdef DBG_PACKED
  print_small(std::cout);
  print_big(std::cout);
  print(std::cout);
#endif
}

void PackedA::refresh_block_deltas() {
  _block_Delta.assign(_num_blocks, 0);

  for (int k = 0; k < _inst.num_small_machines; ++k) {
    const size_t c0 = _small_offset[static_cast<size_t>(k)];
    const size_t c1 = _small_offset[static_cast<size_t>(k + 1)];
    Scal block_delta = 0;
    if (c1 > c0 && _hatS.size() > 0) {
      const auto width = static_cast<Eigen::Index>(c1 - c0);
      block_delta = _hatS.leftCols(width).maxCoeff();
      if (_num_big_cols != 0 || !small_id_small_omit_dummy) {
        const int dummy_max = _inst.t[k] - _hatS_cost[0];
        block_delta = std::max(block_delta, dummy_max);
      }
    }
    _block_Delta[static_cast<size_t>(k)] = block_delta;
  }

  if (_inst.big_residue.size() > 0) {
    Eigen::Map<const Eigen::ArrayXi> big_residues(
        _inst.big_residue.touched.data(), _inst.big_residue.touched.size());
    for (int k = 0; k < _inst.big_residue.size(); ++k) {
      Eigen::ArrayXi d = big_residues(k) - _hatB_col_cost_mod_a.array();
      d += (d < 0).cast<int>() * _inst.a;
      const Scal dummy_delta =
          d.size() > 0 ? static_cast<Scal>(d.maxCoeff()) : 0;
      const Scal block_delta =
          std::max<Scal>(dummy_delta, _hatB.size() > 0 ? _hatB.maxCoeff() : 0);
      _block_Delta[static_cast<size_t>(_inst.num_small_machines + k)] =
          block_delta;
    }
    _block_Delta.back() = _inst.a;
  } else if (!small_id_small_omit_slack) {
    _block_Delta.back() = _inst.a;
  }
}

void PackedA::finalize_small(const std::vector<int> &tmp_conf,
                             const std::vector<int> &tmp_cost) {
  const auto num_cols = tmp_cost.size();

  // 1) sort indices by cost
  // spdlog::info("sort by cost");
  std::vector<int> idx(num_cols);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(std::execution::unseq, idx.begin(), idx.end(),
            [&](int a, int b) { return tmp_cost[a] < tmp_cost[b]; });

  // 2) permute both confs and cost by sorted idx
  // spdlog::info("permute");
  std::vector<int> sorted_cost(tmp_cost.size());
  std::vector<int> sorted_conf(tmp_conf.size());
  const int col_height = _inst.N;
  for (int new_j = 0; new_j < num_cols; ++new_j) {
    const int old_j = idx[new_j];

    // cost
    sorted_cost[new_j] = tmp_cost[old_j];

    // column data in column-major layout
    const int src_offset = old_j * col_height;
    const int dst_offset = new_j * col_height;

    std::copy_n(&tmp_conf[src_offset], col_height, &sorted_conf[dst_offset]);
  }

  // insert confs and costs into local containers
  _hatS_cost = Eigen::Map<const Eigen::VectorXi>(sorted_cost.data(), num_cols);

  const int num_rows = _inst.N;
  _hatS = Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::ColMajor>>(sorted_conf.data(),
                                                           num_rows, num_cols);
}

Scal PackedA::operator()(size_t row, size_t col) const {
  const size_t nrows = rows();
  const size_t ncols = cols();
  if (row >= nrows) [[unlikely]] {
    std::cerr << "[PackedA::operator()] ERROR row=" << row
              << " >= n_rows=" << nrows << "\n";
    std::abort();
  }

  if (col >= ncols) [[unlikely]] {
    std::cerr << "[PackedA::operator()] ERROR col=" << col
              << " >= n_cols=" << ncols << "\n";
    std::abort();
  }

  const size_t dummy_row = _inst.p.size();

  if (col < _num_small_cols) {
    return small(row, col, dummy_row);
  } else if (col < _big_end) {
    return big(row, col, dummy_row);
  } else if (col < _slack_end) {
    return slack(row, col);
  } else [[unlikely]] {
    std::cerr << "[PackedA::operator()] ERROR col=" << col
              << " >= _slack_end=" << _slack_end << "\n";
    std::abort();
  }
}

size_t PackedA::get_hatB_width() const {
  size_t w = 1;
  for (int j = 0; j < _inst.n.size(); ++j) {
    w *= 1 + (std::min(_inst.a - 1, _inst.n(j))) * (j != _inst.idx_a);
  }
  return w;
}

Scal PackedA::small(size_t row, size_t col, size_t dummy_row) const {
  if (row == dummy_row) [[unlikely]] {
    if constexpr (small_id_mixed_zero_dummy)
      // Small Identity
      return 0;
    else {
      size_t k = small_row_to_block(col);
      size_t col_local = col - _small_offset[k];
      return _inst.t[k] - _hatS_cost[col_local];
    }
  }

  size_t k = small_row_to_block(col);
  size_t col_local = col - _small_offset[k];

  return _hatS(row, col_local);
}

size_t PackedA::small_row_to_block(size_t i) const {
  return std::upper_bound(_small_offset.begin(), _small_offset.end(), i) -
         _small_offset.begin() - 1;
}

Scal PackedA::big(size_t row, size_t col, size_t dummy_row) const {
  const size_t big_local = col - _num_small_cols;
  const size_t block_idx = big_local / _hatB.cols();
  const size_t col_local = big_local % _hatB.cols();
  if (row == dummy_row) {
    return (_inst.big_residue.get_for_block(block_idx) + _inst.a -
            _hatB_col_cost_mod_a[col_local]) %
           _inst.a;
  } else if (row == _inst.idx_a) {
    return 0;
  } else {
    return _hatB(row, col_local);
  }
}

void PackedA::print_small(std::ostream &os) const {
  const size_t ncols = _hatS.cols();
  const size_t nrows = _hatS.rows();
  os << "hatS (" << nrows << " x " << ncols << ")\n";
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = 0; j < ncols; ++j) {
      os << _hatS(i, j);
      if (j + 1 < ncols)
        os << ' ';
    }
    os << '\n';
  }
}

void PackedA::print_big(std::ostream &os) const {
  size_t ncols = (_big_built)*_hatB.cols();
  size_t nrows = (_big_built)*_hatB.rows();
  os << "hatB (" << nrows << " x " << ncols << ")\n";
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = 0; j < ncols; ++j) {
      os << _hatB(i, j);
      if (j + 1 < ncols)
        os << ' ';
    }
    os << '\n';
  }
}

void PackedA::print(std::ostream &os) const {
  const size_t ncols = cols();
  const size_t nrows = rows();
  os << "packedA (" << nrows << " x " << ncols << ")\n";
  for (size_t i = 0; i < nrows; ++i) {
    for (size_t j = 0; j < ncols; ++j) {
      os << (*this)(i, j);
      if (j + 1 < ncols)
        os << ' ';
    }
    os << '\n';
  }
}
