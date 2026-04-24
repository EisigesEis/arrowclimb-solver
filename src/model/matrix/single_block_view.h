#pragma once

#include "model/matrix/packed.h"

#include <algorithm>
#include <cstddef>
#include <utility>

class SingleBlockAView {
public:
  SingleBlockAView(const PackedA &A, int block_index)
      : A_(A), block_index_(block_index) {
    const auto [c0, c1] =
        A_.block_col_range(static_cast<std::size_t>(block_index_));
    c0_ = static_cast<int>(c0);
    c1_ = static_cast<int>(c1);
  }

  std::size_t rows() const { return A_.rows(); }
  std::size_t cols() const {
    return static_cast<std::size_t>(std::max(0, c1_ - c0_));
  }
  std::size_t num_blocks() const { return 1; }
  int block_maxCoeff(std::size_t k) const {
    if (k == 0)
      return A_.block_maxCoeff(static_cast<std::size_t>(block_index_));
    return 0;
  }
  std::pair<std::size_t, std::size_t> block_col_range(std::size_t k) const {
    if (k == 0)
      return {0, cols()};
    return {cols(), cols()};
  }
  int maxCoeff() const {
    return A_.block_maxCoeff(static_cast<std::size_t>(block_index_));
  }
  int operator()(std::size_t row, std::size_t col) const {
    return A_(row, static_cast<std::size_t>(c0_ + static_cast<int>(col)));
  }

private:
  const PackedA &A_;
  int block_index_ = 0;
  int c0_ = 0;
  int c1_ = 0;
};
