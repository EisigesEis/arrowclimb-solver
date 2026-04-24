#pragma once

#include "model/matrix/packed.h"

#include <algorithm>

namespace convolve {

/*
  Iterates columns in the first n block ranges of A.

  This is the cheap path for callers that only care about true block columns and
  know that columns outside those ranges are irrelevant.
*/
template <class MatrixLike, class OnBlockCol, class OnBlockEnd>
inline void for_each_block_col(const MatrixLike &A, int n,
                               OnBlockCol &&on_block_col,
                               OnBlockEnd &&on_block_end) {
  const int C = A.cols();

  for (int e = 0; e < n; ++e) {
    auto [c0, c1] = A.block_col_range(e);
    const int lo = std::clamp<int>((int)c0, 0, C);
    const int hi = std::clamp<int>((int)c1, 0, C);
    for (int j = lo; j < hi; ++j)
      on_block_col(e, j);
    on_block_end(e);
  }
}

/*
  Iterates all columns of A, partitioned by the first n block ranges.

  A column j is "covered" if it lies in some interval A.block_col_range(e) for
  e in [0, n). A column is "uncovered" if it lies outside the union of those
  first-n block intervals.

  Visit order:
  - uncovered columns before block 0
  - columns of block 0, then on_block_end(0)
  - uncovered columns between block 0 and block 1
  - columns of block 1, then on_block_end(1)
  - ...
  - uncovered columns after the last processed block
*/
template <class MatrixLike, class OnBlockCol, class OnBlockEnd, class OnUncoveredCol>
inline void for_each_block_and_uncovered_col(const MatrixLike &A, int n,
                                             OnBlockCol &&on_block_col,
                                             OnBlockEnd &&on_block_end,
                                             OnUncoveredCol &&on_uncovered) {
  const int C = A.cols();
  int next_uncovered = 0;

  for (int e = 0; e < n; ++e) {
    auto [c0, c1] = A.block_col_range(e);
    const int lo = std::clamp<int>((int)c0, 0, C);
    const int hi = std::clamp<int>((int)c1, 0, C);
    for (int j = next_uncovered; j < lo; ++j)
      on_uncovered(j);
    for (int j = lo; j < hi; ++j)
      on_block_col(e, j);
    next_uncovered = std::max(next_uncovered, hi);
    on_block_end(e);
  }

  for (int j = next_uncovered; j < C; ++j)
    on_uncovered(j);
}

} // namespace convolve
