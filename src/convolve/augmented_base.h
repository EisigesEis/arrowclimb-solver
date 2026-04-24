#pragma once

#include "convolve/block_cols.h"
#include "convolve/sumset.h"

namespace convolve {

template <class MatrixLike, class OnVec, class OnBlockEnd>
inline void for_each_aug_vec(const MatrixLike &A, int r, int n,
                             OnVec &&on_vec,
                             OnBlockEnd &&on_block_end) {
  for_each_block_col(
      A, n,
      [&](int e, int j) {
        Vec v{};
        for (int k = 0; k < r; ++k)
          v.x[(size_t)k] = A(k, j);
        v.x[(size_t)(r + e)] = 1;
        on_vec(v);
      },
      std::forward<OnBlockEnd>(on_block_end));
}

template <class MatrixLike, class OnVec>
inline void for_each_aug_vec(const MatrixLike &A, int r, int n,
                             OnVec &&on_vec) {
  for_each_aug_vec(
      A, r, n, std::forward<OnVec>(on_vec), [](int) {});
}

template <class MatrixLike>
inline void build_aug_base(const MatrixLike &A, int r, int n,
                           const Bounds &bounds,
                           const Eigen::Ref<const Eigen::VectorXi> &ub,
                           SumSet &out) {
  out.clear();
  out.len = 1;
  out.hashed = false;
  out.vec.reserve((size_t)A.cols() + 1);

  auto add_if_valid = [&](const Vec &v) {
    if (vec_in_bounds(v, r + n, bounds, ub))
      out.vec.push_back(v);
  };

  add_if_valid(Vec{});
  for_each_aug_vec(A, r, n, add_if_valid);

  out.sort_unique();
  out.ensure_hashed();
}

} // namespace convolve
