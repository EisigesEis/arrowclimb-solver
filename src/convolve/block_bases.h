#pragma once

#include "convolve/sumset.h"
#include "model/matrix/packed.h"

#include <utility>
#include <vector>

namespace convolve {

// Build one base SumSet per block e from block column vectors of A.
inline void build_block_bases(const PackedA &A, int r, int n,
                              std::vector<SumSet> &base) {
  base.assign((size_t)n, SumSet{});
  for (int e = 0; e < n; ++e) {
    SumSet S;
    S.len = 1;
    S.hashed = false;

    auto [c0, c1] = A.block_col_range(e);
    S.vec.reserve((size_t)(c1 - c0));
    for (int col = c0; col < c1; ++col) {
      Vec v{};
      for (int row = 0; row < r; ++row)
        v.x[row] = A(row, col);
      S.vec.push_back(v);
    }

    S.sort_unique();
    S.ensure_hashed();
    base[(size_t)e] = std::move(S);
  }
}

} // namespace convolve
