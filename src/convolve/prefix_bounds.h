#pragma once

#include "convolve/sumset.h"

#include <algorithm>

namespace convolve {

// Generic affine prefix tube:
//   lo_k = max(0, floor(len * b_up[k] / q) + lo_shift)
//   hi_k = min(    ceil(len * b_up[k] / q) + hi_shift, b_up[k])
inline Bounds compute_prefix_bounds_asym_affine(
    const Eigen::Ref<const Eigen::VectorXi> &b_up, int m_dim, int q, int len,
    long long lo_shift, long long hi_shift) {
  Bounds B;
  B.vacuous = false;
  B.lo_active = true;
  B.hi_active = true;

  for (int k = 0; k < m_dim; ++k) {
    const long long num = 1LL * len * (long long)b_up[k];
    const long long floor_center = num / q;
    const long long ceil_center = (num + (q - 1)) / q;

    const long long lo = std::max(0LL, floor_center + lo_shift);
    const long long hi =
        std::min(ceil_center + hi_shift, (long long)b_up[k]);
    B.lo[k] = lo;
    B.hi[k] = hi;
  }
  return B;
}

} // namespace convolve
