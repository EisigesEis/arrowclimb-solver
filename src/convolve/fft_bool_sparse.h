#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace convolve {

inline std::size_t count_ones(const std::vector<std::uint8_t> &bits) {
  std::size_t ones = 0;
  for (std::size_t i = 0; i < bits.size(); ++i)
    ones += (bits[i] != 0);
  return ones;
}

inline bool should_use_sparse_self_conv(std::size_t N, std::size_t ones) {
  if (N == 0 || ones == 0)
    return false;
  // Exact sparse pair-sum is O(ones^2); FFT path is ~O(N log N).
  const double lhs = (double)ones * (double)ones;
  const double rhs = 2.0 * (double)N * std::max(1.0, std::log2((double)N));
  return lhs <= rhs;
}

inline void mul_bool_sparse_pair_sum(std::vector<std::uint8_t> &out,
                                     const std::vector<std::uint8_t> &a,
                                     std::vector<std::size_t> &active_out) {
  const std::size_t N = a.size();
  if (out.size() != N)
    out.assign(N, 0);
  else
    std::fill(out.begin(), out.end(), std::uint8_t{0});
  active_out.clear();

  std::vector<std::size_t> active;
  active.reserve(count_ones(a));
  for (std::size_t i = 0; i < N; ++i) {
    if (a[i])
      active.push_back(i);
  }

  const std::size_t s = active.size();
  for (std::size_t p = 0; p < s; ++p) {
    const std::size_t i = active[p];
    for (std::size_t q = 0; q < s; ++q) {
      const std::size_t j = active[q];
      const std::size_t sum = i + j;
      if (sum >= N)
        continue;
      if (!out[sum]) {
        out[sum] = 1;
        active_out.push_back(sum);
      }
    }
  }
}

} // namespace convolve
