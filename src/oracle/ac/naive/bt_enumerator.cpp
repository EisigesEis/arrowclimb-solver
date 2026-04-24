#include "oracle/ac/naive/bt_enumerator.h"

#include "convolve/sumset.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace oracle::ac_naive {

SumSet compute_base_sumset_for_block(const PackedA &A, int k,
                                     Eigen::Ref<const Eigen::VectorXi> target,
                                     int s, PowTable & /*pow_table*/) {
  const int r = static_cast<int>(A.rows());
  const int c0 = static_cast<int>(A.get_block_offset(static_cast<std::size_t>(k)));
  const int c1 =
      static_cast<int>(A.get_block_offset(static_cast<std::size_t>(k + 1)));
  const int t = c1 - c0;

  if (target.size() != r) {
    assert(false &&
           "compute_base_sumset_for_block: target.size() must equal A.rows()");
  }

  SumSet out;
  out.len = 0;
  out.hashed = false;

  if (r == 0 || t == 0) {
    if (s == 0) {
      reset_unit_sumset(out);
    }
    return out;
  }

  if (s == 0) {
    reset_unit_sumset(out);
    return out;
  }

  Eigen::VectorXi caps = target;
  const int Delta_local = A.block_maxCoeff(static_cast<std::size_t>(k));
  for (int i = 0; i < r; ++i) {
    const long long tmp = 1LL * s * Delta_local;
    if (tmp < caps[i])
      caps[i] = static_cast<int>(tmp);
  }

  std::vector<Vec> cur;
  cur.reserve(1);
  cur.push_back(Vec{});

  for (int pick = 1; pick <= s; ++pick) {
    std::vector<Vec> next;
    const std::size_t want = cur.size() * static_cast<std::size_t>(c1 - c0);
    next.reserve(std::min<std::size_t>(want, 1'000'000ULL));

    for (const Vec &base : cur) {
      for (int j = c0; j < c1; ++j) {
        bool ok = true;
        Vec z = base;
        for (int i = 0; i < r; ++i) {
          const int aij = A(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
          const int sum = z.x[static_cast<std::size_t>(i)] + aij;
          if (sum > caps[i]) {
            ok = false;
            break;
          }
          z.x[static_cast<std::size_t>(i)] = sum;
        }
        if (ok)
          next.push_back(z);
      }
    }

    out.vec = std::move(next);
    out.sort_unique();
    cur.swap(out.vec);
    if (cur.empty())
      break;
  }

  out.vec = std::move(cur);
  out.sort_unique();
  out.ensure_hashed();
  return out;
}

} // namespace oracle::ac_naive
