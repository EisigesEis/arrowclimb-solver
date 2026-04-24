#include "oracle/ac/batch/bt_enumerator.h"

#include "convolve/sumset.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace oracle::ac_batch {

SumSet compute_base_sumset_for_block(const PackedA &A, int k,
                                     Eigen::Ref<const Eigen::VectorXi> target,
                                     int s, PowTable &pow_table) {
  using Eigen::VectorXi;
  SumSet empty;

  const int r = static_cast<int>(A.rows());
  const auto [c0, c1] = A.block_col_range(k);
  const int t = c1 - c0;

  if (target.size() != r) {
    assert(false &&
           "compute_base_table_for_block: target.size() must equal A.rows()");
  }

  if (r == 0 || t == 0) {
    if (s == 0) {
      reset_unit_sumset(empty);
      return empty;
    }
    return empty;
  }
  if (s == 0) {
    reset_unit_sumset(empty);
    return empty;
  }

  // invalidate pow table cache on block change
  if (!pow_table.matches_digit_key(target, s))
    pow_table.reset_for_digit_key(target, s);

  VectorXi caps = target;
  const int Delta_local = A.block_maxCoeff(static_cast<std::size_t>(k));
  for (int i = 0; i < r; ++i) {
    const long long tmp = 1LL * s * Delta_local;
    if (tmp < caps[i])
      caps[i] = static_cast<int>(tmp);
  }

  SumSet base;
  base.len = 1;
  base.hashed = false;
  base.vec.reserve(static_cast<std::size_t>(std::max(0, t)));

  for (int j = c0; j < c1; ++j) {
    bool ok = true;
    Vec v{};
    for (int i = 0; i < r; ++i) {
      const int aij = A(i, j);
      if (aij > caps[i]) {
        ok = false;
        break;
      }
      v.x[(std::size_t)i] = aij;
    }
    if (ok)
      base.vec.push_back(v);
  }

  if (base.vec.empty())
    return empty;
  base.sort_unique();
  base.ensure_hashed();

  SumSet cur;
  reset_unit_sumset(cur);

  SumSet tmp_merge;
  SumSet tmp_pow;
  Bounds vac;
  vac.vacuous = true;

  auto cnt = static_cast<std::uint32_t>(s);
  while (cnt) {
    const int bit = __builtin_ctz(cnt);
    cnt &= (cnt - 1);

    ensure_pow_level(pow_table, bit, base, r, caps, 1, 0, 0, tmp_pow);
    merge_sets_into(cur, pow_table.p[(std::size_t)bit], r, caps, vac, tmp_merge);
    std::swap(cur, tmp_merge);
    if (cur.size() == 0)
      return empty;
  }

  return cur;
}

} // namespace oracle::ac_batch
