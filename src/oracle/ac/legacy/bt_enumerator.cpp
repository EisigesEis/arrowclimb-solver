#include "oracle/ac/legacy/bt_enumerator.h"

#include "oracle/ac/common/Dedup.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <vector>

namespace {
static thread_local DedupScratch g_dedup_ctx;
}

namespace oracle::ac_legacy {

std::vector<Eigen::VectorXi>
compute_base_table_for_block(const PackedA &A, std::size_t k,
                             Eigen::Ref<const Eigen::VectorXi> target, int s) {
  using Eigen::VectorXi;

  const int r = static_cast<int>(A.rows());
  const int c0 = static_cast<int>(A.get_block_offset(k));
  const int c1 = static_cast<int>(A.get_block_offset(k + 1));
  const int t = c1 - c0;

  if (target.size() != r) {
    assert(false &&
           "compute_base_table_for_block: target.size() must equal A.rows()");
  }

  if (r == 0 || t == 0) {
    return (s == 0) ? std::vector<VectorXi>{VectorXi()}
                    : std::vector<VectorXi>{};
  }

  if (s == 0) {
    return {VectorXi::Zero(r)};
  }

  VectorXi caps = target;
  const int Delta_local = A.block_maxCoeff(k);
  for (int i = 0; i < r; ++i) {
    const long long tmp = 1LL * s * Delta_local;
    if (tmp < caps[i])
      caps[i] = static_cast<int>(tmp);
  }

  std::vector<VectorXi> cur;
  cur.reserve(1);
  cur.emplace_back(VectorXi::Zero(r));

  VectorXi rem(r);
  VectorXi col(r);

  for (int pick = 1; pick <= s; ++pick) {
    std::vector<VectorXi> next;

    const std::size_t want = cur.size() * static_cast<std::size_t>(c1 - c0);
    next.reserve(std::min<std::size_t>(want, 1'000'000ULL));

    for (const VectorXi &base : cur) {
      rem = caps - base;

      for (int j = c0; j < c1; ++j) {
        bool ok = true;
        for (int i = 0; i < r; ++i) {
          const int v = A(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
          if (v > rem[i]) {
            ok = false;
            break;
          }
          col[i] = v;
        }

        if (ok)
          next.emplace_back(base + col);
      }
    }

    dedupe_inplace(next, r, g_dedup_ctx);
    cur.swap(next);
    if (cur.empty())
      break;
  }

  return cur;
}

} // namespace oracle::ac_legacy
