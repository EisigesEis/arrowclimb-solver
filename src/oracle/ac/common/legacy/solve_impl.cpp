#include "oracle/ac/common/Dedup.h"
#include "oracle/ac/common/legacy/dynamic_program.h"
#include "model/matrix/model.h"
#include "model/matrix/packed.h"
#include "oracle/ac/naive/util.h"
#include <Eigen/Dense>
#include <algorithm>
#include <spdlog/spdlog.h>

// #define AC_DEBUG

using Eigen::MatrixXi;

static DedupScratch g_dedup_ctx;

namespace oracle::ac_common {

bool solve_impl(const LPModel<PackedA> &m, const BaseTableFn &bt) {
  const int Delta_global = m.A.maxCoeff();
  const auto &A = m.A;
  const auto &b = m.b;

  if (A.num_blocks() == 0) {
    return (b.array() == 0).all();
  }

  const auto &r = A.rows();
  const auto &h = A.cols();
  const auto &n = A.num_blocks();

  const auto &b_up = b.head(r);
  const auto &b_down = b.tail(A.num_blocks());

  // if (b_up[r-1] < 0) {
  //   return false; // can't reach negative rhs
  // }

  // 1) Determine global parameters
  const int K = std::max(
      1,
      int(floor(2.0 * (r + 1) * log2(4.0 * (r + 1) * double(Delta_global)))));
  const auto D = n * K * Delta_global;
  const auto b_down_max = b_down.maxCoeff();
  const int I =
      std::max(1, int(ceil(log2(double(b_down_max + K) / (2 * K + 1)))) + 1);
  assert(K >= 1 && I >= 1 && D >= 1);

// 2) Direct solve on base case
#ifdef AC_DEBUG
  spdlog::info("Computed params: K={} Delta={} I={}", K, Delta_global, I);
#endif
#ifdef AC_USE_DIRECT_SOLVE
  spdlog::debug("Calling solve direct.");
  return solve_direct(Delta_global, BM, b_full);
#endif

  // 3) Alg.2 get_num_machines to obtain problem split
  MatrixXi b_down_tilde = MatrixXi::Zero(n, I);
#ifdef AC_DEBUG
  std::cerr << "b_down=" << b_down.transpose() << std::endl;
#endif
  get_num_machines(I, n, K, b_down, b_down_tilde);
#ifdef AC_DEBUG
  std::cerr << "b_down_tilde=" << b_down_tilde.transpose() << std::endl;
#endif

#ifdef AC_DEBUG
  std::cerr << "b_up=" << b_up.transpose() << std::endl;
#endif
  const MatrixXi b_up_tilde = get_num_jobs(I, b_up);
  const auto LO = (b_up_tilde.array() - D).matrix().cwiseMax(0);
#ifdef AC_DEBUG
  std::cerr << "b_up_tilde=" << b_up_tilde.transpose() << std::endl;
#endif

  // Frontier holds r-vectors at the current digit scale.
  std::vector<VectorXi> merged;
  if (I == 1) {
    std::vector<VectorXi> Ni =
        dynamic_program(m.A, b_up_tilde.col(0), b_down_tilde.col(0), bt);
    for (const auto &y : Ni) {
      // std::cout << "[AC] checking " << y.transpose() << " ?= " << b_up.transpose() << std::endl;
      if ((y.array() == b_up.array()).all())
        return true;
    }
    return false;
  }

  for (int i = 0; i < I; ++i) {

#ifdef AC_DEBUG
    std::cerr << "[AC] digit " << i
              << " job_target=" << b_up_tilde.col(i).transpose()
              << " picks_by_block=" << b_down_tilde.col(i).transpose();
#endif

    // Base: Get Ni from DP, already unique
    std::vector<VectorXi> Ni =
        dynamic_program(m.A, b_up_tilde.col(i), b_down_tilde.col(i), bt);
#ifdef AC_DEBUG
    std::cerr << "    raw Ni.size()=" << Ni.size();
    if (!Ni.empty())
      std::cerr << "  sample=" << Ni.front().transpose();
    std::cerr << "\n";
#endif

    if (Ni.empty()) {
#ifdef AC_DEBUG
      std::cerr << "    FAIL: Ni empty after dedup -> infeasible\n";
#endif
      return false;
    }

    const auto hiA = b_up_tilde.col(i).array();
    const auto loA = LO.col(i).array();

    if (i == 0) {
      std::vector<VectorXi> base;
      base.reserve(Ni.size());
#ifdef AC_DEBUG
      size_t kept = 0, dropped = 0;
#endif
      for (const auto &y : Ni) {
        if (in_window(y.array(), loA, hiA)) {
          base.push_back(2 * y);
#ifdef AC_DEBUG
          ++kept;
        } else {
          ++dropped;
#endif
        }
      }
#ifdef AC_DEBUG
      std::cerr << "    base keep=" << kept << " drop=" << dropped
                << " -> merged.size()=" << base.size() << "\n";
#endif
      if (base.empty()) {
#ifdef AC_DEBUG
        std::cerr << "    FAIL: no candidates survive window at base digit\n";
#endif
        return false;
      }
      merged.swap(base);
      continue;
    }

    // Induction: combine Ni, Nim1
    std::vector<VectorXi> next;
    next.reserve(
        std::max<size_t>(1, merged.size() * std::min<size_t>(Ni.size(), 8)));

#ifdef AC_DEBUG
    size_t acc = 0, rej = 0, final_hits = 0;
#endif
    for (const auto &x : merged) {
      for (const auto &y : Ni) {
        const auto sum = x + y;
        if (i < I - 1) {
          if (in_window(sum.array(), loA, hiA)) {
            next.push_back(2 * sum);
#ifdef AC_DEBUG
            ++acc;
          } else {
            ++rej;
#endif
          }
        } else {
          // Final: compare equality
          if ((sum.array() == b_up.array()).all()) {
#ifdef AC_DEBUG
            std::cerr << "    SUCCESS: final exact match " << sum.transpose()
                      << "\n";
#endif
            return true;
#ifdef AC_DEBUG
          } else {
            ++rej;
#endif
          }
        }
      }
    }
#ifdef AC_DEBUG
    std::cerr << "    combine accepted=" << acc << " rejected=" << rej
              << " -> next.size()=" << next.size() << "\n";
#endif

    if (next.empty()) {
#ifdef AC_DEBUG
      std::cerr << "    FAIL: frontier died at digit " << i << "\n";
#endif
      return false;
    }

    // Deduplicate
    dedupe_inplace(next, r, g_dedup_ctx);
#ifdef AC_DEBUG
    std::cerr << "    deduped next.size()=" << next.size();
    if (!next.empty())
      std::cerr << "  sample=" << next.front().transpose();
    std::cerr << "\n";
#endif

    merged.swap(next);
  }

#ifdef AC_DEBUG
  std::cerr << "[AC] FAIL: reached end with no exact match\n";
#endif
  return false;
}

} // namespace oracle::ac_common
