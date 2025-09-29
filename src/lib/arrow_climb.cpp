#include <Eigen/Dense>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "BlockedMatrix.h"
#include "Dedup.h"
#include "dynamic_program.h"
#include "Config.h"

static DedupScratch g_dedup_ctx;

EIGEN_STRONG_INLINE MatrixXi get_num_jobs(int I, const VectorXi &b_up) {
  const auto r = b_up.size();
  MatrixXi b_up_tilde = MatrixXi::Zero(r, I);

  // we expect b_up non-empty and \ge 0 entries here, otherwise unsafe:
  const int steps = std::min(I, int(std::bit_width(static_cast<unsigned>(b_up.maxCoeff()))));
  if (steps == 0)
    return b_up_tilde;

  VectorXi rem = b_up;
  for (int i = I - 1; i >= I - steps; --i) {
    b_up_tilde.col(i) = rem;
    rem = (rem.array() / 2).matrix(); // floor-halving
  }
  return b_up_tilde;
}

template <class WArr, class ArrLo, class ArrHi>
EIGEN_STRONG_INLINE static bool in_window(
    const WArr &wA,
    const ArrLo &loA, const ArrHi &hiA) {
  return (wA <= hiA).all() && (wA >= loA).all();
}

bool arrow_climb_combine(const BlockedMatrix &BM,
                         int K,
                         int D,
                         const VectorXi &b_up,
                         const MatrixXi &b_down_tilde) {
  const int I = b_down_tilde.cols();
  const int r = b_up.size();

  // Build row-side per-digit targets
#ifdef AC_DEBUG
  std::cerr << "b_up=" << b_up.transpose() << std::endl;
#endif
  const MatrixXi b_up_tilde = get_num_jobs(I, b_up);
  const auto LO = (b_up_tilde.array() - D).matrix().cwiseMax(0);
#ifdef AC_DEBUG
  std::cerr << "b_up_tilde=" << b_up_tilde.transpose() << std::endl;
#endif

  // Frontier holds r-vectors at the current digit scale.
  vector<VectorXi> merged;

  if (I == 1) {
    vector<VectorXi> Ni =
      dynamic_program(BM, K, b_up_tilde.col(0), b_down_tilde.col(0));
    for (const auto &y : Ni) {
        if ((y.array() == b_up.array()).all()) return true;
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
    vector<VectorXi> Ni = dynamic_program(BM, K, b_up_tilde.col(i), b_down_tilde.col(i));
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
      vector<VectorXi> base;
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
    vector<VectorXi> next;
    next.reserve(std::max<size_t>(1, merged.size() * std::min<size_t>(Ni.size(), 8)));

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
            std::cerr << "    SUCCESS: final exact match " << sum.transpose() << "\n";
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