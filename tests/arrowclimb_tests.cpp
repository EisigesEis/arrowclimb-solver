#include <Eigen/Dense>
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "BlockedMatrix.h"
#include "bt_enumerator.h"
#include "dynamic_program.h"
#include "feasibility.h"
#include "matrix.h"

using Eigen::MatrixXi;
using Eigen::VectorXi;
using std::vector;

static inline VectorXi v(std::initializer_list<int> xs) {
  VectorXi r((int)xs.size());
  int i = 0;
  for (int x : xs) r[i++] = x;
  return r;
}

// --- Helpers to mirror the solver's parameterization ---
static inline bool same_vec(Eigen::Ref<const Eigen::VectorXi> a,
                            Eigen::Ref<const Eigen::VectorXi> b) {
  return a.size() == b.size() && (a - b).squaredNorm() == 0;
}
static inline int compute_K(int Delta_global, int r) {
  // unchanged utility; used only to reconstruct I for b_down splitting
  return std::max(1, (int)std::floor(2.0 * (r + 1) * std::log2(4.0 * (r + 1) * (double)Delta_global)));
}
static inline int compute_I(int K, const VectorXi &b_down) {
  const int bmax = (b_down.size() > 0) ? b_down.maxCoeff() : 0;
  return std::max(1, (int)std::ceil(std::log2((2.0 * K + 1.0) * (bmax + 1.0))));
}

// Build a BlockedMatrix with r rows and the sum of widths columns; each block filled with 'tag'.
static inline BlockedMatrix makeBM(int r, const std::vector<std::pair<int, int>> &tag_widths) {
  int total_cols = 0;
  for (auto [tag, w] : tag_widths) (void)tag, total_cols += w;
  BlockedMatrix BM(r, total_cols);
  for (auto [tag, w] : tag_widths) {
    MatrixXi A = MatrixXi::Constant(r, w, tag);
    BM.add_block(A);
  }
  return BM;
}

// ---------------- TESTS ----------------

// (1) Infeasible when second coordinate never appears
TEST(SolveArrowClimb, InfeasibleWhenSecondCoordinateUnavailable) {
  const int r = 2, W = 4, Delta = 50;
  BlockedMatrix BM = makeBM(r, {{1, W}, {2, W}}); // two blocks

  VectorXi b_up   = v({1, 1});
  VectorXi b_down = v({10, 8});
  VectorXi b_full(b_up.size() + b_down.size());
  b_full << b_up, b_down;

  const int K = compute_K(Delta, r);
  const int I = compute_I(K, b_down);
  if (I < 2) GTEST_SKIP();

  // Provider returns only {(1,0)}
  BaseTableFn e1_only = [](Eigen::Ref<const MatrixXi> A,
                           Eigen::Ref<const VectorXi>,
                           int) {
    (void)A;
    return std::vector<VectorXi>{ v({1, 0}) };
  };

  EXPECT_FALSE(solve_arrow_climb(Delta, BM, b_full, e1_only));
}

// (2) Single block sanity (direct vs arrow-climb pipelines agree)
TEST(SolveArrowClimb, SingleBlockSmallFeasible) {
  MatrixXi A(4, 12);
  A << 0,  0,  1,  0,  1,  1,  0,  0,  1,  0,  1,  1,
       2,  1,  2,  0,  1,  0,  2,  1,  2,  0,  1,  0,
       2,  3,  2,  4,  3,  4,  3,  4,  3,  5,  4,  5,
      35, 31, 31, 27, 27, 23, 15, 11, 11,  7,  7,  3;

  BlockedMatrix BM(4, 12);
  BM.add_block(A);
  const int Delta = A.maxCoeff();

  // b_full: last entries are b_down; earlier are b_up
  VectorXi b_full(5);
  b_full <<  1,  2, 12, 45,  3;

  EXPECT_EQ(solve_direct(Delta, BM, b_full),
            solve_arrow_climb(Delta, BM, b_full, compute_base_table_for_block));
}