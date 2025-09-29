#include <Eigen/Dense>
#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "bt_enumerator.h"

using Eigen::MatrixXi;
using Eigen::VectorXi;
using std::vector;

// ---------- tiny helpers ----------

static inline VectorXi v(std::initializer_list<int> xs) {
  VectorXi r((int)xs.size());
  int i = 0;
  for (int x : xs) r[i++] = x;
  return r;
}

static std::string key(const VectorXi &u) {
  std::ostringstream oss;
  for (int i = 0; i < u.size(); ++i) {
    if (i) oss << ',';
    oss << u[i];
  }
  return oss.str();
}

static std::set<std::string> as_set(const vector<VectorXi> &U) {
  std::set<std::string> S;
  for (const auto &u : U) S.insert(key(u));
  return S;
}

// Sum of the s largest nonnegative entries per row (tight local cap, no outside info).
static Eigen::VectorXi row_caps_repetition(const Eigen::MatrixXi& A, int s) {
  const int r = A.rows();
  Eigen::VectorXi out(r);
  for (int i = 0; i < r; ++i) {
    const int rowMax = A.row(i).maxCoeff();
    out[i] = s * std::max(0, rowMax);
  }
  return out;
}

// Brute-force (only for small t,s). Enumerate x ∈ Z_{\ge 0}^t with sum x = s,
// compute A x, keep if each coord <= caps where caps = min(target, topS).
static vector<VectorXi> brute_base_table_vec_cap(const MatrixXi &A,
                                                 Eigen::Ref<const VectorXi> target,
                                                 int s) {
  const int r = A.rows(), t = A.cols();
  vector<VectorXi> out;
  if (r <= 0 || t <= 0 || s < 0) return out;

  VectorXi topS = row_caps_repetition(A, s);
  VectorXi caps = target.cwiseMin(topS);

  // recursive enumeration of compositions of s into t parts
  vector<int> x(t, 0);
  std::function<void(int, int)> rec = [&](int pos, int rem) {
    if (pos == t - 1) {
      x[pos] = rem;
      // compute Ax
      VectorXi nu = VectorXi::Zero(r);
      for (int j = 0; j < t; ++j) {
        int mul = x[j];
        if (mul == 0) continue;
        for (int i = 0; i < r; ++i)
          nu[i] += A(i, j) * mul;
      }
      bool ok = true;
      for (int i = 0; i < r; ++i)
        if (nu[i] > caps[i]) { ok = false; break; }
      if (ok) out.emplace_back(std::move(nu));
      return;
    }
    for (int take = 0; take <= rem; ++take) {
      x[pos] = take;
      rec(pos + 1, rem - take);
    }
  };
  rec(0, s);

  // dedup
  std::sort(out.begin(), out.end(), [](const VectorXi &a, const VectorXi &b) {
    for (int i = 0; i < a.size(); ++i) if (a[i] != b[i]) return a[i] < b[i];
    return false;
  });
  out.erase(std::unique(out.begin(), out.end(), [](const VectorXi &a, const VectorXi &b) {
              return (a.size() == b.size()) && ((a - b).squaredNorm() == 0);
            }),
            out.end());
  return out;
}

// ---------- tests ----------

// (1) s = 0 -> only zero vector if r,t > 0
TEST(BTEnumerator, ZeroSumGivesOnlyZero) {
  const int r = 3, t = 2, s = 0;
  MatrixXi A = MatrixXi::Ones(r, t);
  VectorXi target = VectorXi::Zero(r); // large target
  auto got = compute_base_table_for_block(A, target, s);
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0], VectorXi::Zero(r));
}

// (2) Single column, variety of s and target
TEST(BTEnumerator, SingleColumnMatchesScaledColumnWithinTarget) {
  const int r = 2, t = 1;
  MatrixXi A(r, t);
  A << 2,
       3;

  // Case a) s=5, target very large -> expect { 5*(2,3) } = (10,15)
  {
    int s = 5;
    VectorXi target = VectorXi::Constant(r, 1000);
    auto got = compute_base_table_for_block(A, target, s);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0], v({10, 15}));
  }

  // Case b) s=5, target too small -> expect {}
  {
    int s = 5;
    VectorXi target(2); target << 9, 100; // first coord 10 > 9
    auto got = compute_base_table_for_block(A, target, s);
    EXPECT_TRUE(got.empty());
  }
}

// (3) Two columns, small s; hand enumeration
TEST(BTEnumerator, TwoColumnsHandCheck) {
  // Columns: a1=(1,0,2), a2=(0,1,1)
  const int r = 3, t = 2, s = 3;
  MatrixXi A(r, t);
  A.col(0) = v({1, 0, 2});
  A.col(1) = v({0, 1, 1});

  // Large target so caps = topS; topS per row with s=3:
  // row0: only positives {1,0} -> top3 sum = 1
  // row1: {0,1} -> top3 sum = 1
  // row2: {2,1} -> take top 3 with repetition allowed via selections: but topS is over columns, so it's 2+1 (only two positives) = 3
  // However the DP with s=3 can realize (3,0,6), (2,1,5), (1,2,4), (0,3,3).
  // To avoid clipping by topS, give a target >= these values:
  VectorXi target(3); target << 3, 3, 6;

  auto got = compute_base_table_for_block(A, target, s);
  auto S = as_set(got);

  std::set<std::string> expect = {"3,0,6", "2,1,5", "1,2,4", "0,3,3"};
  EXPECT_EQ(S, expect);
}

// (4) Target cap filters but keeps feasible
TEST(BTEnumerator, TargetCapFiltersButKeepsFeasible) {
  // Same A as previous; set target = (1,3,4) so only those <= survive.
  const int r = 3, t = 2, s = 3;
  MatrixXi A(r, t);
  A.col(0) = v({1, 0, 2});
  A.col(1) = v({0, 1, 1});
  VectorXi target = v({1, 3, 4});

  auto got = compute_base_table_for_block(A, target, s);
  auto S = as_set(got);

  // From prior list, only (1,2,4) and (0,3,3) are <= target.
  std::set<std::string> expect = {"1,2,4", "0,3,3"};
  EXPECT_EQ(S, expect);
}

// (5) Duplicated columns; deduplication of outputs (no duplicates)
TEST(BTEnumerator, DuplicatedColumnsProduceNoDuplicateNu) {
  const int r = 2, t = 3, s = 2;
  MatrixXi A(r, t);
  // three identical columns (1,1)
  A.col(0) = v({1, 1});
  A.col(1) = v({1, 1});
  A.col(2) = v({1, 1});

  VectorXi target = VectorXi::Constant(r, 10);
  auto got = compute_base_table_for_block(A, target, s);
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0], v({2, 2}));
}

// (6) Randomized differential test vs brute force (small sizes)
TEST(BTEnumerator, RandomizedMatchesBruteForceSmall) {
  std::mt19937 rng(12345);

  for (int trial = 0; trial < 60; ++trial) {
    int r = 1 + (rng() % 3); // 1..3
    int t = 1 + (rng() % 4); // 1..4
    int s = 0 + (rng() % 4); // 0..3 to keep brute manageable

    MatrixXi A = MatrixXi::Zero(r, t);
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < t; ++j)
        A(i, j) = (rng() % 4); // 0..3

    VectorXi target = VectorXi::Constant(r, 1 + (rng() % 6)); // 1..6
    auto got = compute_base_table_for_block(A, target, s);
    auto want = brute_base_table_vec_cap(A, target, s);

    auto Sg = as_set(got);
    auto Sw = as_set(want);

    ASSERT_EQ(Sg, Sw) << "Mismatch in trial "
                      << trial
                      << " with r=" << r
                      << " t=" << t
                      << " s=" << s
                      << "\nA=\n" << A
                      << "\ntarget=" << target.transpose();
  }
}

// (7) Monotonicity in target: if target_small <= target_big (componentwise),
// results under small are subset of results under big.
TEST(BTEnumerator, MonotoneInTarget) {
  const int r = 2, t = 3, s = 3;
  MatrixXi A(r, t);
  A.col(0) = v({1, 0});
  A.col(1) = v({0, 1});
  A.col(2) = v({1, 1});

  VectorXi target_small = v({2, 2});
  VectorXi target_big   = v({100, 100});

  auto smallT = compute_base_table_for_block(A, target_small, s);
  auto bigT   = compute_base_table_for_block(A, target_big,   s);

  auto S_small = as_set(smallT);
  auto S_big   = as_set(bigT);

  for (const auto &keyv : S_small) {
    ASSERT_TRUE(S_big.count(keyv)) << "Vector " << keyv << " missing under larger target";
  }
}

// (8) Upper row bound respected: every result <= min(target, topS)
TEST(BTEnumerator, UpperRowBoundRespected) {
  const int r = 2, t = 2, s = 4;
  MatrixXi A(r, t);
  A.col(0) = v({2, 1});
  A.col(1) = v({3, 5});

  VectorXi target = VectorXi::Constant(r, 10000);
  VectorXi topS = row_caps_repetition(A, s);
  VectorXi caps = target.cwiseMin(topS);

  auto got = compute_base_table_for_block(A, target, s);

  for (const auto &nu : got) {
    EXPECT_LE(nu[0], caps[0]);
    EXPECT_LE(nu[1], caps[1]);
  }
}

// (9) t==0: for s>0, no way to pick s items -> empty
TEST(BTEnumerator, ZeroColumnsAndPositiveSIsEmpty) {
  const int r = 3, t = 0, s = 2;
  MatrixXi A(r, t); // 3x0
  VectorXi target = VectorXi::Constant(r, 10);
  auto got = compute_base_table_for_block(A, target, s);
  EXPECT_TRUE(got.empty());
}

// (10) SingleBlock_Sanity adapted to vector target
TEST(BTEnumerator, SingleBlock_Sanity) {
  Eigen::MatrixXi A(4, 12);
  A << 0,  0,  1,  0,  1,  1,  0,  0,  1,  0,  1,  1,
       2,  1,  2,  0,  1,  0,  2,  1,  2,  0,  1,  0,
       2,  3,  2,  4,  3,  4,  3,  4,  3,  5,  4,  5,
      35, 31, 31, 27, 27, 23, 15, 11, 11,  7,  7,  3;

  const int s = 3;
  // Use a uniform target so row 3 is capped to 94 (min with topS=97).
  VectorXi target = VectorXi::Constant(4, 94);

  auto N0 = compute_base_table_for_block(A, target, s);

  ASSERT_FALSE(N0.empty());
  EXPECT_LE(N0.size(), 364u); // C(12+3-1, 3) = 364

  for (const auto &v : N0) ASSERT_EQ(v.size(), 4);

  VectorXi topS = row_caps_repetition(A, s);
  VectorXi caps = target.cwiseMin(topS);
  for (const auto &nu : N0)
    for (int i = 0; i < 4; ++i)
      EXPECT_LE(nu[i], caps[i]);

  // Known present: 3 × column 1 = [0,3,9,93]
  VectorXi expected(4); expected << 0, 3, 9, 93;
  bool found_expected = false;
  for (const auto &v : N0) if ((v.array() == expected.array()).all()) { found_expected = true; break; }
  EXPECT_TRUE(found_expected) << "Expected candidate [0,3,9,93] not found";

  // Known absent: 3 × column 0 -> [0,6,6,105] exceeds target on row 3
  VectorXi forbidden(4); forbidden << 0, 6, 6, 105;
  bool found_forbidden = false;
  for (const auto &v : N0) if ((v.array() == forbidden.array()).all()) { found_forbidden = true; break; }
  EXPECT_FALSE(found_forbidden) << "Forbidden candidate [0,6,6,105] should be pruned by cap";
}