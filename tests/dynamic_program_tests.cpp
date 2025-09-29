#include <Eigen/Dense>
#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <set>
#include <sstream>
#include <vector>

#include "BlockedMatrix.h"
#include "bt_enumerator.h"
#include "dynamic_program.h"

using Eigen::MatrixXi;
using Eigen::VectorXi;
using std::vector;

static inline VectorXi v(std::initializer_list<int> xs) {
  VectorXi r((int)xs.size());
  int i = 0;
  for (int x : xs)
    r[i++] = x;
  return r;
}
static inline bool eq(const VectorXi &a, const VectorXi &b) {
  return a.size() == b.size() && (a - b).squaredNorm() == 0;
}
static std::string key(const VectorXi &u) {
  std::ostringstream oss;
  for (int i = 0; i < u.size(); ++i) {
    if (i)
      oss << ',';
    oss << u[i];
  }
  return oss.str();
}
static std::set<std::string> as_set(const vector<VectorXi> &U) {
  std::set<std::string> S;
  for (const auto &u : U)
    S.insert(key(u));
  return S;
}
static inline bool same_vec(Eigen::Ref<const Eigen::VectorXi> a,
                            Eigen::Ref<const Eigen::VectorXi> b) {
  return a.size() == b.size() && (a - b).squaredNorm() == 0;
}

// ---------- Providers ----------

static BaseTableFn testBT_forward_upper(std::vector<Eigen::VectorXi> *seen_uppers,
                                        const Eigen::VectorXi *expected_upper) {
  return [seen_uppers, expected_upper](Eigen::Ref<const Eigen::MatrixXi> A,
                                       Eigen::Ref<const Eigen::VectorXi> upper,
                                       int /*s*/) -> std::vector<VectorXi> {
    seen_uppers->push_back(upper);
    EXPECT_TRUE(same_vec(upper, *expected_upper))
        << "Provider saw mismatching 'upper': got " << upper.transpose()
        << " expected " << expected_upper->transpose();

    const int tag = A(0, 0);
    if (A.rows() == 2) {
      if (tag == 1)
        return {v({1, 0}), v({0, 1})};
      if (tag == 2)
        return {v({1, 1})};
      if (tag == 3)
        return {v({2, 0}), v({0, 2})};
      if (tag == 9)
        return {v({0, 0})};
    }
    return {};
  };
}

static BaseTableFn simple_testBT = [](Eigen::Ref<const Eigen::MatrixXi> A,
                                      Eigen::Ref<const Eigen::VectorXi>,
                                      int /*s*/) {
  const int tag = A(0, 0);
  if (A.rows() == 2) {
    if (tag == 1)
      return vector<VectorXi>{v({1, 0}), v({0, 1})};
    if (tag == 2)
      return vector<VectorXi>{v({1, 1})};
    if (tag == 3)
      return vector<VectorXi>{v({2, 0}), v({0, 2})};
    if (tag == 9)
      return vector<VectorXi>{v({0, 0})};
  }
  return vector<VectorXi>{};
};

// ---------- Tests ----------

TEST(DynamicProgramProps, OutputsAreUniqueAndWithinUpper) {
  const int r = 2;
  BlockedMatrix BM(r, 3);
  BM.add_block(MatrixXi::Constant(r, 1, 1));
  BM.add_block(MatrixXi::Constant(r, 1, 2));
  BM.add_block(MatrixXi::Constant(r, 1, 9));
  const VectorXi upper = v({3, 2});
  const VectorXi target = upper;

  std::vector<Eigen::VectorXi> seen_uppers;
  auto out = dynamic_program(BM, /*K=*/0, upper, target,
                             testBT_forward_upper(&seen_uppers, &upper));

  ASSERT_FALSE(seen_uppers.empty());
  for (const auto &u : seen_uppers)
    EXPECT_TRUE(same_vec(u, upper));

  std::set<std::string> seen;
  for (const auto &x : out) {
    ASSERT_EQ(x.size(), r);
    for (int i = 0; i < r; ++i) {
      EXPECT_GE(x[i], 0);
      EXPECT_LE(x[i], upper[i]);
    }
    const auto k = key(x);
    ASSERT_FALSE(seen.count(k)) << "Duplicate vector " << k;
    seen.insert(k);
  }
}

TEST(DynamicProgramProps, OrderIndependenceOfBlocks) {
  const int r = 2;
  BlockedMatrix BMa(r, 2);
  BMa.add_block(MatrixXi::Constant(r, 1, 2));
  BMa.add_block(MatrixXi::Constant(r, 1, 1));
  BlockedMatrix BMb(r, 2);
  BMb.add_block(MatrixXi::Constant(r, 1, 1));
  BMb.add_block(MatrixXi::Constant(r, 1, 2));
  const VectorXi upper = v({2, 2});
  const VectorXi target = upper;

  std::vector<Eigen::VectorXi> seenA, seenB;
  auto outA = dynamic_program(BMa, /*K=*/0, upper, target,
                              testBT_forward_upper(&seenA, &upper));
  auto outB = dynamic_program(BMb, /*K=*/0, upper, target,
                              testBT_forward_upper(&seenB, &upper));

  EXPECT_EQ(as_set(outA), as_set(outB));
}

static vector<VectorXi> brute_convolution(const vector<vector<VectorXi>> &choices,
                                          const VectorXi &upper) {
  const int n = (int)choices.size();
  const int r = (int)upper.size();
  vector<VectorXi> acc{VectorXi::Zero(r)};
  for (int k = 0; k < n; ++k) {
    vector<VectorXi> next;
    for (const auto &a : acc) {
      for (const auto &b : choices[k]) {
        if (b.size() != r)
          continue;
        VectorXi s = a + b;
        bool ok = true;
        for (int i = 0; i < r; ++i)
          if (s[i] > upper[i]) {
            ok = false;
            break;
          }
        if (ok)
          next.emplace_back(std::move(s));
      }
    }
    std::sort(next.begin(), next.end(), [](const VectorXi &A, const VectorXi &B) {
      for (int i = 0; i < A.size(); ++i)
        if (A[i] != B[i])
          return A[i] < B[i];
      return false;
    });
    next.erase(std::unique(next.begin(), next.end(), [](const VectorXi &A, const VectorXi &B) { return A.size() == B.size() && (A - B).squaredNorm() == 0; }), next.end());
    acc.swap(next);
  }
  return acc;
}

TEST(DynamicProgramProps, MatchesBruteConvolution) {
  const int r = 2;
  const VectorXi upper = v({3, 2});
  const VectorXi target = upper;
  BlockedMatrix BM(r, 3);
  BM.add_block(MatrixXi::Constant(r, 1, 1));
  BM.add_block(MatrixXi::Constant(r, 1, 2));
  BM.add_block(MatrixXi::Constant(r, 1, 1));

  vector<vector<VectorXi>> choices = {
      {v({1, 0}), v({0, 1})},
      {v({1, 1})},
      {v({1, 0}), v({0, 1})}};

  std::vector<Eigen::VectorXi> seen_uppers;
  auto dp_out = dynamic_program(BM, /*K=*/0, upper, target,
                                testBT_forward_upper(&seen_uppers, &upper));
  auto brute = brute_convolution(choices, upper);

  EXPECT_EQ(as_set(dp_out), as_set(brute));
}

TEST(DynamicProgramProps, RandomizedSmallFuzzMatchesBrute) {
  std::mt19937 rng(1337);
  const int trials = 20;
  for (int tcase = 0; tcase < trials; ++tcase) {
    int r = 1 + (rng() % 3);
    int n = 1 + (rng() % 3);
    BlockedMatrix BM(r, n);
    std::vector<std::vector<VectorXi>> choices(n);
    for (int k = 0; k < n; ++k) {
      int tag = 100 + k;
      BM.add_block(Eigen::MatrixXi::Constant(r, 1, tag));
      int pick = rng() % 3;
      if (pick == 0) {
        int m = std::min(2, r);
        for (int i = 0; i < m; ++i) {
          VectorXi e = VectorXi::Zero(r);
          e[i] = 1;
          choices[k].push_back(e);
        }
      } else if (pick == 1) {
        choices[k].push_back(VectorXi::Ones(r));
      } else {
        choices[k].push_back(VectorXi::Zero(r));
      }
    }
    VectorXi upper(r);
    for (int i = 0; i < r; ++i)
      upper[i] = rng() % 4;
    const VectorXi target = upper;

    std::vector<Eigen::VectorXi> seen;
    BaseTableFn mirror = [choices, &seen, &upper](Eigen::Ref<const Eigen::MatrixXi> A,
                                                  Eigen::Ref<const Eigen::VectorXi> up,
                                                  int) {
      seen.push_back(up);
      int tag = A(0, 0);
      int k = tag - 100;
      if (k >= 0 && k < (int)choices.size())
        return choices[k];
      return std::vector<VectorXi>{};
    };

    auto out_dp = dynamic_program(BM, /*K=*/0, upper, target, mirror);
    auto out_ref = brute_convolution(choices, upper);
    EXPECT_EQ(as_set(out_dp), as_set(out_ref));
  }
}

TEST(DynamicProgramProps, ZeroOnlyProviderYieldsOnlyZero) {
  const int r = 2;
  BlockedMatrix BM(r, 2);
  BM.add_block(MatrixXi::Constant(r, 1, 9));
  BM.add_block(MatrixXi::Constant(r, 1, 9));
  const VectorXi upper = v({0, 0});
  const VectorXi target = upper;
  BaseTableFn zeros = [](Eigen::Ref<const Eigen::MatrixXi>, Eigen::Ref<const Eigen::VectorXi>, int) {
    return vector<VectorXi>{v({0, 0}), v({0, 0})};
  };
  auto out = dynamic_program(BM, /*K=*/0, upper, target, zeros);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_TRUE(eq(out[0], v({0, 0})));
}

TEST(DynamicProgramProps, UpperSaturationExactHit) {
  const int r = 2;
  BlockedMatrix BM(r, 2);
  BM.add_block(MatrixXi::Constant(r, 1, 1));
  BM.add_block(MatrixXi::Constant(r, 1, 2));
  const VectorXi upper = v({2, 2});
  const VectorXi target = upper;
  std::vector<Eigen::VectorXi> seen_uppers;
  auto out = dynamic_program(BM, /*K=*/0, upper, target,
                             testBT_forward_upper(&seen_uppers, &upper));
  auto S = as_set(out);
  ASSERT_TRUE(!S.count("2,2"));
}