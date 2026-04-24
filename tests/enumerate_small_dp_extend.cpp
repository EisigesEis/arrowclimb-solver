// tests/small_dp_extend_test.cpp

#include <gtest/gtest.h>

#include "instance/types.h"
#include "enumerate/small.h"
#include "small_old.h"

#include <Eigen/Dense>
#include <cstring>
#include <random>
#include <vector>


using Eigen::VectorXi;

using MatB = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>;

using enumerate_small::build_suffix_dp_exists_le_optimized;
using enumerate_small::extend_suffix_dp_exists_le_optimized;

static bool dp_equal(const MatB &a, const MatB &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols())
    return false;
  const std::size_t total =
      static_cast<std::size_t>(a.rows()) * static_cast<std::size_t>(a.cols());
  return std::memcmp(a.data(), b.data(), total * sizeof(unsigned char)) == 0;
}

// Helper: one correctness check for given (p,u,C) and chosen C_base values.
static void check_extend_correctness_for_instance(const VectorXi &p,
                                                  const VectorXi &u, int C) {
  ASSERT_GE(C, 0);

  MatB dp_full = build_suffix_dp_exists_le_optimized(p, u, C);

  std::vector<int> bases;
  bases.push_back(0);
  bases.push_back(std::max(1, C / 4));
  bases.push_back(std::max(1, C / 2));
  bases.push_back(std::max(1, (3 * C) / 4));

  for (int C_base : bases) {
    if (C_base >= C)
      continue;

    MatB dp_base = build_suffix_dp_exists_le_optimized(p, u, C_base);
    MatB dp_ext = dp_base;

    extend_suffix_dp_exists_le_optimized(dp_ext, p, u, C_base, C);

    ASSERT_TRUE(dp_equal(dp_full, dp_ext))
        << "extend_suffix_dp_exists_le_optimized produced different DP table "
        << "for C_base=" << C_base << " and C=" << C;
  }
}

// --- Tests ---

TEST(SmallDp, Fixed) {
  VectorXi p(4), u(4);
  p << 3, 5, 7, 2;
  u << 1, 2, 3, 4;

  for (int C : {0, 1, 5, 10, 20, 40}) {
    check_extend_correctness_for_instance(p, u, C);
  }
}

TEST(SmallDp, Bench) {
  VectorXi p(8), u(8);
  p << 776, 972, 484, 301, 614, 120, 273, 479;
  u << 1, 3, 4, 13, 3, 3, 1, 200;

  int C = 20000;
  check_extend_correctness_for_instance(p, u, C);
}

TEST(SmallDp, Random) {
  std::mt19937 rng(123456);

  std::uniform_int_distribution<int> dist_N(1, 10);
  std::uniform_int_distribution<int> dist_p(1, 50);
  std::uniform_int_distribution<int> dist_u(0, 10);

  for (int trial = 0; trial < 10; ++trial) {
    int N = dist_N(rng);
    VectorXi p(N), u(N);

    int max_load = 0;
    for (int i = 0; i < N; ++i) {
      p[i] = dist_p(rng);
      u[i] = dist_u(rng);
      max_load += p[i] * u[i];
    }

    int C = std::min(max_load, 2000);
    if (C <= 0)
      continue;

    check_extend_correctness_for_instance(p, u, C);
  }
}
