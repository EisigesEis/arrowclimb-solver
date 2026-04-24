#pragma once
#include <Eigen/Dense>
#include "instance/types.h"
#include <gtest/gtest.h>
#include <random>

using Eigen::VectorXi;
using Scal = int;
using Eigen::MatrixXi;

inline void EXPECT_UNIQUE(const Eigen::MatrixXi &M) {
  const int rows = M.rows(); // length of each config
  const size_t col_bytes = size_t(rows) * sizeof(int);

  // index array 0..cols_used-1
  std::vector<int> idx(M.cols());
  std::iota(idx.begin(), idx.end(), 0);

  // helper: pointer to column c
  auto col_ptr = [&](int c) -> const int * {
    // In column-major, each column is contiguous
    return M.data() + size_t(c) * rows;
  };

  // lexicographic compare via memcmp
  auto lex_less = [&](int a, int b) {
    const int *pa = col_ptr(a);
    const int *pb = col_ptr(b);
    return std::memcmp(pa, pb, col_bytes) < 0;
  };

  std::sort(std::execution::par_unseq, idx.begin(), idx.end(), lex_less);

  // count unique columns
  bool ok = true;
  for (int k = 1; k < M.cols(); ++k) {
    const int *prev = col_ptr(idx[k - 1]);
    const int *curr = col_ptr(idx[k]);
    if (std::memcmp(prev, curr, col_bytes) == 0) {
      ok = false;
      Eigen::Map<const Eigen::VectorXi> v_prev(prev, rows);
      Eigen::Map<const Eigen::VectorXi> v_curr(curr, rows);
      ADD_FAILURE()
          << "Duplicate configuration detected:\n"
          << "  sorted position k-1 = " << (k - 1)
          << ", original column = " << idx[k - 1] << "\n"
          << "  sorted position k   = " << k
          << ", original column = " << idx[k] << "\n"
          << "  prev = " << v_prev.transpose() << "\n"
          << "  curr = " << v_curr.transpose();
      break;
    }
  }

  EXPECT_TRUE(ok) << "Found duplicate configurations in matrix";
}

inline void EXPECT_UNIQUE(const Eigen::MatrixXi &M, int cols_used) {
  ASSERT_LE(cols_used, M.cols()) << "cols_used > M.cols()";

  const int rows  = M.rows();
  const size_t bytes = size_t(rows) * sizeof(int);

  // index array 0..cols_used-1
  std::vector<int> idx(cols_used);
  std::iota(idx.begin(), idx.end(), 0);

  auto col_ptr = [&](int c) -> const int * {
    // column-major, each column contiguous
    return M.data() + size_t(c) * rows;
  };

  auto lex_less = [&](int a, int b) {
    const int *pa = col_ptr(a);
    const int *pb = col_ptr(b);
    return std::memcmp(pa, pb, bytes) < 0;
  };

  std::sort(std::execution::par_unseq, idx.begin(), idx.end(), lex_less);

  bool ok = true;
  for (int k = 1; k < cols_used; ++k) {
    const int *prev = col_ptr(idx[k - 1]);
    const int *curr = col_ptr(idx[k]);
    if (std::memcmp(prev, curr, bytes) == 0) {
      ok = false;
      Eigen::Map<const Eigen::VectorXi> v_prev(prev, rows);
      Eigen::Map<const Eigen::VectorXi> v_curr(curr, rows);

      ADD_FAILURE()
          << "Duplicate configuration detected:\n"
          << "  sorted position k-1 = " << (k - 1)
          << ", original column = " << idx[k - 1] << "\n"
          << "  sorted position k   = " << k
          << ", original column = " << idx[k] << "\n"
          << "  prev = " << v_prev.transpose() << "\n"
          << "  curr = " << v_curr.transpose();
    }
  }

  EXPECT_TRUE(ok) << "Found duplicate configurations in matrix";
}


inline size_t get_hatB_width(ProblemInstance &inst) {
  size_t w = 1;
  for (int j = 0; j < inst.p.size(); ++j) {
    w *= 1 + (std::min(inst.a - 1, inst.n(j))) * (j != inst.idx_a);
  }
  return w;
}

inline Eigen::VectorXi uniform_random_vec(int n, int lo, int hi) {
  assert(lo <= hi);
  Eigen::VectorXi v =
      Eigen::VectorXi::Random(n);
  const int range = hi - lo + 1;

  v = v.unaryExpr([&](int x) {
    x = std::abs(x);
    return lo + (x % range);
  });

  return v;
}

static std::mt19937 global_rng(123456u);

inline ProblemInstance random_instance(unsigned seed = std::random_device{}()) {
  ProblemInstance inst;

  std::uniform_int_distribution<int> dist_NM(1, 4);
  inst.M = dist_NM(global_rng);
  inst.N = dist_NM(global_rng);
  std::uniform_int_distribution<int> dist_zM(0, inst.M - 1);
  inst.num_small_machines = dist_zM(global_rng);

  inst.m = uniform_random_vec(inst.M, 1, 4);
  inst.s = uniform_random_vec(inst.M, 5, 100);
  inst.t = uniform_random_vec(inst.M, 5, 100);

  inst.n = uniform_random_vec(inst.N, 1, 20);
  inst.p = uniform_random_vec(inst.N, 1, 20);

  int idx = 0;
  inst.n.maxCoeff(&idx);
  inst.idx_a = static_cast<size_t>(idx);
  inst.a = inst.p[inst.idx_a];

  return inst;
}
