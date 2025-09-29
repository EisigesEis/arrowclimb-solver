#include "matrix.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

using Eigen::MatrixXi;
using Eigen::VectorXi;

static VectorXi V(std::initializer_list<int> xs) {
  VectorXi v(xs.size());
  int i = 0;
  for (int x : xs)
    v[i++] = x;
  return v;
}

TEST(GetNumMachines, ProducesExpectedColumns_KeqMaxSupp21) {
  // Case: K (=maxSupp) is large enough so every entry is a "zero-iteration" immediately.
  // b_down = [20,10]^T, I = 5, K = 21 (== maxSupp)
  // Expect: only the LAST column (col 4) is nonzero and equals b_down.
  const int I = 5;
  const VectorXi b_down = V({20, 10});
  const int r = static_cast<int>(b_down.size());
  const int K = 21; // K == maxSupp

  MatrixXi got = MatrixXi::Zero(r, I);
  get_num_machines(I, r, K, b_down, got);

  MatrixXi want(r, I);
  want << 0, 0, 0, 0, 20,
      0, 0, 0, 0, 10;

  EXPECT_EQ(got, want) << "b_down_tilde should have only the last column nonzero when K>=all b[k].";
}

TEST(GetNumMachines, ProducesExpectedColumns_KeqMaxSupp1) {
  // Case: small K (=maxSupp=1) forces the full binary-like carry sequence.
  // b_down = [20,10]^T, I = 5, K = 1 (== maxSupp)
  //
  // Hand-calculated from the paper’s recurrence:
  // Row for 20: [1,0,1,0,0]  (col0..col4)
  // Row for 10: [0,1,0,1,0]
  const int I = 5;
  const VectorXi b_down = V({20, 10});
  const int r = static_cast<int>(b_down.size());
  const int K = 1; // K == maxSupp

  MatrixXi got = MatrixXi::Zero(r, I);
  get_num_machines(I, r, K, b_down, got);

  MatrixXi want(r, I);
  want << 1, 0, 1, 0, 0,
      0, 1, 0, 1, 0;

  EXPECT_EQ(got, want) << "b_down_tilde columns must match the paper’s getNumMachines with K=maxSupp=1.";
}