#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

#include "Types.h"
#include "bfs.h"
// #include "eigen_bitwise_ext.h"

#include <Eigen/Dense>
#include <cassert>
#include <vector>

using MatrixXi = Eigen::MatrixXi;
class BlockedMatrix;

/**
 * Alg.2 from the paper.
 * Given I, Delta, K
 * Determine lower entries (number of each machine type) of each sub-problem for
 * each iteration of solving Fill b_down, b_down_tilde, b_down_hat, z
 */
EIGEN_STRONG_INLINE void get_num_machines(int I, int r, int K,
                                          const Eigen::VectorXi &b_down,
                                          Eigen::MatrixXi &B) {
  assert(B.rows()==r && B.cols()==I && "dimensional mismatch of b_down_tilde in get_num_machines");
  Eigen::VectorXi b = b_down;
  Eigen::VectorXi nb(r); nb.setZero();
  Eigen::VectorXi til(r);

  for (int i = I; i >= 2; --i) {
    for (int k = 0; k < r; ++k) {
      const int bk = b[k];
      if (bk <= K) { til[k] = bk; nb[k] = 0; }
      else {
        const int z = ((bk & 1) != (K & 1)) ? 1 : 0;
        til[k] = K - z;
        nb[k]  = (bk - til[k]) >> 1;
      }
    }
    B.col(i-1) = til;      // r×1 -> r×1, no transpose
    if (i == 2) B.col(0) = nb;
    b.swap(nb);
    nb.setZero();
  }

  if (I == 1) { // meeting 290425: Falls I=1 ist b_down_tilde = b_down in Index I
    B.col(0) = b_down;
  }
}

/**
 * Construct Sparse Matrix with all configuration blocks
 *
 * \param instance problem instance
 * \param a pivot element
 * \param l number of dummy jobs
 * \param mod_frequency how often a each modulo remainder mod a is present
 * \param candidates_S all candidate configs feasible for biggest small machine
 * \param candidates_B all candidate configs feasible for big machines
 * \param b rhs (to be filled)
 */
bool construct_ilp(ProblemInstance &instance, int a, int l,
                   VectorXi mod_frequency,
                   vector<Config> &candidates_S,
                   vector<Config> &candidates_B, VectorXi &b,
                   BlockedMatrix &bm);