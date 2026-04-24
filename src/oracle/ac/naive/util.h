#include <Eigen/Dense>
#include <cassert>
#include <omp.h>
#include <vector>


/**
 * Alg.2 from the paper.
 * Given I, Delta, K
 * Determine lower entries (number of each machine type) of each sub-problem for
 * each iteration of solving Fill b_down, b_down_tilde, b_down_hat, z
 */
inline void get_num_machines(const int I, const int r, const int K,
                             const Eigen::VectorXi &b_down,
                             Eigen::MatrixXi &B) {
  assert(B.rows() == r && B.cols() == I &&
         "dimensional mismatch of b_down_tilde in get_num_machines");
  Eigen::ArrayXi b = b_down;
  Eigen::ArrayXi nb(r);
  Eigen::ArrayXi til(r);

  const int K_par = (K & 1);

  for (int i = I; i >= 2; --i) {
#pragma omp simd
    for (int k = 0; k < r; ++k) {
      const int bk = b[k];
      const int big = (bk > K);

      const int z = big & ((bk & 1) ^ K_par);

      const int til_k = big ? (K - z) : bk;
      const int nb_k = big ? ((bk - til_k) >> 1) : 0;

      til[k] = til_k;
      nb[k] = nb_k;
    }
    B.col(i - 1) = til;
    if (i == 2)
      B.col(0) = nb;

    if (nb.isZero()) {
      if (i > 2) {
        B.leftCols(i - 2).setZero();
      }
      return;
    }

    b.swap(nb);
  }

  if (I == 1) {
    B.col(0) = b_down;
  }
}

inline Eigen::MatrixXi get_num_jobs(int I, const Eigen::VectorXi &b_up) {
  const auto r = b_up.size();
  Eigen::MatrixXi b_up_tilde = Eigen::MatrixXi::Zero(r, I);

  // we expect b_up non-empty and \ge 0 entries here, otherwise unsafe:
  const int steps =
      std::min(I, int(std::bit_width(static_cast<unsigned>(b_up.maxCoeff()))));
  if (steps == 0)
    return b_up_tilde;

  Eigen::ArrayXi rem = b_up;
  for (int i = I - 1; i >= I - steps; --i) {
    b_up_tilde.col(i) = rem;
    rem /= 2; // floor-halving
  }
  return b_up_tilde;
}

template <class WArr, class ArrLo, class ArrHi>
constexpr bool in_window(const WArr &wA, const ArrLo &loA,
                                          const ArrHi &hiA) {
  return (wA <= hiA).all() && (wA >= loA).all();
}