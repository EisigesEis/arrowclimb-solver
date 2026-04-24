#pragma once
#include "instance/types.h"
#include "model/matrix/packed.h"
#include "io/csv_main.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <ostream>
#include "bench.h"

// #define DBG_MODEL

/*
  Model for ILP of arrow climb
  only responsible for rhs b.
  matrix is off-loaded to MatrixImpl
*/
template <class MatrixImpl> inline void finalize_after_mat_update(MatrixImpl &) {}

template <> inline void finalize_after_mat_update<PackedA>(PackedA &A) {
  A.refresh_block_deltas();
}

template <class MatrixImpl> struct LPModel {
  ProblemInstance &inst;
  MatrixImpl A;
  Eigen::VectorXi b;

  explicit LPModel(ProblemInstance &instance) : inst(instance), A(inst) {}

  /*
    after an iteration the guessed makespan vector inst.t has changed
    update b and A accordingly
  */
  void update() {
    build_rhs();

    BENCH_RUN("mat_update", A.update());
    finalize_after_mat_update(A);
    log_proxy_s();
  }

private:
  /*
    build rhs vector according to guessed makespans
    if no big machines are present, according to small identity, we delete ell, calN
  */
  inline void build_rhs() {
    bool big_machines_exist = (inst.num_small_machines != inst.m.size());

    bool add_ell = (MatrixImpl::small_id_small_omit_dummy) ? big_machines_exist : true;
    bool add_calN = (MatrixImpl::small_id_small_omit_slack) ? big_machines_exist : true;

    b.setZero(inst.p.size() + add_ell + inst.num_small_machines +
              inst.big_residue.size() + add_calN); // n, ell, m, calN

    b.head(inst.n.size()) = inst.n;
#ifdef DBG_MODEL
    std::cout << "added " << inst.n.transpose() << " to " << b.transpose()
              << std::endl;
#endif

    b.segment(inst.p.size() + add_ell, inst.num_small_machines) =
        inst.m.head(inst.num_small_machines);
#ifdef DBG_MODEL
    std::cout << "added " << inst.m.head(inst.num_small_machines).transpose()
              << " to " << b.transpose() << std::endl;
#endif

    if (big_machines_exist) {
      int ell = get_ell();

      b[inst.p.size()] = ell;
      b[b.size() - 1] = get_num_slack(ell);

#ifdef DBG_MODEL
      std::cout << "added ell=" << ell << " and calN=" << b[b.size() - 1]
                << " to " << b.transpose() << std::endl;
#endif
    } else if constexpr (!MatrixImpl::small_id_mixed_zero_dummy) {
      int ell = get_ell();
      if (add_ell) {
        b[inst.p.size()] = ell;
        if (ell < 0) {
          spdlog::error(
              "[model] derive gave inherently infeasible makespan guess");
          std::abort();
        }
      }
      if (add_calN) {
        b[b.size() - 1] = get_num_slack(ell);
      }
    }

    for (int i = 0; i < inst.big_residue.size(); ++i) {
      b[inst.p.size() + 1 + inst.num_small_machines + i] =
          inst.big_residue.get_cnt_for_block(i);
#ifdef DBG_MODEL
      std::cout << "added big machine r=" << inst.big_residue.get_for_block(i)
                << " with cnt=" << inst.big_residue.get_cnt_for_block(i)
                << std::endl;
#endif
    }

#ifdef DBG_MODEL
    print(std::cout);
#endif
  }

  void print(std::ostream &os) {
    os << "RHS b(" << b.size() << ") transposed:" << std::endl
       << b.transpose() << std::endl;
  }

  inline int get_ell() {
    return inst.t.dot(inst.m) - inst.total_load;
  }

  // Compute calN for the RHS b.
  inline int get_num_slack(int ell) {
    int D_idxA;
    int sum_ceils = 0;

#ifdef DBG_MODEL
    std::cout << "inst.idx_a=" << inst.idx_a << std::endl;
#endif

    for (int j = 0; j < inst.n.size(); ++j) {
      int Hj = 0;
      // H_small_j
      for (int k = 0; k < inst.num_small_machines; ++k) {
        Hj += inst.m[k] * (inst.t[k] / inst.p[j]); // integer division floor
      }

      // H_big_j
      Hj += (j == inst.idx_a) ? 0 : (inst.a - 1) * inst.M_B;

      int Dj = std::max(0, inst.n[j] - Hj);
#ifdef DBG_MODEL
      std::cout << "D_" << j << " = " << Dj << std::endl;
#endif

      if (j == inst.idx_a) {
        D_idxA = Dj;
      } else {
        sum_ceils += (Dj + inst.a - 1) / inst.a; // ceil(Dj / a)
      }
    }

    const int dummy_bundles = (ell + inst.a - 1) / inst.a; // ceil(ell / a)

    return D_idxA + std::max(dummy_bundles, sum_ceils);
  }

  inline void log_proxy_s() {
    const auto d = static_cast<long long>(A.rows());
    const auto Delta = static_cast<long long>(A.maxCoeff());
    const auto ell = static_cast<long long>(get_ell());
    const auto calN = static_cast<long long>(get_num_slack((int)ell));

    io::csv::main_set("d", d);
    io::csv::main_set("Delta", Delta);
    io::csv::main_set("ell", ell);
    io::csv::main_set("calN", calN);
  }

  /*
    Currently unused attempt to tighten ell according to small identity.
    If small and big machines are present, the dummy row on small machines could be set to zero.
    Then only ell = ell_small + ell_big = ell_big has to be bound.
    Potential to decrease \Delta.

    Example showing why ell needs a divisibility adjustment:
    \calA =
      0 1 1 0 1 0 0
      0 0 2 4 0 a 0
    \b_up^\top =
      1 \ell
    \b_down =
      1 1 \calN

    For even a and odd \ell, this problem is infeasible. With nonzero dummy
    jobs on small machines, \calA.col(0) = (0 1)^\top restores feasibility.

    C_big is a lower bound for ell_big, and it must be rounded up to preserve
    feasibility in cases with this parity mismatch.
  */
  int get_ell_small_identity() {
    const auto num_big_machines = inst.t.size() - inst.num_small_machines;
    const auto m_big = inst.m.tail(num_big_machines);
    const auto C_big = inst.t.tail(num_big_machines).dot(m_big);
    const auto &P = inst.total_load;
    const auto &a = inst.a;

    const Eigen::Map<Eigen::VectorXi> big_residues(
        inst.big_residue.touched.data(), inst.big_residue.touched.size());
    const Scal R = big_residues.dot(m_big);

    const Scal Lambda = C_big - P;
    const Scal D0 = std::max(Lambda, R);

    const Scal delta = D0 - R;
    const Scal delta_mod = delta % a;
    const Scal adjust = (a - delta_mod) % a;
    const Scal ell_big = D0 + adjust;

    // Scal ell_big = R + a * ((D0 - R + a - 1) / a);

    return ell_big;

    // Scal rem = (D0 - R) % a;
    // return D0 + (rem != 0) * (a - rem);
  }
};
