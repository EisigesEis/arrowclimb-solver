#include "oracle/discrep/solve.h"

#include "model/matrix/single_block_view.h"
#include "oracle/discrep/detail/fft_solver_core.h"

namespace oracle::discrep {

bool solve(const LPModel<PackedA> &m) {
  static thread_local detail::FFTWorkspace ws;
  return detail::run_solver(m.A, m.b, &m, ws, "discrepancy::solve");
}

void warmup(const LPModel<PackedA> &m) { detail::warmup(m); }

bool solve_single_block(const PackedA &A, int block_index,
                        const Eigen::Ref<const Eigen::VectorXi> &target,
                        int picks) {
  Eigen::VectorXi b(target.size() + 1);
  b.head(target.size()) = target;
  b[target.size()] = picks;

  SingleBlockAView view(A, block_index);
  static thread_local detail::FFTWorkspace ws;
  return detail::run_solver(view, b, nullptr, ws,
                            "ac_discrepancy::single_block");
}

} // namespace oracle::discrep
