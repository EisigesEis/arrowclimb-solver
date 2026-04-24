#include "gupta_core.h"
#include "common.h"
#include "convolve/block_bases.h"
#include "convolve/prefix_bounds.h"
#include "model/matrix/model.h"

/*
  Feasibility adjusted version of Alg.3 from 2507.03766.
  We implicitly construct the layered DAG from Construction 1:
  - Layers j = 0..q "steps" are positions in balanced schedule Msig
  - Vertices at layer j are integer vectors v\in\Z^r bounded by Lemma 5
  - Directed edges from layer j-1 to j add one column vector from block Msig[j]

  The implementation performs layered DP for feasibility without materializing
  the DAG.
  Expected O(|V| + |E|) with hash-based dedup.
  Return true iff b^{(0)} is reachable at layer q.
*/
namespace oracle::gupta {

bool solve(const LPModel<PackedA> &m) {
  return oracle::gupta_common::run_with(
      m, [](const oracle::gupta_common::Context &ctx, SumSet &cur,
            SumSet &tmp) {
        Eigen::VectorXi msig;
        gupta_core::build_msig(ctx.b_down, msig);

        std::vector<SumSet> base;
        oracle::gupta_common::build_base(ctx, base);

        for (int step = 1; step <= ctx.q; ++step) {
          const int block = msig[step - 1];
          const Bounds bounds = convolve::compute_prefix_bounds_asym_affine(
              ctx.b_up, ctx.r, ctx.q, step, ctx.lo_const, ctx.hi_const);
          merge_sets_into(cur, base[(size_t)block], ctx.r, ctx.b_up, bounds,
                          tmp);
          std::swap(cur, tmp);
          if (cur.size() == 0)
            return false;
        }

        return true;
      });
}

} // namespace oracle::gupta
