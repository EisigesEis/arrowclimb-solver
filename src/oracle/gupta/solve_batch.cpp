#include "common.h"
#include "convolve/block_bases.h"
#include "convolve/prefix_bounds.h"
#include "gupta_core.h"
#include "model/matrix/model.h"

#include <cstdint>
#include <vector>

namespace oracle::gupta_batch {

using gupta_core::RunBatch;

// Feasibility-adjusted Alg.3 from 2507.03766. Msig is compressed into
// consecutive block runs and scheduled in powers of two.
bool solve(const LPModel<PackedA> &model) {
  return oracle::gupta_common::run_with(
      model,
      [](const oracle::gupta_common::Context &ctx, SumSet &cur, SumSet &tmp) {
        assert(ctx.r <= MAX_M &&
               "Increase MAX_M in discr_helper.h or use dynamic Vec");

        Eigen::VectorXi msig;
        std::vector<RunBatch> runs;
        Eigen::VectorXi max_run;

        // Build regular msig schedule. Then compress to batches.
        gupta_core::build_msig(ctx.b_down, msig);
        gupta_core::compress_msig(msig, ctx.n, runs, max_run);

        std::vector<SumSet> base;
        oracle::gupta_common::build_base(ctx, base);

        std::vector<PowTable> pow(ctx.n);
        SumSet pow_tmp;

        for (const auto &rb : runs) {
          const int e = rb.e;
          uint32_t cnt = rb.cnt;
          assert(cnt > 0);

          /*
          Schedule runs as power of two iteration (ctz) and self convolution
          using ensure_pow_level.
          */
          while (cnt) {
            const int bit = __builtin_ctz(cnt);
            cnt &= (cnt - 1);

            ensure_pow_level(pow[e], bit, base[e], ctx.r, ctx.b_up, ctx.q,
                             ctx.lo_const, ctx.hi_const, pow_tmp);

            const int add_len = 1 << bit;
            const int new_len = cur.len + add_len;
            const Bounds bounds = convolve::compute_prefix_bounds_asym_affine(
                ctx.b_up, ctx.r, ctx.q, new_len, ctx.lo_const, ctx.hi_const);

            merge_sets_into(cur, pow[e].p[bit], ctx.r, ctx.b_up, bounds, tmp);
            std::swap(cur, tmp);

            if (cur.size() == 0)
              return false;
          }
        }

        return true;
      });
}

} // namespace oracle::gupta_batch
