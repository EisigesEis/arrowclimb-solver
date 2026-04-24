#include "oracle/ac/common/legacy/dynamic_program.h"
#include "oracle/ac/common/Dedup.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <vector>

#include <spdlog/spdlog.h>

using Eigen::MatrixXi;
using Eigen::Ref;
using Eigen::VectorXi;
using std::vector;

static thread_local DedupScratch g_dedup_ctx;

std::vector<VectorXi>
dynamic_program(const PackedA &BM,
                Eigen::Ref<const VectorXi> job_target,
                Eigen::Ref<const VectorXi> picks_by_block,
                const BaseTableFn &bt) {
  const int B = BM.num_blocks();
  assert(picks_by_block.size() == B && "picks_by_block must have one entry per block");

  const int r = BM.rows();
  assert(job_target.size() == r && "job_target must have r entries (jobs+slack)");

  vector<VectorXi> Dt, next;
  Dt.reserve(128);
  Dt.emplace_back(VectorXi::Zero(r));

  for (int k = 0; k < B; ++k) {
    vector<VectorXi> Bk = bt(BM, k, job_target, picks_by_block(k));
    if (Bk.empty()) return {};

    next.clear();
    // reserve heuristic
    {
      size_t guess = (size_t)Dt.size() * (size_t)Bk.size();
      if (guess < 128)
        guess = 128;
      if (guess > (1ull << 26))
        guess = (1ull << 26);
      next.reserve(guess);
    }

    // SIMD without repacking
    const auto target = job_target.array();
    for (const auto &x : Dt) {
      for (const auto &y : Bk) {
        Eigen::VectorXi z = x + y;
        if ((z.array() <= target).all()) {
          // if ((z.array() == target).all()) {
          //   spdlog::info("[DT ingest] block {} sees target vector", k);
          // }

          next.emplace_back(std::move(z));
        }
      }
    }

    dedupe_inplace(next, r, g_dedup_ctx);

    Dt.swap(next);
  }

  // Emit frontier within job_target after all blocks.
  return Dt;
}
