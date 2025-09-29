#include "dynamic_program.h"
#include "BlockedMatrix.h"
#include "Dedup.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <vector>

using Eigen::MatrixXi;
using Eigen::Ref;
using Eigen::VectorXi;
using std::vector;

static thread_local DedupScratch g_dedup_ctx;

std::vector<VectorXi>
dynamic_program(const BlockedMatrix &BM,
                int /*K*/,
                Eigen::Ref<const VectorXi> job_target,
                Eigen::Ref<const VectorXi> picks_by_block,
                const BaseTableFn &bt) {
  const int B = BM.num_blocks();
  assert(picks_by_block.size() == B && "picks_by_block must have one entry per block");

  const MatrixXi A0 = MatrixXi(BM.get_block(0));
  const int r = A0.rows();
  assert(job_target.size() == r && "job_target must have r entries (jobs+slack)");

  vector<VectorXi> Dt, next;
  Dt.reserve(128);
  Dt.emplace_back(VectorXi::Zero(r));

  for (int k = 0; k < B; ++k) {
    MatrixXi A_k = MatrixXi(BM.get_block(k));
    assert(A_k.rows() == r && "Block A_k shape must have r rows (jobs+slack)");

    vector<VectorXi> Bk = bt(A_k, job_target, picks_by_block(k));
    if (Bk.empty()) return {};

    // We expect bt to respect job_target.

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
        if ((z.array() <= target).all())
          next.emplace_back(std::move(z));
      }
    }

    dedupe_inplace(next, r, g_dedup_ctx);

    Dt.swap(next);
  }

  // Emit frontier within job_target after all blocks.
  return Dt;
}