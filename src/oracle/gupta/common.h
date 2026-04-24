#pragma once

#include "convolve/block_bases.h"
#include "convolve/prefix_bounds.h"
#include "convolve/sumset.h"
#include "model/matrix/model.h"
#include "model/matrix/packed.h"

#include <cassert>
#include <utility>
#include <vector>

namespace oracle::gupta_common {

enum class Status { Ready, SolvedTrue, SolvedFalse };

struct Context {
  const PackedA *A = nullptr;
  int r = 0;
  int n = 0;
  int q = 0;
  int Delta = 0;
  long long lo_const = 0;
  long long hi_const = 0;
  Eigen::VectorXi b_up;
  Eigen::VectorXi b_down;
  Vec target{};

  const PackedA &matrix() const {
    assert(A != nullptr);
    return *A;
  }
};

inline Status prepare(const LPModel<PackedA> &m, Context &ctx) {
  const PackedA &A = m.A;
  const auto &b = m.b;

  ctx.A = &A;
  ctx.r = A.rows();
  ctx.n = A.num_blocks();
  ctx.b_up = b.head(ctx.r);
  ctx.b_down = b.tail(ctx.n);
  ctx.q = (int)ctx.b_down.sum();

  if (ctx.q == 0)
    return ctx.b_up.isZero() ? Status::SolvedTrue : Status::SolvedFalse;

  ctx.Delta = A.maxCoeff();
  ctx.lo_const = -1LL * ctx.n * (long long)ctx.Delta * (ctx.n + 2LL * ctx.r);
  ctx.hi_const = 1LL * ctx.n * (long long)ctx.Delta * (1 + 2LL * ctx.r);

  ctx.target = Vec{};
  for (int k = 0; k < ctx.r; ++k)
    ctx.target.x[(size_t)k] = ctx.b_up[k];

  return Status::Ready;
}

inline void build_base(const Context &ctx, std::vector<SumSet> &base) {
  convolve::build_block_bases(ctx.matrix(), ctx.r, ctx.n, base);
}

template <class Runner>
inline bool run_with(const LPModel<PackedA> &m, Runner &&runner) {
  Context ctx;
  switch (prepare(m, ctx)) {
  case Status::SolvedTrue:
    return true;
  case Status::SolvedFalse:
    return false;
  case Status::Ready:
    break;
  }

  SumSet cur;
  SumSet tmp;
  reset_unit_sumset(cur);

  if (!std::forward<Runner>(runner)(ctx, cur, tmp))
    return false;

  if (!cur.hashed)
    cur.sort_unique();
  return cur.contains(ctx.target);
}

} // namespace oracle::gupta_common
