#pragma once
#include "convolve/sumset.h"

#include <cassert>
#include <limits>
#include <queue>
#include <vector>

namespace oracle::gupta_core {

struct RunBatch {
  int cnt;
  int e;
};

static inline long long score_ll(long long k, long long j, long long q,
                                 long long m) {
  return k * q - j * m;
}

static inline long long max_run_vs(long long ke, long long me, int e,
                                   long long kf, long long mf, int f,
                                   long long q, long long j0) {
  const long long base = (ke - kf) * q - j0 * (me - mf);
  const long long den = q - (me - mf);
  const long long bound = (e < f) ? 0LL : -1LL;
  assert(den > 0);
  return floor_div_ll(bound - base, den) + 1;
}

// Alg. 1 from 2507.03766 in O(nq)
static inline void build_msig(const Eigen::Ref<const Eigen::VectorXi> &b_down,
                              Eigen::VectorXi &Msig) {
  const int n = (int)b_down.size();
  const int q = (int)b_down.sum();
  Msig.resize(q);

  Eigen::VectorXi occ = Eigen::VectorXi::Zero(n);

  // At step j (1..q), choose argmin_k occ[k]*q - j*b_down[k]
  for (int j = 1; j <= q; ++j) {
    int best = 0;
    long long best_val = static_cast<long long>(occ[0]) * q -
                         j * static_cast<long long>(b_down[0]);

    for (int k = 1; k < n; ++k) {
      const long long val = static_cast<long long>(occ[k]) * q -
                            j * static_cast<long long>(b_down[k]);
      if (val < best_val) {
        best = k;
        best_val = val;
      }
    }

    Msig[j - 1] = best; // store best block for layer
    ++occ[best];
  }
}

// compress regular msig into batched by (cnt, e), cnt <= m[e], e\in
static inline void compress_msig(const Eigen::Ref<const Eigen::VectorXi> &Msig,
                                 int n, std::vector<RunBatch> &runs,
                                 Eigen::VectorXi &max_run) {
  runs.clear();
  max_run = Eigen::VectorXi::Zero(n);
  if (Msig.size() == 0)
    return;

  int cur = Msig[0];
  int cnt = 1;
  for (int i = 1; i < Msig.size(); ++i) {
    if (Msig[i] == cur) {
      ++cnt;
    } else {
      runs.push_back(RunBatch{cnt, cur});
      max_run[cur] = std::max(max_run[cur], cnt);
      cur = Msig[i];
      cnt = 1;
    }
  }
  runs.push_back(RunBatch{cnt, cur});
  max_run[cur] = std::max(max_run[cur], cnt);
}

/*
Two attempts were made to lower cost of batch msig creation by direct
computation of batches (build_msig_batch, build_msig_heap).

While having better theoretical complexity, both revealed to be less performant
currently when running .\build\bin\bench_gupta_sched.exe. So build_msig+compress
is used in production.
*/

// Alg.1 as batch scan.
static inline void build_msig_batch(const Eigen::Ref<const Eigen::VectorXi> &m,
                                    std::vector<RunBatch> &runs,
                                    Eigen::VectorXi &max_run) {
  const int n = (int)m.size();
  long long q = 0;
  runs.clear();
  max_run = Eigen::VectorXi::Zero(n);

  // validate multiplicities and total length q
  for (int e = 0; e < n; ++e) {
    assert(m[e] > 0);
    q += (long long)m[e];
  }
  if (q == 0)
    return;

  // k[e] = emitted copies of e
  std::vector<int> k(n, 0);

  long long produced = 0;
  while (produced < q) {
    // choose argmin_e k[e]*q - j*m[e] at current step j
    const long long j0 = produced + 1;
    int best_e = -1;
    long long best_s = std::numeric_limits<long long>::max();

    for (int e = 0; e < n; ++e) {
      if (k[e] >= m[e])
        continue;
      const long long se = score_ll((long long)k[e], j0, q, (long long)m[e]);
      if (best_e < 0 || se < best_s || (se == best_s && e < best_e)) {
        best_e = e;
        best_s = se;
      }
    }

    assert(best_e >= 0);
    const int e = best_e;
    const long long me = m[e];
    const long long ke = k[e];
    const long long remaining = me - ke;
    assert(remaining >= 1);

    // maximal run while e stays optimal against every active block
    long long rlen = remaining;
    for (int f = 0; f < n; ++f) {
      if (f == e || k[f] >= m[f])
        continue;
      const long long rf =
          max_run_vs(ke, me, e, (long long)k[f], (long long)m[f], f, q, j0);
      assert(rf >= 1);
      rlen = std::min(rlen, rf);
      if (rlen == 1)
        break;
    }

    // append or extend run of block e
    const int rr = (int)rlen;
    if (!runs.empty() && runs.back().e == e) {
      runs.back().cnt += rr;
      max_run[e] = std::max(max_run[e], runs.back().cnt);
    } else {
      runs.push_back(RunBatch{rr, e});
      max_run[e] = std::max(max_run[e], rr);
    }

    // advance e
    k[e] = (int)(ke + rlen);
    produced += rlen;
  }
}

// Alg.1 as heap scan
static inline void
build_msig_batch_heap(const Eigen::Ref<const Eigen::VectorXi> &m,
                      std::vector<RunBatch> &runs, Eigen::VectorXi &max_run) {
  const int n = (int)m.size();
  long long q = 0;
  runs.clear();
  max_run = Eigen::VectorXi::Zero(n);

  // validate multiplicities and total length q
  for (int e = 0; e < n; ++e) {
    assert(m[e] > 0);
    q += (long long)m[e];
  }
  if (q == 0)
    return;

  std::vector<int> k(n, 0);

  long long produced = 0;
  while (produced < q) {
    // heap over exact scores at current step j
    const long long j0 = produced + 1;
    using Score = std::pair<long long, int>;
    std::priority_queue<Score, std::vector<Score>, std::greater<Score>> pq;
    for (int e = 0; e < n; ++e) {
      if (k[e] >= m[e])
        continue;
      pq.emplace(score_ll((long long)k[e], j0, q, (long long)m[e]), e);
    }

    assert(!pq.empty());
    const int e = pq.top().second;

    const long long me = m[e];
    const long long ke = k[e];
    const long long remaining = me - ke;
    assert(remaining >= 1);

    // maximal run while e stays optimal against every active block
    long long rlen = remaining;
    for (int f = 0; f < n; ++f) {
      if (f == e || k[f] >= m[f])
        continue;
      const long long rf =
          max_run_vs(ke, me, e, (long long)k[f], (long long)m[f], f, q, j0);
      assert(rf >= 1);
      rlen = std::min(rlen, rf);
      if (rlen == 1)
        break;
    }

    // append or extend run of block e
    const int rr = (int)rlen;
    if (!runs.empty() && runs.back().e == e) {
      runs.back().cnt += rr;
      max_run[e] = std::max(max_run[e], runs.back().cnt);
    } else {
      runs.push_back(RunBatch{rr, e});
      max_run[e] = std::max(max_run[e], rr);
    }

    k[e] = (int)(ke + rlen);
    produced += rlen;
  }

  assert(produced == q);
}

} // namespace oracle::gupta_core
