#pragma once
#include "AlignedSet.h"

struct DedupScratch {
  AlignedSet set;     // persistent buckets
  size_t buckets = 0; // last rehash bucket count
  // ensure load factor ~ 0.7; grow/shrink when far off
  void ensure(size_t expect) {
    const size_t min_bkt = 16;
    if (buckets == 0) {
      buckets = std::max(min_bkt, next_pow2((size_t)std::ceil(expect / 0.7)));
      set.rehash(buckets);
      return;
    }
    const double lf = expect * 1.0 / (buckets ? buckets : 1.0);
    if (lf > 0.75 || lf < 0.10) {
      buckets = std::max(min_bkt, next_pow2((size_t)std::ceil(expect / 0.7)));
      set.rehash(buckets);
    }
  }
  static size_t next_pow2(size_t x) {
    if (x <= 1)
      return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
    x |= x >> 32;
#endif
    return x + 1;
  }
};

// Heuristic: for very large arrays (all size==r), sort+unique can beat hashing
constexpr size_t DEDUP_SORT_THRESHOLD = 20000;

EIGEN_STRONG_INLINE
static void dedupe_inplace(std::vector<Eigen::VectorXi> &arr,
                    int r,
                    DedupScratch &ctx) {
  if (arr.empty())
    return;

  // assume same sizes r > 0, otherwise unsafe

  if (arr.size() >= DEDUP_SORT_THRESHOLD) {
    // sort + unique path
    std::sort(std::execution::par_unseq, arr.begin(), arr.end(), LexLess{});
    auto it = std::unique(std::execution::par_unseq, arr.begin(), arr.end(),
                          [](const Eigen::VectorXi &a, const Eigen::VectorXi &b) {
                            return (a - b).squaredNorm() == 0;
                          });
    arr.erase(it, arr.end());
    return;
  }


  ctx.set.clear();

  ctx.ensure(arr.size());

  std::vector<Eigen::VectorXi> out;
  out.reserve(arr.size());

  for (auto &v : arr) {
    auto ins = ctx.set.insert(v);
    if (ins.second)
      out.push_back(std::move(v));
  }

  arr.swap(out);
}

EIGEN_STRONG_INLINE
void dedupe_inplace(std::vector<Eigen::VectorXi>& arr, int r) {
  static thread_local DedupScratch tls;
  dedupe_inplace(arr, r, tls);
}