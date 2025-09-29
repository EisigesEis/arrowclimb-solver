#pragma once
#include <Eigen/Dense>
#include <execution>
#include <unordered_set>
// #include "absl/container/flat_hash_set.h"

struct VecHash {
  size_t operator()(const Eigen::VectorXi& v) const noexcept {
    uint64_t h = 1469598103934665603ull;
    const int n = v.size();
    for (int i = 0; i < n; ++i) {
      uint64_t x = static_cast<uint64_t>(static_cast<int64_t>(v(i)));
      h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    }
    return static_cast<size_t>(h);
  }
};
struct VecEq {
  bool operator()(const Eigen::VectorXi& a, const Eigen::VectorXi& b) const noexcept {
    return (a - b).squaredNorm() == 0;
  }
};
struct LexLess {
  bool operator()(const Eigen::VectorXi& a, const Eigen::VectorXi& b) const noexcept {
    const int n = a.size();
    for (int i = 0; i < n; ++i) {
      const int ai = a(i), bi = b(i);
      if (ai < bi) return true;
      if (ai > bi) return false;
    }
    return false;
  }
};

using AlignedSet = std::unordered_set<
    Eigen::VectorXi, VecHash, VecEq,
    Eigen::aligned_allocator<Eigen::VectorXi>>;
// using AlignedSet = absl::flat_hash_set<Eigen::VectorXi, VecHash, VecEq>;