#include "Eigen/Dense"
#include <gtest/gtest.h>
#include <random>
#include "AlignedSet.h"
#include "Dedup.h"

using Eigen::VectorXi;

// ---- helpers ----
static VectorXi V(std::initializer_list<int> xs) {
  VectorXi v((int)xs.size());
  int i = 0;
  for (int x : xs)
    v[i++] = x;
  return v;
}

// ---------- tests ----------
TEST(DedupInplace, EmptyInputNoCrash) {
  std::vector<Eigen::VectorXi> a;
  DedupScratch s;
  dedupe_inplace(a, 0, s);
  EXPECT_TRUE(a.empty());
}

TEST(DedupInplace, AllUniqueKept) {
  std::vector<Eigen::VectorXi> a = { V({1,2}), V({1,3}), V({2,2}) };
  DedupScratch s;
  dedupe_inplace(a, 2, s);
  EXPECT_EQ(a.size(), 3u);
}

TEST(DedupInplace, AllDuplicatesReducedToOne) {
  std::vector<Eigen::VectorXi> a = { V({1,2,3}), V({1,2,3}), V({1,2,3}) };
  DedupScratch s;
  dedupe_inplace(a, 3, s);
  EXPECT_EQ(a.size(), 1u);
}

TEST(DedupInplace, MixedDuplicatesAndUniques) {
  std::vector<Eigen::VectorXi> a = { V({0,0}), V({1,0}), V({0,0}), V({1,1}), V({1,0}), V({2,2}) };
  DedupScratch s;
  dedupe_inplace(a, 2, s);
  EXPECT_EQ(a.size(), 4u);
}

TEST(DedupInplace, ReuseScratchAcrossCalls) {
  DedupScratch s;
  std::vector<Eigen::VectorXi> a1 = { V({1}), V({1}), V({2}) };
  std::vector<Eigen::VectorXi> a2 = { V({2}), V({3}), V({3}) };
  dedupe_inplace(a1, 1, s);
  dedupe_inplace(a2, 1, s);
  EXPECT_EQ(a1.size(), 2u);
  EXPECT_EQ(a2.size(), 2u);
}

// TEST(DedupInplace, LargeRandomWithPlantedDups) {
//   // ... build `a` ...
//   DedupScratch s;
//   dedupe_inplace(a, /*r=*/4, s);
//   EXPECT_LE(a.size(), 2000u);
// }