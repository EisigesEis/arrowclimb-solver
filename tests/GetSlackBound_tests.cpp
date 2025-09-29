#include "Types.h"
#include "slack_bound.h"
#include <gtest/gtest.h>

TEST(SlackBound, ZeroResidualsGivesZero) {
  ProblemInstance I;
  // jobs: j* has p=a=3
  I.jobs.push_back(Job{2, 3}); // j*=0
  I.jobs.push_back(Job{0, 2});
  // one small machine, T=6 (t=6), multiplicity n=1
  I.machines.push_back(Machine{1, 0}); I.machines.back().t = 6;
  I.num_small_machines = 1;

  const int a = 3, idxA = 0, ell = 0;
  EXPECT_EQ(get_num_slack(I, a, idxA, ell), 0);
}

TEST(SlackBound, EllDominatesBundles) {
  ProblemInstance I;
  I.jobs.push_back(Job{0, 3}); // j*=0
  I.jobs.push_back(Job{1, 2});
  // big machines only: 1 type with multiplicity 1 (M_B=1)
  I.machines.push_back(Machine{1, 0}); I.machines.back().t = 100; // big; t irrelevant for big
  I.num_small_machines = 0;

  const int a = 3, idxA = 0, ell = 5; // ceil(5/3)=2
  EXPECT_EQ(get_num_slack(I, a, idxA, ell), 2);
}

TEST(SlackBound, SumOfCeilsDominatesEll) {
  ProblemInstance I;
  I.jobs.push_back(Job{0, 3}); // j*=0
  I.jobs.push_back(Job{5, 2}); // needs ceil(5/3)=2 bundles if no capacity
  // no machines
  I.num_small_machines = 0;

  const int a = 3, idxA = 0, ell = 0;
  EXPECT_EQ(get_num_slack(I, a, idxA, ell), 2);
}

TEST(SlackBound, SinglesForPivotTypeCountSeparately) {
  ProblemInstance I;
  I.jobs.push_back(Job{4, 3}); // j*=0
  I.jobs.push_back(Job{0, 5});
  // one small machine: T=3 -> floor(3/3)=1
  I.machines.push_back(Machine{1, 0}); I.machines.back().t = 3;
  I.num_small_machines = 1;

  const int a = 3, idxA = 0, ell = 0;
  // D_{j*} = 4 - 1 = 3; no bundles needed
  EXPECT_EQ(get_num_slack(I, a, idxA, ell), 3);
}

TEST(SlackBound, UsesFloorForSmallBlocks) {
  ProblemInstance I;
  I.jobs.push_back(Job{2, 3}); // j*=0
  I.jobs.push_back(Job{3, 2});
  // one small machine T=5 -> floor(5/3)=1, floor(5/2)=2
  I.machines.push_back(Machine{1, 0}); I.machines.back().t = 5;
  I.num_small_machines = 1;

  const int a = 3, idxA = 0, ell = 0;
  // D_{j*}=max(0,2-1)=1; D_2=max(0,3-2)=1 -> ceil(1/3)=1
  EXPECT_EQ(get_num_slack(I, a, idxA, ell), 1 + 1);
}

TEST(SlackBound, RoundsEllUp) {
  ProblemInstance I;
  I.jobs.push_back(Job{0, 3}); // j*=0
  I.jobs.push_back(Job{0, 2});
  // no machines
  I.num_small_machines = 0;

  const int a = 3, idxA = 0, ell = 1; // ceil(1/3)=1
  EXPECT_EQ(get_num_slack(I, a, idxA, ell), 1);
}

TEST(SlackBound, BigBlockCapsForNonPivot) {
  ProblemInstance I;
  I.jobs.push_back(Job{0, 3}); // j*=0
  I.jobs.push_back(Job{10, 2});
  // big machines: one type with multiplicity 2 -> M_B=2 -> H_big=(a-1)*M_B=4
  I.machines.push_back(Machine{2, 0}); I.machines.back().t = 100;
  I.num_small_machines = 0;

  const int a = 3, idxA = 0, ell = 0;
  // D_2 = 10 - 4 = 6 -> ceil(6/3)=2
  EXPECT_EQ(get_num_slack(I, a, idxA, ell), 2);
}

TEST(SlackBound, MixedSmallAndBigMachines) {
  ProblemInstance I;
  // a=4, j*=0
  I.jobs.push_back(Job{12, 4}); // j*=0
  I.jobs.push_back(Job{40, 3});
  I.jobs.push_back(Job{7,  5});

  // small machines: T=10 (m=1), T=7 (m=2)
  I.machines.push_back(Machine{1, 0}); I.machines.back().t = 10;
  I.machines.push_back(Machine{2, 0}); I.machines.back().t = 7;
  I.num_small_machines = 2;

  // big machines: one type with multiplicity 3 -> M_B=3
  I.machines.push_back(Machine{3, 0}); I.machines.back().t = 100;

  const int a = 4, idxA = 0, ell = 5; // ceil(5/4)=2
  // From earlier hand calc: calN = 14
  EXPECT_EQ(get_num_slack(I, a, idxA, ell), 14);
}

TEST(SlackBound, AEqualsOneEdgeCase) {
  ProblemInstance I;
  // a=1 ⇒ bundles are unit columns; H_big=(a-1)M_B=0
  I.jobs.push_back(Job{3, 1}); // j*=0
  I.jobs.push_back(Job{5, 2});
  // one small machine T=2 (m=1)
  I.machines.push_back(Machine{1, 0}); I.machines.back().t = 2;
  I.num_small_machines = 1;

  const int a = 1, idxA = 0, ell = 2;
  // floor: j0:2/1=2, j1:2/2=1 -> D0=1, D1=4 -> \sum ceil(D/a)=4; calN=1+max(2,4)=5
  EXPECT_EQ(get_num_slack(I, a, idxA, ell), 5);
}