#pragma once
#include "Types.h"
#define SLACK_CAP 0 // 0 => structural; 1 => \widetilde\calN_1; 2 => \widetilde\calN_2

EIGEN_STRONG_INLINE int get_num_slack(ProblemInstance &instance,
                                                 int a, int idxA, int l) {
#if SLACK_CAP == 2
  // \widetilde\calN_2 = n_{j*} + ceil(\ell / a) + \sum_{j\ne j*} ceil(n_j / a)
  int sum_ceils = 0;
  for (int j = 0; j < (int)instance.jobs.size(); ++j) {
    if (j == idxA) continue;
    sum_ceils += ceil_div_int(instance.jobs[j].n, a);
  }
  return instance.jobs[idxA].n + ceil_div_int(ell, a) + sum_ceils;
#else

  // number of big machines in total
  int M_B = 0;
  for (int k = instance.num_small_machines; k < instance.machines.size(); ++k) {
    M_B += instance.machines[k].n;
  }

  int D_idxA = 0;    // D_{j*}
  int sum_ceils = 0; // \sum_{j\ne idxA} \ceil{D_j / a}

  for (int j = 0; j < instance.jobs.size(); ++j) {
    int Hj = 0;
    // H_small_j
    for (int k = 0; k < instance.num_small_machines; ++k) {
      Hj += instance.machines[k].n *
            (instance.machines[k].t /
             instance.jobs[j].p); // integer division floor
    }

    // H_big_j
    Hj += (j == idxA) ? 0 : (a - 1) * M_B;

    int Dj = max(0, instance.jobs[j].n - Hj);

    if (j == idxA) {
      D_idxA = Dj;
    } else {
      sum_ceils += (Dj + a - 1) / a; // ceil(Dj / a)
    }
  }

  const int dummy_bundles = (l + a - 1) / a;
#if SLACK_CAP == 1
  // \widetilde\calN_1 = D_{j*} + ceil(\ell / a) + \sum_{j \ne j*} ceil(D_j / a)
  return D_idxA + dummy_bundles + sum_ceils;
#else
  // structurally tight bound
  return D_idxA + max(dummy_bundles, sum_ceils);
#endif
#endif
}