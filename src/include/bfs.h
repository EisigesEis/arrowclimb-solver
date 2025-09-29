#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <queue>
#include <sstream>
#include <vector>

#include "Types.h"
#include "Util.h"
#include <execution>

using namespace std;
using namespace Eigen;

// A config for a machine
struct Config {
  VectorXi vec; // how many jobs of each type
  size_t cost;  // total cost of the config: \sum_{i \in M} vec_i * s_i

  static bool comparator(const Config& a, const Config& b) {
    if (a.cost != b.cost) return a.cost < b.cost;
    const int n = a.vec.size(), m = b.vec.size();
    const int k = std::min(n, m);
    for (int i = 0; i < k; ++i) {
      int ai = a.vec.coeff(i), bi = b.vec.coeff(i);
      if (ai != bi) return ai < bi;
    }
    return n < m;
  }

  static bool eq(const Config& a, const Config& b) {
    return a.cost == b.cost
        // && a.vec.size() == b.vec.size()
        && (a.vec.array() == b.vec.array()).all();
  }
};

// A state in the BFS
struct BFSState {
  Config conf;
  int next_index;  // next field index to assign
  bool feasible_S; // partial config is still feasible for small machine
  bool feasible_B; // partial config is still feasible for big machine

  // checks cost condition for small machine candidate config
  inline bool check_feasible_S(int t_i_max) {
    return this->conf.cost <= t_i_max;
  }

  // checks conditions for big machine candidate config
  inline bool check_feasible_B(int i, int a) { return this->conf.vec[i] < a; }
};

// Overload stream operator for std::vector<int>
std::ostream &operator<<(std::ostream &os, const std::vector<int> &vec);

// Overload the stream operator for Config
std::ostream &operator<<(std::ostream &os, const Config &c);

// Overload the stream operator for State
std::ostream &operator<<(std::ostream &os, const BFSState &s);

void bfsGenerateCandidatesAll(ProblemInstance &instance, int a,
                           VectorXi &mod_frequency,
                           vector<Config> &candidates_S,
                           vector<Config> &candidates_B,
                           const bool external_init_feas_S,
                           const bool external_init_feas_B);

#ifndef __CUDACC__
/**
 * BFS search for all possible configurations of small and big machines
 *
 * We want to get all candidate configs with their
 *
 * \param instance problem instance
 * \param a current pivot element
 * \param mod_frequency number of machines with that modulo remainder, that is:
 * mod_frequency[i] = |\{ j\in B | machines[j].t % a = i \}|
 * \param candidates_S all candidate configs feasible for biggest small machine
 * \param candidates_B all candidate configs feasible for big machines
 * \param external_regenerate_S if candidates_S requires regeneration
 * \param external_regenerate_B if candidate_B requires regeneration
 */
constexpr void bfsGenerateCandidates(ProblemInstance &instance, int a,
                                     VectorXi &mod_frequency,
                                     vector<Config> &candidates_S,
                                     vector<Config> &candidates_B,
                                     const bool external_regenerate_S,
                                     const bool external_regenerate_B) {
  if (external_regenerate_S)
    candidates_S.clear();
  if (external_regenerate_B)
    candidates_B.clear();
    
  bfsGenerateCandidatesAll(instance, a, mod_frequency, candidates_S, candidates_B, external_regenerate_S, external_regenerate_B);

  if (external_regenerate_S) {
    sort(std::execution::par_unseq, candidates_S.begin(), candidates_S.end(),
      Config::comparator);
    auto it = unique(std::execution::par_unseq, candidates_S.begin(), candidates_S.end(), Config::eq);
    candidates_S.erase(it, candidates_S.end());
  }

  if (external_regenerate_B) {
    sort(std::execution::par_unseq, candidates_B.begin(), candidates_B.end(),
      Config::comparator);
    auto it = unique(std::execution::par_unseq, candidates_B.begin(), candidates_B.end(), Config::eq);
    candidates_B.erase(it, candidates_B.end());
  }
};
#endif