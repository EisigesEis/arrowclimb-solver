#pragma once
#include <cassert>
#include <string>
#include <vector>
#include <Eigen/Dense>
using namespace std;
using Eigen::MatrixXi;

typedef int uint; // at this point let's just ctrl+r to int...

struct Machine {
  uint n; // number of machine type
  uint s; // machine processing power
  uint t; // guessed machine load
  Machine() {}
  Machine(uint n, uint s) : n(n), s(s) {}
  Machine(uint n, uint s, uint t) : n(n), s(s), t(t) {}

  static bool comparator_s(const Machine &a, const Machine &b) {
    return a.s < b.s;
  }
};

struct Job {
  uint n; // number of job type
  uint p; // job cost
  Job() {}
  Job(uint n, uint p) : n(n), p(p) {}

  static bool comparator_p(const Job &a, const Job &b) {
    return a.p < b.p;
  }

  static bool comparator_n_index(const size_t i1, const size_t i2, const std::vector<Job> &jobs) {
    return jobs[i1].n > jobs[i2].n;
  }
};

struct ProblemInstance {
  uint num_jobs;
  std::vector<Job> jobs;
  std::vector<int> jobs_nsort; // job indices sorted by j.n (descending)
  uint num_machines;
  std::vector<Machine> machines;

  uint num_small_machines;

  double avg_makespan;
  long scaled_lb; // lower bound for makespan
  long scaled_ub; // upper bound for makespan

  int delta; // maximum coefficient in config matrix A

  long s_lcm;

  long p_sum;
  uint p_max;
  uint p_max2;
  uint p_max4;

  // upper bound on slack blocks
  // uint N_base = 0;
  // constexpr void init_N_safe(const int a) {
  //   for (int i = 0; i < this->jobs.size(); ++i) {
  //     if (this->jobs_nsort[0] != i) {
  //       this->N_base += this->jobs[i].n * 1.0 / a;
  //     }
  //   }
  // }
  // constexpr int N_safe(const int a, const int p_max_2_B) {
  //   return this->N_base + max(0, this->jobs[this->jobs_nsort[0]].n - p_max_2_B);
  // }
  // uint N_tight = 0;
  // constexpr void update_N_tight(const int a) {
  //   for (int i = 0; i < this->num_small_machines; ++i) {
  //     // const int u_small = floor(this->machines[i].)
  //   }
  // }
};