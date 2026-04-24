#pragma once
#include "compact_types.h"
#include <Eigen/Dense>
#include <cassert>
#include <execution>

using Scal = int;

struct GenResidueCounter {
  Eigen::VectorXi cnt;
  Eigen::VectorXi gen;
  std::vector<Scal> touched;
  int cur_gen = 0;

  inline void init(Scal a) {
    cnt.setZero(a);
    gen.setZero(a);
    touched.clear();
    cur_gen = 0;
  }

  inline void add(Scal r, Scal v) {
    assert(r < gen.size());
    if (gen[r] != cur_gen) {
      gen[r] = cur_gen;
      cnt[r] = v;
      touched.push_back(r);
    } else {
      cnt[r] += v;
    }
  }

  inline void init_next_iter() {
    ++cur_gen;
    if (cur_gen == 0) { // wrap around on overflow
      cur_gen = 1;
      gen.setZero();
    }
    touched.clear();
  }

  inline void finalize() {
    std::sort(std::execution::unseq, touched.begin(), touched.end(),
              std::less<Scal>());
  }

  [[nodiscard]] constexpr Scal get_for_block(size_t i) const { return touched[i]; }

  [[nodiscard]] constexpr Scal get_cnt_for_block(size_t i) const { return cnt[touched[i]]; }
  [[nodiscard]] constexpr Scal get_cnt_for_residue(Scal r) const { return cnt[r]; }

  [[nodiscard]] constexpr size_t size() const { return touched.size(); }
};

struct ProblemInstance {
  // filename without extension
  std::string name;

  // machine type counts
  Eigen::VectorXi m;
  // machine type speeds
  Eigen::VectorXi s;
  // guessed makespan distributed to each machine type
  Eigen::VectorXi t;
  // job type counts
  Eigen::VectorXi n;
  // job type costs
  Eigen::VectorXi p;
  // required dummy job count for rhs dummy row
  Scal ell = 0;

  // denominator for distributing guessed t and inherent infeasible checks
  Eigen::VectorXi cover_den;
  // minimal k_min approx where ell >= 0 ("non-inherently infeasible")
  long long k_gate = 0;

  bool machines_shape_ok() const noexcept {
    return (m.size() == s.size()) && (t.size() == s.size());
  }
  bool jobs_shape_ok() const noexcept { return n.size() == p.size(); }

  // number of machine types
  std::size_t M = 0;
  // number of big machine types
  std::size_t M_B = 0;
  // number of job types
  std::size_t N = 0;

  // n.dot(p)
  long long total_load = 0;

  // big machine types grouped by residue t \mod a
  GenResidueCounter big_residue;

  // ub and lb for binary search C* \in [scaled_lb / s_lcm, scaled_ub / s_lcm]
  long long scaled_lb = 0, scaled_ub = 0;
  // m.dot(s) / n.dot(p)
  double avg_makespan = 0.0;

  Scal p_min = 0, p_max = 0, p_max2 = 0, p_max4 = 0;
  Scal a = 0;
  size_t idx_a = 0, num_small_machines = 0;

  Scal s_min = 0, s_max = 0, s_lcm = 0;

  size_t delta = 0;

  void clear_raw() {
    name.clear();
    m.setZero();
    s.setZero();
    t.setZero();
    n.setZero();
    p.setZero();
    clear_derived();
  }
  void clear_derived() {
    total_load = 0;
    p_min = p_max = p_max2 = 0;
    s_min = s_max = 0;
  }

  void sort() {
    sort_by_s();
    sort_by_p();
  }

private:
  /*
    apply index reordering idx to vector a
  */
  inline void reorder(Eigen::VectorXi &a, const std::vector<size_t> &idx,
                      Eigen::VectorXi &tmp) {
    for (size_t i = 0; i < a.size(); ++i)
      tmp[i] = a[idx[i]];
    a.swap(tmp);
  }

  /*
    sort m and s by s ASC
  */
  inline void sort_by_s() {
    std::vector<size_t> idx(M);
    std::iota(idx.begin(), idx.end(), 0);

    auto cmp = [&](size_t i, size_t j) {
      if (s[i] != s[j])
        return s[i] < s[j];
      if (m[i] != m[j])
        return (m[i] < m[j]);
      return i < j;
    };
    std::sort(std::execution::par_unseq, idx.begin(), idx.end(), cmp);

    Eigen::VectorXi tmp(M);
    reorder(m, idx, tmp);
    reorder(s, idx, tmp);
  }

  /*
    sort n and p by p ASC
  */
  inline void sort_by_p() {
    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    auto cmp = [&](size_t i, size_t j) {
      if (p[i] != p[j])
        return p[i] < p[j];
      if (n[i] != n[j])
        return n[i] < n[j];
      return i < j;
    };
    std::sort(std::execution::par_unseq, idx.begin(), idx.end(), cmp);

    Eigen::VectorXi tmp(N);
    reorder(n, idx, tmp);
    reorder(p, idx, tmp);
  }
};
