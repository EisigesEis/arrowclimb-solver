#include "bfs.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>

#include <spdlog/spdlog.h>

#include "AlignedSet.h"
#include "Config.h"

std::ostream& operator<<(std::ostream& os, const std::vector<int>& vec) {
  os << "[";
  for (std::size_t i = 0; i < vec.size(); ++i) {
    if (i != 0) os << ", ";
    os << vec[i];
  }
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Config& c) {
  return os << "Config {"
            << "cost = " << c.cost << ", "
            << "vec = {" << c.vec.transpose() << "} "
            << "}";
}

std::ostream& operator<<(std::ostream& os, const BFSState& s) {
  return os << "State {"
            << "conf = " << s.conf << ", "
            << "next_index = " << s.next_index << ", "
            << "feasible_S = " << s.feasible_S << ", "
            << "feasible_B = " << s.feasible_B << "}";
}

void bfsGenerateCandidatesAll(ProblemInstance& instance,
                              int a,
                              VectorXi& /*mod_frequency*/,
                              std::vector<Config>& candidates_S,
                              std::vector<Config>& candidates_B,
                              const bool external_init_feas_S,
                              const bool external_init_feas_B) {
  std::queue<BFSState> q;

  const int config_length = static_cast<int>(instance.jobs.size());

  // Initial state
  BFSState init;
  init.conf.vec = VectorXi::Zero(config_length);
  init.conf.cost = 0;
  init.next_index = 0;

#ifdef BFS_TRUST_LB
  const int small_machine_load_lb = max(0, (int)instance.avg_makespan - instance.p_max); // floor by int cast
#else
  const int small_machine_load_lb = 0;
#endif
  int small_machine_load_ub = 0;

  init.feasible_S = (external_init_feas_S && (instance.num_small_machines != 0));
  if (init.feasible_S) {
    const int last_small_T = instance.machines[instance.num_small_machines - 1].t;
    const int avg_plus_pmax = instance.avg_makespan + instance.p_max;
    small_machine_load_ub = std::min(last_small_T, avg_plus_pmax);
#ifdef DBG_BFS
    spdlog::debug("small_machine_load_lb = {}, small_machine_load_ub = {}",
                  small_machine_load_lb, small_machine_load_ub);
#endif
  }

  // Big machines feasible only if some exist
  init.feasible_B = external_init_feas_B &&
                    (instance.num_small_machines != static_cast<int>(instance.machines.size()));

  q.push(init);

  while (!q.empty()) {
    BFSState s = q.front();
    q.pop();

    if (s.next_index == config_length) {
      // Full vector formed: decide where it can go
#ifdef DBG_BFS
      std::cout << s.next_index << ". determining where to put " << s << '\n';
#endif
      if (s.feasible_S && s.conf.cost >= small_machine_load_lb) {
        candidates_S.push_back(s.conf);
#ifdef DBG_BFS
        std::cout << "   ===> put on small machine (" << s.conf.cost
                  << " >= " << small_machine_load_lb << ")\n";
#endif
      }
#ifdef DBG_BFS
      else {
        std::cout << "   ===> not on small machine (feas_S=" << s.feasible_S
                  << ", cost=" << s.conf.cost
                  << ", lb=" << small_machine_load_lb << ")\n";
      }
#endif

      if (s.feasible_B) {
        candidates_B.push_back(s.conf);
#ifdef DBG_BFS
        std::cout << "   ===> put on big machine\n";
#endif
      }
#ifdef DBG_BFS
      else {
        if (instance.num_small_machines != static_cast<int>(instance.machines.size())) {
          std::cout << "   ===> not on big machine (feas_B=false)\n";
        } else {
          std::cout << "   ===> not on big machine: no big machines exist\n";
        }
      }
      std::cout << '\n';
#endif
      continue;
    }

    // Extend partial vector
    BFSState ns;
    ns.conf.cost = s.conf.cost;
    ns.conf.vec = s.conf.vec;
    ns.next_index = s.next_index + 1;

    const int j = s.next_index;
    const int cost_next_index = instance.jobs[j].p;
    const int n_cap_job = instance.jobs[j].n;

    const int cap_small = std::min<int>(instance.p_max4, n_cap_job);

    // Pivot index
    const bool is_pivot_pos =
        // (instance.jobs[j].p == a) ||
        (j == instance.jobs_nsort[0]);

    if (s.feasible_S) {
      if (is_pivot_pos) {
        ns.feasible_B = s.feasible_B && ns.check_feasible_B(j, a);
        ns.feasible_S = ns.check_feasible_S(small_machine_load_ub);
        if (ns.feasible_S || ns.feasible_B) {
          q.push(ns);

          // Branching tells us to only enumerate S combinations, not B
          ns.feasible_B = false;
          for (; ns.conf.vec[j] <= cap_small; ++ns.conf.vec[j]) {
            if (!(ns.feasible_S = ns.check_feasible_S(small_machine_load_ub))) {
              break;
            }
            q.push(ns);
#ifdef DBG_BFS
            std::cout << ns.conf.vec[j] << ". found " << ns << '\n';
#endif
            ns.conf.cost += cost_next_index;
          }
        }
      } else {
        for (; ns.conf.vec[j] <= cap_small; ++ns.conf.vec[j]) {
          ns.feasible_B = s.feasible_B && ns.check_feasible_B(j, a);
          ns.feasible_S = ns.check_feasible_S(small_machine_load_ub);
          if (ns.feasible_S || ns.feasible_B) {
            q.push(ns);
#ifdef DBG_BFS
            std::cout << ns.conf.vec[j] << ". found " << ns << '\n';
#endif
          } else {
            break;
          }
          ns.conf.cost += cost_next_index;
        }
      }
      continue;
    }

    // If S is not feasible but B is, we only branch for big machines
    if (s.feasible_B) {
      ns.feasible_B = true;
      ns.feasible_S = false;

      if (!is_pivot_pos) {
        // Keep original big-machine cap logic: counts up to a-1 (and does not use n_j bound)
        const int cap_big = std::min<int>(a - 1, n_cap_job * 1);
        for (; ns.conf.vec[j] <= cap_big; ++ns.conf.vec[j]) {
          q.push(ns);
#ifdef DBG_BFS
          std::cout << ns.conf.vec[j] << ". found " << ns << '\n';
#endif
          ns.conf.cost += cost_next_index;
        }
      } else {
        q.push(ns);
      }
    }
  }
}
