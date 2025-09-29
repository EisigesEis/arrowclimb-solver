#include "matrix.h"
#include "BlockedMatrix.h"
#include <spdlog/spdlog.h>

#include "Config.h"
#include "slack_bound.h"

bool construct_ilp(ProblemInstance &instance, int a, int l,
                   VectorXi mod_frequency, vector<Config> &candidates_S,
                   vector<Config> &candidates_B, VectorXi &b,
                   BlockedMatrix &BM) {
  // fill matrix blocks with the configurations
  std::vector<Matrix<int, Eigen::Dynamic, Eigen::Dynamic>> blocks;
  const uint block_height = instance.jobs.size() + 1;

#ifdef MAT_DONT_TRUST_BFS
  for (size_t u = 1; u < candidates_S.size(); ++u)
    if (candidates_S[u-1].cost > candidates_S[u].cost) {
      spdlog::error("candidates_S is not sorted: {} costs more than {} but is put before it in candidates",
                        candidates_S[u-1].cost, candidates_S[u].cost);
      exit(1);
    };
#endif

  // fill small machine blocks
  uint small_machines_width = 0;
  if (instance.num_small_machines != 0) {
    for (int j = 0; j < instance.num_small_machines; ++j) {
      const Machine &m = instance.machines[j];
      const auto it_end = std::upper_bound(
          candidates_S.begin(), candidates_S.end(), m.t,
          [](int key, const Config &c) { return key < c.cost; });
      const auto block_size = std::distance(candidates_S.begin(), it_end);
#ifdef DBG_MAT
      spdlog::debug("machine j={}: block_size={}.", j, block_size);
      // spdlog::debug("machine j={}: block_size={}. Found first cost={} which
      // doesn't fit for m[j].t={}. Last inserted cost={}", j, block_size,
      // (*it_end).cost, instance.machines[j].t);
#endif

      if (block_size == 0)
        continue;
      // ++final_height;

      MatrixXi new_block = MatrixXi::Zero(block_height, block_size);

      // TODO Speedup:
      // - Check if this re-use is beneficial
      // - Use submat for these recurring matrix parts

      int c = 0;
      for (auto it = candidates_S.begin(); it < it_end; ++it, ++c) {
        const int y = instance.machines[j].t - (*it).cost;
#ifdef MAT_DONT_TRUST_BFS
        if (y < 0) {
          spdlog::error("Negative slack in small block: T={} cost={} y={}",
                        instance.machines[j].t, candidates_S[c].cost, y);
          continue;
        }
#endif
        new_block.col(c).head(instance.jobs.size()) = (*it).vec;
        new_block(new_block.rows() - 1, c) = y;
      }

      blocks.push_back(new_block);

      small_machines_width += new_block.cols();
    }
  }

  // fill big machine blocks
  uint big_machines_width = 0;
  if (instance.num_small_machines != (uint)instance.machines.size()) {
    for (int j = 0; j < a; ++j) {
      if (mod_frequency[j] > 0) {
        big_machines_width += candidates_B.size();
        const auto &block_size = candidates_B.size();

        if (block_size == 0)
          continue;

        Eigen::MatrixXi new_block = MatrixXi::Zero(block_height, block_size);

        // // count of cols which cannot be covered by dummy jobs
        // int rejected_confs = 0;
        for (int c = 0; c < candidates_B.size(); ++c) {
#ifdef DBG_MAT_DETAIL
          cout << candidates_B[c].vec.transpose() << endl;
          cout << "block.rows()=" << new_block.rows()
               << " block.cols()=" << new_block.cols()
               << " vec.rows()=" << candidates_B[c].vec.rows()
               << " vec.cols()=" << candidates_B[c].vec.cols() << endl;
#endif
          new_block.col(c).head(instance.jobs.size()) = candidates_B[c].vec;
          const int x = abs((int)(j - candidates_B[c].cost));
          const int y = x % a;
          // rejected_confs += (y > l);
          new_block(new_block.rows() - 1, c) = y;
          // assert((candidates_B[c].cost + y) % a == j);

#ifdef DBG_MAT_DETAIL
          cout << "assembling using " << candidates_B[c].vec.transpose()
               << endl;
#endif
        }
        // if (rejected_confs == candidates_B.size())
        //   return false; // all confs for this big machine have been rejected,
        //                 // ILP infeasible

        blocks.push_back(new_block);
      }
    }

    // fill slack block
    big_machines_width += instance.jobs.size() + 2;
    MatrixXi new_block = MatrixXi::Zero(
        instance.jobs.size() + 1, instance.jobs.size() + 2); // one zero column
    for (int i = 0; i < instance.jobs.size() + 1; ++i) {
      new_block(i, i) = i == instance.jobs_nsort[0] ? 1 : a;
    }
    blocks.push_back(new_block);
  }

  // fill actual whole matrix with the blocks
#ifdef DBG_MAT
  cout << "big_machines_width=" << big_machines_width
       << " small_machines_width=" << small_machines_width << endl;
#endif
  BM.mat_.resize(block_height, big_machines_width + small_machines_width);
  int c = 0;
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto &cur_block = blocks[i];
#ifdef DBG_MAT
    cout << "Placing block i=" << i << " at starting c=" << c << " of size ("
         << cur_block.rows() << ", " << cur_block.cols() << ")" << endl;
    cout << cur_block << endl;
#endif
    BM.add_block(cur_block);

    // Row of 1s is now taken care of inside gurobi api
    // cout << "Placing row of 1s at c=" << block_height + i << " high 1 and r="
    // << c << " wide " << b.cols() << endl; A.block(block_height + i, c, 1,
    // b.cols()).setOnes(); cout << A << endl;

    // cout << "Placed row of 1s at y=" << block_height + i << " high 1 and r="
    // << c << " wide " << b.cols() << endl;

    c += cur_block.cols();
  }
#ifdef DBG_MAT_BRIEF
  cout << "final matrix:\n" << BM.mat_ << endl;
#endif
#ifdef DBG_RHS
  cout << endl << "=== Fill RHS ===" << endl;
#endif

  if (blocks.size() == 0) {
    cerr << "Scheduling without a valid config for one of the machines. This "
            "should not be possible."
         << endl;
    exit(1);
  }

  b = VectorXi::Zero(block_height + blocks.size());
  for (size_t i = 0; i < instance.jobs.size(); ++i) {
    b[i] = instance.jobs[i].n; // number of each regular job
  }
#ifdef RHS_TRUST_A_REMOVAL
  b[instance.jobs_nsort[0]] = max(0, (int)(b[instance.jobs_nsort[0]] - instance.p_max2 * (instance.jobs.size() - instance.num_small_machines)));
#endif

#ifdef DBG_RHS
  cout << "dummy jobs" << endl;
#endif
  b[instance.jobs.size()] = l; // number of dummy jobs
#ifdef DBG_RHS
  cout << b.transpose() << endl;
  cout << "small machines" << endl;
#endif
  for (int i = 0; i < instance.num_small_machines; ++i) {
    b[instance.jobs.size() + 1 + i] =
        instance.machines[i].n; // number of small machines
  }
#ifdef DBG_RHS
  cout << b.transpose() << endl;
  cout << "big machines" << endl;
#endif
  if (instance.num_small_machines != instance.machines.size()) {
    // add number of big machines with congruency i = m.t mod a
    for (int i = 0, j = block_height + instance.num_small_machines;
         i < a && j < b.size(); ++i) {
      if (mod_frequency[i] > 0) {
#ifdef DBG_RHS
        cout << i << " has high enough modulo remainder frequency" << endl;
#endif
        b[j++] = mod_frequency[i];
#ifdef DBG_RHS
        cout << b.transpose() << endl;
#endif
      }
    }

    // add slack block frequency in worst case
    // TODO Optimization: Instead of bulk operation, make structural arguments
    // using how configurations are generated. i.e. small machines take up to
    // max(n_i, floor(T_k / p_i) * m_k) for job type i big machines take up to
    // max(n_i, a) * m_k for job type i where p_i != a big machines take 0 for
    // job type i where p_i = a and include additional criteria which are also
    // used in the bfs
    //     int slack_needed_typewise = 0;
    //     VectorXi hostable = VectorXi::Zero(instance.jobs.size());
    //     for (int i = 0; i < BM.num_blocks() - 1; ++i) {
    //       const MatrixXi &Ak = BM.get_block(i);
    //       const VectorXi &max_per_row = Ak.rowwise().maxCoeff();
    //       for (int j = 0; j < instance.jobs.size(); ++j) {
    //         const int assignable = b(instance.jobs.size() + 1 + i);
    //         const int &max_c = max_per_row(j);
    //         hostable[j] += assignable * max_c;
    // #ifdef DBG_SLACK_N
    //         spdlog::info("job type j={} can be put assignable={} * max_c={} =
    //         {} "
    //                      "times on machine {} ",
    //                      j, assignable, max_c, assignable * max_c, i);
    // #endif
    //       }
    //     }
    //     for (int j = 0; j < instance.jobs.size(); ++j) {
    //       const int deficit = max(0, b[j] - hostable[j]);
    //       slack_needed_typewise +=
    //           ceil(deficit * 1.0 / (j == instance.jobs_nsort[0] ? 1 : a));
    // #ifdef DBG_SLACK_N
    //       spdlog::info("job j={} has deficit={} and needs slack={}", j,
    //       deficit,
    //                    ceil(deficit / (j == instance.jobs_nsort[0] ? 1 :
    //                    a)));
    // #endif
    //     }
    //     const int &slack_needed_dummy = static_cast<int>(ceil(l * 1.0 / a));
    // #ifdef DBG_SLACK_N
    //     spdlog::info("we need slack_needed_typewise={} and
    //     slack_needed_dummy={}, "
    //                  "in total calN={}",
    //                  slack_needed_typewise, slack_needed_dummy,
    //                  max(0, slack_needed_typewise + slack_needed_dummy));
    // #endif
    //     b[b.size() - 1] = max(0, slack_needed_typewise + slack_needed_dummy);
    b[b.size() - 1] = get_num_slack(instance, a, instance.jobs_nsort[0], l);
  }

  return true;
}