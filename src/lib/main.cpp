#include <algorithm>
#include <cassert> // explicit include for msvc
#include <cmath>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream> // TODO: Move to custom logger with log levels and output file

#include <numeric>
#include <omp.h>

#include <Eigen/Sparse>

#include "BlockedMatrix.h"
#include "Config.h"
#include "CsvLogger.h"
#include "Types.h"
#include "Util.h"
#include "bfs.h"
#include "dynamic_program.h"
#include "feasibility.h"
#include "matrix.h"
// #include "pooled_solver.h"
#include "temperature.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

inline void print_instance(ProblemInstance &instance) {
  std::ostringstream oss;
  oss << "Machines: " << endl;
  for (Machine &m : instance.machines)
    oss << m.n << ',' << m.s << endl;
  oss << endl;

  oss << "Jobs: " << endl;
  for (Job &j : instance.jobs)
    oss << j.n << ',' << j.p << endl;
  oss << endl;

  oss << "Jobs_nsort: " << endl;
  for (int i : instance.jobs_nsort)
    oss << i << endl;
  oss << endl;

  spdlog::info(oss.str());
}

std::string config_vector_to_string_multiline(const std::vector<Config> &vec) {
  std::ostringstream oss;
  for (const Config &c : vec) {
    oss << "cost: " << c.cost << " vec: " << c.vec.transpose() << endl;
  }
  return oss.str();
}

inline void print_candidates(ProblemInstance &instance,
                             vector<Config> &candidates_S,
                             vector<Config> &candidates_B) {
  // cout << "Candidates (" << candidates_S.size() << ") for Small Machines ("
  //      << instance.num_small_machines << "): " << endl;
  // for (const Config &c : candidates_S) {
  // cout << "cost: " << c.cost << " vec: " << c.vec.transpose() << endl;
  // }
  // cout << endl;
#ifdef LOG_CANDIDATES_DETAILED
  spdlog::debug("Candidates ({}) for Small Machines ({}): \n{}",
                candidates_S.size(), instance.num_small_machines,
                config_vector_to_string_multiline(candidates_S));
#else
  spdlog::debug("Candidates ({}) for Small Machines ({}).", candidates_S.size(),
                instance.num_small_machines);
#endif

  // cout << "Candidates (" << candidates_B.size() << ") for Big Machines ("
  //      << instance.machines.size() - instance.num_small_machines
  //      << "): " << endl;
  // for (const Config &c : candidates_B) {
  // cout << "cost: " << c.cost << " vec: " << c.vec.transpose() << endl;
  // }
  // cout << endl;
#ifdef LOG_CANDIDATES_DETAILED
  spdlog::debug("Candidates ({}) for Big Machines ({}): \n{}",
                candidates_B.size(),
                instance.machines.size() - instance.num_small_machines,
                config_vector_to_string_multiline(candidates_B));
#else
  spdlog::debug("Candidates ({}) for Big Machines ({}).", candidates_B.size(),
                instance.machines.size() - instance.num_small_machines);
#endif
}

// TODO:
// - performance gain from insertionsort at insert?
// - error handling of getline, my_stoi and such
// - getline calls may cost a bit?
void parse_file(ifstream &file, ProblemInstance &instance) {
  string line;
  long s_sum = 0;

  // parse machines
  getline(file, line);
  auto M = my_stoi(line.c_str());
  instance.machines.reserve(M);
  instance.num_machines = 0;
  csv_logger.seti("M", M);
  instance.num_machines = 0;
  for (uint i = 0; i < M; ++i) {
    getline(file, line, ',');
    const uint n = my_stoi(line.c_str());
    getline(file, line);
    const uint s = my_stoi(line.c_str());
    instance.machines.push_back(Machine(n, s));

    s_sum += n * s; // sum up total processing power for calculating Lower Bound
    // update lCM of machine processing power
    if (i != 0) {
      instance.s_lcm = (instance.s_lcm * s) * 1.0 /
                       my_gcd((int)instance.s_lcm,
                              (int)s); // TODO: __gcd() is compiler intrinsic
                                       // and may not be optimal
    } else {
      instance.s_lcm = s;
    }

    instance.num_machines += n;
  }
  csv_logger.seti("machine_count", instance.num_machines);
  csv_logger.seti("s_sum", s_sum);

  // parse jobs
  getline(file, line);
  auto N = my_stoi(line.c_str());
  instance.jobs.reserve(N);
  csv_logger.seti("N", N);
  instance.p_sum = 0;
  instance.num_jobs = 0;
  for (uint i = 0; i < N; ++i) {
    getline(file, line, ',');
    const uint n = my_stoi(line.c_str());
    getline(file, line);
    const uint p = my_stoi(line.c_str());
    instance.jobs.push_back(Job(n, p));

    instance.p_sum +=
        n * p; // sum up total job cost for calculating Upper Bound

    instance.num_jobs += n;
  }
  csv_logger.seti("job_count", instance.num_jobs);
  csv_logger.seti("p_sum", instance.p_sum);

  // all sort happens using introsort for O(n log(n))
  sort(std::execution::par_unseq, instance.machines.begin(),
       instance.machines.end(), Machine::comparator_s);
  sort(std::execution::par_unseq, instance.jobs.begin(), instance.jobs.end(),
       Job::comparator_p);

  // initialize index vector for jobs sorted by j.n (descending)
  instance.jobs_nsort.resize(instance.jobs.size());
  iota(instance.jobs_nsort.begin(), instance.jobs_nsort.end(), 0);
  sort(std::execution::par_unseq, instance.jobs_nsort.begin(),
       instance.jobs_nsort.end(), [&instance](size_t i1, size_t i2) {
         return Job::comparator_n_index(i1, i2, instance.jobs);
       });

  // update p_max's
  instance.p_max = instance.jobs.back().p;
  instance.p_max2 = instance.p_max * instance.p_max;
  instance.p_max4 = instance.p_max2 * instance.p_max2;
  csv_logger.set("p_max", instance.p_max);

  auto s_max = instance.machines.back().s;
  csv_logger.set("s_max", s_max);

  // calculate upper and lower bounds
  instance.avg_makespan = instance.p_sum * 1.0 / s_sum;
  // instance.scaled_lb = std::floor(instance.s_lcm * instance.avg_makespan);
  // instance.scaled_ub = instance.s_lcm * (instance.avg_makespan +
  // instance.p_max);
  instance.scaled_lb = static_cast<long>(std::ceil(
      instance.s_lcm * std::max(static_cast<double>(instance.p_sum) / s_sum,
                                static_cast<double>(instance.p_max) / s_max)));
  instance.scaled_ub = static_cast<long>(std::ceil(
      instance.s_lcm * (static_cast<double>(instance.p_sum) / s_max)));

  spdlog::info("p_sum={} s_sum={} p_max={} s_max={}", instance.p_sum, s_sum,
               instance.p_max, s_max);
  spdlog::info("Found bounds k_min={} k_max={} using s_lcm={}",
               instance.scaled_lb, instance.scaled_ub, instance.s_lcm);
}

double bsearch_opt_makespan(ProblemInstance &instance,
                            const bool use_gurobi_and_not_arrowclimb
#ifdef MAT_DMP
                            ,
                            ofstream &ofs
#endif
) {
  Stopwatch sw;
  const uint num_machine_types = instance.machines.size();
  const uint num_job_types = instance.jobs.size();

  int highest_small_gen = -1; // highest small machine t our bfs was called at,
                              // used to save bfs calls

  vector<Config>
      candidates_S; // holds all possible configurations for small machines
  vector<Config>
      candidates_B; // holds all possible configurations for big machines

  const uint a = instance.jobs[instance.jobs_nsort[0]].p;
  spdlog::info("Selected pivot a={}", a);
  csv_logger.set("a", a);

  sw.start();
  while (instance.scaled_lb < instance.scaled_ub) {
    // calculate middle of the search interval and the corresponding makespan
    const long k_mid =
        instance.scaled_lb + (instance.scaled_ub - instance.scaled_lb) / 2;
    const double makespan = (double)k_mid * 1.0 / (double)instance.s_lcm;
    csv_logger.set("C_guess", makespan);
    // cout << "guessed makespan: " << makespan << endl;
    spdlog::info("Considering k_mid={} with makespan={}", k_mid, makespan);
    // if (instance.p_max > makespan) {
    //   cout << "Iteration is infeasible by construction. Skipping" << endl;
    //   csv_logger.set("status", "infeasible p_max cover");
    //   instance.scaled_lb = k_mid + 1;
    //   csv_logger.flush();
    // }
    long l = -instance.p_sum; // TODO: does this stay as an int?; Z.482

    // guess load for each machine
    for (int i = 0; i < instance.machines.size(); ++i) {
      Machine &m = instance.machines[i];
      m.t = k_mid * m.s / instance.s_lcm; // Z.478
      l += m.n * m.t;                     // Z.482, max. number of dummy jobs

      spdlog::info("Guessed load for machine {}: {}", i, m.t);
    }
    l = max(0L, l);
    spdlog::info("l={}", l);
    csv_logger.seti("l", l);

    // TODO: somewhere here...
    // - can use K/s_ instead of a/kGV
    // - dummy jobs needed

    // sw.start();

    auto is_small = [&](const Machine &m) { return m.t < instance.p_max4; };
    auto it_first_big = std::partition_point(instance.machines.begin(),
                                             instance.machines.end(), is_small);
    const std::size_t num_small = static_cast<std::size_t>(
        std::distance(instance.machines.begin(), it_first_big));
    instance.num_small_machines = static_cast<int>(num_small);

    // sw.log_and_reset("determine small/big machine split took ");
    // cout << "Found " << instance.num_small_machines << " Small Machines and "
    //      << instance.machines.size() - instance.num_small_machines
    //      << " Big Machines." << endl;

    // Start with the most frequent as pivot. If a <= p_max2_B we don't have
    // anymore frequent jobs left.
    // TODO Correctness:
    // - Do we truly need to try more than one pivot element? Is highest truly
    // the best? How do we deal with better solution inbetween?
    // - Is there always a valid pivot element?
    VectorXi mod_frequency;
    if (instance.num_small_machines != num_machine_types) {
      const uint p_max2_B =
          instance.p_max2 * (num_machine_types - instance.num_small_machines);
      assert(a <= p_max2_B);

      // see which parities need combinations
      mod_frequency = VectorXi::Zero(a);
      for (int i = instance.num_small_machines; i < instance.machines.size();
           ++i) {
        const Machine &m = instance.machines[i];
        mod_frequency[m.t % a] += m.n;
        // cout << "machine i=" << i << " has mod freq " << " m.t % a = " << m.t
        // << " % " << a << " = " << m.t%a << endl;
      }
    }

    // generate all possible configurations
    const bool regenerate_S =
        highest_small_gen == -1 ||
        (instance.num_small_machines == 0
             ? false
             : highest_small_gen <
                   instance.machines[instance.num_small_machines - 1].t);
    const bool regenerate_B =
        candidates_B.empty() &&
        (instance.num_small_machines != instance.machines.size());
    bfsGenerateCandidates(instance, a, mod_frequency, candidates_S,
                          candidates_B, regenerate_S, regenerate_B);
    highest_small_gen =
        instance.num_small_machines == 0
            ? -1
            : instance.machines[instance.num_small_machines - 1].t;
    // sort ASC by cost
    // sort(std::execution::par_unseq, candidates_B.begin(), candidates_B.end(),
    //      [&a](Config &x, Config &y) { // sort ASC by (mod a)
    //        return x.cost % a <= y.cost % a;
    //      });

    sw.log_and_reset("candidate regeneration took");
    print_candidates(instance, candidates_S, candidates_B);

    // TODO: Optimization: redundant blocks
    VectorXi b;
    BlockedMatrix BM;
    const bool feasible_dummy_job_cover = construct_ilp(
        instance, a, l, mod_frequency, candidates_S, candidates_B, b, BM);
    const auto &A = BM.mat_;
    // spdlog::info("A:\n{}\nb.transpose():\n{}", matrix_to_string(A),
    // vector_to_string(b.transpose())); spdlog::info("b = {}",
    // vector_to_string(b.transpose()));
#ifdef MAT_DMP
    ofs << "A:\n" << A << "\n";
    ofs << "b.transpose():\n" << b.transpose() << "\n";
#endif

    if (!feasible_dummy_job_cover) {
      cout << "makespan " << makespan
           << " found infeasible (dummy job cover insufficient)" << endl;
      spdlog::info(
          "makespan {} found infeasible (dummy job cover insufficient)",
          makespan);
      csv_logger.set("status", "infeasible djc");
      csv_logger.flush();
      instance.scaled_lb = k_mid + 1;
      continue;
    }
    // ofs.close();
    // exit(0);
    instance.delta = A.maxCoeff();
    sw.log_and_reset("ILP construction took");
    spdlog::info("b.size()={} jobs.size()={} num_small={} machines={}",
                 b.size(), instance.jobs.size(), instance.num_small_machines,
                 instance.machines.size());

    // --- Debug structural parameters ---
    const auto Delta_no_dummy =
        (A.rows() > 1) ? A.topRows(A.rows() - 1).maxCoeff() : 0;

    const auto n_jobs = static_cast<Eigen::Index>(instance.jobs.size());
    const auto need_up = n_jobs + 1;
    const auto B = b.size();
    if (B < need_up) {
      spdlog::error("b too small: {} < {}", B, need_up);
      throw std::runtime_error("invalid b size");
    }

    const auto b_up_max = b.head(need_up).maxCoeff();
    const auto len_down = (B > need_up) ? (B - need_up) : 0;
    const auto b_down_max = (len_down > 0) ? b.tail(len_down).maxCoeff() : 0;
    const auto b_def = std::min(b_up_max, b_down_max);

    const bool has_small_dummy =
        (instance.num_small_machines != instance.machines.size());
    const auto b_up_max_no_dummy = (n_jobs > 0) ? b.head(n_jobs).maxCoeff() : 0;
    const auto len_down_no_dummy =
        (len_down > 0) ? (len_down - (has_small_dummy ? 1 : 0)) : 0;
    const auto b_down_max_no_dummy =
        (len_down_no_dummy > 0)
            ? b.segment(need_up, len_down_no_dummy).maxCoeff()
            : 0;
    const auto b_def_no_dummy =
        std::min(b_up_max_no_dummy, b_down_max_no_dummy);

    spdlog::info("A.rows={} A.cols={} Delta={} b_def={}", A.rows(), A.cols(),
                 instance.delta, b_def);
    csv_logger.set("A_rows", A.rows());
    csv_logger.set("A_cols", A.cols());
    csv_logger.set("num_blocks", BM.num_blocks());
    csv_logger.set("Delta", instance.delta);
    csv_logger.set("Delta_no_dummy", Delta_no_dummy);
    csv_logger.set("b_def", b_def);
    csv_logger.set("b_def_no_dummy", b_def_no_dummy);
    assert(instance.delta >= 0);
    assert(b_def >= 0);
    assert(A.rows() >= 1 && A.cols() >= 1 && BM.num_blocks() >= 1);

    // --- solving ---
    sw.start();
    const auto feasible = solve_direct(instance.delta, BM, b);
    csv_logger.set("s_gur", sw.log_and_reset_get_time("gurobi took"));
    const auto feasible_ac = solve_arrow_climb(instance.delta, BM, b);
    csv_logger.set("s_ac", sw.log_and_reset_get_time("arrow climb took"));
    // const auto feasible_pm = solve_pooled_master_and_assignment(
    //     BM, instance.jobs, instance.machines, instance.num_small_machines, a,
    //     false);
    // csv_logger.set("s_pm", sw.log_and_reset_get_time("pooled took"));
    // const auto feasible_pms = solve_pooled_master_and_assignment(
    //     BM, instance.jobs, instance.machines, instance.num_small_machines,
    //     instance.jobs_nsort[0], true);
    // csv_logger.set("s_pms", sw.log_and_reset_get_time("pooled small took"));

#ifdef ASSERT_SOLVERS
    if (feasible != feasible_ac) {
      cerr << "arrow climb diverted from expected: feasible=" << feasible
           << " feasible_ac=" << feasible_ac << endl;
#ifdef MAT_DMP
      ofs.close();
#endif
      exit(1);
    }
#endif

    //   if (a > 1) {
    //     if (feasible != feasible_pm) {
    //       cerr << "pooled diverted from expected: feasible=" << feasible
    //             << " feasible_pm=" << feasible_pm << endl;
    // #ifdef MAT_DMP
    //       ofs.close();
    // #endif
    //       exit(1);
    //     }
    //   }

    //     if (feasible != feasible_pms) {
    //       cerr << "pooled small diverted from expected: feasible=" <<
    //       feasible
    //             << " feasible_pm=" << feasible_pms << endl;
    // #ifdef MAT_DMP
    //       ofs.close();
    // #endif
    //       exit(1);
    //     }
    // assert(feasible == feasible_pm && "pooled diverted from expected");
    // assert(feasible == feasible_pms && "pooled small diverted from
    // expected");

    // sw.start();
    // bool feasible;
    // if (!use_gurobi_and_not_arrowclimb) {
    //   feasible = solve_arrow_climb(instance.delta, BM, b);
    // } else {
    //   feasible = solve_direct(instance.delta, BM, b);
    // }
    // sw.log_and_reset("solve took");

    // --- update search interval ---
    if (feasible) {
      // cout << "makespan " << makespan << " found feasible" << endl;
      spdlog::info("makespan {} found feasible", makespan);
      instance.scaled_ub = k_mid;
      csv_logger.set("status", "feasible");
      csv_logger.flush();
    } else {
      // cout << "makespan " << makespan << " found infeasible" << endl;
      spdlog::info("makespan {} found infeasible", makespan);
      instance.scaled_lb = k_mid + 1;
      csv_logger.set("status", "infeasible");
      csv_logger.flush();
    }
  }

  return instance.scaled_ub * 1.0 / (double)instance.s_lcm;
}

int main(int argc, char *argv[]) {
  Stopwatch sw;

  bool use_gurobi_and_not_arrowclimb;
  if (argc == 2) {
    // std::cout << "defaulting to gurobi" << std::endl;
    use_gurobi_and_not_arrowclimb = true;
  } else if (argc == 3) {
    use_gurobi_and_not_arrowclimb = my_stoi(argv[2]);
  } else {
    std::cerr << "Invalid number of arguments. Usage:\n"
                 "  uniformsched <path/to/instance.dat|folder> "
                 "[use_gurobi_and_not_arrowclimb=1]\n";
    return 1;
  }

  try {
    std::filesystem::path p(argv[1]);
    csv_logger.set("instance", p.stem().string());

#if SPDLOG_LOG_TYPE == 1
    const auto console = spdlog::stdout_color_mt("console");
    spdlog::set_default_logger(console);
#elif SPLDOG_LOG_TYPE == 2
    const auto file_logger = spdlog::basic_logger_mt(
        "file_logger", make_log_filename(p, use_gurobi_and_not_arrowclimb));
    spdlog::set_default_logger(file_logger);
    spdlog::drop("console");
#else
    spdlog_disable();
#endif

    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");
    spdlog::set_level(spdlog::level::debug);
  } catch (const spdlog::spdlog_ex &ex) {
    cerr << "Log init failed: " << ex.what() << endl;
    return 1;
  }

  ProblemInstance instance;

  const std::filesystem::path input_path(argv[1]);

  auto process_one_file = [&](const std::filesystem::path &p) -> bool {
    try {
      std::ifstream ifs(p);
      if (!ifs) {
        std::cerr << "Error opening file: " << p << std::endl;
        return false;
      }

      csv_logger.set("instance", p.stem().string());

      ProblemInstance instance;
      sw.start();
      parse_file(ifs, instance);
      ifs.close();
      sw.log_and_reset("File parsing took");

      print_instance(instance);

#ifdef MAT_DMP
      const std::string out_name =
          "matrix_output_" + p.stem().string() + ".txt";
      std::ofstream ofs(out_name);
      if (!ofs) {
        std::cerr << "Error opening output file: " << out_name << std::endl;
        return false;
      }
#endif

      const double opt_makespan =
          bsearch_opt_makespan(instance, use_gurobi_and_not_arrowclimb
#ifdef MAT_DMP
                               ,
                               ofs
#endif
          );
      sw.log_and_reset("bsearch_opt_makespan took");
      std::cout << p.filename().string()
                << " -> optimal makespan = " << opt_makespan << std::endl;
      spdlog::info("{} -> optimal makespan = {}", p.filename().string(),
                   opt_makespan);
#ifdef MAT_DMP
      ofs.close();
#endif
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Exception while processing " << p << ": " << e.what()
                << std::endl;
      return false;
    }
  };

  int processed = 0, succeeded = 0;

  if (std::filesystem::is_directory(input_path)) {
    std::vector<std::filesystem::path> files;
    for (const auto &dirent : std::filesystem::directory_iterator(input_path)) {
      if (!dirent.is_regular_file())
        continue;
      const auto &p = dirent.path();
      if (p.has_extension() && p.extension() == ".dat") {
        files.push_back(p);
      }
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
      std::cerr << "No .dat files found directly under: " << input_path
                << std::endl;
      return 1;
    }

    for (size_t i = 0; i < files.size(); ++i) {
      ++processed;
      if (process_one_file(files[i]))
        ++succeeded;
      if (i + 1 < files.size())
        cooldown_between_jobs_windows();
    }
  } else {
    ++processed;
    if (process_one_file(input_path))
      ++succeeded;
  }

  std::cout << "Finished. Succeeded " << succeeded << " / " << processed
            << " file(s)." << std::endl;
  return (succeeded == processed) ? 0 : 2;
}