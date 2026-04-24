#include "oracle/gupta/gupta_core.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;
using Eigen::VectorXi;
using oracle::gupta_core::RunBatch;

namespace {

struct CLI {
  int reps = 200;
  int warmup = 3;
  int sleep_ms = 0;
};

struct BenchCase {
  std::string name;
  VectorXi m;
};

struct RunView {
  std::vector<RunBatch> runs;
  VectorXi max_run;
};

struct Baseline {
  VectorXi msig;
  RunView runs;
};

struct Result {
  double secs = 0.0;
  std::uint64_t run_hash = 0;
  std::uint64_t seq_hash = 0;
  int runs = 0;
  int peak_run = 0;
};

volatile std::uint64_t g_sink = 0;

static CLI parse_cli(int argc, char **argv) {
  CLI cli;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](int &j) -> std::string {
      return (j + 1 < argc) ? std::string(argv[++j]) : std::string();
    };
    if (a == "--reps")
      cli.reps = std::max(1, std::stoi(next(i)));
    else if (a == "--warmup")
      cli.warmup = std::max(0, std::stoi(next(i)));
    else if (a == "--sleep")
      cli.sleep_ms = std::max(0, std::stoi(next(i)));
  }
  return cli;
}

static VectorXi vec_from(const std::vector<int> &vals) {
  VectorXi out((int)vals.size());
  for (int i = 0; i < out.size(); ++i)
    out[i] = vals[(size_t)i];
  return out;
}

static std::vector<int> repeated(int n, int val) {
  return std::vector<int>((size_t)n, val);
}

static std::vector<int> skewed(int n, int small, int big) {
  std::vector<int> v((size_t)n, small);
  v.back() = big;
  return v;
}

static std::vector<int> two_heavy(int n, int small, int heavy0, int heavy1) {
  std::vector<int> v((size_t)n, small);
  v[(size_t)(n - 2)] = heavy0;
  v[(size_t)(n - 1)] = heavy1;
  return v;
}

static std::vector<int> staircase(int n, int base) {
  std::vector<int> v((size_t)n);
  for (int i = 0; i < n; ++i)
    v[(size_t)i] = base + i;
  return v;
}

static std::vector<BenchCase> make_cases() {
  std::vector<BenchCase> cases;
  cases.push_back({"tiny", vec_from({3, 2, 1, 1})});
  cases.push_back({"balanced_32x64", vec_from(repeated(32, 64))});
  cases.push_back({"stair_32", vec_from(staircase(32, 1))});
  cases.push_back({"skew_32", vec_from(skewed(32, 1, 4096))});
  cases.push_back({"two_heavy_32", vec_from(two_heavy(32, 1, 2048, 2048))});
  return cases;
}

static RunView compress_msig(const VectorXi &msig, int n) {
  RunView out;
  oracle::gupta_core::compress_msig(msig, n, out.runs, out.max_run);
  return out;
}

static std::uint64_t hash_run_view(const RunView &view) {
  std::uint64_t h = 1469598103934665603ull;
  auto mix = [&](std::uint64_t x) {
    h ^= x + 0x9e3779b97f4a7c15ull;
    h = ((h << 13) | (h >> 51)) * 0xff51afd7ed558ccdull;
  };
  mix((std::uint64_t)view.runs.size());
  for (const auto &rb : view.runs) {
    mix((std::uint64_t)rb.cnt);
    mix((std::uint64_t)rb.e);
  }
  for (int i = 0; i < view.max_run.size(); ++i)
    mix((std::uint64_t)view.max_run[i]);
  return h;
}

static std::uint64_t hash_seq(const VectorXi &seq) {
  std::uint64_t h = 1469598103934665603ull;
  for (int i = 0; i < seq.size(); ++i) {
    h ^= (std::uint64_t)seq[i] + 0x9e3779b97f4a7c15ull;
    h = ((h << 13) | (h >> 51)) * 0xff51afd7ed558ccdull;
  }
  return h;
}

static VectorXi expand_runs(const RunView &view) {
  int q = 0;
  for (const auto &rb : view.runs)
    q += rb.cnt;

  VectorXi seq(q);
  int pos = 0;
  for (const auto &rb : view.runs) {
    for (int t = 0; t < rb.cnt; ++t)
      seq[pos++] = rb.e;
  }
  return seq;
}

static VectorXi counts_from_seq(const VectorXi &seq, int n) {
  VectorXi cnt = VectorXi::Zero(n);
  for (int i = 0; i < seq.size(); ++i)
    ++cnt[seq[i]];
  return cnt;
}

static VectorXi max_run_from_seq(const VectorXi &seq, int n) {
  VectorXi out = VectorXi::Zero(n);
  if (seq.size() == 0)
    return out;

  int cur = seq[0];
  int len = 1;
  for (int i = 1; i < seq.size(); ++i) {
    if (seq[i] == cur) {
      ++len;
    } else {
      out[cur] = std::max(out[cur], len);
      cur = seq[i];
      len = 1;
    }
  }
  out[cur] = std::max(out[cur], len);
  return out;
}

static std::string vec_str(const VectorXi &v) {
  std::string s = "[";
  for (int i = 0; i < v.size(); ++i) {
    if (i)
      s += ", ";
    s += std::to_string(v[i]);
  }
  s += "]";
  return s;
}

static std::string run_prefix_str(const RunView &view, int limit = 12) {
  std::string s = "[";
  const int take = std::min(limit, (int)view.runs.size());
  for (int i = 0; i < take; ++i) {
    if (i)
      s += ", ";
    s += "(" + std::to_string(view.runs[(size_t)i].e) + " x" +
         std::to_string(view.runs[(size_t)i].cnt) + ")";
  }
  if ((int)view.runs.size() > take)
    s += ", ...";
  s += "]";
  return s;
}

static std::string diff_runs(const RunView &base, const RunView &cand) {
  const size_t common = std::min(base.runs.size(), cand.runs.size());
  for (size_t i = 0; i < common; ++i) {
    const auto &a = base.runs[i];
    const auto &b = cand.runs[i];
    if (a.e != b.e || a.cnt != b.cnt) {
      return "run[" + std::to_string(i) + "] base=(" + std::to_string(a.e) +
             " x" + std::to_string(a.cnt) + ") cand=(" + std::to_string(b.e) +
             " x" + std::to_string(b.cnt) + ")";
    }
  }
  if (base.runs.size() != cand.runs.size()) {
    return "run count base=" + std::to_string(base.runs.size()) +
           " cand=" + std::to_string(cand.runs.size());
  }
  return "none";
}

static std::string diff_seq(const VectorXi &base, const VectorXi &cand) {
  const int common = std::min(base.size(), cand.size());
  for (int i = 0; i < common; ++i) {
    if (base[i] != cand[i]) {
      return "step " + std::to_string(i) + " base=" + std::to_string(base[i]) +
             " cand=" + std::to_string(cand[i]);
    }
  }
  if (base.size() != cand.size()) {
    return "length base=" + std::to_string(base.size()) +
           " cand=" + std::to_string(cand.size());
  }
  return "none";
}

static std::string validate_candidate(const std::string &name, const VectorXi &m,
                                      const Baseline &base,
                                      const RunView &cand) {
  const VectorXi cand_seq = expand_runs(cand);
  const VectorXi cand_cnt = counts_from_seq(cand_seq, (int)m.size());
  const VectorXi cand_peak = max_run_from_seq(cand_seq, (int)m.size());

  if (cand_seq.size() != base.msig.size() || cand_seq != base.msig ||
      cand.max_run.size() != cand_peak.size() || cand.max_run != cand_peak ||
      cand_cnt.size() != m.size() || cand_cnt != m) {
    std::string msg = name + " mismatch\n";
    msg += "  m        = " + vec_str(m) + "\n";
    msg += "  seq diff = " + diff_seq(base.msig, cand_seq) + "\n";
    msg += "  run diff = " + diff_runs(base.runs, cand) + "\n";
    msg += "  base runs= " + run_prefix_str(base.runs) + "\n";
    msg += "  cand runs= " + run_prefix_str(cand) + "\n";
    msg += "  base cnt = " + vec_str(m) + "\n";
    msg += "  cand cnt = " + vec_str(cand_cnt) + "\n";
    msg += "  base max = " + vec_str(base.runs.max_run) + "\n";
    msg += "  cand max = " + vec_str(cand.max_run) + "\n";
    msg += "  seq max  = " + vec_str(cand_peak);
    return msg;
  }

  return "";
}

static Result run_build_msig(const VectorXi &m) {
  VectorXi msig;
  const auto t0 = Clock::now();
  oracle::gupta_core::build_msig(m, msig);
  const auto t1 = Clock::now();
  const RunView view = compress_msig(msig, (int)m.size());
  const std::uint64_t run_h = hash_run_view(view);
  const std::uint64_t seq_h = hash_seq(msig);
  g_sink ^= (run_h ^ seq_h);
  Result out;
  out.secs = std::chrono::duration<double>(t1 - t0).count();
  out.run_hash = run_h;
  out.seq_hash = seq_h;
  out.runs = (int)view.runs.size();
  out.peak_run = view.max_run.size() ? view.max_run.maxCoeff() : 0;
  return out;
}

static Result run_build_msig_compress(const VectorXi &m) {
  VectorXi msig;
  RunView view;
  const auto t0 = Clock::now();
  oracle::gupta_core::build_msig(m, msig);
  oracle::gupta_core::compress_msig(msig, (int)m.size(), view.runs,
                                    view.max_run);
  const auto t1 = Clock::now();
  const std::uint64_t run_h = hash_run_view(view);
  const std::uint64_t seq_h = hash_seq(msig);
  g_sink ^= (run_h ^ seq_h);
  Result out;
  out.secs = std::chrono::duration<double>(t1 - t0).count();
  out.run_hash = run_h;
  out.seq_hash = seq_h;
  out.runs = (int)view.runs.size();
  out.peak_run = view.max_run.size() ? view.max_run.maxCoeff() : 0;
  return out;
}

static Result run_build_msig_batch(const VectorXi &m) {
  RunView view;
  const auto t0 = Clock::now();
  oracle::gupta_core::build_msig_batch(m, view.runs, view.max_run);
  const auto t1 = Clock::now();
  const VectorXi seq = expand_runs(view);
  const std::uint64_t run_h = hash_run_view(view);
  const std::uint64_t seq_h = hash_seq(seq);
  g_sink ^= (run_h ^ seq_h);
  Result out;
  out.secs = std::chrono::duration<double>(t1 - t0).count();
  out.run_hash = run_h;
  out.seq_hash = seq_h;
  out.runs = (int)view.runs.size();
  out.peak_run = view.max_run.size() ? view.max_run.maxCoeff() : 0;
  return out;
}

static Result run_build_msig_batch_heap(const VectorXi &m) {
  RunView view;
  const auto t0 = Clock::now();
  oracle::gupta_core::build_msig_batch_heap(m, view.runs, view.max_run);
  const auto t1 = Clock::now();
  const VectorXi seq = expand_runs(view);
  const std::uint64_t run_h = hash_run_view(view);
  const std::uint64_t seq_h = hash_seq(seq);
  g_sink ^= (run_h ^ seq_h);
  Result out;
  out.secs = std::chrono::duration<double>(t1 - t0).count();
  out.run_hash = run_h;
  out.seq_hash = seq_h;
  out.runs = (int)view.runs.size();
  out.peak_run = view.max_run.size() ? view.max_run.maxCoeff() : 0;
  return out;
}

static Baseline baseline_view(const VectorXi &m) {
  Baseline out;
  oracle::gupta_core::build_msig(m, out.msig);
  out.runs = compress_msig(out.msig, (int)m.size());
  return out;
}

static void verify_case(const BenchCase &bc) {
  const Baseline base = baseline_view(bc.m);

  RunView scan;
  oracle::gupta_core::build_msig_batch(bc.m, scan.runs, scan.max_run);
  if (const std::string msg = validate_candidate("build_msig_batch", bc.m, base,
                                                 scan);
      !msg.empty())
    throw std::runtime_error("case " + bc.name + "\n" + msg);

  RunView heap;
  oracle::gupta_core::build_msig_batch_heap(bc.m, heap.runs, heap.max_run);
  if (const std::string msg =
          validate_candidate("build_msig_batch_heap", bc.m, base, heap);
      !msg.empty())
    throw std::runtime_error("case " + bc.name + "\n" + msg);
}

static int total_sum(const VectorXi &m) { return (int)m.sum(); }

} // namespace

int main(int argc, char **argv) {
  try {
    const CLI cli = parse_cli(argc, argv);
    const auto cases = make_cases();

    struct Variant {
      const char *name;
      Result (*run)(const VectorXi &m);
    };
    const std::vector<Variant> variants = {
        {"build_msig", run_build_msig},
        {"build_msig+compress", run_build_msig_compress},
        {"build_msig_batch", run_build_msig_batch},
        {"build_msig_batch_heap", run_build_msig_batch_heap},
    };

    for (const auto &bc : cases) {
      verify_case(bc);
      const Baseline base = baseline_view(bc.m);
      const int q = total_sum(bc.m);
      const int n = (int)bc.m.size();
      const int peak_run =
          base.runs.max_run.size() ? base.runs.max_run.maxCoeff() : 0;

      std::cout << "\n== " << bc.name << " | n=" << n << " | q=" << q
                << " | runs=" << base.runs.runs.size()
                << " | peak_run=" << peak_run << " | m=" << vec_str(bc.m)
                << " ==\n";

      for (const auto &variant : variants) {
        for (int w = 0; w < cli.warmup; ++w)
          (void)variant.run(bc.m);

        std::vector<double> times;
        times.reserve((size_t)cli.reps);
        Result last;

        for (int r = 0; r < cli.reps; ++r) {
          last = variant.run(bc.m);
          times.push_back(last.secs);
        }

        std::sort(times.begin(), times.end());
        const double p50 = times[times.size() / 2];
        const double p90 = times[(times.size() * 9) / 10];
        const double layers_per_s = (p50 > 0.0) ? (q / p50) : 0.0;

        std::cout << std::left << std::setw(21) << variant.name
                  << " p50=" << std::fixed << std::setprecision(6) << p50 << "s"
                  << "  p90=" << p90 << "s"
                  << "  layers/s=" << std::setprecision(1) << layers_per_s
                  << "  runs=" << last.runs
                  << "  peak=" << last.peak_run
                  << "  run_hash=0x" << std::hex << last.run_hash
                  << "  seq_hash=0x" << last.seq_hash << std::dec << "\n";

        if (cli.sleep_ms > 0)
          std::this_thread::sleep_for(std::chrono::milliseconds(cli.sleep_ms));
      }
    }

    if (g_sink == 0)
      std::cerr << "";
  } catch (const std::exception &e) {
    std::cerr << "bench_gupta_sched failed: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
