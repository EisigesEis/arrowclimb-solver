#pragma once
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <sstream>
#include <Eigen/Dense>
#include <filesystem>
#include <spdlog/spdlog.h>

unsigned int my_stoi(const char *str);

inline void vec_zero_to_n(std::vector<int> &v);

int my_gcd(int m, int n);

std::string matrix_to_string(const Eigen::MatrixXi& mat);

std::string vector_to_string(const Eigen::VectorXi &v);

#ifndef __CUDACC__
std::string make_log_filename(const std::filesystem::path &input_path, const bool use_gurobi_and_not_arrowclimb);
#endif

inline long ceil_div(long num, long den) {
  return (num + den - 1) / den;
};

class Stopwatch {
private:
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point stop_time;

public:
  void start() { start_time = std::chrono::steady_clock::now(); };
  void stop() { stop_time = std::chrono::steady_clock::now(); };
  inline double get_time() const { return std::chrono::duration<double>(stop_time - start_time).count(); };
  void log_and_reset(std::string msg) {
    stop();
    spdlog::info("{} {}", msg, pretty_duration(duration_cast<std::chrono::nanoseconds>(stop_time-start_time)));
    start();
  }
  double log_and_reset_get_time(std::string msg) {
    stop();
    const double t = get_time();
    spdlog::info("{} {}", msg, pretty_duration(duration_cast<std::chrono::nanoseconds>(stop_time-start_time)));
    start();
    return t;
  }
  inline std::string pretty_duration(std::chrono::nanoseconds ns) {
    static const char* units[] = {"ns", "us", "ms", "s"};
    long double v = static_cast<long double>(ns.count());
    int idx = 0;
    while (v >= 1000.0 && idx < 3) {
        v /= 1000.0;
        ++idx;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << v << " " << units[idx];
    return oss.str();
  }
};