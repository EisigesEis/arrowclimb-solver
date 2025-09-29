#include "Util.h"
#include "bfs.h"
#include <iomanip>

// https://tinodidriksen.com/2010/02/cpp-convert-string-to-int-speed/
unsigned int my_stoi(const char *str) {
  int val = 0;
  while (*str) {
    val = val * 10 + (*str++ - '0');
  }
  return val;
}

inline void vec_zero_to_n(std::vector<int> &v) {
  return std::iota(v.begin(), v.end(), 0);
}

int my_gcd(int m, int n) {
  while (n != 0) {
    int t = m % n;
    m = n;
    n = t;
  }
  return m;
}

std::string matrix_to_string(const Eigen::MatrixXi &mat) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < mat.rows(); ++i) {
    if (i > 0)
      oss << "; ";
    for (int j = 0; j < mat.cols(); ++j) {
      if (j > 0)
        oss << ", ";
      oss << mat(i, j);
    }
  }
  oss << "]";
  return oss.str();
}

std::string vector_to_string(const Eigen::VectorXi &v) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < v.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << v[i];
  }
  oss << "]";
  return oss.str();
}

inline std::string get_log_folder(const std::filesystem::path &input_path) {
  const std::vector<std::string> valid_folders = {"BIG", "E1", "E2", "E3",
                                                  "E4"};
  std::filesystem::path current_path = input_path;

  if (current_path.has_parent_path()) {
    current_path = current_path.parent_path();

    for (const auto &folder : valid_folders) {
      // cout << "current_path.filename()=" << current_path.filename() <<
      // "folder=" << folder << endl;
      if (current_path.filename() == folder) {
        return folder;
      }
    }
  }

  return ".";
}

std::string make_log_filename(const std::filesystem::path &input_path,
                              const bool use_gurobi_and_not_arrowclimb) {
  const auto base = input_path.filename();
  const std::string log_folder = get_log_folder(input_path);
  auto now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &now);
#else
  localtime_r(&now, &tm);
#endif
  std::ostringstream oss;
  oss << "logs/" << (log_folder / base).string() << "_"
      << (use_gurobi_and_not_arrowclimb ? "GUR" : "ACC")
      //  << "_" << std::put_time(&tm, "%Y%m%d_%H%M%S")
      << ".log";
  cout << "Writing to logfile " << oss.str() << endl;
  return oss.str();
}