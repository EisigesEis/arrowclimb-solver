#include "run.h"
#include "io/csv_main.h"
#include "io/log_utils.h"
#include <filesystem>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <string_view>

int main(int argc, char **argv) {
  bool log_to_console = true;
  bool write_csv = true;
  proc::RunMode run_mode = proc::RunMode::Main;
  std::filesystem::path input_path;
  std::optional<std::regex> file_pattern;
  bool saw_mode_flag = false;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "-l") {
      if (!saw_mode_flag) {
        log_to_console = false;
        write_csv = false;
        saw_mode_flag = true;
      }
      log_to_console = true;
      continue;
    }
    if (arg == "-c") {
      if (!saw_mode_flag) {
        log_to_console = false;
        write_csv = false;
        saw_mode_flag = true;
      }
      write_csv = true;
      continue;
    }
    if (arg == "-acdc") {
      run_mode = proc::RunMode::AcDc;
      continue;
    }

    if (input_path.empty()) {
      input_path = argv[i];
      continue;
    }

    if (!file_pattern.has_value()) {
      const std::string pattern_arg(argv[i]);
      try {
        file_pattern.emplace(pattern_arg, std::regex::ECMAScript);
      } catch (const std::regex_error &e) {
        std::cerr << "Invalid regex '" << pattern_arg << "': " << e.what()
                  << "\nUsage:\n"
                     "  uniformsched [-l] [-c] [-acdc] <path/to/instance.dat|folder> [filename-regex]\n"
                     "  default: log to console and write csv\n";
        return 1;
      }
      continue;
    }

    if (file_pattern.has_value()) {
      std::cerr << "Invalid arguments. Usage:\n"
                   "  uniformsched [-l] [-c] [-acdc] <path/to/instance.dat|folder> [filename-regex]\n"
                   "  default: log to console and write csv\n";
      return 1;
    }
  }

  if (input_path.empty()) {
    std::cerr << "Invalid number of arguments. Usage:\n"
                 "  uniformsched [-l] [-c] [-acdc] <path/to/instance.dat|folder> [filename-regex]\n"
                 "  default: log to console and write csv\n";
    return 1;
  }

  io::configure_spdlog_console(log_to_console);
  io::csv::set_main_enabled(write_csv);

  const auto [processed, succeeded] =
      proc::run(input_path, run_mode, file_pattern);
  if (processed == 0)
    return 1;

  std::cout << "Finished. Succeeded " << succeeded << " / " << processed
            << " file(s)." << std::endl;
  return (succeeded == processed) ? 0 : 2;
}
