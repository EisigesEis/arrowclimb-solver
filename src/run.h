#pragma once
#include <filesystem>
#include <optional>
#include <regex>
#include <tuple>

namespace proc {

enum class RunMode { Main, AcDc };

} // namespace proc

#include "instance/parse.h"
#include "bsearch.h"
#include "io/csv_main.h"
#include <iostream>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

namespace proc {

inline std::filesystem::path main_csv_path_for_input(
    const std::filesystem::path &input, RunMode mode = RunMode::Main) {
  const char *filename = mode == RunMode::AcDc ? "acdc.csv" : "main.csv";
  if (std::filesystem::is_directory(input)) {
    return input / filename;
  }
  if (const auto parent = input.parent_path(); !parent.empty()) {
    return parent / filename;
  }
  return filename;
}

inline io::csv::MainSchema csv_schema_for_run_mode(RunMode mode) {
  return mode == RunMode::AcDc ? io::csv::MainSchema::AcDc
                               : io::csv::MainSchema::Main;
}

inline bool single(std::filesystem::path input_path, RunMode mode) {
  ParseResult parse_result = parse_file(input_path);

  if (!parse_result.ok) {
    if (parse_result.err.has_value()) {
      const auto &err = parse_result.err.value();
      if (err.line > 0) {
        spdlog::error("Failed to parse '{}': {} (line {})", input_path.string(),
                      err.message, err.line);
      } else {
        spdlog::error("Failed to read '{}': {}", input_path.string(),
                      err.message);
      }
    } else {
      spdlog::error("Failed to parse '{}': unknown parse error",
                    input_path.string());
    }
    return false;
  }

  log_main_instance_fields(parse_result.inst);

  double makespan = 0.0;
  try {
    makespan = proc::bsearch_c_max(parse_result.inst, mode);
  } catch (const std::exception &e) {
    spdlog::error("Failed while solving '{}': {}", input_path.string(),
                  e.what());
    return false;
  } catch (...) {
    spdlog::error("Failed while solving '{}': unknown exception",
                  input_path.string());
    return false;
  }

  spdlog::info("For inst={} found makespan={}", input_path.filename().string(),
               makespan);

  return true;
}

inline std::tuple<int, int> run(std::filesystem::path input_path,
                                RunMode mode = RunMode::Main,
                                const std::optional<std::regex> &file_pattern = std::nullopt) {
  int processed = 0, succeeded = 0;

  std::error_code ec;
  const bool exists = std::filesystem::exists(input_path, ec);
  if (ec) {
    std::cerr << "Failed to access path '" << input_path
              << "': " << ec.message() << std::endl;
    return std::make_tuple(0, 0);
  }
  if (!exists) {
    std::cerr << "Path does not exist: " << input_path << std::endl;
    return std::make_tuple(0, 0);
  }

  const bool is_dir = std::filesystem::is_directory(input_path, ec);
  if (ec) {
    std::cerr << "Failed to inspect path '" << input_path
              << "': " << ec.message() << std::endl;
    return std::make_tuple(0, 0);
  }

  if (!is_dir) {
    if (file_pattern.has_value()) {
      std::cerr << "The optional regex argument is only supported for directory runs."
                << std::endl;
      return std::make_tuple(0, 0);
    }

    if (!std::filesystem::is_regular_file(input_path, ec) || ec) {
      std::cerr << "Path is neither a regular file nor a directory: "
                << input_path
                << std::endl;
      return std::make_tuple(0, 0);
    }

    io::csv::init_main(main_csv_path_for_input(input_path, mode),
                       csv_schema_for_run_mode(mode));
    processed = 1;
    succeeded = proc::single(input_path, mode) == true;
  } else {
    io::csv::init_main(main_csv_path_for_input(input_path, mode),
                       csv_schema_for_run_mode(mode));

    std::vector<std::filesystem::path> files;
    for (const auto &dirent :
         std::filesystem::directory_iterator(input_path,
                                             std::filesystem::directory_options::none, ec)) {
      if (ec) {
        std::cerr << "Failed to iterate directory '" << input_path
                  << "': " << ec.message() << std::endl;
        return std::make_tuple(processed, succeeded);
      }
      if (!dirent.is_regular_file())
        continue;
      const auto &p = dirent.path();
      if (!p.has_extension() || p.extension() != ".dat")
        continue;

      if (file_pattern.has_value() &&
          !std::regex_search(p.filename().string(), *file_pattern)) {
        continue;
      }

      files.push_back(p);
    }
    std::sort(files.begin(), files.end(),
              [](const auto &lhs, const auto &rhs) {
                return lhs.filename().string() < rhs.filename().string();
              });

    if (files.empty()) {
      if (file_pattern.has_value()) {
        std::cerr << "No .dat files matched the regex directly under: "
                  << input_path
                  << std::endl;
      } else {
        std::cerr << "No .dat files found directly under: " << input_path
                  << std::endl;
      }
      return std::make_tuple(0, 0);
    }

    for (size_t i = 0; i < files.size(); ++i) {
      ++processed;

      if (proc::single(files[i], mode))
        ++succeeded;
    }
  }

  io::csv::reset_main();
  return std::make_tuple(processed, succeeded);
}

} // namespace proc
