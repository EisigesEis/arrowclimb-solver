#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <unordered_set>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/null_sink.h>

class CsvLogger {
public:
  explicit CsvLogger(const std::string &filename, bool persist_fields = true)
      : filename_(filename),
        header_written_(false),
        persist_fields_(persist_fields) {
    std::error_code ec;
    if (std::filesystem::exists(filename_, ec) &&
        std::filesystem::file_size(filename_, ec) > 0) {
      header_written_ = true;
    }
  }

  void setPersist(bool on) { persist_fields_ = on; }

  // Scalar setters
  void set(const std::string &key, const std::string &value) {
    ensureColumn(key);
    current_row_[key] = value;
  }
  void set(const std::string &key, const char *value) {
    ensureColumn(key);
    current_row_[key] = value ? std::string(value) : std::string();
  }
  void set(const std::string &key, int value) {
    ensureColumn(key);
    current_row_[key] = std::to_string(value);
  }
  void set(const std::string &key, long long value) {
    ensureColumn(key);
    current_row_[key] = std::to_string(value);
  }
  void set(const std::string &key, double value) {
    ensureColumn(key);
    current_row_[key] = to_string_fixed(value);
  }
  template <typename T>
  void seti(const std::string &key, T value) {
    ensureColumn(key);
    current_row_[key] = std::to_string(static_cast<int>(value));
  }

  void set(const std::string &key, const std::vector<double> &values) {
    ensureColumn(key);
    std::string joined;
    for (size_t i = 0; i < values.size(); ++i) {
      if (i)
        joined += ';';
      joined += to_string_fixed(values[i]);
    }
    current_row_[key] = joined;
  }

  void flush() {
    std::ofstream out(filename_, std::ios::app);
    if (!out) {
      std::cerr << "CsvLogger: cannot open " << filename_ << " for append.\n";
      return;
    }

    if (!header_written_) {
      for (size_t i = 0; i < columns_.size(); ++i) {
        if (i)
          out << ",";
        out << csv_escape(columns_[i]);
      }
      out << "\n";
      header_written_ = true;
    }

    for (size_t i = 0; i < columns_.size(); ++i) {
      if (i)
        out << ",";
      const auto &col = columns_[i];

      std::string value;
      auto it_now = current_row_.find(col);
      if (it_now != current_row_.end()) {
        value = it_now->second;
      } else if (persist_fields_) {
        auto it_prev = last_row_.find(col);
        if (it_prev != last_row_.end())
          value = it_prev->second;
      }
      out << csv_escape(value);
    }
    out << "\n";

    if (persist_fields_) {
      for (auto &kv : current_row_)
        last_row_[kv.first] = kv.second;
    }
    current_row_.clear();
  }

  void clearAll() {
    columns_.clear();
    current_row_.clear();
    last_row_.clear();
    header_written_ = false;
  }

private:
  static std::string to_string_fixed(double x) {
    return std::to_string(x);
  }

  void ensureColumn(const std::string &key) {
    if (seen_.insert(key).second) {
      columns_.push_back(key);
    }
  }

  static std::string csv_escape(const std::string &s) {
    bool need_quotes = false;
    for (char c : s) {
      if (c == ',' || c == '"' || c == '\n' || c == '\r') {
        need_quotes = true;
        break;
      }
    }
    if (!need_quotes)
      return s;
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
      if (c == '"')
        out.push_back('"');
      out.push_back(c);
    }
    out.push_back('"');
    return out;
  }

  std::string filename_;
  bool header_written_;
  bool persist_fields_;

  std::vector<std::string> columns_;
  std::map<std::string, std::string> current_row_;
  std::map<std::string, std::string> last_row_;
  std::unordered_set<std::string> seen_;
};

extern CsvLogger csv_logger;

inline void spdlog_disable() {
  auto null_sink  = std::make_shared<spdlog::sinks::null_sink_mt>();
  auto null_logger = std::make_shared<spdlog::logger>("null", null_sink);
  spdlog::set_default_logger(null_logger);
  spdlog::set_level(spdlog::level::off);
}