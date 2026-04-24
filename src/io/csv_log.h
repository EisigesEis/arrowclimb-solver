#pragma once

#include "io/csv_file.h"
#include <utility>
#include <utility>

#ifndef UNIFORMSCHED_ENABLE_CSV_LOGGING
#define UNIFORMSCHED_ENABLE_CSV_LOGGING 1
#endif

namespace io::csv {

class Logger : public File {
public:
  using File::File;
  using File::set;

  void set_persist(bool on) { set_persist_fields(on); }
  void set_persist_field(std::string field) {
    File::set_persist_field(std::move(field));
  }

  template <typename T> void seti(std::string_view key, T value) {
    set_enum(key, value);
  }

  void clear_all() { clear(); }
};

class Log {
public:
  Log() = default;
  explicit Log(Logger &logger) : logger_(&logger) {}

  template <typename T> void set(std::string_view key, T &&value) {
    if (logger_ != nullptr) {
      logger_->set(key, std::forward<T>(value));
    }
  }

  template <typename T> void seti(std::string_view key, T value) {
    if (logger_ != nullptr) {
      logger_->seti(key, value);
    }
  }

  void flush() {
    if (logger_ != nullptr) {
      logger_->flush();
    }
  }

  void clear_all() {
    if (logger_ != nullptr) {
      logger_->clear_all();
    }
  }

private:
  Logger *logger_ = nullptr;
};

} // namespace io::csv
