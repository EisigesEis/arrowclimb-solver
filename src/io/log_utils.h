#pragma once

#include <spdlog/spdlog.h>

namespace io {

inline void enable_spdlog_info() { spdlog::set_level(spdlog::level::info); }

inline void errors_only_spdlog() { spdlog::set_level(spdlog::level::err); }

inline void disable_spdlog() { spdlog::set_level(spdlog::level::off); }

inline void configure_spdlog_console(bool log_to_console) {
  if (log_to_console) {
    enable_spdlog_info();
  } else {
    errors_only_spdlog();
  }
}

} // namespace io
