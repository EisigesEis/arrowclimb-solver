#pragma once

#include "io/csv_log.h"

#include <filesystem>
#include <string_view>
#include <utility>

namespace io::csv {

enum class MainSchema {
  Main,
  AcDc,
};

void set_main_enabled(bool enabled);
void init_main(const std::filesystem::path &path,
               MainSchema schema = MainSchema::Main);
void reset_main();
Log main();
void main_persist(std::string_view key);

template <typename T> inline void main_set(std::string_view key, T &&value) {
  main().set(key, std::forward<T>(value));
}

inline void main_flush() { main().flush(); }

} // namespace io::csv
