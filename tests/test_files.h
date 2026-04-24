#pragma once

#include <filesystem>
#include <system_error>

namespace test_support {

inline void remove_if_exists(const std::filesystem::path &path) {
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}

} // namespace test_support
