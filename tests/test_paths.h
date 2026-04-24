#pragma once

#include <filesystem>
#include <optional>

namespace test_support {

inline bool looks_like_repo_root(const std::filesystem::path &path) {
  return std::filesystem::exists(path / "src") &&
         std::filesystem::exists(path / "tests") &&
         std::filesystem::exists(path / "instances");
}

inline std::optional<std::filesystem::path>
find_repo_root_from(std::filesystem::path path) {
  if (path.empty())
    path = std::filesystem::current_path();
  if (std::filesystem::is_regular_file(path))
    path = path.parent_path();

  for (;;) {
    if (looks_like_repo_root(path))
      return path;
    const auto parent = path.parent_path();
    if (parent == path || parent.empty())
      return std::nullopt;
    path = parent;
  }
}

inline std::filesystem::path test_repo_root() {
  if (const auto root = find_repo_root_from(std::filesystem::path(__FILE__)))
    return *root;
  if (const auto root = find_repo_root_from(std::filesystem::current_path()))
    return *root;
  return std::filesystem::path(__FILE__).parent_path().parent_path();
}

} // namespace test_support
