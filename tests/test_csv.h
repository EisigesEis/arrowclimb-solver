#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace test_support {

inline std::vector<std::string> split_row(std::string_view line, char separator,
                                          bool keep_trailing_empty) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == separator) {
      out.push_back(cur);
      cur.clear();
      continue;
    }
    cur.push_back(c);
  }
  if (keep_trailing_empty || !cur.empty() || !line.empty())
    out.push_back(cur);
  return out;
}

inline std::vector<std::string> split_csv_row(std::string_view line) {
  return split_row(line, ',', true);
}

inline std::vector<std::string> split_semicolon_row(std::string_view line) {
  return split_row(line, ';', false);
}

inline std::map<std::string, std::string>
read_csv_first_row(const std::filesystem::path &path) {
  std::ifstream in(path);
  EXPECT_TRUE(in.is_open()) << path.string();

  std::string header;
  std::string row;
  EXPECT_TRUE(static_cast<bool>(std::getline(in, header))) << path.string();
  EXPECT_TRUE(static_cast<bool>(std::getline(in, row))) << path.string();

  const std::vector<std::string> keys = split_csv_row(header);
  const std::vector<std::string> values = split_csv_row(row);
  EXPECT_EQ(keys.size(), values.size()) << path.string();

  std::map<std::string, std::string> out;
  for (std::size_t i = 0; i < std::min(keys.size(), values.size()); ++i)
    out.emplace(keys[i], values[i]);
  return out;
}

inline std::optional<std::map<std::string, std::string>>
read_csv_first_row_with_value(const std::filesystem::path &path,
                              std::string_view key) {
  std::ifstream in(path);
  EXPECT_TRUE(in.is_open()) << path.string();

  std::string header;
  EXPECT_TRUE(static_cast<bool>(std::getline(in, header))) << path.string();
  const std::vector<std::string> keys = split_csv_row(header);

  std::string row;
  while (std::getline(in, row)) {
    const std::vector<std::string> values = split_csv_row(row);
    if (values.size() != keys.size())
      continue;
    std::map<std::string, std::string> out;
    for (std::size_t i = 0; i < keys.size(); ++i)
      out.emplace(keys[i], values[i]);
    const auto it = out.find(std::string(key));
    if (it != out.end() && !it->second.empty())
      return out;
  }
  return std::nullopt;
}

} // namespace test_support
