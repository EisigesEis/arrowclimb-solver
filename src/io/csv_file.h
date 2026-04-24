#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace io::csv {

class File {
public:
  File() = default;
  File(std::filesystem::path path, std::vector<std::string> fields = {},
       bool persist_fields = true);

  void set_persist_fields(bool on) { persist_fields_ = on; }
  void set_persist_field(std::string field);
  void register_field(std::string field);
  void register_fields(const std::vector<std::string> &fields);

  void set(std::string_view key, std::string_view value);
  void set(std::string_view key, const std::string &value);
  void set(std::string_view key, const char *value);
  void set(std::string_view key, bool value);
  void set(std::string_view key, int value);
  void set(std::string_view key, long long value);
  void set(std::string_view key, double value);

  template <typename T> void set_enum(std::string_view key, T value) {
    set(key, static_cast<int>(value));
  }

  void set(std::string_view key, const std::vector<double> &values);

  void flush();
  void clear();

  const std::filesystem::path &path() const { return path_; }
  const std::vector<std::string> &fields() const { return fields_; }

private:
  void ensure_field(std::string_view key);
  std::string build_header_line() const;
  void ensure_header_matches_existing_file_or_throw() const;

  static std::string to_string_fixed(double value);
  static std::string csv_escape(const std::string &value);

  std::filesystem::path path_;
  bool persist_fields_ = true;
  std::vector<std::string> fields_;
  std::unordered_set<std::string> seen_fields_;
  std::unordered_set<std::string> persist_fields_selected_;
  std::map<std::string, std::string, std::less<>> current_row_;
  std::map<std::string, std::string, std::less<>> previous_row_;
};

class Registry {
public:
  File &register_file(std::string name, std::filesystem::path path,
                      std::vector<std::string> fields = {},
                      bool persist_fields = true);

  bool contains(std::string_view name) const;
  File *find(std::string_view name);
  const File *find(std::string_view name) const;

  File &at(std::string_view name);
  const File &at(std::string_view name) const;

  void clear();

private:
  std::unordered_map<std::string, File, std::hash<std::string>, std::equal_to<>>
      files_;
};

} // namespace io::csv
