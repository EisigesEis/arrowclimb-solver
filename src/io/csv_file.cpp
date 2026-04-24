#include "io/csv_file.h"

#include <iomanip>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace io::csv {

File::File(std::filesystem::path path, std::vector<std::string> fields,
           bool persist_fields)
    : path_(std::move(path)), persist_fields_(persist_fields),
      fields_(std::move(fields)) {
  seen_fields_.insert(fields_.begin(), fields_.end());
}

void File::set_persist_field(std::string field) {
  register_field(field);
  persist_fields_selected_.insert(std::move(field));
}

void File::register_field(std::string field) {
  if (seen_fields_.insert(field).second) {
    fields_.push_back(std::move(field));
  }
}

void File::register_fields(const std::vector<std::string> &fields) {
  for (const auto &field : fields) {
    register_field(field);
  }
}

void File::set(std::string_view key, std::string_view value) {
  ensure_field(key);
  current_row_[std::string(key)] = std::string(value);
}

void File::set(std::string_view key, const std::string &value) {
  set(key, std::string_view(value));
}

void File::set(std::string_view key, const char *value) {
  set(key, value ? std::string_view(value) : std::string_view());
}

void File::set(std::string_view key, bool value) {
  ensure_field(key);
  current_row_[std::string(key)] = value ? "1" : "0";
}

void File::set(std::string_view key, int value) {
  ensure_field(key);
  current_row_[std::string(key)] = std::to_string(value);
}

void File::set(std::string_view key, long long value) {
  ensure_field(key);
  current_row_[std::string(key)] = std::to_string(value);
}

void File::set(std::string_view key, double value) {
  ensure_field(key);
  current_row_[std::string(key)] = to_string_fixed(value);
}

void File::set(std::string_view key, const std::vector<double> &values) {
  ensure_field(key);

  std::string joined;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      joined.push_back(';');
    }
    joined += to_string_fixed(values[i]);
  }

  current_row_[std::string(key)] = std::move(joined);
}

void File::flush() {
  if (const auto parent = path_.parent_path(); !parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      throw std::runtime_error("Failed to create CSV parent directory '" +
                               parent.string() + "': " + ec.message());
    }
  }

  std::error_code exists_ec;
  const bool exists = std::filesystem::exists(path_, exists_ec);
  if (exists_ec) {
    throw std::runtime_error("Failed to stat CSV file '" + path_.string() +
                             "': " + exists_ec.message());
  }

  bool has_data = false;
  if (exists) {
    std::error_code size_ec;
    has_data = std::filesystem::file_size(path_, size_ec) > 0;
    if (size_ec) {
      throw std::runtime_error("Failed to get CSV file size for '" +
                               path_.string() + "': " + size_ec.message());
    }
    if (has_data) {
      ensure_header_matches_existing_file_or_throw();
    }
  }

  std::ofstream out(path_, std::ios::app);
  if (!out) {
    throw std::runtime_error("Failed to open CSV file for append: " +
                             path_.string());
  }

  if (!has_data) {
    out << build_header_line() << "\n";
  }

  for (std::size_t i = 0; i < fields_.size(); ++i) {
    if (i != 0) {
      out << ",";
    }

    const auto &field = fields_[i];
    std::string value;

    if (const auto it = current_row_.find(field); it != current_row_.end()) {
      value = it->second;
    } else if (persist_fields_ || persist_fields_selected_.contains(field)) {
      if (const auto it = previous_row_.find(field);
          it != previous_row_.end()) {
        value = it->second;
      }
    }

    out << csv_escape(value);
  }
  out << "\n";

  if (persist_fields_) {
    for (const auto &[key, value] : current_row_) {
      previous_row_[key] = value;
    }
  } else if (!persist_fields_selected_.empty()) {
    for (const auto &[key, value] : current_row_) {
      if (persist_fields_selected_.contains(key)) {
        previous_row_[key] = value;
      }
    }
  }

  current_row_.clear();
}

void File::clear() {
  current_row_.clear();
  previous_row_.clear();
}

void File::ensure_field(std::string_view key) {
  register_field(std::string(key));
}

std::string File::to_string_fixed(double value) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(std::numeric_limits<double>::max_digits10)
         << value;
  return stream.str();
}

std::string File::csv_escape(const std::string &value) {
  bool needs_quotes = false;
  for (char c : value) {
    if (c == ',' || c == '"' || c == '\n' || c == '\r') {
      needs_quotes = true;
      break;
    }
  }

  if (!needs_quotes) {
    return value;
  }

  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (char c : value) {
    if (c == '"') {
      escaped.push_back('"');
    }
    escaped.push_back(c);
  }
  escaped.push_back('"');
  return escaped;
}

std::string File::build_header_line() const {
  std::string header;
  for (std::size_t i = 0; i < fields_.size(); ++i) {
    if (i != 0) {
      header.push_back(',');
    }
    header += csv_escape(fields_[i]);
  }
  return header;
}

void File::ensure_header_matches_existing_file_or_throw() const {
  std::ifstream in(path_);
  if (!in) {
    throw std::runtime_error("Failed to open CSV file for header validation: " +
                             path_.string());
  }

  std::string existing_header;
  std::getline(in, existing_header);
  if (!existing_header.empty() && existing_header.back() == '\r') {
    existing_header.pop_back();
  }

  const std::string expected_header = build_header_line();
  if (existing_header != expected_header) {
    throw std::runtime_error("CSV header mismatch for '" + path_.string() +
                             "'. Existing: [" + existing_header +
                             "], expected: [" + expected_header + "]");
  }
}

File &Registry::register_file(std::string name, std::filesystem::path path,
                              std::vector<std::string> fields,
                              bool persist_fields) {
  auto [it, inserted] =
      files_.try_emplace(std::move(name), path, fields, persist_fields);
  if (!inserted) {
    it->second = File(std::move(path), std::move(fields), persist_fields);
  }
  return it->second;
}

bool Registry::contains(std::string_view name) const {
  return files_.contains(std::string(name));
}

File *Registry::find(std::string_view name) {
  if (auto it = files_.find(std::string(name)); it != files_.end()) {
    return &it->second;
  }
  return nullptr;
}

const File *Registry::find(std::string_view name) const {
  if (auto it = files_.find(std::string(name)); it != files_.end()) {
    return &it->second;
  }
  return nullptr;
}

File &Registry::at(std::string_view name) {
  return files_.at(std::string(name));
}

const File &Registry::at(std::string_view name) const {
  return files_.at(std::string(name));
}

void Registry::clear() { files_.clear(); }

} // namespace io::csv
