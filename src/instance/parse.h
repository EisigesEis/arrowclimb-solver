#pragma once
#include "instance/derive.h"
#include "instance/types.h"
#include <cstddef>
#include <filesystem>
#include <istream>
#include <optional>
#include <string>

struct ParseError {
  std::string message;
  std::size_t line = 0;
};
struct ParseResult {
  bool ok = false;
  ProblemInstance inst;
  std::optional<ParseError> err;
};

ParseResult parse_file_raw(const std::filesystem::path &path);
ParseResult parse_stream_raw(std::istream &in, std::string instance_name);

inline ParseResult parse_file(const std::filesystem::path &path) {
  auto R = parse_file_raw(path);
  if (R.ok) {
    R.inst.sort();
    derive_inital(R.inst);
  }
  return R;
}

inline ParseResult parse_stream(std::istream &in, std::string name) {
  auto R = parse_stream_raw(in, std::move(name));
  if (R.ok) {
    R.inst.sort();
    derive_inital(R.inst);
  }
  return R;
}
