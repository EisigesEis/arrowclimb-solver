#pragma once

#include <gtest/gtest.h>

#include "instance/derive.h"
#include "instance/parse.h"

#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace test_support {

inline ProblemInstance make_midpoint_instance(const std::filesystem::path &path) {
  const ParseResult parsed = parse_file(path);
  EXPECT_TRUE(parsed.ok) << path.string();

  ProblemInstance inst = parsed.inst;
  const long long k_mid =
      inst.scaled_lb + (inst.scaled_ub - inst.scaled_lb) / 2;
  EXPECT_FALSE(derive_for_guess(inst, k_mid)) << path.string();
  return inst;
}

inline std::optional<ProblemInstance>
try_make_midpoint_instance(const std::filesystem::path &path) {
  const ParseResult parsed = parse_file(path);
  if (!parsed.ok)
    return std::nullopt;

  ProblemInstance inst = parsed.inst;
  const long long k_mid =
      inst.scaled_lb + (inst.scaled_ub - inst.scaled_lb) / 2;
  if (derive_for_guess(inst, k_mid))
    return std::nullopt;
  return inst;
}

inline ProblemInstance make_instance_at_guess(std::string_view contents,
                                              long long scaled_guess,
                                              std::string name) {
  std::string storage(contents);
  std::istringstream in(storage);
  const ParseResult parsed = parse_stream(in, std::move(name));
  EXPECT_TRUE(parsed.ok)
      << (parsed.err.has_value() ? parsed.err->message : std::string{});

  ProblemInstance inst = parsed.inst;
  EXPECT_FALSE(derive_for_guess(inst, scaled_guess));
  return inst;
}

} // namespace test_support
