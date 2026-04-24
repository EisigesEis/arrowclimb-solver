#include "instance/parse.h"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string_view>

namespace {

constexpr char SEP = ',';
constexpr char COMMENT = '#';
// Forbids extra data after the declared N job lines
#define PARSE_STRICT_TRAILING_CONTENT 0

inline std::string_view ltrim(std::string_view s) {
  std::size_t i = 0, n = s.size();
  while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r'))
    ++i;
  return s.substr(i);
}
inline std::string_view rtrim(std::string_view s) {
  std::size_t n = s.size();
  while (n && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\r'))
    --n;
  return s.substr(0, n);
}
inline std::string_view trim(std::string_view s) { return rtrim(ltrim(s)); }
inline std::string_view strip_inline_comment(std::string_view s) {
  const auto pos = s.find(COMMENT);
  if (pos == std::string_view::npos)
    return s;
  return s.substr(0, pos);
}

inline bool split_once(std::string_view s, char sep, std::string_view &lhs,
                       std::string_view &rhs) {
  const auto pos = s.find(sep);
  if (pos == std::string_view::npos)
    return false;
  lhs = trim(s.substr(0, pos));
  rhs = trim(s.substr(pos + 1));
  return true;
}

inline bool getline_counted(std::istream &in, std::string &dst,
                            std::size_t &line_no) {
  if (!std::getline(in, dst))
    return false;
  ++line_no;
  return true;
}

inline bool is_skippable(std::string_view sv) {
  if (sv.empty())
    return true;

  for (char c : sv) {
    if (c == ' ' || c == '\t' || c == '\r')
      continue;
    return c == COMMENT;
  }
  return false;
}

inline bool parse_u32_token(std::string_view sv, u32 &out) {
  unsigned long long tmp = 0ULL;
  const char *first = sv.data();
  const char *last = first + sv.size();
  auto [p, ec] = std::from_chars(first, last, tmp);
  if (ec != std::errc() || p != last || tmp > std::numeric_limits<u32>::max())
    return false;
  out = static_cast<u32>(tmp);
  return true;
}

// Parse into a ProblemInstance's raw fields
ParseResult parse_core(std::istream &in, std::string instance_name) {
  ParseResult R;
  R.ok = false;
  R.err.reset();
  R.inst.clear_raw();
  R.inst.name = std::move(instance_name);

  auto &m = R.inst.m;
  auto &s = R.inst.s;
  auto &t = R.inst.t;
  auto &n = R.inst.n;
  auto &p = R.inst.p;

  auto &M = R.inst.M;
  auto &N = R.inst.N;

  std::string line;
  std::size_t line_no = 0;
  std::string_view sv;

  auto fail = [&](std::string msg) -> ParseResult {
    R.ok = false;
    R.err = ParseError{std::move(msg), line_no};
    return R;
  };

  auto next_data_line = [&](std::string &out, std::string_view &sv) -> bool {
    while (getline_counted(in, out, line_no)) {
      sv = trim(strip_inline_comment(out));
      if (!is_skippable(sv))
        return true;
    }
    return false;
  };

  // --- Read machine type count M
  if (!next_data_line(line, sv))
    return fail("Missing machine count (M).");
  u32 Mcnt = 0;
  if (!parse_u32_token(sv, Mcnt))
    return fail("Invalid machine count (M).");
  M = Mcnt;
  m.resize(Mcnt);
  s.resize(Mcnt);
  t.setZero(Mcnt);

  // --- Read M lines: "m,s"
  for (u32 i = 0; i < Mcnt; /* ++i when accepted */) {
    if (!getline_counted(in, line, line_no))
      return fail("Unexpected EOF in machine section.");
    sv = trim(strip_inline_comment(line));
    if (is_skippable(sv))
      continue;

    std::string_view a, b;
    if (!split_once(sv, SEP, a, b)) {
      u32 single = 0;
      if (parse_u32_token(sv, single)) {
        return fail("Expected 'm,s' for machine line, but found a single "
                    "integer. M may be too large or a machine line is missing.");
      }
      return fail("Expected 'm,s' for machine line.");
    }

    u32 mv = 0, svv = 0;
    if (!parse_u32_token(a, mv) || !parse_u32_token(b, svv))
      return fail("Invalid integer in machine line.");

    m[i] = mv;
    s[i] = svv;
    ++i;
  }

  // --- Read job type count N
  if (!next_data_line(line, sv))
    return fail("Missing job count (N).");
  u32 Ncnt = 0;
  if (!parse_u32_token(sv, Ncnt))
    return fail("Invalid job count (N).");
  N = Ncnt;
  n.resize(Ncnt);
  p.resize(Ncnt);

  // --- Read N lines: "n,p"
  for (u32 i = 0; i < Ncnt; /* ++i when accepted */) {
    if (!getline_counted(in, line, line_no))
      return fail("Unexpected EOF in job section.");
    sv = trim(strip_inline_comment(line));
    if (is_skippable(sv))
      continue;

    std::string_view a, b;
    if (!split_once(sv, SEP, a, b)) {
      u32 single = 0;
      if (parse_u32_token(sv, single)) {
        return fail("Expected 'n,p' for job line, but found a single integer. "
                    "N may be too large or a job line is missing.");
      }
      return fail("Expected 'n,p' for job line.");
    }

    u32 nv = 0, pv = 0;
    if (!parse_u32_token(a, nv) || !parse_u32_token(b, pv))
      return fail("Invalid integer in job line.");

    n[i] = nv;
    p[i] = pv;
    ++i;
  }

  // Reject trailing non-comment/non-empty content
#if PARSE_STRICT_TRAILING_CONTENT
  while (getline_counted(in, line, line_no)) {
    sv = trim(strip_inline_comment(line));
    if (is_skippable(sv))
      continue;
    return fail("Unexpected trailing content after declared job section.");
  }
#endif

  // Final shape checks
  if (!R.inst.machines_shape_ok())
    return fail(
        "Machines shape mismatch: m.size()!=s.size() or t has wrong size.");
  if (!R.inst.jobs_shape_ok())
    return fail("Jobs shape mismatch: n.size()!=p.size().");
  if (R.inst.s.size() == 0)
    return fail("No machines parsed.");
  if (R.inst.p.size() == 0)
    return fail("No jobs parsed.");

  R.ok = true;
  return R;
}

} // namespace

ParseResult parse_file_raw(const std::filesystem::path &path) {
  std::ifstream fin(path, std::ios::in | std::ios::binary);
  ParseResult R;
  if (!fin) {
    R.ok = false;
    R.err = ParseError{"Failed to open file: " + path.string(), 0};
    return R;
  }
  return parse_core(fin, path.stem().string());
}

ParseResult parse_stream_raw(std::istream &in, std::string instance_name) {
  return parse_core(in, std::move(instance_name));
}
