#pragma once
#include "instance/types.h"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace ac::debug {

// print a compact head...tail preview, e.g. [3,5,8, ..., 21,34] (len=17)
inline void preview_vec(std::ostream &os, const std::vector<u32> &v,
                        std::size_t max_elems = 8) {
  os << '[';
  const std::size_t n = v.size();
  if (n == 0) {

  } else if (n <= max_elems) {
    for (std::size_t i = 0; i < n; ++i) {
      if (i)
        os << ',';
      os << v[i];
    }
  } else {
    const std::size_t k = max_elems / 2;
    for (std::size_t i = 0; i < k; ++i) {
      if (i)
        os << ',';
      os << v[i];
    }
    os << ", …, ";
    for (std::size_t i = n - (max_elems - k); i < n; ++i) {
      if (i != n - (max_elems - k))
        os << ',';
      os << v[i];
    }
  }
  os << "] (len=" << n << ")";
}

// Parsed-instance debug dump
inline void dump_parsed(const ProblemInstance &I, std::ostream &os = std::cout,
                        std::size_t max_preview = 8) {
  os << "ProblemInstance (parsed)\n";
  os << "  name: " << I.name << "\n";
  os << "  M (machine types): " << I.M << "   N (job types): " << I.N
     << "\n";

  // shape
  os << "  shape machines OK: " << (I.machines_shape_ok() ? "yes" : "NO")
     << "\n";
  os << "  shape jobs OK:     " << (I.jobs_shape_ok() ? "yes" : "NO") << "\n";

  os << "  m (counts per machine type): ";
  preview_vec(os, I.m, max_preview);
  os << "\n";
  os << "  s (speed  per machine type): ";
  preview_vec(os, I.s, max_preview);
  os << "\n";
  if (!I.t.empty()) {
    os << "  t (cap    per machine type): ";
    preview_vec(os, I.t, max_preview);
    os << "\n";
  } else {
    os << "  t (cap    per machine type): [omitted → defaults to 0] (len=0)\n";
  }
  os << "  n (counts per job type):     ";
  preview_vec(os, I.n, max_preview);
  os << "\n";
  os << "  p (size   per job type):     ";
  preview_vec(os, I.p, max_preview);
  os << "\n";
  os.flush();
}

} // namespace ac::debug
