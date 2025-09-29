#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>

#include "Dedup.h"
#include "Config.h"

using Eigen::MatrixXi;
using Eigen::VectorXi;
using Eigen::Ref;

namespace {
#if (DBG_BT || DBG_CAPS || DBG_COLS || DBG_LEVELS || DBG_S0)
static inline std::string vec_str(const VectorXi &v) {
  std::ostringstream os;
  os << "[" << v.transpose() << "]";
  return os.str();
}
static inline std::string mat_shape_str(const MatrixXi &A) {
  std::ostringstream os;
  os << "(" << A.rows() << "x" << A.cols() << ")";
  return os.str();
}
static inline void dump_mat_head(const char* name, const MatrixXi& A, int rows=6, int cols=8) {
  std::cerr << name << " " << mat_shape_str(A) << "\n";
  const int r = std::min(rows, (int)A.rows());
  const int c = std::min(cols, (int)A.cols());
  for (int i = 0; i < r; ++i) {
    std::cerr << "  ";
    for (int j = 0; j < c; ++j) {
      std::cerr << A(i,j) << (j+1==c? "": " ");
    }
    if (c < A.cols()) std::cerr << " ...";
    std::cerr << "\n";
  }
  if (r < A.rows()) std::cerr << "  ...\n";
}
static inline void dump_vec_labeled(const char* name, const VectorXi& v) {
  std::cerr << name << " (n=" << v.size() << ") = " << vec_str(v) << "\n";
}
#endif

static thread_local DedupScratch g_dedup_ctx;

}

std::vector<VectorXi>
compute_base_table_for_block(const MatrixXi &A_k,
                             Ref<const VectorXi> target,
                             int s)
{
  const int r = A_k.rows();
  const int t = A_k.cols();

#if DBG_BT
  std::cerr << "\n[BT] enter compute_base_table_for_block  A_k=" << mat_shape_str(A_k)
            << "  s=" << s << "  target.size=" << target.size() << "\n";
  dump_mat_head("A_k", A_k);
#endif

  if (target.size() != r) {
#if DBG_BT
    std::cerr << "[BT][ERROR] shape mismatch: target.size()=" << target.size()
              << " vs A_k.rows()=" << r << "\n";
#endif
    assert(false && "compute_base_table_for_block: target.size() must equal A_k.rows()");
  }

  if (r == 0 || t == 0) {
#if DBG_BT
    std::cerr << "[BT] degenerate block (r==0 || t==0). ";
    if (s == 0) std::cerr << "Return {[]}.\n"; else std::cerr << "Return {}.\n";
#endif
    if (s == 0) return { VectorXi() };
    return {};
  }

  if (s == 0) {
#if DBG_S0
    std::cerr << "[BT][s==0] feasible: return single zero vector of size " << r << "\n";
#endif
    return { VectorXi::Zero(r) };
  }

  const VectorXi caps = target.array()
             .min( (s * A_k.rowwise().maxCoeff().array()).max(0) )
             .matrix();

#if DBG_CAPS
  dump_vec_labeled("caps  ", caps);
#endif

  // --- filter usable columns (each column must be <= caps component-wise) ---
  std::vector<int> usable_cols;
  usable_cols.reserve(t);
  const auto capsA = caps.array();

  for (int j = 0; j < t; ++j) {
    if ( (A_k.col(j).array() <= capsA).all() ) {
      usable_cols.push_back(j);
    }
  }

#if DBG_COLS
  std::cerr << "[BT] usable_cols " << usable_cols.size() << "/" << t << "\n";
  if (!usable_cols.empty()) {
    std::cerr << "     first: " << vec_str(A_k.col(usable_cols.front())) << "\n";
    if (usable_cols.size() > 1)
      std::cerr << "     last : " << vec_str(A_k.col(usable_cols.back())) << "\n";
  } else {
    std::cerr << "     none under caps -> infeasible.\n";
  }
#endif

  if (usable_cols.empty()) return {};

  // --- enumerate exactly s picks with repetition, prune by caps, dedupe each level ---
  std::vector<VectorXi> cur;
  cur.reserve(1 + usable_cols.size());
  cur.push_back(VectorXi::Zero(r)); // 0 picks

  for (int pick = 1; pick <= s; ++pick) {
    std::vector<VectorXi> next;
    next.reserve(cur.size() * usable_cols.size());

    // Build next layer by adding one more column (with repetition allowed).
    for (const VectorXi& base : cur) {
      const auto remA = (caps - base).array();
      for (const int j : usable_cols) {
        const auto col = A_k.col(j);
        if ((col.array() <= remA).all()) {
          next.emplace_back(base + col);
        }
      }
    }

    dedupe_inplace(next, r, g_dedup_ctx);

#if DBG_LEVELS
    std::cerr << "[BT] level " << pick
              << " size=" << next.size()
              << " (after prune+dedup)  from prev=" << cur.size()
              << "  usable_cols=" << usable_cols.size() << "\n";
    if (!next.empty()) {
      const int show = std::min<int>(10, (int)next.size());
      std::cerr << "     samples (" << show << "): ";
      for (int i = 0; i < show; ++i) std::cerr << vec_str(next[i]) << " ";
      std::cerr << "\n";
    }
#endif

    cur.swap(next);
    if (cur.empty()) {
#if DBG_LEVELS
      std::cerr << "[BT] exhausted at level " << pick << " — infeasible ahead.\n";
#endif
      break;
    }
  }

#if DBG_BT
  std::cerr << "[BT] exit with " << cur.size() << " vectors (exact " << s << " picks)\n";
#endif
  return cur;
}