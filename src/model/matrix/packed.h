#pragma once
#include "Eigen/Core"
#include "enumerate/big.h"
#include "enumerate/small.h"
#include "instance/types.h"
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>


using Mat = Eigen::MatrixXi;
using Eigen::VectorXi;
using Scal = int;

class PackedA {
public:
  explicit PackedA(ProblemInstance &inst) : _inst(inst) {
    size_t hatB_width = this->get_hatB_width();
    _hatB = Mat::Zero(hatB_width, inst.n.size());
    _hatB_col_cost_mod_a = VectorXi::Zero(hatB_width);
  }

  /*
    look up element at \calA(row, col)
  */
  Scal operator()(size_t row, size_t col) const;

  /*
    width of \calA
  */
  constexpr size_t cols() const { return _slack_end; }

  /*
    height of \calA
  */
  constexpr size_t rows() const {
    if constexpr (small_id_small_omit_dummy) {
      return _inst.p.size() + (_num_big_cols != 0);
    } else {
      return _inst.p.size() + 1;
    }
  }

  /*
    total number of blocks (small, big, slack combined)
  */
  constexpr size_t num_blocks() const { return _num_blocks; }
  constexpr size_t num_small_blocks() const { return _inst.num_small_machines; }
  constexpr size_t num_big_blocks() const { return _inst.big_residue.size(); }
  constexpr bool has_slack_block() const {
    return _num_blocks > num_small_blocks() + num_big_blocks();
  }

  /*
    returns offset allowing lookup in block with local col index:
    \calA(r_global, c_local + offset) = \calA.block(k)(r_global, c_local)
  */
  constexpr size_t get_block_offset(size_t k) const {
    if (k < _inst.num_small_machines)
      return _small_offset[k];

    if (k < _inst.num_small_machines + _inst.big_residue.size())
      return _num_small_cols + (k - _inst.num_small_machines) * _hatB.cols();

    if (k < _inst.num_small_machines + _inst.big_residue.size() + 1)
      return _big_end;
    return _slack_end;
  }

  // Global column interval [c0, c1) for block k.
  inline std::pair<size_t, size_t> block_col_range(size_t k) const {
    assert(k < _num_blocks);
    const size_t c0 = get_block_offset(k);
    const size_t c1 = get_block_offset(k + 1);
    return {c0, c1};
  }

  // Max coefficient in block k of \calA.
  inline Scal block_maxCoeff(size_t k) const {
    assert(k < _block_Delta.size());
    return _block_Delta[k];
  }

  // \calA.maxCoeff().
  constexpr Scal maxCoeff() const { return _Delta; }

  // Update \calA for the current makespan guess.
  void update();

  // Refresh cached per-block maxima after update().
  void refresh_block_deltas();

  void print(std::ostream &os) const;

  void print_small(std::ostream &os) const;

  void print_big(std::ostream &os) const;

  // Small-Big mixed: zero small-machine dummy jobs and adjust ell.
  static constexpr bool small_id_mixed_zero_dummy = false;

  // Small-only: omit the dummy row and slack block.
  static constexpr bool small_id_small_omit_dummy = true;

  // Small-only: omit slack block.
  static constexpr bool small_id_small_omit_slack = true;

private:
  // Width of all big-machine configs, excluding the dummy job.
  // In contrast to small machine configs, number of big machine configs
  // is pre-determined, to then later write into constant size matrix,
  // instead of intermediate buffer.
  size_t get_hatB_width() const;

  // Look up a small-block entry for global \calA(row, col).
  Scal small(size_t row, size_t col, size_t dummy_row) const;

  // Small block index containing global column i.
  size_t small_row_to_block(size_t i) const;

  // Look up a big-block entry for global \calA(row, col).
  Scal big(size_t row, size_t col, size_t dummy_row) const;

  // Look up a slack-block entry for global \calA(row, col).
  constexpr Scal slack(size_t row, size_t col) const noexcept {
    return ((col - _big_end) == row) *
           (1 + (_inst.a - 1) * (row != _inst.idx_a));
  }

  /*
    store small machine config buffers into memory
  */
  void finalize_small(const std::vector<int> &tmp_conf,
                      const std::vector<int> &tmp_cost);

  const ProblemInstance &_inst; // link to instance
  Mat _hatB;                    // holds all big machine configs
  size_t _num_big_cols = 0; // width of all (real) big machine blocks combined
  VectorXi _hatB_col_cost_mod_a; // cache \forall i: _hatB.col(i) \circ p \mod a
  bool _big_built = false;       // build big machine confs only once

  // width of all (real) small machine blocks combined
  size_t _num_small_cols = 0;
  // holds all small machine configs of biggest small machine
  Mat _hatS = Mat::Zero(0, 0);
  // small block col offset (see get_block_offset() def and comment)
  std::vector<size_t> _small_offset;
  // cache cost of each entry in _hatS:
  // _hatS_cost[i] = _hatS.col(i).dot(inst.p)
  VectorXi _hatS_cost;
  // maximal makespan small machine configs were last generated for
  size_t _last_C = 0;

  // first col which is after last big block
  size_t _big_end,
      // first col which is after slack block
      _slack_end,
      // total number of blocks (small, big, slack combined)
      _num_blocks;

  // maximum coefficient in \calA
  Scal _Delta,
      // maximum coefficient in _hatB and slack (cache this once as stays
      // unchanged after _hatB is generated) then only need to check _hatS,
      // small block dummy row, big block dummy row for updating global \Delta
      _hatB_slack_Delta;
  std::vector<Scal> _block_Delta;
};
