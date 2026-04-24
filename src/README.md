# Source Overview

The main execution path is:

`main.cpp` -> `run.h` -> `bsearch.h`

## Components

`convolve` contains FFT and `SumSet` convolution helpers, including binary powering through `PowTable` and `ensure_pow_level`.

`enumerate` contains DFS enumerators for small and big machine configurations.

`instance` owns parsing and derived instance fields, including modulo relaxation from [2409.04212]:
- `types`: hold parsed and derived instance data
- `parse`: input files to `ProblemInstance`
- `derive`: initialize static fields (`p_max`, `avg_makespan`, ...), plus per-guess fields such as `m.T`

`io` contains CSV logging helpers.

`model/matrix` contains the ILP data structures. `PackedA` exploits matrix redundancies:
- $\hat S := \hat A_{|S|-1} \supseteq \dotsc \supseteq \hat A_{1}$ is regenerated only when the largest small-machine guess grows; otherwise existing columns are segmented.
- $\hat B := \hat A_{|S| + 1} = \dotsc = \hat A_{|S| + |\mathfrak{B}|}$ is constant across runs and built once.
- The dummy row is computed on access from the cost vectors of $\hat S$, $\hat B$, and `m.T`.
- The slack block is a constant diagonal block computed on access.

## Solvers

The AC solvers implement [2501.04859]. `ac_legacy` keeps the earlier AC implementation. The other AC solvers share the `ac_modern` backend and differ in base-table construction:
- `ac_naive`: direct convolution
- `ac_batch`: binary powering via `PowTable` and `ensure_pow_level`
- `ac_fft`: dense FFT grid per block row, repeated for `b_down[k]`, followed by inverse FFT extraction
- `ac_discrep`: discrepancy [1803.04744] solver calls for each $b \in (K \Delta)^r$, as described in [2501.04859]

`gupta` follows the balanced schedule $M_{\sigma}$ over the implicit DAG. `gupta_batch` compresses consecutive blocks in that schedule and applies binary powering.

`discrep` embeds the discrepancy DP [1803.04744] into an FFT grid and convolves $\ell$ layers.

`gur` uses Gurobi as a correctness reference.

## Results Notes

All regular solvers were tested on dataset `E1`.

`build/plots/mat_update_vs_gur_bands_detail.png` Small benchmarks or tests of single components cannot account for all side effects occurring in real data. To boost test iterations on real data (E1 benchmark takes 4h+), ILP generation time was reduced to the absolute minimum.

`build/plots/proxy_s/(selected_solver|branch)_.*\.png`: Binary powering is the strongest practical optimization in these experiments. `gupta_batch` and `ac_batch` are the fastest solvers on most of `E1`. Gupta benefits less because large batches appear only in skewed sections of the balanced schedule. Further optimization is possible with reusing schedule patterns $M_{\sigma} = AAABAAABAAA = (A^{(3)}B)^{(2)} + A^{(3)}$ can reduce the number of convolutions, but such patterns are instance-dependent. The current comparison covers feasibility, so our benchmarks do not evaluate optimization sections from the Gupta paper.

`build/plots/fft_utilization_overview.png`: `discrep`, `ac_fft`, and `ac_discrep` apply FFT to sparse signals. Dummy inflation expands the dense grid, which makes Rohwedder's modulo relaxation expensive for FFT-based methods. `discrep` encodes machine counts directly in the FFT grid, while `ac_fft` encodes them through repeated convolution by `b_down[k]`. This additional grid size is visible when solvers are ordered by `machine_types`: among the measured solvers, only `discrep` correlates strongly. Several `discrep` instances exceed the digit grid limit (`50 million`). Average useful-grid utilization is about `1.2%` for `discrep` and `8.4%` for `ac_fft`, and practical runtime mainly increases with un-utilized grid size, so most FFT work is spent on empty grid space.

`build/plots_acdc/proxy_s/selected_solver_bands.png`: `ac_batch` and `ac_discrep` were compared on selected instances: 20 from the first batch via `M3_N6_U1_20_0((0|1)\d|20)\.dat$` and 5 from the last batch via `M5_N25_U20_50_00[1-5]\.dat$`. `ac_discrep` consistently adds about `10^2 ms` of overhead while total `ac_discrep` processing times are up to `10^5 ms`. AC needs algorithms that extract all feasible RHS values for a block directly, instead of single feasibility oracles like `discrep`, as the sheer number of calls $(K \Delta)^r$ outweighs any optimization spend in the underlying single feasibility oracle.

## Future Work

The lower bound of binary search can likely be tightened. For now we reject any makespan guess with negative $\ell = \text{total\_capacity} - \text{total\_load}$ at `derive_for_guess`.

The remaining unexplained anomalies are the advantage of `gupta` over `ac_batch` on lower-complexity instances and the binary-powering spike on highly complex instances.

Promising follow-up work:
- Use matrix redundancies from bachelor thesis directly within solver algorithms.
- Reuse information across binary-search iterations.
- Reduce dummy inflation, i.e. by limiting $\ell$ to big-machine dummy jobs.
- Treat the slack block as a divisibility constraint, outside combinatorial ILP.
