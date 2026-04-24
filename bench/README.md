## Overview of Benchmarks
### big enumerate
`.\build\bin\bench_big_odometer.exe --reps 20 --warmup 1 ` Test the advantage of reordering and naive versions with batch cost against incremental cost update (B1). Includes correctness checks (count or hash compare). Ultimately B1 wins here.

### Small enumerate
`.\build\bin\bench_small_dp.exe --reps 20 --warmup 3 --sleep 100` Test DFS tables vs. incremental remaining capacity. Includes correctness checks (count or hash compare). The more advanced tables are close to incremental remainder. Considering this enumerate is not called often (we reuse on smaller biggest small machine capacity by segmenting small configs from older guess with binary search) and that tables grow too large on higher capacities, we use S7.
#### Small dp build
`.\build\bin\bench_small_dp_build.exe` Test the speed of build vs. recover remainder table. Ultimately went unused as S7 is superior (see above).

### gupta msig batching
`.\build\bin\bench_gupta_sched.exe` For our `gupta_batch` we want to have a Msig of the form [(e_0, count_0), (e_1, count_1), ...]. We compare merging to batches of existing schedule against methods with better theoretical runtime using for example heap. In our data regime, merging existing schedule has proven superior, so it is used in production.

### Deduplication
`.\build\bin\bench_discr_merge.exe --maxn 1024 --cases 20 --repeat 1 --warmup 1 > merge.csv` compare sort+unique against hashing to find turning point where hashing beats sort+unique on larger data. Plot was used for `SumSet` turning point.
`.\build\bin\bench_discr_helper.exe > out.csv` Covers the same comparison. I believe `bench_discr_merge` output was used for the `SumSet` threshold because it can be visualized with `view_merge_sets.html` and `out.csv` does not have a visualizer.
