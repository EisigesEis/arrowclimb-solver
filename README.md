Be sure to also read the other READMEs after this one:
- [bench/README.md](bench) - benchmark overview with resulting decisions
- [instances/README.md](instances) - file structure adjustments compared to previous project
- [src/README.md](src) - code overview with results and future work

## Installation

### Dependencies
| Library | Installation Method | Tested on Version |
|-|-|-|
| GTest | Pulled via CMake on build | `1.17.0` |
| Gurobi | [Standalone install](https://www.gurobi.com/product/download-center) | `12.0.3` |
| Abseil | Install via vcpkg | `20260107.1#2` |
| Eigen3 | Install via vcpkg | `3.4.0` |
| spdlog | Install via vcpkg | `1.17.0` |
| FFTW3 | Install via vcpkg | `3.3.10` |

### CMake toolchain
This project used [vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started). Follow their documentation, then install the required packages:
- `%VCPKG_ROOT%\vcpkg.exe install abseil eigen3 spdlog fftw3`

The checked-in `default` preset expects `VCPKG_ROOT` to point at the same vcpkg checkout where the packages above were installed. It forwards that path both as `CMAKE_TOOLCHAIN_FILE` and `UNIFORMSCHED_VCPKG_ROOT`, so `find_package(...)` also searches `<vcpkg>/installed/x64-windows` during reconfigure. If your dependencies live elsewhere, adjust [CMakePresets.json](./CMakePresets.json) or pass the paths manually. For manual FFTW installs outside vcpkg, pass `-DFFTW3_ROOT=<path>`. For Gurobi, set `GUROBI_HOME` or pass `-DGUROBI_DIR=<path>`.

### Build
Build via CMake presets:
- `cmake --preset default`
- `cmake --build --preset default`

Compiled binaries will live within `${CMAKE_BUILD_DIR}/bin`.

## Support
The code was tested on Windows `24H2 26100.8246` with clang `22.1.0`, MSVC `19.50.35728` and the stated library versions under Dependencies. Other versions and operating systems are experimental and may break. This code is provided as is, without any guarantees of safety or fitness for a particular purpose.

## Usage
`uniformsched.exe` solves one instance file or all matching instance files in a folder.

Command:
- `.\build\bin\uniformsched.exe [-l] [-c] [-acdc] <path\to\instance.dat|folder> [filename-regex]`

Examples:
- all solvers test single file `.\build\bin\uniformsched.exe -l .\instances\E1\M3_N15_U1_20_001.dat`
- benchmark all solvers on dataset `.\build\bin\uniformsched.exe -c .\instances\E1\`
- benchmark all solvers on dataset, only first 20 files of each batch `.\build\bin\uniformsched.exe -c .\instances\E1\ '_(0[0-1]\d|\d|020)\.dat$'`
- benchmark ac_discrepancy vs. ac_batch on selected first/last batch files `.\build\bin\uniformsched.exe .\instances\E1 -acdc '(M3_N6_U1_20_0((0|1)\d|20)|M5_N25_U20_50_00[1-5])\.dat$'`

Flags:
- `-l` enable console logging and disable csv
- `-c` write `main.csv` and disable logging
- `-l -c` (default, so same as not supplying `-l -c`) enable both console logging and CSV output
- `-acdc` run only ac_discrepancy and ac_batch and output to `acdc.csv`

When a folder is provided, the optional `filename-regex` filters which files are processed.

### Plots
Use the plotting script as follows:
- `python scripts\plot_main_suite.py --input instances\E1\main.csv --x-axis proxy_S,ell,machine_types,job_types --force`
- `python scripts\plot_main_suite.py --input instances\E1\acdc.csv --x-axis proxy_S,ell,machine_types,job_types --force`
