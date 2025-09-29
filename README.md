## Installation
We actively use the following C++ libraries:
- Eigen3 (header-only, needs to be in includePath)
- Gtest (cmake fetches)
- Gurobi (standard installation path, may need adjustments in top level cmake)
- spdlog (header-only, needs to be in includePath)

Build via cmake.
`mkdir build`
`cmake -DCMAKE_BUILD_TYPE:STRING=Release --no-warn-unused-cli -S . -B ./build`

Compiled binaries will live within `${CMAKE_BUILD_DIR}/bin`.

## Support
The code was tested on Windows `24H2 26100.6584` with clang `21.1.0`, MSVC `19.29.30159`, Gurobi `12.0.3`, Eigen3 `3.4.0`, absl `LTS 20250814.0`, gtest `1.17.0`. Other version or operating system support is experimental and may break.

## Usage
Single file:
`.\build\bin\uniformsched.exe .\instances_uniform\E1\<some_instance>.dat`

Bulk dataset:
`.\build\bin\uniformsched.exe .\instances_uniform\E1\`

Standard configuration will log output in `result.csv` in execution location. Adjust debug levels and all other compile-time config variables in `.\src\include\Config.h`.