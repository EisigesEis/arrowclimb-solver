## Installation
We actively use the following C++ libraries:
- Eigen3 (header-only, needs to be in includePath)
- Gtest (cmake fetches)
- Gurobi (standard installation path, may need adjustments in top level cmake)
- spdlog (header-only, needs to be in includePath)

Installation via cmake. Compiled binaries within `${CMAKE_BUILD_DIR}/bin`.

## Usage
Single file:
`.\build\bin\uniformsched.exe .\instances_uniform\E1\<some_instance>.dat`

Bulk dataset:
`.\build\bin\uniformsched.exe .\instances_uniform\E1\`

Standard configuration will log output in `result.csv` in execution location. Adjust debug levels and all other compile-time config variables in `.\src\include\Config.h`.