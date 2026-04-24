cmake_minimum_required(VERSION 3.20)

project(uniformSched LANGUAGES CXX)

set(UNIFORMSCHED_VCPKG_ROOT "" CACHE PATH
    "Optional vcpkg root used to derive the toolchain file and CMAKE_PREFIX_PATH during configure.")

if(UNIFORMSCHED_VCPKG_ROOT)
  set(UNIFORMSCHED_VCPKG_TOOLCHAIN
      "${UNIFORMSCHED_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
  set(UNIFORMSCHED_VCPKG_TRIPLET_DIR
      "${UNIFORMSCHED_VCPKG_ROOT}/installed/x64-windows")

  if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND
     EXISTS "${UNIFORMSCHED_VCPKG_TOOLCHAIN}")
    set(CMAKE_TOOLCHAIN_FILE "${UNIFORMSCHED_VCPKG_TOOLCHAIN}" CACHE FILEPATH
        "vcpkg toolchain file" FORCE)
  endif()

  if(EXISTS "${UNIFORMSCHED_VCPKG_TRIPLET_DIR}")
    list(PREPEND CMAKE_PREFIX_PATH "${UNIFORMSCHED_VCPKG_TRIPLET_DIR}")
  endif()
endif()

# ---- Global toolchain settings ------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

cmake_policy(SET CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# option(ENABLE_CLANG_TIDY "Enable clang-tidy analysis during build" OFF)

# if(ENABLE_CLANG_TIDY)
#   find_program(CLANG_TIDY_EXE NAMES clang-tidy)

#   if(CLANG_TIDY_EXE)
#     message(STATUS "clang-tidy enabled: ${CLANG_TIDY_EXE}")
#     set(CMAKE_CXX_CLANG_TIDY
#         ${CLANG_TIDY_EXE};
#         -extra-arg=-std=c++20
#     )
#   else()
#     message(WARNING "clang-tidy requested but not found")
#   endif()
# endif()

# Prefer new timestamp policy for FetchContent
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# ---- GUROBI -------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
find_package(GUROBI REQUIRED)
if(GUROBI_FOUND)
  message(STATUS "Found Gurobi include dir: ${GUROBI_INCLUDE_DIRS}")
  message(STATUS "Found Gurobi C lib:       ${GUROBI_LIBRARY}")
  if(NOT GUROBI_CXX_LIBRARY)
    message(FATAL_ERROR "Gurobi C++ library not found. Set GUROBI_HOME or GUROBI_DIR.")
  else()
    message(STATUS "Found Gurobi C++ lib:     ${GUROBI_CXX_LIBRARY}")
  endif()
else()
  message(FATAL_ERROR "Gurobi not found")
endif()

include_directories(${GUROBI_INCLUDE_DIRS})

# ---- GoogleTest via FetchContent ----------------------------------------------
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "")
include(FetchContent)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)
set(gtest_disable_pthreads ON CACHE BOOL "Disable pthreads in GoogleTest")

# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
# )
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(googletest)

# Silence char-conversion diagnostics in gtest only
foreach(tgt IN ITEMS gtest gtest_main gmock gmock_main)
  if (TARGET ${tgt})
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_compile_options(${tgt} PRIVATE -Wno-character-conversion -Wno-error=character-conversion)
    endif()
    if (MSVC)
      target_compile_options(${tgt} PRIVATE /WX-)
    endif()
  endif()
endforeach()

# ---- Abseil (prebuilt) ---------------------------
find_package(absl CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  function(uniformsched_fix_absl_link_flags)
    get_property(_imported_targets DIRECTORY PROPERTY IMPORTED_TARGETS)
    foreach(_tgt IN LISTS _imported_targets)
      if(NOT _tgt MATCHES "^absl::")
        continue()
      endif()

      get_target_property(_libs ${_tgt} INTERFACE_LINK_LIBRARIES)
      if(NOT _libs)
        continue()
      endif()

      set(_new_libs "${_libs}")
      string(REPLACE "$<LINK_ONLY:-ignore:4221>" "" _new_libs "${_new_libs}")
      string(REPLACE "-ignore:4221" "" _new_libs "${_new_libs}")

      if(NOT _new_libs STREQUAL _libs)
        set_target_properties(${_tgt} PROPERTIES
          INTERFACE_LINK_LIBRARIES "${_new_libs}")
        target_link_options(${_tgt} INTERFACE "LINKER:/ignore:4221")
      endif()
    endforeach()
  endfunction()

  uniformsched_fix_absl_link_flags()
endif()

# --- Abseil (fetched) -----------------------------
# set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
# set(ABSL_ENABLE_INSTALL ON CACHE BOOL "" FORCE)
# set(ABSL_USE_EXTERNAL_GOOGLETEST ON CACHE BOOL "" FORCE)
# FetchContent_Declare(
#   abseil
#   GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
#   GIT_TAG        20250814.1
# )
# FetchContent_MakeAvailable(abseil)


# ---- FFTW (imported targets) --------------------------------------------------
set(FFTW3_ROOT "" CACHE PATH
    "Optional FFTW install root for manual installs outside the active toolchain.")
find_path(FFTW3_INCLUDE fftw3.h
  HINTS "${FFTW3_ROOT}" ENV FFTW3_ROOT
  PATH_SUFFIXES include)
find_library(FFTW3_LIBRARY
  NAMES libfftw3-3 libfftw3 fftw3
  HINTS "${FFTW3_ROOT}" ENV FFTW3_ROOT
  PATH_SUFFIXES lib)
find_library(FFTW3F_LIBRARY
  NAMES libfftw3f-3 libfftw3f fftw3f
  HINTS "${FFTW3_ROOT}" ENV FFTW3_ROOT
  PATH_SUFFIXES lib)
find_library(FFTW3L_LIBRARY
  NAMES libfftw3l-3 libfftw3l fftw3l
  HINTS "${FFTW3_ROOT}" ENV FFTW3_ROOT
  PATH_SUFFIXES lib)
if(WIN32)
  find_file(FFTW3_DLL
    NAMES libfftw3-3.dll libfftw3.dll fftw3.dll
    HINTS "${FFTW3_ROOT}" ENV FFTW3_ROOT
    PATH_SUFFIXES bin)
  find_file(FFTW3F_DLL
    NAMES libfftw3f-3.dll libfftw3f.dll fftw3f.dll
    HINTS "${FFTW3_ROOT}" ENV FFTW3_ROOT
    PATH_SUFFIXES bin)
  find_file(FFTW3L_DLL
    NAMES libfftw3l-3.dll libfftw3l.dll fftw3l.dll
    HINTS "${FFTW3_ROOT}" ENV FFTW3_ROOT
    PATH_SUFFIXES bin)
endif()

if(NOT FFTW3_INCLUDE)
  message(FATAL_ERROR "Could not find FFTW3_INCLUDE")
endif()
if(NOT FFTW3_LIBRARY)
  message(FATAL_ERROR "Could not find FFTW3_LIBRARY")
endif()
if(NOT FFTW3F_LIBRARY)
  message(FATAL_ERROR "Could not find FFTW3F_LIBRARY (single precision)")
endif()
if(NOT FFTW3L_LIBRARY)
  message(FATAL_ERROR "Could not find FFTW3L_LIBRARY (long double)")
endif()
if(WIN32 AND NOT FFTW3_DLL)
  message(FATAL_ERROR "Could not find FFTW3_DLL")
endif()
if(WIN32 AND NOT FFTW3F_DLL)
  message(FATAL_ERROR "Could not find FFTW3F_DLL (single precision)")
endif()
if(WIN32 AND NOT FFTW3L_DLL)
  message(FATAL_ERROR "Could not find FFTW3L_DLL (long double)")
endif()

if(WIN32)
  add_library(FFTW3::fftw SHARED IMPORTED)
  set_target_properties(FFTW3::fftw PROPERTIES
    IMPORTED_IMPLIB "${FFTW3_LIBRARY}"
    IMPORTED_LOCATION "${FFTW3_DLL}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE}")

  add_library(FFTW3::fftwf SHARED IMPORTED)
  set_target_properties(FFTW3::fftwf PROPERTIES
    IMPORTED_IMPLIB "${FFTW3F_LIBRARY}"
    IMPORTED_LOCATION "${FFTW3F_DLL}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE}")

  add_library(FFTW3::fftwl SHARED IMPORTED)
  set_target_properties(FFTW3::fftwl PROPERTIES
    IMPORTED_IMPLIB "${FFTW3L_LIBRARY}"
    IMPORTED_LOCATION "${FFTW3L_DLL}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE}")
else()
  add_library(FFTW3::fftw UNKNOWN IMPORTED)
  set_target_properties(FFTW3::fftw PROPERTIES
    IMPORTED_LOCATION "${FFTW3_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE}")

  add_library(FFTW3::fftwf UNKNOWN IMPORTED)
  set_target_properties(FFTW3::fftwf PROPERTIES
    IMPORTED_LOCATION "${FFTW3F_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE}")

  add_library(FFTW3::fftwl UNKNOWN IMPORTED)
  set_target_properties(FFTW3::fftwl PROPERTIES
    IMPORTED_LOCATION "${FFTW3L_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE}")
endif()

# ---- -CUDA --------------------------------------------------------------------
# set(CMAKE_CUDA_ARCHITECTURES native)
# find_package(CUDAToolkit REQUIRED)

# ---- CTest --------------------------------------------------------------------
include(CTest)
set(CMAKE_CTEST_VERBOSE FALSE)
set(CTEST_OUTPUT_ON_FAILURE FALSE)
enable_testing()

# ---- Subdirectories -----------------------------------------------------------
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(bench)
