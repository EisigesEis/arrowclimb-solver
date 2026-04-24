# src/CMakeLists.txt
# 1) Gather library sources
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*/_deprecated/.*")
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")

add_library(myapp_lib ${LIB_SOURCES})

# 2) Project headers + optional local include overrides
set(UNIFORMSCHED_EXTRA_INCLUDE_DIRS "" CACHE STRING
    "Optional semicolon-separated include directories for local experiments.")
target_include_directories(myapp_lib
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${GUROBI_INCLUDE_DIRS}"
)

if(UNIFORMSCHED_EXTRA_INCLUDE_DIRS)
  target_include_directories(myapp_lib SYSTEM PUBLIC
    ${UNIFORMSCHED_EXTRA_INCLUDE_DIRS})
endif()

# 3) Link libraries
target_link_libraries(myapp_lib
  PRIVATE
    ${GUROBI_LIBRARY}
    ${GUROBI_CXX_LIBRARY}
  PUBLIC
    Eigen3::Eigen
    FFTW3::fftw
    FFTW3::fftwf
    spdlog::spdlog
    # CUDA::cufft
    # absl::int128          # uint128 ops (/,%)
    # absl::base
    # absl::strings
    # absl::hash
    # absl::raw_hash_set
    # absl::flat_hash_map
    absl::flat_hash_set
    absl::inlined_vector
)

# 3.1) Link temperature libraries
if (WIN32)
  target_compile_definitions(myapp_lib
    PUBLIC
      NOMINMAX
      WIN32_LEAN_AND_MEAN
    PRIVATE
      _WIN32_DCOM
      UNICODE
      _UNICODE
  )
  target_link_libraries(myapp_lib PRIVATE ole32 oleaut32 wbemuuid)
endif()

# if(ENABLE_CLANG_TIDY AND CLANG_TIDY_EXE)
#   set_target_properties(myapp_lib PROPERTIES
#     CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-extra-arg=-std=c++20"
#   )
# endif()

# 3.1) Fix cuda warnings
# target_compile_options(myapp_lib PRIVATE
#   $<$<COMPILE_LANGUAGE:CXX>:/std:c++20 /utf-8 /EHsc /GR /O2 /Ob1 /bigobj>
# )
# target_compile_options(myapp_lib PRIVATE
#   $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20 --expt-relaxed-constexpr -diag-suppress=20012 -Xcompiler=/utf-8>
# )
# target_compile_definitions(myapp_lib PRIVATE
#   $<$<COMPILE_LANGUAGE:CUDA>:BOOST_NO_CUDA>
# )

# 4) Executable
add_executable(uniformsched "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(uniformsched PRIVATE myapp_lib)
