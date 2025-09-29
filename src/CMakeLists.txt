# src/CMakeLists.txt
# 1) Gather library sources
file(GLOB_RECURSE LIB_SOURCES
     CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cu")
# Remove main.cpp explicitly
list(FILTER LIB_SOURCES EXCLUDE REGEX "main.cpp")

add_library(myapp_lib ${LIB_SOURCES})

# 2) Project headers + global include
set(INCLUDE_GLOBAL "C:/Users/User/include" CACHE STRING "Global include path")
target_include_directories(myapp_lib
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${INCLUDE_GLOBAL}"
    "${GUROBI_INCLUDE_DIRS}"
)

# 3) Link libraries
target_link_libraries(myapp_lib
  PRIVATE
    ${GUROBI_LIBRARY}
    ${GUROBI_CXX_LIBRARY}
    # FFTW3::fftw
    # FFTW3::fftwf
    # CUDA::cufft
    # absl::int128          # uint128 ops (/,%)
    # absl::base
    # absl::strings
    # absl::hash
    # absl::raw_hash_set
    # absl::flat_hash_map
)

# 3.1) Link temperature libraries
if (WIN32)
  target_compile_definitions(myapp_lib PRIVATE _WIN32_DCOM UNICODE _UNICODE)
  target_link_libraries(myapp_lib PRIVATE ole32 oleaut32 wbemuuid)
endif()

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
add_executable(uniformsched "${CMAKE_CURRENT_SOURCE_DIR}/lib/main.cpp")
target_link_libraries(uniformsched PRIVATE myapp_lib)