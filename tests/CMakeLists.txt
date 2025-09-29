file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")

add_executable(gtests ${TEST_SOURCES})

target_include_directories(gtests
  PRIVATE
    "${CMAKE_SOURCE_DIR}/src/include"
    "${INCLUDE_GLOBAL}"
)

target_link_libraries(gtests
  PRIVATE
    myapp_lib
    GTest::gtest_main
    # FFTW3::fftw
    # absl::base
    # absl::strings
    # absl::hash
    # absl::raw_hash_set
    # absl::flat_hash_map
)

include(GoogleTest)
gtest_discover_tests(gtests)