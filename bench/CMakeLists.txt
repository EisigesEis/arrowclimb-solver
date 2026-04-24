add_executable(bench_big_odometer big_odometer.cpp)
target_link_libraries(bench_big_odometer PRIVATE myapp_lib)
add_executable(bench_small_dp small_dp.cpp)
target_link_libraries(bench_small_dp PRIVATE myapp_lib)
add_executable(bench_small_dp_build small_dp_build.cpp)
target_link_libraries(bench_small_dp_build PRIVATE myapp_lib)
add_executable(bench_gupta_sched gupta_sched.cpp)
target_link_libraries(bench_gupta_sched PRIVATE myapp_lib)
# add_executable(bench_discr_helper discr_helper.cpp)
# target_link_libraries(bench_discr_helper PRIVATE myapp_lib absl::flat_hash_set
#     absl::inlined_vector)
# add_executable(bench_discr_merge discr_merge.cpp)
# target_link_libraries(bench_discr_merge PRIVATE myapp_lib absl::flat_hash_set
#     absl::inlined_vector)
